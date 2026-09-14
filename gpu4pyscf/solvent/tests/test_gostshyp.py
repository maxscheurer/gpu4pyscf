# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for GPU GOSTSHYP implementation.

These tests verify that the GPU GOSTSHYP kernel produces numerically
correct results by comparing against the CPU reference implementation
from pyscf-forge (pyscf.solvent.gostshyp).
"""

import unittest
from unittest import mock
import numpy as np
import cupy as cp
from pyscf import gto, scf as cpu_scf
from pyscf.solvent.gostshyp import GOSTSHYP as GOSTSHYP_CPU

from gpu4pyscf import scf
from gpu4pyscf.solvent import _attach_solvent
from gpu4pyscf.solvent.gostshyp import GOSTSHYP
from gpu4pyscf.solvent.moist import HAS_MOIST


def cpu_reference(mol, dm, options=None):
    """Run pyscf-forge GOSTSHYP on CPU as reference.

    Returns the CPU GOSTSHYP object with all attributes populated
    (.e, .v, .forces, .amplitudes, .gtilde_expval).
    """
    opts = {**(options or {}), 'direct': False}
    cpu = GOSTSHYP_CPU(mol, options=opts)
    cpu.build()
    cpu.kernel(dm)
    return cpu


def converged_dm(mol):
    """Get a converged SCF density matrix for testing.

    Uses a physical DM to ensure all GOSTSHYP forces are positive,
    avoiding differences in negative-amplitude handling between
    GPU and CPU implementations.
    """
    mf = cpu_scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.make_rdm1()


class TestGOSTSHYPKernel(unittest.TestCase):
    """Test the GOSTSHYP kernel() method against CPU reference."""

    def setUp(self):
        """Create test molecules and converged density matrices."""
        self.mol_hf = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )
        self.dm_hf = converged_dm(self.mol_hf)

        self.mol_water = gto.M(
            atom='''
            O  0.0000000000    -0.0000000000     0.1174000000
            H -0.7570000000    -0.0000000000    -0.4696000000
            H  0.7570000000     0.0000000000    -0.4696000000
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )

    def test_energy_hf(self):
        """Test GOSTSHYP energy matches CPU reference for HF molecule."""
        gostshyp_gpu = GOSTSHYP(self.mol_hf, options={'cavity': 'vdw'})
        gostshyp_gpu.build()

        dm = self.dm_hf
        energy_gpu, fock_gpu = gostshyp_gpu.kernel(dm)
        cpu = cpu_reference(self.mol_hf, dm, options={'cavity': 'vdw'})

        if hasattr(fock_gpu, 'get'):
            fock_gpu = fock_gpu.get()

        np.testing.assert_allclose(energy_gpu, cpu.e, atol=1e-7, rtol=1e-7,
                                   err_msg="Energy mismatch between GPU and CPU")
        np.testing.assert_allclose(fock_gpu, cpu.v, atol=1e-7, rtol=1e-7,
                                   err_msg="Fock matrix mismatch between GPU and CPU")

    def test_forces_hf(self):
        """Test GOSTSHYP forces match CPU reference."""
        gostshyp_gpu = GOSTSHYP(self.mol_hf, options={'cavity': 'vdw'})
        gostshyp_gpu.build()

        dm = self.dm_hf
        gostshyp_gpu.kernel(dm)
        forces_gpu = cp.asnumpy(gostshyp_gpu.forces)

        cpu = cpu_reference(self.mol_hf, dm, options={'cavity': 'vdw'})

        np.testing.assert_allclose(forces_gpu, cpu.forces, atol=1e-9, rtol=1e-9,
                                   err_msg="Forces mismatch between GPU and CPU")

    def test_amplitudes_hf(self):
        """Test GOSTSHYP amplitudes match CPU reference."""
        gostshyp_gpu = GOSTSHYP(self.mol_hf, options={'cavity': 'vdw'})
        gostshyp_gpu.build()

        dm = self.dm_hf
        gostshyp_gpu.kernel(dm)
        amplitudes_gpu = cp.asnumpy(gostshyp_gpu.amplitudes)

        cpu = cpu_reference(self.mol_hf, dm, options={'cavity': 'vdw'})

        np.testing.assert_allclose(amplitudes_gpu, cpu.amplitudes, atol=1e-7, rtol=1e-7,
                                   err_msg="Amplitudes mismatch between GPU and CPU")

    def test_spherical_basis(self):
        """Test GOSTSHYP with spherical harmonics basis."""
        mol_sph = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=False,
            verbose=0
        )
        dm = converged_dm(mol_sph)

        gostshyp_gpu = GOSTSHYP(mol_sph, options={'cavity': 'vdw'})
        gostshyp_gpu.build()

        energy_gpu, fock_gpu = gostshyp_gpu.kernel(dm)
        cpu = cpu_reference(mol_sph, dm, options={'cavity': 'vdw'})

        if hasattr(fock_gpu, 'get'):
            fock_gpu = fock_gpu.get()

        np.testing.assert_allclose(energy_gpu, cpu.e, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(fock_gpu, cpu.v, atol=1e-7, rtol=1e-7)


class TestGOSTSHYPSCF(unittest.TestCase):
    """Test GOSTSHYP integration with SCF methods."""

    def test_rhf_scf_convergence(self):
        """Test RHF+GOSTSHYP SCF converges and gives correct energy."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10

        gostshyp = GOSTSHYP(mol, options={'cavity': 'vdw'})
        mf = _attach_solvent._for_scf(mf, gostshyp)

        e_tot = mf.kernel()

        self.assertTrue(mf.converged, "SCF did not converge")

        # Reference energy: HF/6-31g with GOSTSHYP 50GPa, vdw cavity
        np.testing.assert_allclose(e_tot, -99.8941733641653, atol=1e-7,
                                   err_msg="SCF energy does not match reference")

    def test_uhf_scf(self):
        """Test UHF+GOSTSHYP SCF works correctly."""
        mol = gto.M(
            atom='H 0 0 0; H 1 0 0',
            basis='sto-3g',
            spin=0,
            verbose=0
        )

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-10

        gostshyp = GOSTSHYP(mol)
        mf = _attach_solvent._for_scf(mf, gostshyp)

        e_tot = mf.kernel()
        self.assertTrue(mf.converged, "UHF+GOSTSHYP did not converge")

    def test_custom_cutoff(self):
        """Test GOSTSHYP with a custom overlap_cutoff converges and gives reasonable energy."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )

        # Run with default cutoff
        mf_default = scf.RHF(mol)
        mf_default.conv_tol = 1e-10
        sol_default = GOSTSHYP(mol)
        mf_default = _attach_solvent._for_scf(mf_default, sol_default)
        e_default = mf_default.kernel()
        self.assertTrue(mf_default.converged, "SCF with default cutoff did not converge")

        # Run with tighter cutoff (1e-14)
        mf_tight = scf.RHF(mol)
        mf_tight.conv_tol = 1e-10
        sol_tight = GOSTSHYP(mol)
        sol_tight.overlap_cutoff = 1e-14
        mf_tight = _attach_solvent._for_scf(mf_tight, sol_tight)
        e_tight = mf_tight.kernel()
        self.assertTrue(mf_tight.converged, "SCF with cutoff=1e-14 did not converge")

        # Energies should be very close (cutoff only affects negligible integrals)
        np.testing.assert_allclose(e_tight, e_default, atol=1e-7,
                                   err_msg="Energy with cutoff=1e-14 differs from default")

    def test_different_pressure(self):
        """Test GOSTSHYP with different pressure values."""
        mol = gto.M(
            atom='''
            O  0.0000000000    -0.0000000000     0.1174000000
            H -0.7570000000    -0.0000000000    -0.4696000000
            H  0.7570000000     0.0000000000    -0.4696000000
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )
        dm = converged_dm(mol)

        energies = []
        for pressure in [10000, 50000, 100000]:
            gostshyp = GOSTSHYP(mol, options={'pressure_mpa': pressure})
            gostshyp.build()

            energy, fock = gostshyp.kernel(dm)
            energies.append(energy)

            if hasattr(fock, 'get'):
                fock = fock.get()

            # Check Fock matrix is symmetric
            np.testing.assert_allclose(fock, fock.T, atol=1e-10)

        # Higher pressure should give larger magnitude energy
        self.assertLess(abs(energies[0]), abs(energies[1]))
        self.assertLess(abs(energies[1]), abs(energies[2]))


class TestGOSTSHYPBuild(unittest.TestCase):
    """Test GOSTSHYP build() and surface construction."""

    def test_surface_generation(self):
        """Test surface is generated correctly."""
        mol = gto.M(
            atom='''
            O  0.0000000000    -0.0000000000     0.1174000000
            H -0.7570000000    -0.0000000000    -0.4696000000
            H  0.7570000000     0.0000000000    -0.4696000000
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )

        gostshyp = GOSTSHYP(mol)
        gostshyp.build()

        self.assertGreater(gostshyp.n_gaussian, 0)
        self.assertEqual(len(gostshyp.areas), gostshyp.n_gaussian)
        self.assertEqual(len(gostshyp.widths), gostshyp.n_gaussian)
        self.assertEqual(gostshyp.grid_coords.shape, (gostshyp.n_gaussian, 3))
        self.assertEqual(gostshyp.surface_normals.shape, (gostshyp.n_gaussian, 3))

    def test_surface_normals_normalized(self):
        """Test surface normals are unit vectors."""
        mol = gto.M(
            atom='''
            O  0.0000000000    -0.0000000000     0.1174000000
            H -0.7570000000    -0.0000000000    -0.4696000000
            H  0.7570000000     0.0000000000    -0.4696000000
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )

        gostshyp = GOSTSHYP(mol)
        gostshyp.build()

        norms = cp.linalg.norm(gostshyp.surface_normals, axis=1)
        np.testing.assert_allclose(cp.asnumpy(norms), 1.0, atol=1e-10)

    def test_widths_positive(self):
        """Test Gaussian widths are positive."""
        mol = gto.M(
            atom='''
            O  0.0000000000    -0.0000000000     0.1174000000
            H -0.7570000000    -0.0000000000    -0.4696000000
            H  0.7570000000     0.0000000000    -0.4696000000
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )

        gostshyp = GOSTSHYP(mol)
        gostshyp.build()

        self.assertTrue(np.all(gostshyp.widths > 0))

    def test_invalid_cavity(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        with self.assertRaisesRegex(ValueError, 'cavity must be'):
            GOSTSHYP(mol, options={'cavity': 'unknown'}).build()

    def test_reset(self):
        """Test reset() clears cached data."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )

        gostshyp = GOSTSHYP(mol)
        gostshyp.build()

        mol_new = gto.M(
            atom='H 0 0 0; H 1.5 0 0',
            basis='sto-3g',
            verbose=0
        )

        gostshyp.reset(mol_new)

        self.assertEqual(gostshyp.mol, mol_new)
        self.assertIsNone(gostshyp._gtilde)
        self.assertIsNone(gostshyp._force_operators)


class TestGOSTSHYPGradient(unittest.TestCase):
    """Test GOSTSHYP gradient against CPU reference."""

    def setUp(self):
        self.mol_hf = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )
        self.dm_hf = converged_dm(self.mol_hf)

        self.mol_water = gto.M(
            atom='''
            O  0.0000  0.0000  0.1174
            H -0.7570  0.0000 -0.4696
            H  0.7570  0.0000 -0.4696
            ''',
            basis='sto-3g',
            cart=True,
            verbose=0
        )
        self.dm_water = converged_dm(self.mol_water)

    def _compute_gpu_gradient(self, mol, dm, cavity=None):
        from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients
        opts = {}
        if cavity is not None:
            opts['cavity'] = cavity
        gostshyp = GOSTSHYP(mol, options=opts)
        gostshyp.build()
        gostshyp.kernel(dm)
        return GOSTSHYPGradients(gostshyp).kernel(dm)

    def test_gradient_hf_cart(self):
        """Test gradient matches CPU for HF/6-31g/cart."""
        dm = self.dm_hf

        grad_gpu = self._compute_gpu_gradient(self.mol_hf, dm, cavity='vdw')
        cpu = cpu_reference(self.mol_hf, dm, options={'cavity': 'vdw'})
        grad_cpu = cpu.grad(dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (HF/6-31g/cart)")

    def test_gradient_hf_sph(self):
        """Test gradient matches CPU for HF/6-31g/sph."""
        mol_sph = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=False,
            verbose=0
        )
        dm = converged_dm(mol_sph)

        grad_gpu = self._compute_gpu_gradient(mol_sph, dm, cavity='vdw')
        cpu = cpu_reference(mol_sph, dm, options={'cavity': 'vdw'})
        grad_cpu = cpu.grad(dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (HF/6-31g/sph)")

    def test_gradient_water_cart(self):
        """Test gradient matches CPU for H2O/sto-3g/cart."""
        dm = self.dm_water

        grad_gpu = self._compute_gpu_gradient(self.mol_water, dm, cavity='vdw')
        cpu = cpu_reference(self.mol_water, dm, options={'cavity': 'vdw'})
        grad_cpu = cpu.grad(dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (H2O/sto-3g/cart)")

    def test_gradient_ccpvdz_sph(self):
        """Test gradient with cc-pvdz (d functions, spherical)."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='cc-pvdz',
            cart=False,
            verbose=0
        )
        dm = converged_dm(mol)

        grad_gpu = self._compute_gpu_gradient(mol, dm, cavity='vdw')
        cpu = cpu_reference(mol, dm, options={'cavity': 'vdw'})
        grad_cpu = cpu.grad(dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (HF/cc-pvdz/sph)")

    def test_gradient_translational_invariance(self):
        """Gradient should sum to zero over all atoms (Newton's third law)."""
        dm = self.dm_water

        grad = self._compute_gpu_gradient(self.mol_water, dm)
        total_force = np.sum(grad, axis=0)

        np.testing.assert_allclose(total_force, 0.0, atol=1e-7,
                                   err_msg="Gradient does not satisfy translational invariance")

    def test_gradient_scf_hf_reference(self):
        """Test GOSTSHYP gradient after SCF against validated reference."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        gostshyp = GOSTSHYP(mol, options={'cavity': 'vdw'})
        mf = _attach_solvent._for_scf(mf, gostshyp)
        mf.kernel()

        from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients
        grad = GOSTSHYPGradients(gostshyp).kernel(mf.make_rdm1())

        ref = np.array([
            [-0.00951946951, 0.00951946951],
            [ 0.00000000000, 0.00000000000],
            [ 0.00000000000, 0.00000000000],
        ]).T
        np.testing.assert_allclose(grad, ref, atol=1e-7)

    def test_nuc_grad_method(self):
        """Test mf.nuc_grad_method().kernel() returns combined solute+solvent gradient."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        gostshyp = GOSTSHYP(mol)
        mf = _attach_solvent._for_scf(mf, gostshyp)
        mf.kernel()

        grad_obj = mf.nuc_grad_method()
        de = grad_obj.kernel()
        self.assertEqual(de.shape, (mol.natm, 3))

        # The combined gradient should include both solute and solvent parts
        self.assertIsNotNone(grad_obj.de_solvent)
        self.assertIsNotNone(grad_obj.de_solute)
        np.testing.assert_allclose(de, grad_obj.de_solute + grad_obj.de_solvent, atol=1e-10)

        # Cross-check: solvent part should match a direct call
        dm = mf.make_rdm1()
        if hasattr(dm, 'get'):
            dm = dm.get()
        from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients
        de_solvent_direct = GOSTSHYPGradients(gostshyp).kernel(dm)
        np.testing.assert_allclose(grad_obj.de_solvent, de_solvent_direct, atol=1e-10)

    def test_hessian_not_implemented(self):
        """Test mf.Hessian() raises NotImplementedError for GOSTSHYP."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='sto-3g',
            verbose=0,
        )
        mf = scf.RHF(mol)
        gostshyp = GOSTSHYP(mol)
        mf = _attach_solvent._for_scf(mf, gostshyp)
        with self.assertRaises(NotImplementedError):
            mf.Hessian()

    def test_gradient_finite_diff_convergence(self):
        """Verify O(h^2) convergence of finite differences against analytical gradient."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='sto-3g',
            cart=True,
            verbose=0
        )
        dm = converged_dm(mol)

        grad_anal = self._compute_gpu_gradient(mol, dm)

        errors = []
        for h in [1e-3, 1e-4, 1e-5]:
            grad_fd = np.zeros_like(grad_anal)
            coords = mol.atom_coords().copy()
            for ia in range(mol.natm):
                for ix in range(3):
                    coords_p = coords.copy()
                    coords_p[ia, ix] += h
                    mol_p = mol.copy()
                    mol_p.set_geom_(coords_p, unit='Bohr')
                    mol_p.build()
                    gostshyp_p = GOSTSHYP(mol_p)
                    gostshyp_p.build()
                    e_p, _ = gostshyp_p.kernel(dm)

                    coords_m = coords.copy()
                    coords_m[ia, ix] -= h
                    mol_m = mol.copy()
                    mol_m.set_geom_(coords_m, unit='Bohr')
                    mol_m.build()
                    gostshyp_m = GOSTSHYP(mol_m)
                    gostshyp_m.build()
                    e_m, _ = gostshyp_m.kernel(dm)

                    grad_fd[ia, ix] = (e_p - e_m) / (2 * h)

            errors.append(np.max(np.abs(grad_anal - grad_fd)))

        # Check O(h^2) convergence: error ratio should be ~100 for 10x h reduction
        # The second ratio can be lower due to floating-point noise at small h
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        self.assertGreater(ratio1, 50, "FD convergence not O(h^2)")
        self.assertGreater(ratio2, 10, "FD convergence not O(h^2)")


class TestMOISTAdapter(unittest.TestCase):
    def test_missing_moist_error_from_public_build(self):
        from gpu4pyscf.solvent import moist as moist_adapter
        mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        with mock.patch.object(moist_adapter, 'HAS_MOIST', False), \
             mock.patch.object(moist_adapter, '_MOIST_IMPORT_ERROR', None):
            with self.assertRaisesRegex(ImportError, 'gpu4pyscf\\[moist\\]'):
                GOSTSHYP(mol, options={'cavity': 'drop'}).build()

    def test_broken_moist_import_preserves_cause(self):
        from gpu4pyscf.solvent import moist as moist_adapter
        cause = ImportError('cannot load libmoist.so')
        with mock.patch.object(moist_adapter, 'HAS_MOIST', False), \
             mock.patch.object(moist_adapter, '_MOIST_IMPORT_ERROR', cause):
            with self.assertRaisesRegex(ImportError, 'could not be imported') as ctx:
                moist_adapter._require_moist()
        self.assertIs(ctx.exception.__cause__, cause)


@unittest.skipUnless(HAS_MOIST, 'MOIST library not available')
class TestDROPCavity(unittest.TestCase):
    def setUp(self):
        self.options = {'cavity': 'drop', 'npoints': 26}
        self.mol = gto.M(
            atom='H 0 0 0; F 0 0 1', basis='sto-3g', cart=True,
            verbose=0)
        self.dm = converged_dm(self.mol)

    def test_surface_energy_fock_and_fused_kernels(self):
        from gpu4pyscf.gto import int3c_overlap

        gost = GOSTSHYP(self.mol, options=self.options).build()
        self.assertGreater(gost.n_gaussian, 0)
        self.assertTrue(bool(cp.all(gost.areas > 0)))
        self.assertEqual(gost.grid_coords.shape, (gost.n_gaussian, 3))
        self.assertEqual(gost.atom_idx.shape, (gost.n_gaussian,))
        np.testing.assert_allclose(
            cp.asnumpy(cp.linalg.norm(gost.surface_normals, axis=1)),
            1.0, atol=1e-12)

        with mock.patch.object(
                int3c_overlap, 'get_int3c_overlap_density_contracted_sp',
                wraps=int3c_overlap.get_int3c_overlap_density_contracted_sp
        ) as density_sp, mock.patch.object(
                int3c_overlap, 'get_int3c_overlap_amplitude_contracted_sp',
                wraps=int3c_overlap.get_int3c_overlap_amplitude_contracted_sp
        ) as amplitude_sp:
            energy, fock = gost.kernel(self.dm)

        self.assertEqual(density_sp.call_count, 1)
        self.assertEqual(amplitude_sp.call_count, 1)
        self.assertEqual(density_sp.call_args.args[1].shape,
                         (gost.n_gaussian, 3))
        self.assertEqual(amplitude_sp.call_args.kwargs['amp_p'].shape,
                         (gost.n_gaussian, 3))

        cpu = cpu_reference(self.mol, self.dm, options=self.options)
        np.testing.assert_allclose(energy, cpu.e, atol=1e-7, rtol=1e-7)
        fock = cp.asnumpy(fock)
        np.testing.assert_allclose(fock, cpu.v, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(fock, fock.T, atol=1e-12)
        np.testing.assert_allclose(
            cp.asnumpy(gost.forces), cpu.forces, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(
            cp.asnumpy(gost.amplitudes), cpu.amplitudes,
            atol=1e-7, rtol=1e-7)

    def test_attached_scf_and_frozen_reset(self):
        gost = GOSTSHYP(self.mol, options=self.options)
        mf = scf.RHF(self.mol)
        mf.conv_tol = 1e-9
        mf = _attach_solvent._for_scf(mf, gost)
        mf.kernel()
        self.assertTrue(mf.converged)
        np.testing.assert_allclose(
            cp.asnumpy(gost.v), cp.asnumpy(gost.v).T, atol=1e-12)

        frozen = GOSTSHYP(self.mol, options=self.options)
        frozen_mf = _attach_solvent._for_scf(
            scf.RHF(self.mol), frozen, dm=self.dm)
        energy_before = frozen.e
        potential_before = frozen.v.copy()
        frozen_mf.reset(self.mol)
        self.assertEqual(frozen.e, energy_before)
        np.testing.assert_allclose(
            cp.asnumpy(frozen.v), cp.asnumpy(potential_before), atol=0.0)
        veff = frozen_mf.get_veff(self.mol, cp.asarray(self.dm))
        self.assertEqual(veff.v_solvent.shape, potential_before.shape)

    def test_to_cpu_direct_and_attached(self):
        gost = GOSTSHYP(self.mol, options=self.options).build()
        energy_gpu, _ = gost.kernel(self.dm)
        cpu = gost.to_cpu()
        self.assertEqual(cpu.cavity, 'drop')
        self.assertEqual(cpu._drop_kwargs, {})
        energy_cpu, _ = cpu.kernel(self.dm)
        np.testing.assert_allclose(energy_cpu, energy_gpu, atol=1e-7)

        attached = _attach_solvent._for_scf(
            scf.RHF(self.mol), gost, dm=self.dm)
        attached_cpu = attached.to_cpu()
        self.assertEqual(attached_cpu.with_solvent.cavity, 'drop')
        self.assertTrue(attached_cpu.with_solvent.frozen)
        attached_cpu.get_veff(self.mol, self.dm)

    def test_moist_work_is_cached_outside_kernel(self):
        from gpu4pyscf.solvent import moist as moist_adapter
        from gpu4pyscf.solvent.grad.gostshyp import _get_surface_derivatives

        gost = GOSTSHYP(self.mol, options=self.options).build()
        with mock.patch.object(
                moist_adapter, 'get_anchor_gradient',
                wraps=moist_adapter.get_anchor_gradient) as anchor_gradient:
            gost.kernel(self.dm)
            gost.kernel(self.dm)
            self.assertEqual(anchor_gradient.call_count, 0)

            first = _get_surface_derivatives(gost)
            second = _get_surface_derivatives(gost)
            self.assertIs(first, second)
            self.assertEqual(anchor_gradient.call_count, 1)

            gost.reset()
            _get_surface_derivatives(gost)
            self.assertEqual(anchor_gradient.call_count, 2)


@unittest.skipUnless(HAS_MOIST, 'MOIST library not available')
class TestDROPGradient(unittest.TestCase):
    def _gradient(self, mol, dm):
        from gpu4pyscf.solvent.grad.gostshyp import Gradients
        gost = GOSTSHYP(
            mol, options={'cavity': 'drop', 'npoints': 26}).build()
        gost.kernel(dm)
        return gost, Gradients(gost).kernel(dm)

    def _finite_difference(self, mol, dm, step=1e-3):
        coords = mol.atom_coords().copy()
        gradient = np.zeros((mol.natm, 3))
        for atom in range(mol.natm):
            for axis in range(3):
                energies = []
                for sign in (1.0, -1.0):
                    displaced = mol.copy()
                    displaced_coords = coords.copy()
                    displaced_coords[atom, axis] += sign * step
                    displaced.set_geom_(displaced_coords, unit='Bohr')
                    displaced.build()
                    gost = GOSTSHYP(
                        displaced,
                        options={'cavity': 'drop', 'npoints': 26}).build()
                    energies.append(gost.kernel(dm)[0])
                gradient[atom, axis] = (
                    energies[0] - energies[1]) / (2.0 * step)
        return gradient

    def test_gradient_matches_cpu_and_kernel_sequence(self):
        from gpu4pyscf.gto import int3c_overlap

        mol = gto.M(
            atom='H 0 0 0; F 0 0 1', basis='sto-3g', cart=True,
            verbose=0)
        dm = converged_dm(mol)
        with mock.patch.object(
                int3c_overlap, 'get_int3c_overlap_density_contracted',
                wraps=int3c_overlap.get_int3c_overlap_density_contracted
        ) as density_contracted:
            _, grad_gpu = self._gradient(mol, dm)

        aux_ls = [call.kwargs['aux_l']
                  for call in density_contracted.call_args_list]
        self.assertEqual(aux_ls, [1, 2, 0, 3])

        cpu = cpu_reference(
            mol, dm, options={'cavity': 'drop', 'npoints': 26})
        np.testing.assert_allclose(
            grad_gpu, cpu.grad(dm), atol=1e-6, rtol=1e-6)

    def test_gradient_water_and_translational_invariance(self):
        mol = gto.M(
            atom='O 0 0 0.1174; H -0.757 0 -0.4696; '
                 'H 0.757 0 -0.4696',
            basis='sto-3g', cart=True, verbose=0)
        dm = converged_dm(mol)
        _, grad_gpu = self._gradient(mol, dm)
        cpu = cpu_reference(
            mol, dm, options={'cavity': 'drop', 'npoints': 26})
        np.testing.assert_allclose(
            grad_gpu, cpu.grad(dm), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(grad_gpu.sum(axis=0), 0.0, atol=1e-7)

    def test_gradient_spherical_basis(self):
        mol = gto.M(
            atom='H 0 0 0; F 0 0 1', basis='cc-pvdz', cart=False,
            verbose=0)
        self.assertNotEqual(mol.nao_nr(), mol.nao_nr(cart=True))
        dm = converged_dm(mol)
        gost, grad_gpu = self._gradient(mol, dm)
        cpu = cpu_reference(
            mol, dm, options={'cavity': 'drop', 'npoints': 26})
        np.testing.assert_allclose(gost.e, cpu.e, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(
            cp.asnumpy(gost.v), cpu.v, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(
            grad_gpu, cpu.grad(dm), atol=1e-6, rtol=1e-6)

    def test_gradient_finite_difference(self):
        mol = gto.M(
            atom='H 0 0 0; F 0 0 1', basis='sto-3g', cart=True,
            verbose=0)
        dm = converged_dm(mol)
        _, analytic = self._gradient(mol, dm)
        np.testing.assert_allclose(
            analytic, self._finite_difference(mol, dm),
            atol=1e-5, rtol=1e-5)

    def test_polyatomic_gradient_finite_difference(self):
        mol = gto.M(
            atom='O 0 0 0.1174; H -0.757 0 -0.4696; '
                 'H 0.757 0 -0.4696',
            basis='sto-3g', cart=True, verbose=0)
        dm = converged_dm(mol)
        _, analytic = self._gradient(mol, dm)
        np.testing.assert_allclose(
            analytic, self._finite_difference(mol, dm),
            atol=1e-5, rtol=1e-5)

    def test_surface_response_components_are_active(self):
        from gpu4pyscf.solvent.grad.gostshyp import (
            _apply_grid_coordinate_response, _get_surface_derivatives)

        mol = gto.M(
            atom='O 0 0 0.1174; H -0.757 0 -0.4696; '
                 'H 0.757 0 -0.4696',
            basis='sto-3g', cart=True, verbose=0)
        gost = GOSTSHYP(
            mol, options={'cavity': 'drop', 'npoints': 26}).build()
        dareas, dcoords = _get_surface_derivatives(gost)
        self.assertGreater(float(cp.linalg.norm(dareas)), 0.0)
        self.assertGreater(float(np.linalg.norm(dcoords)), 0.0)

        rng = np.random.default_rng(2)
        shape = (gost.n_gaussian, 3)
        gtilde_grid = cp.asarray(rng.standard_normal(shape))
        force_grid = cp.asarray(rng.standard_normal(shape))
        p_contracted = cp.asarray(rng.standard_normal(shape))
        force_coeffs = cp.asarray(rng.standard_normal(gost.n_gaussian))
        gtilde_grad = cp.zeros((mol.natm, 3))
        force_grad = cp.zeros((mol.natm, 3))
        normal_grad = _apply_grid_coordinate_response(
            gost, gtilde_grad, force_grad, gtilde_grid, force_grid,
            p_contracted, force_coeffs, dcoords)

        self.assertGreater(float(cp.linalg.norm(gtilde_grad)), 0.0)
        self.assertGreater(float(cp.linalg.norm(force_grad)), 0.0)
        self.assertGreater(float(cp.linalg.norm(normal_grad)), 0.0)


if __name__ == "__main__":
    print("Full Tests for GPU GOSTSHYP")
    unittest.main()
