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
correct results by comparing against a CPU reference implementation.
"""

import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.solvent.pcm import gen_surface, modified_Bondi

from gpu4pyscf import scf
from gpu4pyscf.solvent import _attach_solvent
from gpu4pyscf.solvent.gostshyp import GOSTSHYP


def fakemol_for_gaussian(coords, exponents, l=0, cart=True, coeffs=None):
    """Create a fake molecule for auxiliary Gaussians (CPU reference)."""
    nbas = coords.shape[0]
    if coeffs is None:
        coeffs = np.ones_like(exponents)
    angmom = np.zeros_like(exponents)
    angmom[:] = l

    ang_norm = {
        0: 2.0 * np.sqrt(np.pi),
        1: 2.0 * np.sqrt(np.pi / 3),
        2: 1.0,
        3: 1.0,
    }

    fakeatm = np.zeros((nbas, gto.mole.ATM_SLOTS), dtype=np.int32)
    fakebas = np.zeros((nbas, gto.mole.BAS_SLOTS), dtype=np.int32)
    fakeenv = [0] * gto.mole.PTR_ENV_START
    ptr = gto.mole.PTR_ENV_START
    fakeatm[:, gto.mole.PTR_COORD] = np.arange(ptr, ptr + nbas * 3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas * 3
    fakebas[:, gto.mole.ATOM_OF] = np.arange(nbas)
    fakebas[:, gto.mole.ANG_OF] = angmom
    fakebas[:, gto.mole.NPRIM_OF] = 1
    fakebas[:, gto.mole.NCTR_OF] = 1
    fakebas[:, gto.mole.PTR_EXP] = ptr + np.arange(nbas) * 2
    fakebas[:, gto.mole.PTR_COEFF] = ptr + np.arange(nbas) * 2 + 1
    coeff = ang_norm[l] * coeffs
    fakeenv.append(np.vstack((exponents, coeff)).T.ravel())

    fakemol = gto.Mole()
    fakemol.cart = cart
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = np.hstack(fakeenv)
    fakemol._built = True
    return fakemol


def gostshyp_kernel_cpu_reference(mol, dm, pressure_mpa=50000, npoints=110, scaling_factor=1.2):
    """
    CPU reference implementation of GOSTSHYP kernel (integral-direct mode).

    This mirrors the direct=True mode from the reference CPU implementation.
    """
    pressure_au = pressure_mpa * 3.3989309735473356e-08

    # Build surface
    radii = scaling_factor * modified_Bondi
    surface_dict = gen_surface(mol, ng=npoints, rad=radii)
    areas = np.asarray(surface_dict['area'])
    grid_coords = np.asarray(surface_dict['grid_coords'])
    atom_coords = mol.atom_coords()
    gslice_by_atom = surface_dict['gslice_by_atom']

    # Compute atom index and surface normals
    atom_idx = np.zeros(len(areas), dtype=int)
    for ia, (p0, p1) in enumerate(gslice_by_atom):
        atom_idx[p0:p1] = ia

    ref_coords = atom_coords[atom_idx]
    dr = grid_coords - ref_coords
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    surface_normals = dr / dr_norm

    # Compute widths
    widths = np.pi * np.log(2) / areas
    nao = mol.nao
    n_gaussian = len(areas)

    # Build fakemols for integrals
    gmol = fakemol_for_gaussian(grid_coords, widths, l=0, cart=mol.cart)
    gmol_p = fakemol_for_gaussian(grid_coords, widths, l=1, cart=mol.cart,
                                   coeffs=2.0 * widths)
    supermol = mol + gmol
    supermol_p = mol + gmol_p

    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)

    # Compute s-type overlap integrals
    overlap3_s = supermol.intor("int3c1e", shls_slice=slices, aosym="s1")

    # Compute p-type overlap integrals
    naux_p = 3
    overlap3_p = supermol_p.intor("int3c1e", shls_slice=slices, aosym="s1")
    overlap3_p = overlap3_p.reshape(nao, nao, n_gaussian, naux_p)

    # Contract p-type with surface normals to get force operators
    force_operators = np.einsum('ijgc,gc->ijg', overlap3_p, surface_normals, optimize=True)

    # Compute forces
    forces = np.einsum('ij,ijg->g', dm, force_operators, optimize=True)

    # Compute amplitudes
    amplitudes = pressure_au * areas / forces

    # Compute Fock term 1
    fock1 = np.einsum('g,ijg->ij', amplitudes, overlap3_s, optimize=True)

    # Compute gtilde_expval
    gtilde_expval = np.einsum('ij,ijg->g', dm, overlap3_s, optimize=True)

    # Compute Fock term 2 (response)
    response_coeff = -pressure_au * areas * gtilde_expval / (forces ** 2)
    fock2 = np.einsum('g,ijg->ij', response_coeff, force_operators, optimize=True)

    fock = fock1 + fock2
    energy = np.vdot(fock1, dm)

    return energy, fock, forces, amplitudes


class TestGOSTSHYPKernel(unittest.TestCase):
    """Test the GOSTSHYP kernel() method against CPU reference."""

    def setUp(self):
        """Create test molecules."""
        self.mol_hf = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )
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
        gostshyp_gpu = GOSTSHYP(self.mol_hf)
        gostshyp_gpu.build()

        np.random.seed(42)
        nao = self.mol_hf.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        energy_gpu, fock_gpu = gostshyp_gpu.kernel(dm)
        energy_cpu, fock_cpu, _, _ = gostshyp_kernel_cpu_reference(self.mol_hf, dm)

        # Convert CuPy array to numpy if needed
        if hasattr(fock_gpu, 'get'):
            fock_gpu = fock_gpu.get()

        # Use appropriate tolerance for GPU vs CPU numerical differences
        np.testing.assert_allclose(energy_gpu, energy_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Energy mismatch between GPU and CPU")
        np.testing.assert_allclose(fock_gpu, fock_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Fock matrix mismatch between GPU and CPU")

    def test_forces_hf(self):
        """Test GOSTSHYP forces match CPU reference."""
        gostshyp_gpu = GOSTSHYP(self.mol_hf)
        gostshyp_gpu.build()

        np.random.seed(42)
        nao = self.mol_hf.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        gostshyp_gpu.kernel(dm)
        forces_gpu = cp.asnumpy(gostshyp_gpu.forces)

        _, _, forces_cpu, _ = gostshyp_kernel_cpu_reference(self.mol_hf, dm)

        np.testing.assert_allclose(forces_gpu, forces_cpu, atol=1e-9, rtol=1e-9,
                                   err_msg="Forces mismatch between GPU and CPU")

    def test_amplitudes_hf(self):
        """Test GOSTSHYP amplitudes match CPU reference."""
        gostshyp_gpu = GOSTSHYP(self.mol_hf)
        gostshyp_gpu.build()

        np.random.seed(42)
        nao = self.mol_hf.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        gostshyp_gpu.kernel(dm)
        amplitudes_gpu = cp.asnumpy(gostshyp_gpu.amplitudes)

        _, _, _, amplitudes_cpu = gostshyp_kernel_cpu_reference(self.mol_hf, dm)

        # Amplitudes = pressure * areas / forces
        # Small forces magnify numerical differences, so use relative tolerance
        # to handle both large and small amplitudes
        np.testing.assert_allclose(amplitudes_gpu, amplitudes_cpu, atol=1e-5, rtol=2e-3,
                                   err_msg="Amplitudes mismatch between GPU and CPU")

    def test_spherical_basis(self):
        """Test GOSTSHYP with spherical harmonics basis."""
        mol_sph = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=False,
            verbose=0
        )

        gostshyp_gpu = GOSTSHYP(mol_sph)
        gostshyp_gpu.build()

        np.random.seed(42)
        nao = mol_sph.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        energy_gpu, fock_gpu = gostshyp_gpu.kernel(dm)
        energy_cpu, fock_cpu, _, _ = gostshyp_kernel_cpu_reference(mol_sph, dm)

        # Convert CuPy array to numpy if needed
        if hasattr(fock_gpu, 'get'):
            fock_gpu = fock_gpu.get()

        np.testing.assert_allclose(energy_gpu, energy_cpu, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(fock_gpu, fock_cpu, atol=1e-7, rtol=1e-7)


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

        gostshyp = GOSTSHYP(mol)
        mf = _attach_solvent._for_scf(mf, gostshyp)

        e_tot = mf.kernel()

        self.assertTrue(mf.converged, "SCF did not converge")

        # Reference energy: HF/6-31g with GOSTSHYP 50GPa
        # Computed using both GPU and CPU implementations with proper energy accounting
        np.testing.assert_allclose(e_tot, -100.06337794338, atol=1e-7,
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

        energies = []
        for pressure in [10000, 50000, 100000]:
            gostshyp = GOSTSHYP(mol, options={'pressure_mpa': pressure})
            gostshyp.build()

            nao = mol.nao
            dm = np.eye(nao)

            energy, fock = gostshyp.kernel(dm)
            energies.append(energy)

            # Convert CuPy array to numpy if needed
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


def gostshyp_gradient_cpu_reference(mol, dm, pressure_mpa=50000, npoints=110, scaling_factor=1.2):
    """
    CPU reference gradient implementation using PySCF's libcint integrals.

    Returns the total gradient as well as per-term breakdown.
    """
    from pyscf.solvent.grad.pcm import get_dF_dA

    pressure_au = pressure_mpa * 3.3989309735473356e-08
    radii = scaling_factor * modified_Bondi
    surface_dict = gen_surface(mol, ng=npoints, rad=radii)
    areas = np.asarray(surface_dict['area'])
    grid_coords = np.asarray(surface_dict['grid_coords'])
    atom_coords = mol.atom_coords()
    gslice_by_atom = surface_dict['gslice_by_atom']

    atom_idx = np.zeros(len(areas), dtype=int)
    for ia, (p0, p1) in enumerate(gslice_by_atom):
        atom_idx[p0:p1] = ia

    ref_coords = atom_coords[atom_idx]
    dr = grid_coords - ref_coords
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    surface_normals = dr / dr_norm
    widths = np.pi * np.log(2) / areas
    nao = mol.nao
    n_gaussian = len(areas)

    # Energy quantities
    gmol = fakemol_for_gaussian(grid_coords, widths, l=0, cart=mol.cart)
    gmol_p = fakemol_for_gaussian(grid_coords, widths, l=1, cart=mol.cart,
                                   coeffs=2.0 * widths)
    supermol = mol + gmol
    supermol_p = mol + gmol_p
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)

    overlap3_s = supermol.intor("int3c1e", shls_slice=slices, aosym="s1")
    overlap3_p = supermol_p.intor("int3c1e", shls_slice=slices, aosym="s1")
    overlap3_p = overlap3_p.reshape(nao, nao, n_gaussian, 3)
    force_operators = np.einsum('ijgc,gc->ijg', overlap3_p, surface_normals, optimize=True)
    forces = np.einsum('ij,ijg->g', dm, force_operators, optimize=True)
    gtilde_expval = np.einsum('ij,ijg->g', dm, overlap3_s, optimize=True)
    amplitudes = pressure_au * areas / forces

    # Area derivatives
    surface_dict_np = {
        'grid_coords': grid_coords, 'area': areas,
        'gslice_by_atom': gslice_by_atom,
        'R_vdw': np.asarray(surface_dict['R_vdw']),
        'switch_fun': np.asarray(surface_dict['switch_fun']),
        'R_in_J': np.asarray(surface_dict['R_in_J']),
        'R_sw_J': np.asarray(surface_dict['R_sw_J']),
        'atom_coords': np.asarray(surface_dict['atom_coords']),
    }
    _, dareas = get_dF_dA(surface_dict_np)
    dareas = dareas.transpose(1, 2, 0)

    # dE1
    dE1 = pressure_au * np.einsum('acg,g->ac', dareas, gtilde_expval / forces, optimize=True)

    # dE2
    dPQ = supermol.intor("int3c1e_ip1", shls_slice=slices)
    dPQ = np.einsum('xijn,n->xij', dPQ, amplitudes, optimize=True)
    slices_g = (mol.nbas, mol.nbas + gmol.nbas, 0, mol.nbas, 0, mol.nbas)
    dG = supermol.intor("int3c1e_ip1", shls_slice=slices_g)
    aoslice = mol.aoslice_by_atom()
    dgtilde_braket = np.einsum('xij,ij->ix', dPQ, dm, optimize=True)
    dgtilde_braket += np.einsum('xij,ji->ix', dPQ, dm, optimize=True)
    dgtilde_gaussian = np.einsum('xnij,n,ij->nx', dG, amplitudes, dm, optimize=True)
    gtilde_operator_grad = np.asarray(
        [np.sum(dgtilde_braket[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
    np.add.at(gtilde_operator_grad, atom_idx, dgtilde_gaussian)
    gtilde_operator_grad *= -1.0

    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)
    gmol_d = fakemol_for_gaussian(grid_coords, widths, l=2,
                                   coeffs=wgrad_prefs * amplitudes, cart=True)
    supermol_d = mol + gmol_d
    supermol_d.cart = True
    slices_d = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_d.nbas)
    nao_cart = mol.nao_nr(cart=True)
    overlap3d = supermol_d.intor("int3c1e", shls_slice=slices_d).reshape(
        nao_cart, nao_cart, -1, 6)
    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized="sp")
        overlap3d = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s, optimize=True)
    diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
    imd = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
    dE_d = -1.0 * np.einsum('acg,g->ac', dareas, imd, optimize=True)
    dE2 = gtilde_operator_grad + dE_d

    # dE3
    coeffs = -2.0 * pressure_au * areas * gtilde_expval * widths / (forces * forces)
    gmol_p2 = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=coeffs)
    supermol_p2 = mol + gmol_p2
    slices_p2 = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p2.nbas)
    dpq = supermol_p2.intor("int3c1e_ip1", shls_slice=slices_p2).reshape(
        3, nao, nao, -1, 3)
    dpq[:, :, :] *= surface_normals
    slices_g2 = (mol.nbas, mol.nbas + gmol_p2.nbas, 0, mol.nbas, 0, mol.nbas)
    dG2 = supermol_p2.intor("int3c1e_ip1", shls_slice=slices_g2).reshape(
        3, -1, 3, nao, nao)
    dpq_ix = np.einsum('xijnp,ij->ix', dpq, dm, optimize=True)
    dpq_ix += np.einsum('xijnp,ji->ix', dpq, dm, optimize=True)
    dG2 = np.einsum('xnpij,np->xnij', dG2, surface_normals, optimize=True)
    dG2 = np.einsum('xnij,ij->nx', dG2, dm, optimize=True)
    force_operator_grad = np.asarray(
        [np.sum(dpq_ix[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
    np.add.at(force_operator_grad, atom_idx, dG2)
    force_operator_grad *= -1.0

    f_coeffs_val = -2.0 * widths * wgrad_prefs
    gmol_f = fakemol_for_gaussian(grid_coords, widths, l=3, coeffs=f_coeffs_val, cart=True)
    supermol_f = mol + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor("int3c1e", shls_slice=slices_f).reshape(
        nao_cart, nao_cart, -1, 10)
    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized="sp")
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s, optimize=True)
    xf = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    yf = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    zf = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    dx = np.einsum('ijg,ij->g', xf, dm, optimize=True)
    dy = np.einsum('ijg,ij->g', yf, dm, optimize=True)
    dz = np.einsum('ijg,ij->g', zf, dm, optimize=True)
    dr = np.vstack((dx, dy, dz)).T
    rf2 = 1.0 / (forces * forces)
    dr *= surface_normals
    dFdR = dareas * wgrad_prefs * forces / widths
    dFdR += np.einsum('gc,axg->axg', dr, dareas, optimize=True)
    width_grad_ftype = -pressure_au * np.einsum(
        'g,g,axg,g->ax', areas, gtilde_expval, dFdR, rf2, optimize=True)
    dE3 = force_operator_grad + width_grad_ftype

    return dE1 + dE2 + dE3


class TestGOSTSHYPGradient(unittest.TestCase):
    """Test GOSTSHYP gradient against CPU reference."""

    def setUp(self):
        self.mol_hf = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='6-31g',
            cart=True,
            verbose=0
        )
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

    def _compute_gpu_gradient(self, mol, dm):
        from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients
        gostshyp = GOSTSHYP(mol)
        gostshyp.build()
        gostshyp.kernel(dm)
        return GOSTSHYPGradients(gostshyp).kernel(dm)

    def test_gradient_hf_cart(self):
        """Test gradient matches CPU for HF/6-31g/cart."""
        np.random.seed(42)
        nao = self.mol_hf.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        grad_gpu = self._compute_gpu_gradient(self.mol_hf, dm)
        grad_cpu = gostshyp_gradient_cpu_reference(self.mol_hf, dm)

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
        np.random.seed(42)
        nao = mol_sph.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        grad_gpu = self._compute_gpu_gradient(mol_sph, dm)
        grad_cpu = gostshyp_gradient_cpu_reference(mol_sph, dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (HF/6-31g/sph)")

    def test_gradient_water_cart(self):
        """Test gradient matches CPU for H2O/sto-3g/cart."""
        np.random.seed(42)
        nao = self.mol_water.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        grad_gpu = self._compute_gpu_gradient(self.mol_water, dm)
        grad_cpu = gostshyp_gradient_cpu_reference(self.mol_water, dm)

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
        np.random.seed(42)
        nao = mol.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        grad_gpu = self._compute_gpu_gradient(mol, dm)
        grad_cpu = gostshyp_gradient_cpu_reference(mol, dm)

        np.testing.assert_allclose(grad_gpu, grad_cpu, atol=1e-7, rtol=1e-7,
                                   err_msg="Gradient mismatch (HF/cc-pvdz/sph)")

    def test_gradient_translational_invariance(self):
        """Gradient should sum to zero over all atoms (Newton's third law)."""
        np.random.seed(42)
        nao = self.mol_water.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

        grad = self._compute_gpu_gradient(self.mol_water, dm)
        total_force = np.sum(grad, axis=0)

        np.testing.assert_allclose(total_force, 0.0, atol=1e-7,
                                   err_msg="Gradient does not satisfy translational invariance")

    def test_gradient_finite_diff_convergence(self):
        """Verify O(h^2) convergence of finite differences against analytical gradient."""
        mol = gto.M(
            atom='H 1 0 0; F 2 0 0',
            basis='sto-3g',
            cart=True,
            verbose=0
        )
        np.random.seed(42)
        nao = mol.nao
        dm = np.random.randn(nao, nao)
        dm = (dm + dm.T) / 2

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
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        self.assertGreater(ratio1, 50, "FD convergence not O(h^2)")
        self.assertGreater(ratio2, 50, "FD convergence not O(h^2)")


if __name__ == "__main__":
    print("Full Tests for GPU GOSTSHYP")
    unittest.main()
