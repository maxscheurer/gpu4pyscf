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
GPU-accelerated GOSTSHYP pressure model.

References:
    J. Chem. Theory Comput. 2021, 17, 1, 583–597
    https://doi.org/10.1021/acs.jctc.0c01212
"""

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf import gto
from gpu4pyscf.solvent import _attach_solvent
from gpu4pyscf.solvent.pcm import gen_surface, modified_Bondi
from gpu4pyscf.gto.int3c1e import VHFOpt
from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.lib import logger


@lib.with_doc(_attach_solvent._for_scf.__doc__)
def gostshyp_for_scf(mf, solvent_obj=None, dm=None):
    """Attach GOSTSHYP solvent model to SCF method."""
    if solvent_obj is None:
        solvent_obj = GOSTSHYP(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


# Pressure conversion: MPa to atomic units
# 1 MPa = 3.3989309735473356e-08 Hartree/Bohr^3
MPA_TO_AU = 3.3989309735473356e-08


class GOSTSHYP(lib.StreamObject):
    """
    GPU-accelerated GOSTSHYP solvation model.

    The GOSTSHYP model computes cavity energy under pressure using:
    1. Surface tessellation with Gaussian-weighted grid points
    2. 3-center overlap integrals with AO basis
    3. Force-based amplitude determination

    Attributes
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    pressure_mpa : float
        Applied pressure in MPa (default: 50000 MPa = 50 GPa)
    npoints : int
        Number of grid points per atom for surface (default: 110)
    scaling_factor : float
        VDW radii scaling factor (default: 1.2)
    """

    from gpu4pyscf.lib.utils import to_gpu, device, to_cpu

    _keys = {
        'mol', 'pressure_mpa', 'npoints', 'scaling_factor',
        'surface', 'intopt', 'frozen', 'equilibrium_solvation',
        'e', 'v', 'amplitudes', 'forces', 'gtilde_expval'
    }

    def __init__(self, mol, options=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        # GOSTSHYP parameters
        if options is None:
            options = {}
        self.pressure_mpa = options.get('pressure_mpa', 50_000)  # 50 GPa default
        self.npoints = options.get('npoints', 110)
        self.scaling_factor = options.get('scaling_factor', 1.2)

        # Internal state
        self.surface = {}
        self.intopt = None
        self.frozen = False
        self.equilibrium_solvation = False

        # Results
        self.e = None
        self.v = None
        self.amplitudes = None
        self.forces = None

        # Cached operators
        self._gtilde = None
        self._force_operators = None

        # Timing (GPU ms accumulated via CUDA events)
        self._t_gpu_ms = 0.0
        self._t_wall = 0.0
        self._n_kernel = 0

    @property
    def pressure_au(self):
        """Pressure in atomic units (Hartree/Bohr^3)."""
        return self.pressure_mpa * MPA_TO_AU

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'pressure = %.1f MPa (%.6e a.u.)',
                    self.pressure_mpa, self.pressure_au)
        logger.info(self, 'npoints = %d', self.npoints)
        logger.info(self, 'scaling_factor = %.2f', self.scaling_factor)
        logger.info(self, 'frozen = %s', self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        if self.surface:
            ngrids = len(self.surface['area'])
            logger.info(self, 'n_surface_points = %d', ngrids)
        return self

    def build(self, mol=None):
        """
        Build surface tessellation and integral options.

        Parameters
        ----------
        mol : pyscf.gto.Mole, optional
            Molecular object (uses self.mol if not provided)
        """
        if mol is not None:
            self.mol = mol
        mol = self.mol

        # Generate surface using PCM infrastructure
        rad = self.scaling_factor * modified_Bondi
        self.surface = gen_surface(mol, ng=self.npoints, rad=rad)

        # Compute Gaussian widths from areas (eq. 4 in GOSTSHYP paper)
        # width = pi * ln(2) / area
        # Keep grid data on GPU; gen_surface returns cupy arrays.
        self.areas = self.surface['area']              # cupy [ngrids]
        self.grid_coords = self.surface['grid_coords'] # cupy [ngrids, 3]
        atom_coords = self.surface['atom_coords']      # cupy [natm, 3]
        self.widths = cp.float64(np.pi * np.log(2)) / self.areas  # cupy [ngrids]
        gslice_by_atom = self.surface['gslice_by_atom']

        ngrids = len(self.areas)
        atom_idx = np.zeros(ngrids, dtype=np.int32)
        for ia, (p0, p1) in enumerate(gslice_by_atom):
            atom_idx[p0:p1] = ia
        self.atom_idx = atom_idx  # numpy (small, used for CPU scatter)

        # Compute inward-pointing normals on GPU
        atom_idx_gpu = cp.asarray(atom_idx)
        ref_coords = atom_coords[atom_idx_gpu]
        dr = ref_coords - self.grid_coords
        dr_norm = cp.linalg.norm(dr, axis=1, keepdims=True)
        self.surface_normals = dr / dr_norm  # cupy [ngrids, 3]

        # Build VHFOpt for AO shell pairs
        self.intopt = VHFOpt(mol)
        self.intopt.build(1e-20, aosym=True)

        # Clear cached operators
        self._gtilde = None
        self._force_operators = None

        logger.info(self, 'GOSTSHYP: %d surface Gaussians', ngrids)
        return self

    @property
    def n_gaussian(self):
        """Number of surface Gaussians."""
        return len(self.areas) if hasattr(self, 'areas') else 0

    def _compute_gtilde(self):
        """Compute s-type 3-center overlap integrals (cached)."""
        if self._gtilde is not None:
            return self._gtilde

        mol = self.mol
        nao = mol.nao
        ngrids = self.n_gaussian

        # Compute full tensor for s-type aux
        gtilde = int3c_overlap.get_int3c_overlap(
            mol, self.grid_coords, self.widths, aux_l=0, intopt=self.intopt)

        # Shape: [ngrids, 1, nao, nao] -> [nao, nao, ngrids]
        self._gtilde = gtilde[:, 0, :, :].transpose(1, 2, 0)
        return self._gtilde

    def _compute_force_operators(self):
        """Compute p-type 3-center overlap contracted with normals (cached)."""
        if self._force_operators is not None:
            return self._force_operators

        mol = self.mol
        nao = mol.nao
        ngrids = self.n_gaussian

        # p-type coefficients: 2 * width (derivative of Gaussian)
        p_coeffs = 2.0 * self.widths

        # Compute full tensor for p-type aux
        overlap3_p = int3c_overlap.get_int3c_overlap(
            mol, self.grid_coords, self.widths, aux_l=1, intopt=self.intopt)

        # Shape: [ngrids, 3, nao, nao] -> [nao, nao, ngrids, 3]
        overlap3_p = overlap3_p.transpose(2, 3, 0, 1)

        # Contract with surface normals: F_ij_g = sum_c overlap_ij_g_c * n_g_c
        # Also apply p-type coefficient
        force_ops = np.einsum('ijgc,gc,g->ijg', overlap3_p, self.surface_normals,
                              p_coeffs, optimize=True)

        self._force_operators = force_ops
        return self._force_operators

    def kernel(self, dm):
        """
        Compute GOSTSHYP energy and Fock matrix contribution.

        This implementation uses GPU-native contraction kernels for efficiency,
        avoiding materialization of full [nao, nao, ngrids] tensors.

        Parameters
        ----------
        dm : ndarray of shape (nao, nao) or (2, nao, nao)
            Density matrix (spin-traced for UHF/ROHF)

        Returns
        -------
        energy : float
            GOSTSHYP contribution to total energy
        fock : ndarray of shape (nao, nao)
            GOSTSHYP contribution to Fock matrix
        """
        # Always record wall + GPU time for accumulation, regardless of verbose
        _e0 = cp.cuda.Event(); _e0.record()
        _w0 = logger.perf_counter()
        t0 = logger.init_timer(self)
        if not hasattr(self, 'areas') or self.areas is None:
            self.build()

        # Ensure dm is a CuPy array on GPU
        dm = cp.asarray(dm)
        if dm.ndim == 3 and dm.shape[0] == 2:
            dm = dm[0] + dm[1]

        mol = self.mol
        nao = mol.nao
        assert dm.shape == (nao, nao), f"Expected dm shape ({nao}, {nao}), got {dm.shape}"

        # All grid data (areas, widths, surface_normals, grid_coords) is already on GPU.
        p_coeffs = 2.0 * self.widths  # cupy [ngrids]

        # Step 1: Forces via density-contracted p-type integrals
        p_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, self.grid_coords, self.widths, aux_l=1, dm=dm, intopt=self.intopt)
        forces = cp.sum(p_contracted * self.surface_normals * p_coeffs[:, None], axis=1)

        if cp.any(forces <= 0):
            logger.warn(self, 'GOSTSHYP: Some forces are non-positive, '
                        'results may be unreliable')

        # Step 2: Compute amplitudes = P * A_g / forces
        amplitudes = self.pressure_au * self.areas / forces

        # Step 3: Fock term 1 via amplitude-contracted s-type integrals
        fock1 = int3c_overlap.get_int3c_overlap_amplitude_contracted(
            mol, self.grid_coords, self.widths, aux_l=0,
            amplitudes=amplitudes[:, None], intopt=self.intopt)

        # Step 4: gtilde_expval via density-contracted s-type integrals
        s_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, self.grid_coords, self.widths, aux_l=0, dm=dm, intopt=self.intopt)
        gtilde_expval = s_contracted[:, 0]

        # Step 5: Fock term 2 via amplitude-contracted p-type integrals
        response_coeff = -self.pressure_au * self.areas * gtilde_expval / (forces ** 2)
        weighted_amp = response_coeff[:, None] * self.surface_normals * p_coeffs[:, None]
        fock2 = int3c_overlap.get_int3c_overlap_amplitude_contracted(
            mol, self.grid_coords, self.widths, aux_l=1,
            amplitudes=weighted_amp, intopt=self.intopt)

        fock = fock1 + fock2
        energy = float(cp.vdot(fock1, dm))

        # Store results on GPU
        self.forces = forces
        self.amplitudes = amplitudes
        self.gtilde_expval = gtilde_expval
        self.e = energy
        self.v = fock

        self._n_kernel += 1
        _e1 = cp.cuda.Event(); _e1.record(); _e1.synchronize()
        self._t_wall += logger.perf_counter() - _w0
        self._t_gpu_ms += cp.cuda.get_elapsed_time(_e0, _e1)
        logger.info(self, 'GOSTSHYP energy: %.10f', energy)
        logger.timer(self, 'GOSTSHYP kernel', *t0)
        return energy, self.v

    def reset(self, mol=None):
        """Reset molecule and rebuild surface (for geometry optimization)."""
        if mol is not None:
            self.mol = mol
            self._gtilde = None
            self._force_operators = None
            self.build()
        return self

    def nuc_grad_method(self):
        """Return gradient object for nuclear gradients."""
        from gpu4pyscf.solvent.grad import gostshyp as gostshyp_grad
        return gostshyp_grad.Gradients(self)

    def grad(self, dm):
        '''Compute GOSTSHYP solvent gradient contribution for the given
        density matrix. Called by WithSolventGrad.kernel().
        '''
        from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients
        grad_obj = GOSTSHYPGradients(self)
        result = grad_obj.kernel(dm)
        self._grad_t_wall = grad_obj.t_wall
        return result


def GOSTSHYP_factory(method_or_mol, solvent_obj=None, dm=None):
    """
    Initialize GOSTSHYP model.

    Examples
    --------
    >>> mf = GOSTSHYP(scf.RHF(mol))
    >>> mf.kernel()

    >>> sol = GOSTSHYP(mol)
    >>> mf = sol.for_scf(scf.RHF(mol))
    """
    from pyscf import gto as pyscf_gto
    from gpu4pyscf import scf

    if isinstance(method_or_mol, pyscf_gto.mole.Mole):
        return GOSTSHYP(method_or_mol)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return gostshyp_for_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError(f'GOSTSHYP model does not support {method_or_mol}')


# Inject GOSTSHYP to SCF classes
from gpu4pyscf import scf
scf.hf.RHF.GOSTSHYP = gostshyp_for_scf
scf.uhf.UHF.GOSTSHYP = gostshyp_for_scf
