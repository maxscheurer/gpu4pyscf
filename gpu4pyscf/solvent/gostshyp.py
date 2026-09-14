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
from pyscf.data.radii import BOHR


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
        'e', 'v', 'amplitudes', 'forces', 'gtilde_expval',
        'overlap_cutoff', 'cavity', 'r_ext', 'drop_kwargs',
        'grid_coords', 'areas', 'widths', 'atom_idx', 'surface_normals',
        'surface_distances',
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
        self.overlap_cutoff = options.get('overlap_cutoff', 1e-14)
        self.cavity = options.get('cavity', 'vdw/occ')
        self.r_ext = options.get('r_ext', 0.4724)          # Bohr (0.25 Ang)
        self.drop_kwargs = dict(options.get('drop_kwargs') or {})

        # Internal state
        self.surface = {}
        self._outer_surface = None
        self._occ_ratio_sq = None
        self._drop_cavity = None
        self._surface_derivatives = None
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
        self._grad_t_wall = 0.0
        self._t_moist_build = 0.0
        self._t_moist_grad = 0.0
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
        logger.info(self, 'overlap_cutoff = %.2e', self.overlap_cutoff)
        logger.info(self, 'cavity = %s', self.cavity)
        if self.cavity == 'vdw/occ':
            logger.info(self, 'r_ext = %.4f Bohr (%.4f Ang)', self.r_ext, self.r_ext * 0.529177)
        elif self.cavity == 'drop':
            logger.info(self, 'using MOIST DROPSvdW cavity')
        logger.info(self, 'frozen = %s', self.frozen)
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        if hasattr(self, 'areas') and self.areas is not None:
            logger.info(self, 'n_surface_points = %d', len(self.areas))
        return self

    def check_sanity(self):
        if self.pressure_mpa <= 0:
            raise ValueError(f'pressure_mpa must be positive, got {self.pressure_mpa}')
        if self.scaling_factor <= 0:
            raise ValueError(
                f'scaling_factor must be positive, got {self.scaling_factor}')
        if self.cavity not in ('vdw', 'vdw/occ', 'drop'):
            raise ValueError(
                "cavity must be 'vdw', 'vdw/occ', or 'drop', "
                f"got '{self.cavity}'")
        if self.cavity == 'vdw/occ' and self.r_ext <= 0:
            raise ValueError(
                f'r_ext must be positive for vdw/occ, got {self.r_ext}')
        return self

    def _build_gen_surface(self, mol):
        """Build a PCM surface and return its common surface arrays."""
        rad = self.scaling_factor * modified_Bondi
        if self.cavity == 'vdw/occ':
            self.surface = gen_surface(
                mol, ng=self.npoints, rad=rad + self.r_ext)
            self._outer_surface = dict(self.surface)

            norm_vec = self.surface['norm_vec']
            grid_outer = self.surface['grid_coords']
            grid_inner = grid_outer - self.r_ext * norm_vec
            R_outer = self.surface['R_vdw']
            R_inner = R_outer - self.r_ext
            self._occ_ratio_sq = (R_inner / R_outer) ** 2

            self.surface['grid_coords_outer'] = grid_outer
            self.surface['grid_coords'] = grid_inner
            self.surface['area'] = (
                self.surface['area'] * self._occ_ratio_sq)
            self.surface['R_vdw'] = R_inner
        else:
            self.surface = gen_surface(mol, ng=self.npoints, rad=rad)
            self._outer_surface = None
            self._occ_ratio_sq = None

        atom_idx = np.empty(len(self.surface['area']), dtype=np.int32)
        for ia, (p0, p1) in enumerate(self.surface['gslice_by_atom']):
            atom_idx[p0:p1] = ia
        self._drop_cavity = None
        return self.surface['grid_coords'], self.surface['area'], atom_idx

    def _build_drop_surface(self, mol):
        """Build a MOIST DROP surface and return its common surface arrays."""
        from gpu4pyscf.solvent.moist import build_drop_cavity, get_surface_data

        t0 = logger.perf_counter()
        self._drop_cavity = build_drop_cavity(
            mol, nleb=self.npoints, **self.drop_kwargs)
        grid_coords, areas, atom_idx = get_surface_data(self._drop_cavity)
        self._t_moist_build += logger.perf_counter() - t0
        self.surface = None
        self._outer_surface = None
        self._occ_ratio_sq = None
        return grid_coords, areas, atom_idx

    def _finalize_surface(self, mol, grid_coords, areas, atom_idx):
        """Normalize cavity output and initialize shared GPU state."""
        self.grid_coords = cp.ascontiguousarray(
            cp.asarray(grid_coords, dtype=cp.float64))
        self.areas = cp.ascontiguousarray(cp.asarray(areas, dtype=cp.float64))
        self.atom_idx = np.ascontiguousarray(atom_idx, dtype=np.int32)

        ngrids = len(self.areas)
        if self.grid_coords.shape != (ngrids, 3):
            raise ValueError(
                f'Expected grid coordinates ({ngrids}, 3), '
                f'got {self.grid_coords.shape}')
        if self.atom_idx.shape != (ngrids,):
            raise ValueError(
                f'Expected {ngrids} surface owners, got {self.atom_idx.shape}')
        if ngrids == 0 or bool(cp.any(self.areas <= 0)):
            raise ValueError('DROP/VDW cavity has no grids or non-positive areas')
        if np.any(self.atom_idx < 0) or np.any(self.atom_idx >= mol.natm):
            raise ValueError('Surface owner index is outside the molecule')

        self.widths = cp.float64(np.pi * np.log(2)) / self.areas
        owner_gpu = cp.asarray(self.atom_idx)
        atom_coords = cp.asarray(mol.atom_coords(), dtype=cp.float64)
        displacement = atom_coords[owner_gpu] - self.grid_coords
        distance = cp.linalg.norm(displacement, axis=1)
        if bool(cp.any(distance < 1e-14)):
            raise ValueError('Surface grid point coincides with its owner atom')
        self.surface_distances = distance
        self.surface_normals = displacement / distance[:, None]

        self.intopt = VHFOpt(mol)
        self.intopt.build(1e-20, aosym=True)
        self._gtilde = None
        self._force_operators = None
        self._surface_derivatives = None
        return ngrids

    def build(self, mol=None):
        """Build the selected cavity and common integral state."""
        if mol is not None:
            self.mol = mol
        mol = self.mol
        self.check_sanity()

        if self.cavity == 'drop':
            surface_data = self._build_drop_surface(mol)
        else:
            surface_data = self._build_gen_surface(mol)
        ngrids = self._finalize_surface(mol, *surface_data)

        logger.info(self, 'GOSTSHYP: %d surface Gaussians (cavity=%s)',
                    ngrids, self.cavity)
        return self

    @property
    def n_gaussian(self):
        """Number of surface Gaussians."""
        return len(self.areas) if hasattr(self, 'areas') else 0

    def export_cavity_xyz(self, filename):
        """Export cavity surface grid points as an XYZ file.

        Parameters
        ----------
        filename : str
            Path to the output XYZ file.
        """
        if not hasattr(self, 'grid_coords') or self.grid_coords is None:
            raise RuntimeError('No surface data. Call build() first.')

        coords_ang = self.grid_coords.get() * BOHR
        areas = self.areas.get()
        ngrids = len(coords_ang)
        total_area = float(areas.sum()) * BOHR**2

        with open(filename, 'w') as f:
            f.write(f'{ngrids}\n')
            f.write(f'cavity={self.cavity}  pressure={self.pressure_mpa:.1f} MPa  '
                    f'area={total_area:.4f} Ang^2\n')
            for i in range(ngrids):
                sym = self.mol.atom_symbol(int(self.atom_idx[i]))
                f.write(f'{sym:2s} {coords_ang[i,0]:16.10f} {coords_ang[i,1]:16.10f} '
                        f'{coords_ang[i,2]:16.10f}\n')

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

        cutoff = self.overlap_cutoff

        # Fused pass 1: density-contracted s+p integrals in one kernel pass
        s_contracted, p_contracted = int3c_overlap.get_int3c_overlap_density_contracted_sp(
            mol, self.grid_coords, self.widths, dm=dm, intopt=self.intopt, cutoff=cutoff)
        gtilde_expval = s_contracted[:, 0]
        forces = cp.sum(p_contracted * self.surface_normals * p_coeffs[:, None], axis=1)

        # Compute amplitudes = P * A_g / forces
        amplitudes = self.pressure_au * self.areas / forces

        # Check negative forces: only warn if their energy contribution > 1%
        neg_mask = forces <= 0
        if cp.any(neg_mask):
            neg_contrib = cp.sum(amplitudes[neg_mask] * gtilde_expval[neg_mask])
            total_contrib = cp.sum(amplitudes * gtilde_expval)
            if abs(float(total_contrib)) > 0:
                frac = abs(float(neg_contrib)) / abs(float(total_contrib))
            else:
                frac = 0.0
            n_neg = int(cp.sum(neg_mask))
            if frac > 0.01:
                logger.warn(self, 'GOSTSHYP: %d grid points have non-positive '
                            'forces (%.1f%% of energy), results may be unreliable',
                            n_neg, 100.0 * frac)
            else:
                logger.debug(self, 'GOSTSHYP: %d grid points have non-positive '
                             'forces (%.4f%% of energy)', n_neg, 100.0 * frac)

        # Energy: sum_g amplitudes_g * gtilde_expval_g
        # (equivalent to vdot(fock1, dm) since fock1 = sum_g a_g * S_ij0_g)
        energy = float(cp.sum(amplitudes * gtilde_expval))

        # Fused pass 2: amplitude-contracted s+p integrals in one kernel pass
        response_coeff = -self.pressure_au * self.areas * gtilde_expval / (forces ** 2)
        weighted_amp = response_coeff[:, None] * self.surface_normals * p_coeffs[:, None]
        fock = int3c_overlap.get_int3c_overlap_amplitude_contracted_sp(
            mol, self.grid_coords, self.widths,
            amp_s=amplitudes, amp_p=weighted_amp,
            intopt=self.intopt, cutoff=cutoff)

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
        """Reset molecule and rebuild a responsive surface.

        A frozen solvent represents a fixed external potential.  Preserve its
        cached energy and potential across the standard SCF reset path.
        """
        if mol is not None:
            self.mol = mol
        if self.frozen:
            return self
        self.e = None
        self.v = None
        self.amplitudes = None
        self.forces = None
        self.gtilde_expval = None
        self._surface_derivatives = None
        return self.build()

    def to_cpu(self):
        """Create the corresponding pyscf-forge GOSTSHYP object."""
        from pyscf.solvent.gostshyp import GOSTSHYP as CPU_GOSTSHYP

        options = {
            'pressure_mpa': self.pressure_mpa,
            'npoints': self.npoints,
            'scaling_factor': self.scaling_factor,
            'cavity': self.cavity,
            'r_ext': self.r_ext,
            'drop_kwargs': dict(self.drop_kwargs),
        }
        out = CPU_GOSTSHYP(self.mol, options=options)
        out.frozen = self.frozen
        for name in ('e', 'v', 'amplitudes', 'forces', 'gtilde_expval'):
            value = getattr(self, name, None)
            if isinstance(value, cp.ndarray):
                value = cp.asnumpy(value)
            setattr(out, name, value)
        return out

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
