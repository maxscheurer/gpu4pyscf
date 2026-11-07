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
from gpu4pyscf.solvent.pcm import modified_Bondi
from gpu4pyscf.gto.int3c1e import VHFOpt, GPU_AO_LMAX, libgint
from gpu4pyscf.gto.mole import basis_seg_contraction
from gpu4pyscf.df.int3c2e import sort_mol, _split_l_ctr_groups
from gpu4pyscf.scf.int4c2e import BasisProdCache
from gpu4pyscf.__config__ import num_devices
from pyscf.scf import _vhf
from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.lib import logger
from pyscf.data.radii import BOHR


def _gen_surface(mol, ng=110, rad=modified_Bondi):
    """Vectorized equivalent of gpu4pyscf.solvent.pcm.gen_surface.

    Same quantities and ordering, but all atoms are processed in a few batched
    GPU operations with a single host synchronization instead of a Python loop
    with two synchronizations per atom.
    """
    from pyscf.dft.LebedevGrid import MakeAngularGrid
    from gpu4pyscf.solvent.pcm import switch_h, XI, PI
    from gpu4pyscf.lib.cupy_helper import dist_matrix
    unit_sphere = cp.asarray(MakeAngularGrid(ng))

    natm = mol.natm
    atom_coords = cp.asarray(mol.atom_coords(unit='B'))
    charges = mol.atom_charges()
    N_J = ng * cp.ones(natm)
    R_J = cp.asarray([rad[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J

    r_atm = np.array([rad[gto.charge(mol.atom_symbol(ia))] for ia in range(natm)])
    r_atm_gpu = cp.asarray(r_atm)
    w = unit_sphere[:, 3] * 4.0 * PI
    swf = cp.empty((natm, ng))
    # keep temporaries < 100 MB so they come from the CuPy pool
    chunk = max(1, (1 << 23) // max(1, ng * natm))
    for a0 in range(0, natm, chunk):
        a1 = min(natm, a0 + chunk)
        na = a1 - a0
        atom_grid = (r_atm_gpu[a0:a1, None, None] * unit_sphere[None, :, :3]
                     + atom_coords[a0:a1, None, :])
        riJ = dist_matrix(atom_grid.reshape(-1, 3), atom_coords).reshape(na, ng, natm)
        diJ = (riJ - R_in_J) / R_sw_J
        own = cp.arange(na)
        diJ[own, :, a0 + own] = 1.0
        diJ[diJ < 1e-8] = 0.0
        swf[a0:a1] = cp.prod(switch_h(diJ), axis=2)

    mask = (w[None, :] * swf) > 1e-12
    counts = cp.asnumpy(mask.sum(axis=1))
    offsets = np.append(0, np.cumsum(counts))
    gslice_by_atom = [[int(offsets[i]), int(offsets[i + 1])] for i in range(natm)]

    owner = cp.repeat(cp.arange(natm), ng).reshape(natm, ng)[mask]
    sph = cp.broadcast_to(unit_sphere[None], (natm, ng, 4))[mask]
    r_own = r_atm_gpu[owner]
    w_sel = sph[:, 3] * 4.0 * PI
    swf_sel = swf[mask]
    grid_coords = r_own[:, None] * sph[:, :3] + atom_coords[owner]
    return {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': w_sel,
        'charge_exp': XI[ng] / (r_own * w_sel**0.5),
        'switch_fun': swf_sel,
        'R_vdw': r_own,
        'norm_vec': cp.ascontiguousarray(sph[:, :3]),
        'area': w_sel * r_own**2 * swf_sel,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords,
    }


_pair_bound_kernel = cp.ElementwiseKernel(
    'float64 xi, float64 yi, float64 zi, float64 ai, float64 ci, '
    'float64 xj, float64 yj, float64 zj, float64 aj, float64 cj',
    'float64 q',
    '''
    const double dx = xi - xj, dy = yi - yj, dz = zi - zj;
    q = ci * cj * exp(-(ai * aj / (ai + aj)) * (dx * dx + dy * dy + dz * dz));
    ''', 'gost_pair_bound')


class _OverlapPairOpt(VHFOpt):
    """VHFOpt whose shell pairs are selected by a rigorous overlap bound.

    Every primitive prefactor evaluated by the 3-center overlap kernels is
    bounded by |e12| = fac(li) fac(lj) |ci cj| exp(-ai aj / (ai + aj) R^2).
    Pairs with max |e12| < 0.5 * cutoff contribute nothing and are dropped.
    This replaces the CPU ERI Schwarz diagonal (int2e) of the base class.
    """

    def __init__(self, mol, cutoff):
        VHFOpt.__init__(self, mol, qcondname=None)
        self._overlap_cutoff = cutoff

    def _shell_bound_data(self):
        """Per-shell (min exponent, scaled max |coeff|, atom) of the sorted mol."""
        mol = self._sorted_mol
        bas = mol._bas
        nbas = mol.nbas
        nprim = bas[:, gto.NPRIM_OF]
        npmax = int(nprim.max())
        k = np.arange(npmax)
        valid = k[None, :] < nprim[:, None]
        pe = np.where(valid, bas[:, gto.PTR_EXP][:, None] + k, 0)
        pc = np.where(valid, bas[:, gto.PTR_COEFF][:, None] + k, 0)
        amin = np.where(valid, mol._env[pe], np.inf).min(axis=1)
        cmax = np.where(valid, np.abs(mol._env[pc]), 0.0).max(axis=1)
        l = bas[:, gto.ANG_OF]
        cmax[l == 0] *= 0.282094791773878143
        cmax[l == 1] *= 0.488602511902919921
        return (cp.asarray(amin), cp.asarray(cmax),
                cp.asarray(mol.atom_coords()[bas[:, gto.ATOM_OF]]))

    def _overlap_pairing(self, offsets):
        """Shell pairs (i >= j) whose bound
            max|ci| max|cj| exp(-mu_min R^2) >= max |e12|
        reaches 0.5 * cutoff; mu = ai aj / (ai + aj) is increasing in both
        exponents, so the smallest exponents give the largest exponential.
        Ordering matches get_pairing(aosym=True, diag_block_with_triu=False).
        """
        amin, cmax, xyz = self._shell_bound_data()
        thresh = 0.5 * self._overlap_cutoff
        log_qs, pair2bra, pair2ket = [], [], []
        for p0, p1 in zip(offsets[:-1], offsets[1:]):
            for q0, q1 in zip(offsets[:-1], offsets[1:]):
                if q0 > p0:
                    continue
                q = _pair_bound_kernel(
                    xyz[p0:p1, None, 0], xyz[p0:p1, None, 1], xyz[p0:p1, None, 2],
                    amin[p0:p1, None], cmax[p0:p1, None],
                    xyz[None, q0:q1, 0], xyz[None, q0:q1, 1], xyz[None, q0:q1, 2],
                    amin[None, q0:q1], cmax[None, q0:q1])
                mask = q >= thresh
                if q0 == p0:
                    mask &= cp.tri(p1 - p0, dtype=bool)
                ish, jsh = cp.nonzero(mask)
                if q0 == p0 and len(ish) == 0:
                    continue
                pair2bra.append(cp.asnumpy(ish) + p0)
                pair2ket.append(cp.asnumpy(jsh) + q0)
                log_q = cp.log(cp.minimum(q[ish, jsh] * (1e-14 / thresh)
                                          if thresh > 0 else q[ish, jsh], 1.0))
                log_qs.append(cp.asnumpy(log_q))
        return log_qs, pair2bra, pair2ket

    def build(self, cutoff=1e-13, group_size=None, diag_block_with_triu=False, aosym=True):
        """Copy of gpu4pyscf.gto.int3c1e.VHFOpt.build with overlap-bound pairing."""
        original_mol = self.mol
        mol = basis_seg_contraction(original_mol, allow_replica=True)[0]

        log = logger.new_logger(original_mol, original_mol.verbose)
        cput0 = log.init_timer()
        _sorted_mol, sorted_idx, uniq_l_ctr, l_ctr_counts = sort_mol(mol, log=log)
        self._sorted_mol = _sorted_mol

        if group_size is not None :
            uniq_l_ctr, l_ctr_counts = _split_l_ctr_groups(uniq_l_ctr, l_ctr_counts, group_size)
        self.l_ctr_counts = l_ctr_counts

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, _sorted_mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        # Shell pairs from the rigorous overlap bound, evaluated per block pair
        # on the GPU (same pairs and ordering as get_pairing on the dense
        # bound matrix, without materializing it).
        cput1 = log.timer_debug1('Initialize', *cput0)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        assert aosym and not diag_block_with_triu
        log_qs, pair2bra, pair2ket = self._overlap_pairing(l_ctr_offsets)
        self.log_qs = log_qs.copy()
        cput1 = log.timer_debug1('Get AO pairing', *cput1)

        # contraction coefficient for ao basis
        cart_ao_loc = _sorted_mol.ao_loc_nr(cart=True)
        sph_ao_loc = _sorted_mol.ao_loc_nr(cart=False)
        self.cart_ao_loc = [cart_ao_loc[cp] for cp in l_ctr_offsets]
        self.sph_ao_loc = [sph_ao_loc[cp] for cp in l_ctr_offsets]
        self.angular = [l[0] for l in uniq_l_ctr]
        if any(np.array(self.angular) > GPU_AO_LMAX):
            raise NotImplementedError("H orbital or higher (5Z basis or higher) is not supported")

        # Sorted AO indices
        ao_loc = mol.ao_loc_nr(cart=original_mol.cart)
        ao_idx = np.array_split(np.arange(original_mol.nao), ao_loc[1:-1])
        self._ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        cput1 = log.timer_debug1('AO indices', *cput1)

        ao_loc = cart_ao_loc

        self.pair2bra = pair2bra
        self.pair2ket = pair2ket
        self.l_ctr_offsets = l_ctr_offsets
        bas_pair2shls = np.hstack(pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        bas_pairs_locs = np.append(0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.bas_pair2shls = bas_pair2shls
        self.bas_pairs_locs = bas_pairs_locs
        ncptype = len(self.log_qs)

        n_uniq_l_ctr = len(uniq_l_ctr)
        if aosym:
            # This symmetric case is different from self.cp_idx, whose length is ncptype = n_uniq_l_ctr ** 2
            cp_idx, cp_jdx = np.tril_indices(n_uniq_l_ctr)
        else:
            cp_idx, cp_jdx = np.unravel_index(np.arange(n_uniq_l_ctr**2), (n_uniq_l_ctr, n_uniq_l_ctr))
        l_ij = list(zip(uniq_l_ctr[cp_idx, 0], uniq_l_ctr[cp_jdx, 0]))
        self.l_ij = np.asarray(l_ij)
        def get_n_hermite_density_of_angular_pair(l):
            return (l + 1) * (l + 2) * (l + 3) // 6
        n_density_per_pair = np.array([ get_n_hermite_density_of_angular_pair(li + lj) for (li, lj) in l_ij ])
        n_density_per_angular_pair = (bas_pairs_locs[1:] - bas_pairs_locs[:-1]) * n_density_per_pair
        self.density_offset = np.append(0, np.cumsum(n_density_per_angular_pair)).astype(np.int32)

        self._bpcache = {}
        for n in range(num_devices):
            with cp.cuda.Device(n):
                bpcache = ctypes.POINTER(BasisProdCache)()
                scale_shellpair_diag = 1.0
                libgint.GINTinit_basis_prod(
                    ctypes.byref(bpcache), ctypes.c_double(scale_shellpair_diag),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
                    bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
                    _sorted_mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.natm),
                    _sorted_mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(_sorted_mol.nbas),
                    _sorted_mol._env.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_void_p(cp.cuda.get_current_stream().ptr))
                self._bpcache[n] = bpcache

        cput1 = log.timer_debug1('Initialize GPU cache', *cput1)
        self.aosym = aosym
        if aosym:
            self.cp_idx, self.cp_jdx = np.tril_indices(ncptype)
        else:
            nl = int(round(np.sqrt(ncptype)))
            self.cp_idx, self.cp_jdx = np.unravel_index(np.arange(ncptype), (nl, nl))

        if original_mol.cart:
            self.ao_loc = self.cart_ao_loc
        else:
            self.ao_loc = self.sph_ao_loc



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

        # Cached screening lists (see get_screen)
        self._screen = None

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
            self.surface = _gen_surface(
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
            self.surface = _gen_surface(mol, ng=self.npoints, rad=rad)
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

        self.intopt = _OverlapPairOpt(mol, self.overlap_cutoff)
        self.intopt.build(1e-20, aosym=True, group_size=None)
        self._surface_derivatives = None
        self._screen = None
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

    def get_screen(self):
        """Spatial (shell pair, grid tile) screening lists (cached)."""
        scr = self._screen
        force_dense = getattr(self, '_force_dense', False)
        if scr is None or not scr.matches(self.intopt, self.grid_coords,
                                          self.overlap_cutoff, force_dense):
            scr = self._screen = int3c_overlap.OverlapScreen(
                self.intopt, self.grid_coords, self.widths, self.overlap_cutoff,
                force_dense=force_dense)
            if scr.dense and not force_dense and self.overlap_cutoff > 0:
                logger.info(self, 'GOSTSHYP: screening lists exceed the memory '
                            'budget, using dense (unscreened-list) kernels')
        return scr

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

        screen = self.get_screen()

        # Fused pass 1: density-contracted s+p integrals in one kernel pass
        sp_contracted = screen.density(mol, dm, kmax=1)
        gtilde_expval = sp_contracted[:, 0]
        p_contracted = sp_contracted[:, 1:4]
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
        fock = screen.fock_sp(mol, amplitudes, weighted_amp)

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


# Inject GOSTSHYP to SCF classes
from gpu4pyscf import scf
scf.hf.RHF.GOSTSHYP = gostshyp_for_scf
scf.uhf.UHF.GOSTSHYP = gostshyp_for_scf
