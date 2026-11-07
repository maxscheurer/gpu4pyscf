# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
Gradient implementation for GOSTSHYP solvation model.

The GOSTSHYP gradient dE/dR has three contributions:
  dE1: area derivative — P * sum_g (dA_g/dR) * <D|S_g> / F_g
  dE2: gtilde operator gradient — AO center + aux center + d-type width gradient
  dE3: force operator gradient — AO center + aux center + f-type width gradient
"""

import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.solvent.grad.pcm import get_dF_dA
from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.gto import int3c_overlap_ip
from gpu4pyscf.lib import logger


def _per_atom_sum(per_ao, aoslice):
    """Sum per-AO contributions [nao, 3] to per-atom [natm, 3] on GPU."""
    natm = aoslice.shape[0]
    per_atom = cp.empty((natm, 3), dtype=per_ao.dtype)
    for ia in range(natm):
        p0, p1 = int(aoslice[ia, 2]), int(aoslice[ia, 3])
        per_atom[ia] = cp.sum(per_ao[p0:p1], axis=0)
    return per_atom


def _scatter_add(target, indices, values):
    """Scatter-add values[g, :] into target[indices[g], :] on GPU."""
    cp.add.at(target, cp.asarray(indices), values)


def _get_surface_derivatives(gostshyp):
    """Return cached area and optional non-rigid coordinate derivatives."""
    if gostshyp._surface_derivatives is not None:
        return gostshyp._surface_derivatives

    if gostshyp.cavity == 'drop':
        from gpu4pyscf.solvent.moist import get_anchor_gradient
        t0 = logger.perf_counter()
        dareas, dcoords = get_anchor_gradient(gostshyp._drop_cavity)
        gostshyp._t_moist_grad += logger.perf_counter() - t0
        expected_da = (gostshyp.mol.natm, 3, gostshyp.n_gaussian)
        expected_dx = (3, 3, gostshyp.mol.natm, gostshyp.n_gaussian)
        if dareas.shape != expected_da or dcoords.shape != expected_dx:
            raise ValueError(
                'Unexpected MOIST derivative shapes: '
                f'{dareas.shape}, {dcoords.shape}; expected '
                f'{expected_da}, {expected_dx}')
        result = (cp.asarray(dareas), dcoords)
    else:
        surface = (gostshyp._outer_surface
                   if gostshyp._outer_surface is not None
                   else gostshyp.surface)
        _, dareas = get_dF_dA(surface)
        dareas = dareas.transpose(2, 0, 1)
        if gostshyp._occ_ratio_sq is not None:
            dareas *= gostshyp._occ_ratio_sq[None, None, :]
        result = (dareas, None)

    gostshyp._surface_derivatives = result
    return result


def _contract_dareas(gostshyp, weights):
    """Return sum_g dA_g/dR_(a,x) * weights_g as [natm, 3].

    For PCM-type cavities (vdw, vdw/occ) this evaluates the area derivative of
    pcm.grad.get_dF_dA with one CUDA thread per surface point over the
    neighbour atoms of its owner, without materializing the dense
    [natm, 3, ngrids] tensor. DROP cavities use the cached MOIST derivatives.
    """
    if gostshyp.cavity == 'drop':
        dareas, _ = _get_surface_derivatives(gostshyp)
        return cp.einsum('acg,g->ac', dareas, weights)

    surface = (gostshyp._outer_surface
               if gostshyp._outer_surface is not None
               else gostshyp.surface)
    atom_coords = cp.ascontiguousarray(surface['atom_coords'])
    grid_coords = cp.ascontiguousarray(surface['grid_coords'])
    R_in_J = cp.ascontiguousarray(surface['R_in_J'], dtype=np.float64)
    R_sw_J = cp.ascontiguousarray(surface['R_sw_J'], dtype=np.float64)
    natm = atom_coords.shape[0]
    ngrids = grid_coords.shape[0]
    wA = weights * surface['area']
    if gostshyp._occ_ratio_sq is not None:
        wA = wA * gostshyp._occ_ratio_sq
    wA = cp.ascontiguousarray(wA, dtype=np.float64)
    owner = cp.asarray(gostshyp.atom_idx, dtype=np.int32)

    # Only atoms with 0 < d_iJ < 1 contribute (grad_switch_h vanishes for
    # d >= 1 and the displacement is zeroed for d < 1e-8). Candidate atoms J
    # for points of atom a: |R_a - R_J| - r_max(a) < R_in_J + R_sw_J.
    ptr, nbr = _owner_neighbours(gostshyp, surface)
    out = cp.zeros((natm, 3))
    _dareas_kernel(((ngrids + 127) // 128,), (128,),
                   (grid_coords, atom_coords, R_in_J, R_sw_J, wA, owner,
                    ptr, nbr, out, np.int32(ngrids)))
    return out


_dareas_kernel = cp.RawKernel(r'''
__device__ __forceinline__ double sw_h(double x) {
    if (x < 0) return 0.0;
    if (x > 1) return 1.0;
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
}
__device__ __forceinline__ double dsw_h(double x) {
    if (x < 0 || x > 1) return 0.0;
    return 30.0 * x * x - 60.0 * x * x * x + 30.0 * x * x * x * x;
}
extern "C" __global__
void gost_contract_dareas(const double* __restrict__ grid, const double* __restrict__ atm,
                          const double* __restrict__ R_in, const double* __restrict__ R_sw,
                          const double* __restrict__ wA, const int* __restrict__ owner,
                          const int* __restrict__ ptr, const int* __restrict__ nbr,
                          double* __restrict__ out, const int ngrids)
{
    // sum_g w_g dA_g/dR (pcm.grad.get_dF_dA, area part), one thread per point:
    //   owner atom: + A_g sum_J dfiJ;  atom J: - A_g dfiJ
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids) return;
    const int own = owner[i];
    const double w = wA[i];
    const double gx = grid[i * 3 + 0], gy = grid[i * 3 + 1], gz = grid[i * 3 + 2];
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int n = ptr[own]; n < ptr[own + 1]; ++n) {
        const int J = nbr[n];
        const double dx = gx - atm[J * 3 + 0];
        const double dy = gy - atm[J * 3 + 1];
        const double dz = gz - atm[J * 3 + 2];
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        double d = (r - R_in[J]) / R_sw[J];
        if (d < 1e-8) continue;       // displacement zeroed
        const double g = dsw_h(d);
        if (g == 0.0) continue;       // d >= 1
        const double f = g / (sw_h(d) * r * R_sw[J]);
        const double fx = f * dx, fy = f * dy, fz = f * dz;
        sx += fx; sy += fy; sz += fz;
        atomicAdd(&out[J * 3 + 0], -w * fx);
        atomicAdd(&out[J * 3 + 1], -w * fy);
        atomicAdd(&out[J * 3 + 2], -w * fz);
    }
    atomicAdd(&out[own * 3 + 0], w * sx);
    atomicAdd(&out[own * 3 + 1], w * sy);
    atomicAdd(&out[own * 3 + 2], w * sz);
}
''', 'gost_contract_dareas')


def _owner_neighbours(gostshyp, surface):
    """CSR neighbour lists (atoms != a that can switch points of atom a)."""
    cache = getattr(gostshyp, '_dareas_nbr', None)
    if cache is not None and cache[0] is surface['grid_coords']:
        return cache[1], cache[2]
    atom_coords = surface['atom_coords']
    natm = atom_coords.shape[0]
    owner = cp.asarray(gostshyp.atom_idx)
    r_pt = cp.linalg.norm(surface['grid_coords'] - atom_coords[owner], axis=1)
    r_max = cp.zeros(natm)
    cp.maximum.at(r_max, owner, r_pt)
    d_atm = cp.linalg.norm(atom_coords[:, None, :] - atom_coords[None, :, :], axis=2)
    reach = surface['R_in_J'] + surface['R_sw_J']
    nb = (d_atm - r_max[:, None]) < (reach[None, :] * (1 + 1e-12) + 1e-8)
    nb[cp.arange(natm), cp.arange(natm)] = False
    a, J = cp.nonzero(nb)
    ptr = cp.zeros(natm + 1, dtype=np.int32)
    cp.cumsum(nb.sum(axis=1), out=ptr[1:])
    nbr = cp.ascontiguousarray(J.astype(np.int32))
    gostshyp._dareas_nbr = (surface['grid_coords'], ptr, nbr)
    return ptr, nbr


def _apply_grid_coordinate_response(gostshyp, gtilde_grad, force_grad,
                                    gtilde_grid, force_grid, p_contracted,
                                    force_coeffs, dcoords):
    """Apply grid-center and normal response without duplicating cavity paths."""
    if dcoords is None:
        _scatter_add(gtilde_grad, gostshyp.atom_idx, -gtilde_grid)
        _scatter_add(force_grad, gostshyp.atom_idx, -force_grid)
        return cp.zeros_like(gtilde_grad)

    natm = gostshyp.mol.natm
    ngrids = gostshyp.n_gaussian
    normal_grad = cp.zeros_like(gtilde_grad)

    # Transfer the large host derivative tensor once per grid chunk and use it
    # for both Gaussian-center terms and the normal response.
    free_memory, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_grid = max(1, 9 * natm * np.dtype(np.float64).itemsize)
    chunk_size = max(1, min(ngrids, int(0.1 * free_memory / bytes_per_grid)))

    for p0 in range(0, ngrids, chunk_size):
        p1 = min(ngrids, p0 + chunk_size)
        dcoords_c = cp.asarray(np.ascontiguousarray(dcoords[..., p0:p1]))
        gtilde_grad -= cp.einsum(
            'gx,xaAg->Aa', gtilde_grid[p0:p1], dcoords_c)
        force_grad -= cp.einsum(
            'gx,xaAg->Aa', force_grid[p0:p1], dcoords_c)

        normals_c = gostshyp.surface_normals[p0:p1]
        q = force_coeffs[p0:p1, None] * p_contracted[p0:p1]
        q_tangent = q - normals_c * cp.sum(q * normals_c, axis=1)[:, None]
        q_tangent /= gostshyp.surface_distances[p0:p1, None]

        owners_c = gostshyp.atom_idx[p0:p1]
        _scatter_add(normal_grad, owners_c, q_tangent)
        normal_grad -= cp.einsum('gc,caAg->Aa', q_tangent, dcoords_c)

    return normal_grad


class Gradients(lib.StreamObject):
    """Nuclear gradients for GOSTSHYP solvation model."""

    def __init__(self, gostshyp_obj):
        self.gostshyp = gostshyp_obj
        self.mol = gostshyp_obj.mol

    def kernel(self, dm):
        """
        Compute GOSTSHYP nuclear gradients.

        Parameters
        ----------
        dm : ndarray
            Density matrix

        Returns
        -------
        gradient : ndarray of shape (natm, 3)
            Nuclear gradient contribution from GOSTSHYP
        """
        _w0 = logger.perf_counter()
        t0 = logger.init_timer(self)
        gostshyp = self.gostshyp
        mol = self.mol
        natm = mol.natm

        # Ensure dm is cupy on GPU
        dm = cp.asarray(dm)
        if dm.ndim == 3 and dm.shape[0] == 2:
            dm = dm[0] + dm[1]

        # Cached values from energy calculation (all cupy)
        forces = gostshyp.forces
        amplitudes = gostshyp.amplitudes
        gtilde_expval = gostshyp.gtilde_expval
        widths = gostshyp.widths
        areas = gostshyp.areas
        grid_coords = gostshyp.grid_coords
        normals = gostshyp.surface_normals
        atom_idx = gostshyp.atom_idx  # numpy int array
        intopt = gostshyp.intopt
        P = gostshyp.pressure_au
        cutoff = gostshyp.overlap_cutoff
        ngrids = len(areas)
        aoslice = mol.aoslice_by_atom()

        # Normalize PCM and MOIST derivatives to [natm, 3, ngrids].
        # DROP coordinate derivatives stay on the host and are transferred once
        # per chunk when all coordinate-dependent terms are available.
        if gostshyp.cavity == 'drop':
            _, dcoords = _get_surface_derivatives(gostshyp)
        else:
            dcoords = None

        # Width gradient prefactors
        wgrad_prefs = -cp.float64(np.pi * np.log(2)) / (areas ** 2)

        # ==================================================================
        # Term dE1: Area derivative
        # ==================================================================
        # Collected into w_area below; all area-derivative terms are
        # contracted with dA/dR in a single pass.
        w_area = P * gtilde_expval / forces

        # ==================================================================
        # Term dE2: Gtilde operator gradient
        # ==================================================================

        # One fused screened pass for all density contractions (aux l=0..3),
        # reduced in-kernel to the combinations used below.
        screen = gostshyp.get_screen()
        gd = screen.grad_density(mol, dm, normals)
        s_val = gd[:, 0]
        p_contracted = gd[:, 1:4]
        d_dot_n = gd[:, 4:7]   # sum_q d_xq n_q
        trace_d = gd[:, 7]
        f_trace = gd[:, 8:11]  # (xf, yf, zf)

        coeffs = -2.0 * P * areas * gtilde_expval * widths / (forces * forces)
        weighted_amp = coeffs[:, None] * normals

        # --- Part A (dE2 + dE3): AO center derivatives of the s- and p-type
        # operators, contracted with D + D^T and summed per atom in one pass.
        gtilde_operator_grad = screen.ip1_sp_atoms(
            mol, dm, amplitudes, weighted_amp)
        force_operator_grad = cp.zeros_like(gtilde_operator_grad)

        # --- Part B: Aux center derivative (ip2 of s-type overlap) ---
        # d/dC_x S(l=0) = 2*gamma * S(l=1, px), negated to nabla convention
        dgtilde_gaussian = amplitudes[:, None] * 2.0 * widths[:, None] * p_contracted

        # --- Part C: d-type width gradient ---
        imd = wgrad_prefs * amplitudes * trace_d
        w_area -= imd

        # ==================================================================
        # Term dE3: Force operator gradient
        # ==================================================================

        # sum_p dS_p/dC_x n_p = 2 gamma sum_p d_xp n_p - s n_x
        two_gamma = 2.0 * widths
        dG = coeffs[:, None] * (two_gamma[:, None] * d_dot_n
                                - s_val[:, None] * normals)

        # Apply both Gaussian-center responses together.  For DROP this also
        # adds normal response while reusing p_contracted from dE2.
        dE_normal = _apply_grid_coordinate_response(
            gostshyp, gtilde_operator_grad, force_operator_grad,
            dgtilde_gaussian, dG, p_contracted, coeffs, dcoords)
        gtilde_operator_grad *= -1.0  # dr -> -dR
        force_operator_grad *= -1.0  # dr -> -dR
        dE2 = gtilde_operator_grad

        # --- Part C: f-type width gradient ---
        f_coeffs = -2.0 * widths * wgrad_prefs
        dr = f_trace * f_coeffs[:, None]

        rf2 = 1.0 / (forces * forces)
        dr *= normals

        # f-type width term: -P sum_g A g~ / F^2 (wgrad F / w + sum_c dr_c) dA/dR
        w_area -= P * areas * gtilde_expval * rf2 * (
            wgrad_prefs * forces / widths + dr.sum(axis=1))

        dE3 = force_operator_grad + dE_normal

        gradient = dE2 + dE3 + _contract_dareas(gostshyp, w_area)

        # Full device sync to catch errors from non-default streams used by
        # int3c kernels before returning control to the solute gradient.
        cp.cuda.Device().synchronize()
        self.t_wall = logger.perf_counter() - _w0
        logger.timer(self, 'GOSTSHYP gradient', *t0)

        return cp.asnumpy(gradient)

    grad = kernel


def make_grad(gostshyp_obj, dm):
    """
    Compute GOSTSHYP gradient.

    Parameters
    ----------
    gostshyp_obj : GOSTSHYP
        GOSTSHYP solvent object
    dm : ndarray
        Density matrix

    Returns
    -------
    gradient : ndarray
        Nuclear gradients
    """
    return Gradients(gostshyp_obj).kernel(dm)


def make_grad_object(base_method):
    '''Create nuclear gradients object with solvent contributions for the given
    solvent-attached method based on its gradients method in vacuum
    '''
    from pyscf.grad.rhf import GradientsBase
    if isinstance(base_method, GradientsBase):
        base_method = base_method.base

    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not available for energy gradients')

    vac_grad = base_method.undo_solvent().Gradients()
    vac_grad.base = base_method
    name = with_solvent.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)


class WithSolventGrad:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        logger.debug(self, 'Compute gradients from solvents')
        self.de_solvent = self.base.with_solvent.grad(dm)
        logger.debug(self, 'Compute gradients from solutes')
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent

        if self.verbose >= logger.NOTE:
            from pyscf.grad import rhf as rhf_grad
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        pass
