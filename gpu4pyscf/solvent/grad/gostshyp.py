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


def _contract_dareas(gostshyp, weights, chunk_bytes=1 << 28):
    """Return sum_g dA_g/dR_(a,x) * weights_g as [natm, 3].

    For PCM-type cavities (vdw, vdw/occ) this evaluates the area derivative of
    pcm.grad.get_dF_dA directly in grid chunks without materializing the dense
    [natm, 3, ngrids] tensor. DROP cavities use the cached MOIST derivatives.
    """
    if gostshyp.cavity == 'drop':
        dareas, _ = _get_surface_derivatives(gostshyp)
        return cp.einsum('acg,g->ac', dareas, weights)

    from gpu4pyscf.solvent.pcm import switch_h
    from gpu4pyscf.solvent.grad.pcm import grad_switch_h
    surface = (gostshyp._outer_surface
               if gostshyp._outer_surface is not None
               else gostshyp.surface)
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    R_in_J = surface['R_in_J']
    R_sw_J = surface['R_sw_J']
    natm = atom_coords.shape[0]
    ngrids = grid_coords.shape[0]
    wA = weights * surface['area']
    if gostshyp._occ_ratio_sq is not None:
        wA = wA * gostshyp._occ_ratio_sq
    owner = cp.asarray(gostshyp.atom_idx)

    out = cp.zeros((natm, 3))
    chunk = max(1, chunk_bytes // (natm * 3 * 8))
    for p0 in range(0, ngrids, chunk):
        p1 = min(ngrids, p0 + chunk)
        n = p1 - p0
        own = owner[p0:p1]
        rows = cp.arange(n)
        ri_rJ = grid_coords[p0:p1, None, :] - atom_coords[None, :, :]
        riJ = cp.linalg.norm(ri_rJ, axis=-1)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[rows, own] = 1.0
        small = diJ < 1e-8
        diJ[small] = 0.0
        ri_rJ[rows, own, :] = 0.0
        ri_rJ[small] = 0.0
        fiJ = switch_h(diJ)
        dfiJ = (grad_switch_h(diJ) / (fiJ * riJ * R_sw_J))[:, :, None] * ri_rJ
        w = wA[p0:p1]
        # owner-atom response: + A_i sum_J dfiJ ; other atoms: - A_i dfiJ
        cp.add.at(out, own, w[:, None] * dfiJ.sum(axis=1))
        out -= cp.einsum('i,iJx->Jx', w, dfiJ)
    return out


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

        # One fused screened pass for all density contractions (aux l=0..3)
        screen = gostshyp.get_screen()
        dens = screen.density(mol, dm, kmax=3)
        s_val = dens[:, 0]
        p_contracted = dens[:, 1:4]
        d_contracted = dens[:, 4:10]
        f_contracted = dens[:, 10:20]

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
        trace_d = d_contracted[:, 0] + d_contracted[:, 3] + d_contracted[:, 5]
        imd = wgrad_prefs * amplitudes * trace_d
        w_area -= imd

        # ==================================================================
        # Term dE3: Force operator gradient
        # ==================================================================

        # Build 3x3 derivative matrix on GPU: dS_p_dC[g, x, p]
        two_gamma = 2.0 * widths
        dS_p_dC = cp.zeros((ngrids, 3, 3))
        dS_p_dC[:, 0, 0] = two_gamma * d_contracted[:, 0] - s_val
        dS_p_dC[:, 0, 1] = two_gamma * d_contracted[:, 1]
        dS_p_dC[:, 0, 2] = two_gamma * d_contracted[:, 2]
        dS_p_dC[:, 1, 0] = two_gamma * d_contracted[:, 1]
        dS_p_dC[:, 1, 1] = two_gamma * d_contracted[:, 3] - s_val
        dS_p_dC[:, 1, 2] = two_gamma * d_contracted[:, 4]
        dS_p_dC[:, 2, 0] = two_gamma * d_contracted[:, 2]
        dS_p_dC[:, 2, 1] = two_gamma * d_contracted[:, 4]
        dS_p_dC[:, 2, 2] = two_gamma * d_contracted[:, 5] - s_val

        dG = cp.einsum('gxp,gp,g->gx', dS_p_dC, normals, coeffs)

        # Apply both Gaussian-center responses together.  For DROP this also
        # adds normal response while reusing p_contracted from dE2.
        dE_normal = _apply_grid_coordinate_response(
            gostshyp, gtilde_operator_grad, force_operator_grad,
            dgtilde_gaussian, dG, p_contracted, coeffs, dcoords)
        gtilde_operator_grad *= -1.0  # dr -> -dR
        force_operator_grad *= -1.0  # dr -> -dR
        dE2 = gtilde_operator_grad

        # --- Part C: f-type width gradient ---
        xf = f_contracted[:, 0] + f_contracted[:, 3] + f_contracted[:, 5]
        yf = f_contracted[:, 1] + f_contracted[:, 6] + f_contracted[:, 8]
        zf = f_contracted[:, 2] + f_contracted[:, 7] + f_contracted[:, 9]

        f_coeffs = -2.0 * widths * wgrad_prefs
        dr = cp.stack([xf, yf, zf], axis=1) * f_coeffs[:, None]

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
