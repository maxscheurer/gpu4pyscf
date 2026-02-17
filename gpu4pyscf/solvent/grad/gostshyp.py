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
from pyscf.solvent.grad.pcm import get_dF_dA
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
    """Scatter-add values[g, :] into target[indices[g], :] on GPU.

    indices is a numpy int array (length ngrids), values is cupy (ngrids, 3).
    """
    indices_gpu = cp.asarray(indices)
    cp.add.at(target, indices_gpu, values)


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
        ngrids = len(areas)
        aoslice = mol.aoslice_by_atom()

        # ---- Area derivatives (CPU — PCM infrastructure) ----
        # get_dF_dA is a CPU function; pull the minimal surface data needed.
        surface_dict = {
            'grid_coords': cp.asnumpy(grid_coords),
            'area': cp.asnumpy(areas),
            'gslice_by_atom': gostshyp.surface['gslice_by_atom'],
            'R_vdw': cp.asnumpy(gostshyp.surface['R_vdw']),
            'switch_fun': cp.asnumpy(gostshyp.surface['switch_fun']),
            'R_in_J': cp.asnumpy(gostshyp.surface['R_in_J']),
            'R_sw_J': cp.asnumpy(gostshyp.surface['R_sw_J']),
            'atom_coords': cp.asnumpy(gostshyp.surface['atom_coords']),
        }
        _, dareas_np = get_dF_dA(surface_dict)
        dareas = cp.asarray(dareas_np.transpose(1, 2, 0))  # [natm, 3, ngrids] on GPU

        # Width gradient prefactors
        wgrad_prefs = -cp.float64(np.pi * np.log(2)) / (areas ** 2)

        # ==================================================================
        # Term dE1: Area derivative
        # ==================================================================
        dE1 = P * cp.einsum(
            'acg,g->ac', dareas, gtilde_expval / forces)

        # ==================================================================
        # Term dE2: Gtilde operator gradient
        # ==================================================================

        # --- Part A: AO center derivative (ip1 of s-type overlap) ---
        # ip1 returns nabla_1 = -d/dA (PySCF convention). Result is cupy.
        dPQ = int3c_overlap_ip.get_int3c_overlap_ip1_amplitude_contracted(
            mol, grid_coords, widths, aux_l=0,
            amplitudes=amplitudes[:, None], intopt=intopt)
        # dPQ: cupy [3, nao, nao]

        dgtilde_braket = cp.einsum('xij,ij->ix', dPQ, dm)
        dgtilde_braket += cp.einsum('xij,ji->ix', dPQ, dm)
        gtilde_operator_grad = _per_atom_sum(dgtilde_braket, aoslice)

        # --- Part B: Aux center derivative (ip2 of s-type overlap) ---
        # d/dC_x S(l=0) = 2*gamma * S(l=1, px), negated to nabla convention
        p_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, grid_coords, widths, aux_l=1, dm=dm, intopt=intopt)
        # p_contracted: cupy [ngrids, 3]
        dgtilde_gaussian = amplitudes[:, None] * 2.0 * widths[:, None] * p_contracted
        _scatter_add(gtilde_operator_grad, atom_idx, -dgtilde_gaussian)
        gtilde_operator_grad *= -1.0  # dr -> -dR

        # --- Part C: d-type width gradient ---
        d_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, grid_coords, widths, aux_l=2, dm=dm, intopt=intopt)
        # d_contracted: cupy [ngrids, 6]
        trace_d = d_contracted[:, 0] + d_contracted[:, 3] + d_contracted[:, 5]
        imd = wgrad_prefs * amplitudes * trace_d
        dE_d = -1.0 * cp.einsum('acg,g->ac', dareas, imd)

        dE2 = gtilde_operator_grad + dE_d

        # ==================================================================
        # Term dE3: Force operator gradient
        # ==================================================================

        coeffs = -2.0 * P * areas * gtilde_expval * widths / (forces * forces)

        # --- Part A: AO center derivative (ip1 of p-type overlap) ---
        # ip1 returns nabla_1 = -d/dA (PySCF convention). Result is cupy.
        weighted_amp = coeffs[:, None] * normals
        dpq = int3c_overlap_ip.get_int3c_overlap_ip1_amplitude_contracted(
            mol, grid_coords, widths, aux_l=1,
            amplitudes=weighted_amp, intopt=intopt)
        # dpq: cupy [3, nao, nao]

        dpq_ix = cp.einsum('xij,ij->ix', dpq, dm)
        dpq_ix += cp.einsum('xij,ji->ix', dpq, dm)
        force_operator_grad = _per_atom_sum(dpq_ix, aoslice)

        # --- Part B: Aux center derivative (ip2 of p-type overlap) ---
        s_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, grid_coords, widths, aux_l=0, dm=dm, intopt=intopt)
        s_val = s_contracted[:, 0]  # cupy [ngrids]

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
        _scatter_add(force_operator_grad, atom_idx, -dG)
        force_operator_grad *= -1.0  # dr -> -dR

        # --- Part C: f-type width gradient ---
        f_contracted = int3c_overlap.get_int3c_overlap_density_contracted(
            mol, grid_coords, widths, aux_l=3, dm=dm, intopt=intopt)
        # f_contracted: cupy [ngrids, 10]
        xf = f_contracted[:, 0] + f_contracted[:, 3] + f_contracted[:, 5]
        yf = f_contracted[:, 1] + f_contracted[:, 6] + f_contracted[:, 8]
        zf = f_contracted[:, 2] + f_contracted[:, 7] + f_contracted[:, 9]

        f_coeffs = -2.0 * widths * wgrad_prefs
        dr = cp.stack([xf, yf, zf], axis=1) * f_coeffs[:, None]

        rf2 = 1.0 / (forces * forces)
        dr *= normals

        dFdR = dareas * (wgrad_prefs * forces / widths)[None, None, :]
        dFdR += cp.einsum('gc,axg->axg', dr, dareas)

        width_grad_ftype = -P * cp.einsum(
            'g,g,axg,g->ax', areas, gtilde_expval, dFdR, rf2)

        dE3 = force_operator_grad + width_grad_ftype

        gradient = dE1 + dE2 + dE3
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
