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
3-center overlap integral derivative (ip1) interface for GOSTSHYP gradient.

This module provides GPU-accelerated computation of derivatives of 3-center
overlap integrals with respect to the bra (i) center, following PySCF's
nabla convention:

    nabla_1 S_ijk = -dS_ijk/dA

Functions:
    get_int3c_overlap_ip1_amplitude_contracted:
        Compute sum_k a_k * nabla_1 S_ijk -> [3, nao, nao]
"""

import ctypes
import cupy as cp
import numpy as np
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph
from gpu4pyscf.gto.int3c1e import VHFOpt

libgint = load_library('libgint')


def get_int3c_overlap_ip1_amplitude_contracted(
        mol, aux_coords, aux_exponents, aux_l, amplitudes, intopt):
    """
    Compute amplitude-contracted ip1 3-center overlap derivative
    (PySCF nabla convention):
        sum_k a_k * nabla_1 S_ijk -> [3, nao, nao]

    where nabla_1 = -d/dA (derivative w.r.t. bra center, negated).
    This matches PySCF's int3c1e_ip1 sign convention.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    aux_l : int
        Angular momentum of auxiliary functions
    amplitudes : ndarray of shape (ngrids, ncart_aux)
        Amplitudes for each auxiliary function (Cartesian)
    intopt : VHFOpt
        Integral options with precomputed basis pair data

    Returns
    -------
    fock : ndarray of shape (3, nao, nao)
        Derivative Fock matrix contributions (x, y, z components).
        Note: these are NOT symmetrized -- fock[x, i, j] contains the
        contribution from differentiating bra center of i. The caller
        must handle both bra and ket contributions.
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2

    amplitudes_gpu = cp.asarray(amplitudes, dtype=np.float64).ravel()
    assert amplitudes_gpu.shape[0] == ngrids * ncart_aux

    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')

    nao_cart = intopt._sorted_mol.nao
    fock_x = cp.zeros([nao_cart, nao_cart], dtype=np.float64, order='C')
    fock_y = cp.zeros([nao_cart, nao_cart], dtype=np.float64, order='C')
    fock_z = cp.zeros([nao_cart, nao_cart], dtype=np.float64, order='C')

    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue

        stream = streams[cp_ij_id % len(streams)]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        with stream:
            err = libgint.GINTfill_int3c_overlap_ip1_amplitude_contracted(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(aux_coords_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_exponents_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ngrids),
                ctypes.c_int(aux_l),
                ctypes.cast(amplitudes_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(fock_x.data.ptr, ctypes.c_void_p),
                ctypes.cast(fock_y.data.ptr, ctypes.c_void_p),
                ctypes.cast(fock_z.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id))

        if err != 0:
            raise RuntimeError(
                f'GINTfill_int3c_overlap_ip1_amplitude_contracted failed with error {err}')

    for stream in streams:
        stream.synchronize()

    # The kernel computes both bra and ket derivatives for off-diagonal pairs:
    #   fock[i,j] = d/dA_i S[i,j] (bra derivative)
    #   fock[j,i] = d/dA_j S[j,i] (ket derivative, stored at transposed position)
    # This produces the full asymmetric matrix matching CPU int3c1e_ip1 output.
    # Do NOT symmetrize.

    # Convert to spherical if needed
    if not mol.cart:
        c2s = intopt.cart2sph
        fock_x = c2s.T @ fock_x @ c2s
        fock_y = c2s.T @ fock_y @ c2s
        fock_z = c2s.T @ fock_z @ c2s

    # Unsort orbitals
    fock_x = intopt.unsort_orbitals(fock_x, axis=[0, 1])
    fock_y = intopt.unsort_orbitals(fock_y, axis=[0, 1])
    fock_z = intopt.unsort_orbitals(fock_z, axis=[0, 1])

    # Stack into [3, nao, nao] and negate to match PySCF nabla convention
    # The CUDA kernel computes d/dA (positive); PySCF uses nabla = -d/dA.
    fock = -cp.stack([fock_x, fock_y, fock_z], axis=0)
    return fock
