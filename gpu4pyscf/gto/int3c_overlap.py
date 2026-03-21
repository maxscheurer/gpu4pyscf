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
3-center overlap integral interface for GOSTSHYP solvation model.

This module provides GPU-accelerated computation of 3-center overlap integrals:
    S_ijk = integral chi_i(r) * chi_j(r) * G_k(r) dr

where chi_i, chi_j are AO basis functions and G_k is a surface Gaussian.

Functions:
    get_int3c_overlap: Compute full integral tensor [ngrids, naux, nao, nao]
    get_int3c_overlap_density_contracted: Compute D_ij * S_ijk -> [ngrids, naux]
    get_int3c_overlap_amplitude_contracted: Compute a_k * S_ijk -> [nao, nao]

The auxiliary functions can be Cartesian (naux = ncart) or spherical (naux = 2l+1),
controlled by the aux_cart parameter. Default is Cartesian (aux_cart=True) to match
GOSTSHYP convention.
"""

import ctypes
import cupy as cp
import numpy as np
from pyscf import lib
from pyscf.lib import c_null_ptr
from gpu4pyscf.lib.cupy_helper import load_library, cart2sph, get_avail_mem
from gpu4pyscf.gto.int3c1e import VHFOpt

libgint = load_library('libgint')


def _get_aux_counts(aux_l, aux_cart):
    """Get number of auxiliary functions for Cartesian and spherical cases."""
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2
    nsph_aux = 2 * aux_l + 1
    naux = ncart_aux if aux_cart else nsph_aux
    return ncart_aux, nsph_aux, naux


def get_int3c_overlap(mol, aux_coords, aux_exponents, aux_l, intopt, aux_cart=True, cutoff=1e-14):
    """
    Compute 3-center overlap integrals with GPU acceleration.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary (surface) Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    aux_l : int
        Angular momentum of auxiliary functions (0=s, 1=p, 2=d, 3=f)
    intopt : VHFOpt
        Integral options with precomputed basis pair data
    aux_cart : bool, optional
        If True (default), auxiliary functions are Cartesian (ncart = (l+1)(l+2)/2).
        If False, auxiliary functions are spherical harmonics (nsph = 2l+1).

    Returns
    -------
    int3c : ndarray of shape (ngrids, naux, nao, nao)
        3-center overlap integrals. naux = ncart if aux_cart else nsph.
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]
    ncart_aux, nsph_aux, naux = _get_aux_counts(aux_l, aux_cart)

    # Memory management (use ncart_aux for kernel, convert later if needed)
    total_double_number = ngrids * ncart_aux * nao * nao
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    reserved_memory = avail_mem // 4
    allowed_doubles = reserved_memory // 8

    n_grid_split = max(1, int(np.ceil(total_double_number / allowed_doubles)))
    if n_grid_split > 100:
        raise MemoryError(
            f"Available GPU memory ({avail_mem / 1e9:.1f} GB) insufficient for "
            f"3-center overlap ({total_double_number * 8 / 1e9:.1f} GB required)"
        )
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    # Allocate pinned memory for output (final size with naux)
    buf_size = ngrids * naux * nao * nao
    int3c_pinned = cp.cuda.alloc_pinned_memory(buf_size * 8)
    int3c = np.frombuffer(int3c_pinned, np.float64, buf_size).reshape(
        [ngrids * naux, nao, nao], order='C')

    # Transfer coordinates and exponents to GPU
    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')

    # Set constant memory once before the kernel loop
    libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))

    # Create stream pool for concurrent kernel launches
    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        # Kernel always computes in Cartesian
        int3c_slice = cp.zeros([p1 - p0, ncart_aux, nao, nao], order='C')

        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            if len(log_q_ij) == 0:
                continue

            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            # Use streams in round-robin fashion for concurrent execution
            stream = streams[cp_ij_id % len(streams)]

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
            ni, nj = i1 - i0, j1 - j0

            ao_offsets = np.array([i0, j0], dtype=np.int32)
            strides = np.array([ni, ni * nj], dtype=np.int32)

            int3c_angular = cp.zeros([p1 - p0, ncart_aux, nj, ni], order='C')

            coords_slice = aux_coords_gpu[p0:p1]
            exponents_slice = aux_exponents_gpu[p0:p1]

            with stream:
                err = libgint.GINTfill_int3c_overlap(
                    ctypes.cast(stream.ptr, ctypes.c_void_p),
                    intopt.bpcache,
                    ctypes.cast(coords_slice.data.ptr, ctypes.c_void_p),
                    ctypes.cast(exponents_slice.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(p1 - p0),
                    ctypes.c_int(aux_l),
                    ctypes.cast(int3c_angular.data.ptr, ctypes.c_void_p),
                    strides.ctypes.data_as(ctypes.c_void_p),
                    ao_offsets.ctypes.data_as(ctypes.c_void_p),
                    bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbins),
                    ctypes.c_int(cp_ij_id),
                    ctypes.c_double(cutoff))

            if err != 0:
                raise RuntimeError(f'GINTfill_int3c_overlap failed with error {err}')

            # Synchronize this stream before cart2sph and assignment
            stream.synchronize()

            # Convert AO indices to spherical if needed
            i0s, i1s = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
            j0s, j1s = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
            if not mol.cart:
                int3c_angular = cart2sph(int3c_angular, axis=2, ang=lj)
                int3c_angular = cart2sph(int3c_angular, axis=3, ang=li)

            # Reshape to combine grid and aux dimensions
            int3c_slice[:, :, j0s:j1s, i0s:i1s] = int3c_angular

        # Synchronize all streams before post-processing
        for stream in streams:
            stream.synchronize()

        # Symmetrize: copy from upper triangle [j,i] (j<=i) to lower triangle [i,j]
        # Shell pairs with aosym have j <= i, storing at position [j, i].
        # We copy to [i, j] to symmetrize.
        row, col = cp.triu_indices(nao, k=1)  # upper triangle, excluding diagonal (row < col)
        for k in range(ncart_aux):
            # Copy upper to lower: array[col, row] = array[row, col] where row < col
            int3c_slice[:, k, col, row] = int3c_slice[:, k, row, col]

        # Convert auxiliary to spherical if needed
        if not aux_cart:
            int3c_slice = cart2sph(int3c_slice, axis=1, ang=aux_l)

        # Unsort orbitals and copy to host
        int3c_slice = int3c_slice.reshape([(p1 - p0) * naux, nao, nao])
        int3c_slice = intopt.unsort_orbitals(int3c_slice, axis=[1, 2])
        int3c_slice.get(out=int3c[p0 * naux:(p1) * naux])

    return int3c.reshape([ngrids, naux, nao, nao])


def get_int3c_overlap_density_contracted(mol, aux_coords, aux_exponents, aux_l, dm, intopt, cutoff=1e-14):
    """
    Compute density-contracted 3-center overlap: sum_ij D_ij * S_ijk

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
    dm : ndarray of shape (nao, nao)
        Density matrix
    intopt : VHFOpt
        Integral options with precomputed basis pair data

    Returns
    -------
    forces : ndarray of shape (ngrids, ncart_aux)
        Contracted integrals (e.g., forces per surface point).
        Auxiliary functions are always Cartesian.
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2

    # Prepare density matrix in sorted order
    dm = cp.asarray(dm)
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1] == nao

    dm_sorted = intopt.sort_orbitals(dm, [0, 1])
    if not mol.cart:
        c2s = intopt.cart2sph
        dm_sorted = c2s @ dm_sorted @ c2s.T

    nao_cart = intopt._sorted_mol.nao
    dm_flat = dm_sorted.flatten(order='C')  # Row-major for kernel access

    # Transfer to GPU
    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')
    forces = cp.zeros(ngrids * ncart_aux, dtype=np.float64)

    # Set constant memory once before the kernel loop
    libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))

    # Create stream pool for concurrent kernel launches
    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue

        # Use streams in round-robin fashion for concurrent execution
        stream = streams[cp_ij_id % len(streams)]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        with stream:
            err = libgint.GINTfill_int3c_overlap_density_contracted(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(aux_coords_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_exponents_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ngrids),
                ctypes.c_int(aux_l),
                ctypes.cast(dm_flat.data.ptr, ctypes.c_void_p),
                ctypes.cast(forces.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(cutoff))

        if err != 0:
            raise RuntimeError(f'GINTfill_int3c_overlap_density_contracted failed with error {err}')

    # Synchronize all streams before returning
    for stream in streams:
        stream.synchronize()

    return forces.reshape([ngrids, ncart_aux])


def get_int3c_overlap_amplitude_contracted(mol, aux_coords, aux_exponents, aux_l, amplitudes, intopt, cutoff=1e-14):
    """
    Compute amplitude-contracted 3-center overlap: sum_k a_k * S_ijk

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
        Amplitudes for each auxiliary function.
        Auxiliary functions are always Cartesian.
    intopt : VHFOpt
        Integral options with precomputed basis pair data

    Returns
    -------
    fock : ndarray of shape (nao, nao)
        Fock matrix contribution
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2

    amplitudes_gpu = cp.asarray(amplitudes, dtype=np.float64).ravel()
    assert amplitudes_gpu.shape[0] == ngrids * ncart_aux

    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')

    nao_cart = intopt._sorted_mol.nao
    fock_cart = cp.zeros([nao_cart, nao_cart], dtype=np.float64, order='C')

    # Set constant memory once before the kernel loop
    libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))

    # Create stream pool for concurrent kernel launches
    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue

        # Use streams in round-robin fashion for concurrent execution
        stream = streams[cp_ij_id % len(streams)]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        with stream:
            err = libgint.GINTfill_int3c_overlap_amplitude_contracted(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(aux_coords_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_exponents_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ngrids),
                ctypes.c_int(aux_l),
                ctypes.cast(amplitudes_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(fock_cart.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(cutoff))

        if err != 0:
            raise RuntimeError(f'GINTfill_int3c_overlap_amplitude_contracted failed with error {err}')

    # Synchronize all streams before post-processing
    for stream in streams:
        stream.synchronize()

    # Symmetrize
    row, col = np.tril_indices(nao_cart)
    fock_cart[row, col] = fock_cart[col, row]

    # Convert to spherical if needed
    if not mol.cart:
        c2s = intopt.cart2sph
        fock_cart = c2s.T @ fock_cart @ c2s

    # Unsort orbitals
    fock = intopt.unsort_orbitals(fock_cart, axis=[0, 1])
    return fock


def get_int3c_overlap_density_contracted_sp(mol, aux_coords, aux_exponents, dm, intopt, cutoff=1e-14):
    """
    Fused density-contracted 3-center overlap for s+p in a single kernel pass.

    Computes both s-type (aux_l=0) and p-type (aux_l=1) density contractions
    simultaneously, sharing the recursion work.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    dm : ndarray of shape (nao, nao)
        Density matrix
    intopt : VHFOpt
        Integral options with precomputed basis pair data

    Returns
    -------
    forces_s : ndarray of shape (ngrids, 1)
        s-type (aux_l=0) contracted integrals
    forces_p : ndarray of shape (ngrids, 3)
        p-type (aux_l=1) contracted integrals
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]

    dm = cp.asarray(dm)
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1] == nao

    dm_sorted = intopt.sort_orbitals(dm, [0, 1])
    if not mol.cart:
        c2s = intopt.cart2sph
        dm_sorted = c2s @ dm_sorted @ c2s.T

    nao_cart = intopt._sorted_mol.nao
    dm_flat = dm_sorted.flatten(order='C')

    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')
    forces_s = cp.zeros(ngrids, dtype=np.float64)
    forces_p = cp.zeros(ngrids * 3, dtype=np.float64)

    libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))

    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue

        stream = streams[cp_ij_id % len(streams)]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        with stream:
            err = libgint.GINTfill_int3c_overlap_density_contracted_sp(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(aux_coords_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_exponents_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ngrids),
                ctypes.cast(dm_flat.data.ptr, ctypes.c_void_p),
                ctypes.cast(forces_s.data.ptr, ctypes.c_void_p),
                ctypes.cast(forces_p.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(cutoff))

        if err != 0:
            raise RuntimeError(f'GINTfill_int3c_overlap_density_contracted_sp failed with error {err}')

    for stream in streams:
        stream.synchronize()

    return forces_s.reshape([ngrids, 1]), forces_p.reshape([ngrids, 3])


def get_int3c_overlap_amplitude_contracted_sp(mol, aux_coords, aux_exponents, amp_s, amp_p, intopt, cutoff=1e-14):
    """
    Fused amplitude-contracted 3-center overlap for s+p in a single kernel pass.

    Computes fock[i,j] = sum_g (amp_s[g] * S_ij0_g + sum_k amp_p[g,k] * S_ij1k_g)

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    amp_s : ndarray of shape (ngrids,)
        s-type amplitudes (aux_l=0)
    amp_p : ndarray of shape (ngrids, 3)
        p-type amplitudes (aux_l=1), Cartesian
    intopt : VHFOpt
        Integral options with precomputed basis pair data

    Returns
    -------
    fock : ndarray of shape (nao, nao)
        Fock matrix contribution
    """
    nao = mol.nao
    ngrids = aux_coords.shape[0]

    amp_s_gpu = cp.asarray(amp_s, dtype=np.float64).ravel()
    amp_p_gpu = cp.asarray(amp_p, dtype=np.float64).ravel()
    assert amp_s_gpu.shape[0] == ngrids
    assert amp_p_gpu.shape[0] == ngrids * 3

    aux_coords_gpu = cp.asarray(aux_coords, dtype=np.float64, order='C')
    aux_exponents_gpu = cp.asarray(aux_exponents, dtype=np.float64, order='C')

    nao_cart = intopt._sorted_mol.nao
    fock_cart = cp.zeros([nao_cart, nao_cart], dtype=np.float64, order='C')

    libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))

    n_streams = min(4, len(intopt.log_qs))
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]

    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue

        stream = streams[cp_ij_id % len(streams)]

        nbins = 1
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

        with stream:
            err = libgint.GINTfill_int3c_overlap_amplitude_contracted_sp(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(aux_coords_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_exponents_gpu.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ngrids),
                ctypes.cast(amp_s_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(amp_p_gpu.data.ptr, ctypes.c_void_p),
                ctypes.cast(fock_cart.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_cart),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(cutoff))

        if err != 0:
            raise RuntimeError(f'GINTfill_int3c_overlap_amplitude_contracted_sp failed with error {err}')

    for stream in streams:
        stream.synchronize()

    # Symmetrize
    row, col = np.tril_indices(nao_cart)
    fock_cart[row, col] = fock_cart[col, row]

    # Convert to spherical if needed
    if not mol.cart:
        c2s = intopt.cart2sph
        fock_cart = c2s.T @ fock_cart @ c2s

    # Unsort orbitals
    fock = intopt.unsort_orbitals(fock_cart, axis=[0, 1])
    return fock


def int3c_overlap(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=True,
                  dm=None, amplitudes=None, direct_scf_tol=1e-13, intopt=None,
                  cutoff=1e-14):
    """
    Main interface for 3-center overlap integrals.

    This is analogous to int1e_grids but for pure overlap (no Coulomb operator).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    aux_l : int, optional
        Angular momentum of auxiliary functions (default 0)
    aux_cart : bool, optional
        If True (default), auxiliary functions are Cartesian.
        If False, auxiliary functions are spherical harmonics.
        Only applies to full integral (not dm/amplitude contracted).
    dm : ndarray, optional
        If provided, contracts with density matrix (aux always Cartesian)
    amplitudes : ndarray, optional
        If provided, contracts with amplitudes (aux always Cartesian)
    direct_scf_tol : float, optional
        Tolerance for integral screening
    intopt : VHFOpt, optional
        Pre-built integral options

    Returns
    -------
    result : ndarray
        Full tensor [ngrids, naux, nao, nao] if no contraction,
        forces [ngrids, ncart_aux] if dm provided,
        fock [nao, nao] if amplitudes provided
    """
    if intopt is None:
        intopt = VHFOpt(mol)
        intopt.build(direct_scf_tol, aosym=True)
    else:
        assert isinstance(intopt, VHFOpt), "intopt must be a VHFOpt object"
        assert hasattr(intopt, "density_offset"), "Call intopt.build() first"
        assert intopt.aosym

    assert dm is None or amplitudes is None, \
        "Cannot contract with both dm and amplitudes simultaneously"

    if dm is None and amplitudes is None:
        return get_int3c_overlap(mol, aux_coords, aux_exponents, aux_l, intopt, aux_cart, cutoff)
    elif dm is not None:
        return get_int3c_overlap_density_contracted(
            mol, aux_coords, aux_exponents, aux_l, dm, intopt, cutoff)
    else:
        return get_int3c_overlap_amplitude_contracted(
            mol, aux_coords, aux_exponents, aux_l, amplitudes, intopt, cutoff)
