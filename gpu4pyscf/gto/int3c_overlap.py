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

            coords_slice = aux_coords_gpu[p0:p1]
            exponents_slice = aux_exponents_gpu[p0:p1]

            with stream:
                # Zero the output buffer on the kernel stream to avoid a race
                # condition: cp.zeros uses cudaMemsetAsync on the default stream,
                # but non-blocking streams don't synchronize with the default
                # stream. Allocating and zeroing here ensures the memset and
                # kernel launch are ordered on the same stream.
                int3c_angular = cp.zeros([p1 - p0, ncart_aux, nj, ni], order='C')
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


# ---------------------------------------------------------------------------
# Spatially screened kernels
# ---------------------------------------------------------------------------

OVLP_TILE = int(libgint.GINTovlp_tile_size())  # points per tile (CUDA side)
OVLP_TILES_PER_BLOCK = max(1, 32 // OVLP_TILE)   # 32 points per block
OVLP_TILES_PER_SUPER = max(1, 512 // OVLP_TILE)  # 512 points per superblock


def _part1by2(v):
    """Spread the low 21 bits of v (uint64) to every third bit."""
    v = v & np.uint64(0x1fffff)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100f00f00f00f00f)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10c30c30c30c30c3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def _morton_order(coords):
    lo = coords.min(axis=0)
    span = float((coords.max(axis=0) - lo).max()) + 1e-12
    q = ((coords - lo) * ((2**21 - 1) / span)).astype(cp.uint64)
    code = (_part1by2(q[:, 0]) | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))
    return cp.argsort(code)


_congruence_kernel = cp.RawKernel(r'''
extern "C" __global__
void ovlp_congruence(const double* __restrict__ src, double* __restrict__ dst,
                        const int n_out, const int n_src,
                        const int* __restrict__ rp, const int* __restrict__ ci,
                        const double* __restrict__ v, const double scale)
{
    // dst = scale * T src T^T,  T in CSR (n_out rows)
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_out * n_out) return;
    const int p = (int)(idx / n_out);
    const int q = (int)(idx % n_out);
    double s = 0.0;
    for (int i = rp[p]; i < rp[p + 1]; ++i) {
        const long long a = ci[i];
        const double va = v[i];
        for (int j = rp[q]; j < rp[q + 1]; ++j) {
            const long long b = ci[j];
            s += va * v[j] * src[a * n_src + b];
        }
    }
    dst[idx] = scale * s;
}
''', 'ovlp_congruence')


def _congruence(T, src, out, scale=1.0):
    """out = scale * T src T^T for a CSR matrix T (no temporaries)."""
    n_out, n_src = T.shape
    assert src.shape == (n_src, n_src) and src.flags.c_contiguous
    assert out.shape == (n_out, n_out) and out.flags.c_contiguous
    n = n_out * n_out
    _congruence_kernel(
        ((n + 255) // 256,), (256,),
        (src, out, np.int32(n_out), np.int32(n_src),
         T.indptr, T.indices, T.data, np.float64(scale)))
    return out


class OverlapScreen:
    """Rigorous (shell pair, grid tile) screening lists for one cavity.

    Surface Gaussians are Morton-sorted and grouped into tiles of OVLP_TILE
    points. A tile is kept for a shell pair if an upper bound of any primitive
    prefactor over the tile reaches 0.5 * cutoff, so every primitive skipped by
    screening would also be skipped by the kernels' own cutoff test.
    """

    def __init__(self, intopt, coords, exponents, cutoff):
        coords = cp.ascontiguousarray(cp.asarray(coords, dtype=np.float64))
        exponents = cp.ascontiguousarray(cp.asarray(exponents, dtype=np.float64))
        self.intopt = intopt
        self.cutoff = cutoff
        ngrids = self.ngrids = coords.shape[0]
        perm = _morton_order(coords)
        self.perm = perm
        self.coords = cp.ascontiguousarray(coords[perm])
        self.exps = cp.ascontiguousarray(exponents[perm])

        ntiles = self.ntiles = (ngrids + OVLP_TILE - 1) // OVLP_TILE
        pad = ntiles * OVLP_TILE - ngrids
        idx = cp.concatenate([cp.arange(ngrids), cp.full(pad, ngrids - 1)])
        tc = self.coords[idx].reshape(ntiles, OVLP_TILE, 3)
        tg = self.exps[idx].reshape(ntiles, OVLP_TILE)
        center = tc.mean(axis=1)
        radius = cp.sqrt(((tc - center[:, None]) ** 2).sum(axis=2)).max(axis=1)
        # inflate slightly to absorb rounding of center/radius
        radius = radius * (1 + 1e-12) + 1e-12
        self.tile_sphere = cp.ascontiguousarray(
            cp.hstack([center, radius[:, None]]))
        self.tile_gamma = cp.ascontiguousarray(
            cp.stack([tg.min(axis=1), tg.max(axis=1)], axis=1))

        # blocks and superblocks of consecutive tiles for pair-level pre-tests
        def group(S):
            ngroup = (ntiles + S - 1) // S
            gidx = cp.minimum(cp.arange(ngroup * S), ntiles - 1).reshape(ngroup, S)
            gc = center[gidx]
            gcenter = gc.mean(axis=1)
            gradius = (cp.sqrt(((gc - gcenter[:, None]) ** 2).sum(axis=2))
                       + radius[gidx]).max(axis=1)
            gradius = gradius * (1 + 1e-12) + 1e-12
            sphere = cp.ascontiguousarray(cp.hstack([gcenter, gradius[:, None]]))
            gamma = cp.ascontiguousarray(cp.stack(
                [self.tile_gamma[gidx, 0].min(axis=1),
                 self.tile_gamma[gidx, 1].max(axis=1)], axis=1))
            return sphere, gamma
        self.super_sphere, self.super_gamma = group(OVLP_TILES_PER_SUPER)
        self.block_sphere, self.block_gamma = group(OVLP_TILES_PER_BLOCK)

        npairs = int(intopt.bas_pairs_locs[-1])
        counts = cp.zeros(npairs, dtype=np.int32)
        thresh = 0.5 * cutoff
        libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))
        stream = cp.cuda.get_current_stream()
        null = ctypes.c_void_p(0)
        for fill in (0, 1):
            if fill:
                ptr = cp.zeros(npairs + 1, dtype=np.int32)
                cp.cumsum(counts, out=ptr[1:])
                nnz = int(ptr[-1])
                tiles = cp.empty(max(nnz, 1), dtype=np.int32)
            for cp_ij_id in range(len(intopt.log_qs)):
                err = libgint.GINTovlp_screen(
                    ctypes.c_void_p(stream.ptr), intopt.bpcache, ctypes.c_int(cp_ij_id),
                    ctypes.c_void_p(counts.data.ptr),
                    ctypes.c_void_p(tiles.data.ptr) if fill else null,
                    ctypes.c_void_p(ptr.data.ptr) if fill else null,
                    ctypes.c_void_p(self.tile_sphere.data.ptr),
                    ctypes.c_void_p(self.tile_gamma.data.ptr),
                    ctypes.c_int(ntiles),
                    ctypes.c_void_p(self.super_sphere.data.ptr),
                    ctypes.c_void_p(self.super_gamma.data.ptr),
                    ctypes.c_int(OVLP_TILES_PER_SUPER),
                    ctypes.c_void_p(self.block_sphere.data.ptr),
                    ctypes.c_void_p(self.block_gamma.data.ptr),
                    ctypes.c_int(OVLP_TILES_PER_BLOCK),
                    ctypes.c_double(thresh), ctypes.c_int(fill))
                if err != 0:
                    raise RuntimeError('GINTovlp_screen failed')
        self.pair_ptr = ptr
        self.pair_tiles = tiles
        self.nnz = nnz

    def matches(self, intopt, coords, cutoff):
        return (intopt is self.intopt and cutoff == self.cutoff
                and coords.shape[0] == self.ngrids)

    def _launch(self, fn, *args):
        intopt = self.intopt
        libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(self.cutoff))
        stream = cp.cuda.get_current_stream()
        tail = (ctypes.c_void_p(self.pair_ptr.data.ptr),
                ctypes.c_void_p(self.pair_tiles.data.ptr),
                ctypes.c_void_p(self.coords.data.ptr),
                ctypes.c_void_p(self.exps.data.ptr),
                ctypes.c_int(self.ngrids))
        for cp_ij_id in range(len(intopt.log_qs)):
            err = fn(ctypes.c_void_p(stream.ptr), intopt.bpcache,
                     ctypes.c_int(cp_ij_id), *args, *tail)
            if err != 0:
                raise RuntimeError(f'{fn.__name__} failed')

    def _transform(self, mol):
        """Sparse T (nao_cart_sorted x nao) with dm_cart_sorted = T dm T^T.

        T combines the AO sorting permutation with the block-diagonal
        cart2sph transformation of the sorted basis.
        """
        T = getattr(self, '_T', None)
        if T is not None and self._T_cart == mol.cart:
            return T
        import cupyx.scipy.sparse as sp
        intopt = self.intopt
        idx = cp.asarray(intopt._ao_idx)
        nao = len(intopt._ao_idx)
        nao_cart = intopt._sorted_mol.nao
        if mol.cart:
            rows = cp.arange(nao_cart)
            data = cp.ones(nao_cart)
            cols = idx
        else:
            c2s = sp.coo_matrix(intopt.cart2sph)
            rows, data = c2s.row, c2s.data
            cols = idx[c2s.col]
        T = sp.csr_matrix((data, (rows, cols)), shape=(nao_cart, nao))
        T.sort_indices()
        Tt = T.T.tocsr()
        Tt.sort_indices()
        for M in (T, Tt):
            M.indptr = M.indptr.astype(np.int32)
            M.indices = M.indices.astype(np.int32)
        self._T, self._Tt, self._T_cart = T, Tt, mol.cart
        # persistent work buffers: avoid ~GB cudaMalloc/cudaFree per call
        self._dm_buf = cp.empty((nao_cart, nao_cart))
        self._fock_buf = cp.empty((nao_cart, nao_cart))
        return T

    def _sorted_cart_dm(self, mol, dm, scale=1.0):
        """scale * T dm T^T (sorted Cartesian) in the persistent buffer.

        Only the symmetric part of dm enters the overlap contractions: the
        kernels form D_ij + D_ji themselves, so no symmetrization is needed.
        """
        T = self._transform(mol)
        dm = cp.ascontiguousarray(cp.asarray(dm, dtype=np.float64))
        return _congruence(T, dm, self._dm_buf, scale)

    def _unsort_fock(self, mol, fock_c):
        """T^T F T for the (symmetric) sorted-Cartesian Fock matrix F."""
        self._transform(mol)
        Tt = self._Tt
        out = cp.empty((Tt.shape[0], Tt.shape[0]))
        return _congruence(Tt, fock_c, out)

    def density(self, mol, dm, kmax):
        """sum_ij D_ij S_ij,(g,c) for aux l = 0..kmax -> [ngrids, ncomp]."""
        ncomp = (kmax + 1) * (kmax + 2) * (kmax + 3) // 6
        dm_c = self._sorted_cart_dm(mol, dm)
        nao_cart = dm_c.shape[0]
        out = cp.zeros((ncomp, self.ngrids))  # component-major for coalescing
        self._launch(libgint.GINTovlp_density_screened,
                     ctypes.c_void_p(out.data.ptr), ctypes.c_int(ncomp),
                     ctypes.c_void_p(dm_c.data.ptr), ctypes.c_int(nao_cart),
                     ctypes.c_int(kmax))
        res = cp.empty((self.ngrids, ncomp))
        res[self.perm] = out.T
        return res

    def fock_sp(self, mol, amp_s, amp_p):
        intopt = self.intopt
        amp_s = cp.ascontiguousarray(cp.asarray(amp_s, dtype=np.float64).ravel()[self.perm])
        amp_p = cp.ascontiguousarray(
            cp.asarray(amp_p, dtype=np.float64).reshape(-1, 3)[self.perm])
        nao_cart = intopt._sorted_mol.nao
        self._transform(mol)
        fock = self._fock_buf
        fock.fill(0)
        self._launch(libgint.GINTovlp_fock_sp_screened,
                     ctypes.c_void_p(fock.data.ptr), ctypes.c_int(nao_cart),
                     ctypes.c_void_p(amp_s.data.ptr), ctypes.c_void_p(amp_p.data.ptr))
        # The kernel writes both (i,j) and (j,i), so fock is already symmetric.
        return self._unsort_fock(mol, fock)

    def ip1_sp_atoms(self, mol, dm, amp_s, amp_p):
        """-sum over AO-center derivatives contracted with D+D^T, per atom.

        Equals the per-atom sum of einsum('xij,ij->ix') + einsum('xij,ji->ix')
        of get_int3c_overlap_ip1_amplitude_contracted (s amplitudes amp_s plus
        p amplitudes amp_p).
        """
        dsym = self._sorted_cart_dm(mol, dm)  # kernel forms D + D^T
        amp_s = cp.ascontiguousarray(cp.asarray(amp_s, dtype=np.float64).ravel()[self.perm])
        amp_p = cp.ascontiguousarray(
            cp.asarray(amp_p, dtype=np.float64).reshape(-1, 3)[self.perm])
        grad = cp.zeros((mol.natm, 3))
        self._launch(libgint.GINTovlp_ip1_sp_atoms_screened,
                     ctypes.c_void_p(grad.data.ptr),
                     ctypes.c_void_p(dsym.data.ptr), ctypes.c_int(dsym.shape[0]),
                     ctypes.c_void_p(amp_s.data.ptr), ctypes.c_void_p(amp_p.data.ptr))
        return -grad
