/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Driver functions for 3-center overlap integrals.
 * Provides extern "C" interface for Python ctypes binding.
 *
 * Template strategy: Template on MAX_L_TOTAL = li + lj + lk
 * This gives ~12 instantiations per kernel instead of 100+
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

// Define the __constant__ variable for overlap screening cutoff.
// This is the single definition; g3c_overlap.cu declares it extern.
__constant__ double c_overlap_cutoff;

#include "g3c_overlap.cu"
#include "g3c_overlap_screened.cu"
#include "g3c_overlap_screened_t.cu"

// Dispatch macro for MAX_L_TOTAL templated kernels
#define DISPATCH_BY_L_TOTAL(L_TOTAL, KERNEL_FUNC, ...) \
    case L_TOTAL: KERNEL_FUNC<L_TOTAL><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;

static int GINTfill_int3c_overlap_tasks(
    double* output,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int stride_j, const int stride_ij,
    const int ao_offsets_i, const int ao_offsets_j,
    const double* aux_coords, const double* aux_exponents,
    const cudaStream_t stream
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int l_total = i_l + j_l + k_l;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(0, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(10, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(11, GINTfill_int3c_overlap_kernel_general,
                            output, offsets, i_l, j_l, k_l, nprim_ij, stride_j, stride_ij,
                            ao_offsets_i, ao_offsets_j, aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range (max 11)\n", l_total);
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c_overlap_density_contracted_tasks(
    double* forces,
    const double* dm,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int nao,
    const double* aux_coords, const double* aux_exponents,
    const cudaStream_t stream
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int l_total = i_l + j_l + k_l;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(0, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(10, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(11, GINTfill_int3c_overlap_density_contracted_kernel_general,
                            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range (max 11)\n", l_total);
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c_overlap_amplitude_contracted_tasks(
    double* fock,
    const double* amplitudes,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int nao,
    const double* aux_coords, const double* aux_exponents,
    const cudaStream_t stream
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int l_total = i_l + j_l + k_l;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(0, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(10, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(11, GINTfill_int3c_overlap_amplitude_contracted_kernel_general,
                            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range (max 11)\n", l_total);
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


static int GINTfill_int3c_overlap_density_contracted_sp_tasks(
    double* forces_s,
    double* forces_p,
    const double* dm,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l,
    const int nprim_ij,
    const int nao,
    const double* aux_coords, const double* aux_exponents,
    const cudaStream_t stream
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    // Recursion needs l_total = i_l + j_l + 1 (always k_l=1)
    const int l_total = i_l + j_l + 1;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_density_contracted_sp_kernel_general,
                            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range for SP density kernel (max 9)\n", l_total);
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c_overlap_amplitude_contracted_sp_tasks(
    double* fock,
    const double* amp_s,
    const double* amp_p,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l,
    const int nprim_ij,
    const int nao,
    const double* aux_coords, const double* aux_exponents,
    const cudaStream_t stream
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int l_total = i_l + j_l + 1;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_amplitude_contracted_sp_kernel_general,
                            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
                            aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range for SP amplitude kernel (max 9)\n", l_total);
            return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}


extern "C" {

/*
 * Set constant memory for overlap kernels once, before the cp_ij_id loop.
 * This avoids redundant cudaMemcpyToSymbol calls per kernel launch.
 */
int GINTset_int3c_overlap_constants(
    const BasisProdCache* bpcache,
    const double cutoff
) {
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    checkCudaErrors(cudaMemcpyToSymbol(c_overlap_cutoff, &cutoff, sizeof(double)));
    return 0;
}

/*
 * Compute full 3-center overlap integral tensor.
 */
int GINTfill_int3c_overlap(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const int aux_l,
    double* integrals,
    const int* strides,
    const int* ao_offsets,
    const int* bins_locs_ij,
    int nbins,
    const int cp_ij_id,
    const double cutoff
) {
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int k_l = aux_l;
    const int nprim_ij = cp_ij->nprim_12;

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c_overlap_tasks(
            integrals, offsets, i_l, j_l, k_l, nprim_ij,
            strides[0], strides[1], ao_offsets[0], ao_offsets[1],
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}

/*
 * Compute density-contracted 3-center overlap: sum_ij D_ij * S_ijk -> forces[k]
 */
int GINTfill_int3c_overlap_density_contracted(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const int aux_l,
    const double* dm,
    double* forces,
    const int nao,
    const int* bins_locs_ij,
    int nbins,
    const int cp_ij_id,
    const double cutoff
) {
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int k_l = aux_l;
    const int nprim_ij = cp_ij->nprim_12;

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c_overlap_density_contracted_tasks(
            forces, dm, offsets, i_l, j_l, k_l, nprim_ij, nao,
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}

/*
 * Compute amplitude-contracted 3-center overlap: sum_k amplitude_k * S_ijk -> fock[i,j]
 */
int GINTfill_int3c_overlap_amplitude_contracted(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const int aux_l,
    const double* amplitudes,
    double* fock,
    const int nao,
    const int* bins_locs_ij,
    int nbins,
    const int cp_ij_id,
    const double cutoff
) {
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int k_l = aux_l;
    const int nprim_ij = cp_ij->nprim_12;

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c_overlap_amplitude_contracted_tasks(
            fock, amplitudes, offsets, i_l, j_l, k_l, nprim_ij, nao,
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}

/*
 * Fused density-contracted s+p: computes forces_s and forces_p in one pass
 */
int GINTfill_int3c_overlap_density_contracted_sp(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const double* dm,
    double* forces_s,
    double* forces_p,
    const int nao,
    const int* bins_locs_ij,
    int nbins,
    const int cp_ij_id,
    const double cutoff
) {
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nprim_ij = cp_ij->nprim_12;

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c_overlap_density_contracted_sp_tasks(
            forces_s, forces_p, dm, offsets, i_l, j_l, nprim_ij, nao,
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}

/*
 * Fused amplitude-contracted s+p: computes fock from amp_s and amp_p in one pass
 */
int GINTfill_int3c_overlap_amplitude_contracted_sp(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const double* amp_s,
    const double* amp_p,
    double* fock,
    const int nao,
    const int* bins_locs_ij,
    int nbins,
    const int cp_ij_id,
    const double cutoff
) {
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nprim_ij = cp_ij->nprim_12;

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;

    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) continue;

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c_overlap_amplitude_contracted_sp_tasks(
            fock, amp_s, amp_p, offsets, i_l, j_l, nprim_ij, nao,
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}


/* ---------------- spatially screened kernels ---------------- */

#define OVLP_SWITCH_L(LT, KERNEL, ...)                                         \
    switch (LT) {                                                              \
    case 0: KERNEL<0><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 1: KERNEL<1><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 2: KERNEL<2><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 3: KERNEL<3><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 4: KERNEL<4><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 5: KERNEL<5><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 6: KERNEL<6><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 7: KERNEL<7><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 8: KERNEL<8><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 9: KERNEL<9><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;     \
    case 10: KERNEL<10><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;   \
    case 11: KERNEL<11><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;   \
    default:                                                                   \
        fprintf(stderr, "%s: l_total = %d out of range\n", __func__, LT);      \
        return 1;                                                              \
    }

#define OVLP_TYPE_INFO                                                         \
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;             \
    const int i_l = cp_ij->l_bra;                                              \
    const int j_l = cp_ij->l_ket;                                              \
    const int nprim_ij = cp_ij->nprim_12;                                      \
    const int bas_ij0 = bpcache->bas_pairs_locs[cp_ij_id];                     \
    const int ntasks_ij = bpcache->bas_pairs_locs[cp_ij_id + 1] - bas_ij0;     \
    const int prim_ij0 = bpcache->primitive_pairs_locs[cp_ij_id];              \
    if (ntasks_ij <= 0) return 0;                                              \
    const dim3 threads(32 * OVLP_WARPS_PER_BLOCK);                             \
    const dim3 blocks((ntasks_ij + OVLP_WARPS_PER_BLOCK - 1) / OVLP_WARPS_PER_BLOCK);

// Dispatch over (LI, LJ) in {0,1,2}^2; extra template args follow LI, LJ.
#define OVLP_T_CASE(LI_, LJ_, KERNEL, EXTRA, ...)                              \
    case LI_ * 3 + LJ_:                                                        \
        KERNEL<LI_, LJ_ EXTRA><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;
#define OVLP_T_SWITCH(KERNEL, EXTRA, ...)                                      \
    switch (i_l * 3 + j_l) {                                                   \
    OVLP_T_CASE(0, 0, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(0, 1, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(0, 2, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(1, 0, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(1, 1, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(1, 2, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(2, 0, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(2, 1, KERNEL, EXTRA, __VA_ARGS__)                              \
    OVLP_T_CASE(2, 2, KERNEL, EXTRA, __VA_ARGS__)                              \
    }

extern "C++" {
template <int KMAX>
static void ovlp_launch_density_t(
    const int i_l, const int j_l, const dim3 blocks, const dim3 threads,
    const cudaStream_t stream, double* out, const double* dm, const int nao,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
#define OVLP_DENS_CASE(LI_, LJ_)                                               \
    case LI_ * 3 + LJ_:                                                        \
        GINTovlp_density_screened_t_kernel<LI_, LJ_, KMAX>                     \
            <<<blocks, threads, 0, stream>>>(out, dm, nao, bas_ij0, prim_ij0,  \
            ntasks_ij, nprim_ij, pair_ptr, pair_tiles, coords, exps, ngrids);  \
        break;
    switch (i_l * 3 + j_l) {
    OVLP_DENS_CASE(0, 0) OVLP_DENS_CASE(0, 1) OVLP_DENS_CASE(0, 2)
    OVLP_DENS_CASE(1, 0) OVLP_DENS_CASE(1, 1) OVLP_DENS_CASE(1, 2)
    OVLP_DENS_CASE(2, 0) OVLP_DENS_CASE(2, 1) OVLP_DENS_CASE(2, 2)
    }
#undef OVLP_DENS_CASE
}
}  // extern "C++"

static int ovlp_check(const char* name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", name, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GINTovlp_screen(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    int* counts, int* tiles_out, const int* pair_ptr,
    const double* tile_sphere, const double* tile_gamma, const int ntiles,
    const double* super_sphere, const double* super_gamma, const int tiles_per_super,
    const double thresh, const int fill)
{
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int nprim_ij = cp_ij->nprim_12;
    const int bas_ij0 = bpcache->bas_pairs_locs[cp_ij_id];
    const int ntasks_ij = bpcache->bas_pairs_locs[cp_ij_id + 1] - bas_ij0;
    const int prim_ij0 = bpcache->primitive_pairs_locs[cp_ij_id];
    if (ntasks_ij <= 0) return 0;
    const int nthreads = 128;
    GINTovlp_screen_kernel<<<(ntasks_ij + nthreads - 1) / nthreads, nthreads, 0, stream>>>(
        counts, tiles_out, pair_ptr, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        tile_sphere, tile_gamma, ntiles, super_sphere, super_gamma, tiles_per_super,
        thresh, fill);
    return ovlp_check(__func__);
}

int GINTovlp_density_screened(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    double* out, const int ncomp, const double* dm, const int nao, const int kmax,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
    OVLP_TYPE_INFO
    if (i_l <= 2 && j_l <= 2 && (kmax == 1 || kmax == 3)) {
        if (kmax == 1) {
            ovlp_launch_density_t<1>(i_l, j_l, blocks, threads, stream,
                out, dm, nao, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
                pair_ptr, pair_tiles, coords, exps, ngrids);
        } else {
            ovlp_launch_density_t<3>(i_l, j_l, blocks, threads, stream,
                out, dm, nao, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
                pair_ptr, pair_tiles, coords, exps, ngrids);
        }
        return ovlp_check(__func__);
    }
    const int lt = i_l + j_l + kmax;
    OVLP_SWITCH_L(lt, GINTovlp_density_screened_kernel,
        out, ncomp, dm, nao, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        i_l, j_l, kmax, pair_ptr, pair_tiles, coords, exps, ngrids)
    return ovlp_check(__func__);
}

int GINTovlp_fock_sp_screened(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    double* fock, const int nao, const double* amp_s, const double* amp_p,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
    OVLP_TYPE_INFO
    if (i_l <= 2 && j_l <= 2) {
        OVLP_T_SWITCH(GINTovlp_fock_sp_screened_t_kernel, ,
            fock, nao, amp_s, amp_p, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
            pair_ptr, pair_tiles, coords, exps, ngrids)
        return ovlp_check(__func__);
    }
    const int lt = i_l + j_l + 1;
    OVLP_SWITCH_L(lt, GINTovlp_fock_sp_screened_kernel,
        fock, nao, amp_s, amp_p, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        i_l, j_l, pair_ptr, pair_tiles, coords, exps, ngrids)
    return ovlp_check(__func__);
}

int GINTovlp_ip1_sp_atoms_screened(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    double* grad, const double* dsym, const int nao,
    const double* amp_s, const double* amp_p,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
    OVLP_TYPE_INFO
    if (i_l <= 2 && j_l <= 2) {
        OVLP_T_SWITCH(GINTovlp_ip1_sp_atoms_screened_t_kernel, ,
            grad, dsym, nao, amp_s, amp_p, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
            pair_ptr, pair_tiles, coords, exps, ngrids)
        return ovlp_check(__func__);
    }
    const int lt = i_l + j_l + 3;
    OVLP_SWITCH_L(lt, GINTovlp_ip1_sp_atoms_screened_kernel,
        grad, dsym, nao, amp_s, amp_p, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        i_l, j_l, pair_ptr, pair_tiles, coords, exps, ngrids)
    return ovlp_check(__func__);
}

}  // extern "C"
