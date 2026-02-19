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

}  // extern "C"
