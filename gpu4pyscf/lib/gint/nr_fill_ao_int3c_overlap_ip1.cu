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
 * Driver functions for 3-center overlap ip1 (bra derivative) integrals.
 * Provides extern "C" interface for Python ctypes binding.
 *
 * MAX_L_TOTAL = (li+1) + lj + lk, so dispatch covers 0..12
 * (one more than non-derivative kernels which go up to 11).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "g3c_overlap_ip1.cu"

// Dispatch macro for MAX_L_TOTAL templated kernels
#define DISPATCH_BY_L_TOTAL(L_TOTAL, KERNEL_FUNC, ...) \
    case L_TOTAL: KERNEL_FUNC<L_TOTAL><<<blocks, threads, 0, stream>>>(__VA_ARGS__); break;

static int GINTfill_int3c_overlap_ip1_amplitude_contracted_tasks(
    double* fock_x,
    double* fock_y,
    double* fock_z,
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
    // l_total for the ip1 kernel: (li+1) + (lj+1) + lk (both bra and ket raised)
    const int l_total = (i_l + 1) + (j_l + 1) + k_l;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij + THREADSX - 1) / THREADSX, (ngrids + THREADSY - 1) / THREADSY);

    switch (l_total) {
        DISPATCH_BY_L_TOTAL(0, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(1, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(2, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(3, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(4, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(5, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(6, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(7, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(8, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(9, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(10, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(11, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(12, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        DISPATCH_BY_L_TOTAL(13, GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general,
                            fock_x, fock_y, fock_z, amplitudes, offsets, i_l, j_l, k_l,
                            nprim_ij, nao, aux_coords, aux_exponents)
        default:
            fprintf(stderr, "l_total = %d out of range (max 13) in ip1 kernel\n", l_total);
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
 * Compute amplitude-contracted ip1 3-center overlap:
 *   sum_k amplitude_k * dS_ijk/dA -> fock_x[nao,nao], fock_y[nao,nao], fock_z[nao,nao]
 */
int GINTfill_int3c_overlap_ip1_amplitude_contracted(
    const cudaStream_t stream,
    const BasisProdCache* bpcache,
    const double* aux_coords,
    const double* aux_exponents,
    const int ngrids,
    const int aux_l,
    const double* amplitudes,
    double* fock_x,
    double* fock_y,
    double* fock_z,
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

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    checkCudaErrors(cudaMemcpyToSymbol(c_overlap_cutoff, &cutoff, sizeof(double)));

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

        const int err = GINTfill_int3c_overlap_ip1_amplitude_contracted_tasks(
            fock_x, fock_y, fock_z, amplitudes, offsets,
            i_l, j_l, k_l, nprim_ij, nao,
            aux_coords, aux_exponents, stream);

        if (err != 0) return err;
    }

    return 0;
}

}  // extern "C"
