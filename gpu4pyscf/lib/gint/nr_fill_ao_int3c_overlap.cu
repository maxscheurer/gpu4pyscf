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
 * Host drivers (extern "C", called through ctypes) for the spatially screened
 * 3-center overlap kernels of the GOSTSHYP model, see
 * gpu4pyscf/gto/int3c_overlap.py.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

// Screening threshold of the per-primitive prefactor, set together with
// c_bpcache by GINTset_int3c_overlap_constants.
__constant__ double c_overlap_cutoff;

#include "g3c_overlap_screened.cu"
#include "g3c_overlap_screened_t.cu"

extern "C" {

/*
 * Set the constant memory (c_bpcache, c_overlap_cutoff) used by all kernels
 * below. Called before every batch of launches over cp_ij_id.
 */
int GINTset_int3c_overlap_constants(
    const BasisProdCache* bpcache,
    const double cutoff
) {
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));
    checkCudaErrors(cudaMemcpyToSymbol(c_overlap_cutoff, &cutoff, sizeof(double)));
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

int GINTovlp_tile_size() { return OVLP_TILE; }

int GINTovlp_expand(const cudaStream_t stream, int* tiles_out, const int* pair_ptr,
                    const int* blk, const unsigned* mask, const int* blk_ptr,
                    const int tiles_per_block, const int npairs)
{
    if (npairs <= 0) return 0;
    GINTovlp_expand_kernel<<<(npairs + 3) / 4, 128, 0, stream>>>(
        tiles_out, pair_ptr, blk, mask, blk_ptr, tiles_per_block, npairs);
    return ovlp_check(__func__);
}

int GINTovlp_screen(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    int* counts, int* blk_out, unsigned* mask_out, const int* blk_ptr,
    const double* tile_sphere, const double* tile_gamma, const int ntiles,
    const double* super_sphere, const double* super_gamma, const int tiles_per_super,
    const double* block_sphere, const double* block_gamma, const int tiles_per_block,
    const double thresh, const int fill)
{
    const ContractionProdType* cp_ij = bpcache->cptype + cp_ij_id;
    const int nprim_ij = cp_ij->nprim_12;
    const int bas_ij0 = bpcache->bas_pairs_locs[cp_ij_id];
    const int ntasks_ij = bpcache->bas_pairs_locs[cp_ij_id + 1] - bas_ij0;
    const int prim_ij0 = bpcache->primitive_pairs_locs[cp_ij_id];
    if (ntasks_ij <= 0) return 0;
    const int nthreads = 128;  // 4 warps, one shell pair per warp
    const int nblocks = (ntasks_ij + 3) / 4;
    GINTovlp_screen_kernel<<<nblocks, nthreads, 0, stream>>>(
        counts, blk_out, mask_out, blk_ptr, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        tile_sphere, tile_gamma, ntiles, super_sphere, super_gamma, tiles_per_super,
        block_sphere, block_gamma, tiles_per_block, thresh, fill);
    return ovlp_check(__func__);
}

int GINTovlp_density_screened(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    double* out, const int ncomp, const double* dm, const int nao, const int kmax,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
    OVLP_TYPE_INFO
    if (i_l <= 2 && j_l <= 2 && kmax == 1) {
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

int GINTovlp_grad_density_screened(
    const cudaStream_t stream, const BasisProdCache* bpcache, const int cp_ij_id,
    double* out, const double* dm, const int nao, const double* normals,
    const int* pair_ptr, const int* pair_tiles,
    const double* coords, const double* exps, const int ngrids)
{
    OVLP_TYPE_INFO
    if (i_l <= 2 && j_l <= 2) {
        OVLP_T_SWITCH(GINTovlp_grad_density_screened_t_kernel, ,
            out, dm, nao, normals, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
            pair_ptr, pair_tiles, coords, exps, ngrids)
        return ovlp_check(__func__);
    }
    const int lt = i_l + j_l + 3;
    OVLP_SWITCH_L(lt, GINTovlp_grad_density_screened_kernel,
        out, dm, nao, normals, bas_ij0, prim_ij0, ntasks_ij, nprim_ij,
        i_l, j_l, pair_ptr, pair_tiles, coords, exps, ngrids)
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
