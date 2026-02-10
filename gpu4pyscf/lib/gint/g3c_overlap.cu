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
 * 3-center overlap integral kernels for GOSTSHYP solvation model.
 *
 * Computes (ij|k) = integral of chi_i(r) * chi_j(r) * G_k(r) dr
 * where chi_i, chi_j are AO basis functions and G_k is a surface Gaussian.
 *
 * Template strategy: Template on MAX_L_TOTAL = li + lj + lk to control buffer sizes,
 * but pass angular momenta as runtime parameters. This follows gpu4pyscf's pattern
 * and avoids combinatorial explosion of template instantiations.
 */

#pragma once

#include <cuda_runtime.h>
#include "cint2e.cuh"
#include "gint.h"

// Buffer size for recursion: need (L+2)^3 elements for each of x, y, z
// MAX_L_TOTAL can be up to 4+4+3=11 for g-orbital AO + f-type aux
#define OVERLAP_BUF_SIZE 2744  // (14)^3, supports up to L_total=11

// Use the global constant arrays from gpu4pyscf for Cartesian indexing
// c_idx is laid out as [x0, x1, ...][y0, y1, ...][z0, z1, ...] with TOT_NF elements each
// c_idx[i] gives x exponent, c_idx[TOT_NF + i] gives y, c_idx[2*TOT_NF + i] gives z
// c_l_locs[l] gives the starting index for angular momentum l

__device__ __forceinline__ int cart_idx3(int a, int b, int c, int stride) {
    // 3D index into recursion buffer
    return a * stride * stride + b * stride + c;
}

/*
 * Compute 3-center overlap recursion for a single primitive triplet.
 * Runtime angular momentum version - loops determined by l values at runtime.
 */
template <int MAX_L_TOTAL>
__device__ void compute_3c_overlap_recursion_general(
    const int li, const int lj, const int lk,
    const double GA[3], const double GB[3], const double GC[3],
    const double inv_2zeta,
    double* __restrict__ Sx, double* __restrict__ Sy, double* __restrict__ Sz
) {
    constexpr int STRIDE = MAX_L_TOTAL + 2;

    #define IDX(a, b, c) cart_idx3(a, b, c, STRIDE)

    // Base case
    Sx[IDX(0, 0, 0)] = 1.0;
    Sy[IDX(0, 0, 0)] = 1.0;
    Sz[IDX(0, 0, 0)] = 1.0;

    // Build up base cases for a=1, b=1, c=1
    if (li >= 1) {
        Sx[IDX(1, 0, 0)] = GA[0];
        Sy[IDX(1, 0, 0)] = GA[1];
        Sz[IDX(1, 0, 0)] = GA[2];
    }
    if (lj >= 1) {
        Sx[IDX(0, 1, 0)] = GB[0];
        Sy[IDX(0, 1, 0)] = GB[1];
        Sz[IDX(0, 1, 0)] = GB[2];
    }
    if (lk >= 1) {
        Sx[IDX(0, 0, 1)] = GC[0];
        Sy[IDX(0, 0, 1)] = GC[1];
        Sz[IDX(0, 0, 1)] = GC[2];
    }

    // Upward recursion in c (aux function) for a=b=0
    for (int c = 1; c < lk; ++c) {
        Sx[IDX(0, 0, c + 1)] = GC[0] * Sx[IDX(0, 0, c)] + c * inv_2zeta * Sx[IDX(0, 0, c - 1)];
        Sy[IDX(0, 0, c + 1)] = GC[1] * Sy[IDX(0, 0, c)] + c * inv_2zeta * Sy[IDX(0, 0, c - 1)];
        Sz[IDX(0, 0, c + 1)] = GC[2] * Sz[IDX(0, 0, c)] + c * inv_2zeta * Sz[IDX(0, 0, c - 1)];
    }

    // Upward recursion in b for all c, a=0
    if (lj >= 1) {
        for (int c = 1; c <= lk; ++c) {
            Sx[IDX(0, 1, c)] = GB[0] * Sx[IDX(0, 0, c)] + c * inv_2zeta * Sx[IDX(0, 0, c - 1)];
            Sy[IDX(0, 1, c)] = GB[1] * Sy[IDX(0, 0, c)] + c * inv_2zeta * Sy[IDX(0, 0, c - 1)];
            Sz[IDX(0, 1, c)] = GB[2] * Sz[IDX(0, 0, c)] + c * inv_2zeta * Sz[IDX(0, 0, c - 1)];
        }
        for (int b = 1; b < lj; ++b) {
            Sx[IDX(0, b + 1, 0)] = GB[0] * Sx[IDX(0, b, 0)] + b * inv_2zeta * Sx[IDX(0, b - 1, 0)];
            Sy[IDX(0, b + 1, 0)] = GB[1] * Sy[IDX(0, b, 0)] + b * inv_2zeta * Sy[IDX(0, b - 1, 0)];
            Sz[IDX(0, b + 1, 0)] = GB[2] * Sz[IDX(0, b, 0)] + b * inv_2zeta * Sz[IDX(0, b - 1, 0)];

            for (int c = 1; c <= lk; ++c) {
                Sx[IDX(0, b + 1, c)] = GB[0] * Sx[IDX(0, b, c)] + b * inv_2zeta * Sx[IDX(0, b - 1, c)]
                                      + c * inv_2zeta * Sx[IDX(0, b, c - 1)];
                Sy[IDX(0, b + 1, c)] = GB[1] * Sy[IDX(0, b, c)] + b * inv_2zeta * Sy[IDX(0, b - 1, c)]
                                      + c * inv_2zeta * Sy[IDX(0, b, c - 1)];
                Sz[IDX(0, b + 1, c)] = GB[2] * Sz[IDX(0, b, c)] + b * inv_2zeta * Sz[IDX(0, b - 1, c)]
                                      + c * inv_2zeta * Sz[IDX(0, b, c - 1)];
            }
        }
    }

    // Upward recursion in a for all b, c
    if (li >= 1) {
        for (int c = 1; c <= lk; ++c) {
            Sx[IDX(1, 0, c)] = GA[0] * Sx[IDX(0, 0, c)] + c * inv_2zeta * Sx[IDX(0, 0, c - 1)];
            Sy[IDX(1, 0, c)] = GA[1] * Sy[IDX(0, 0, c)] + c * inv_2zeta * Sy[IDX(0, 0, c - 1)];
            Sz[IDX(1, 0, c)] = GA[2] * Sz[IDX(0, 0, c)] + c * inv_2zeta * Sz[IDX(0, 0, c - 1)];
        }
        for (int b = 1; b <= lj; ++b) {
            Sx[IDX(1, b, 0)] = GA[0] * Sx[IDX(0, b, 0)] + b * inv_2zeta * Sx[IDX(0, b - 1, 0)];
            Sy[IDX(1, b, 0)] = GA[1] * Sy[IDX(0, b, 0)] + b * inv_2zeta * Sy[IDX(0, b - 1, 0)];
            Sz[IDX(1, b, 0)] = GA[2] * Sz[IDX(0, b, 0)] + b * inv_2zeta * Sz[IDX(0, b - 1, 0)];
        }
        for (int a = 1; a < li; ++a) {
            Sx[IDX(a + 1, 0, 0)] = GA[0] * Sx[IDX(a, 0, 0)] + a * inv_2zeta * Sx[IDX(a - 1, 0, 0)];
            Sy[IDX(a + 1, 0, 0)] = GA[1] * Sy[IDX(a, 0, 0)] + a * inv_2zeta * Sy[IDX(a - 1, 0, 0)];
            Sz[IDX(a + 1, 0, 0)] = GA[2] * Sz[IDX(a, 0, 0)] + a * inv_2zeta * Sz[IDX(a - 1, 0, 0)];

            for (int c = 1; c <= lk; ++c) {
                Sx[IDX(a + 1, 0, c)] = GA[0] * Sx[IDX(a, 0, c)] + a * inv_2zeta * Sx[IDX(a - 1, 0, c)]
                                      + c * inv_2zeta * Sx[IDX(a, 0, c - 1)];
                Sy[IDX(a + 1, 0, c)] = GA[1] * Sy[IDX(a, 0, c)] + a * inv_2zeta * Sy[IDX(a - 1, 0, c)]
                                      + c * inv_2zeta * Sy[IDX(a, 0, c - 1)];
                Sz[IDX(a + 1, 0, c)] = GA[2] * Sz[IDX(a, 0, c)] + a * inv_2zeta * Sz[IDX(a - 1, 0, c)]
                                      + c * inv_2zeta * Sz[IDX(a, 0, c - 1)];
            }

            for (int b = 1; b <= lj; ++b) {
                Sx[IDX(a + 1, b, 0)] = GA[0] * Sx[IDX(a, b, 0)] + a * inv_2zeta * Sx[IDX(a - 1, b, 0)]
                                      + b * inv_2zeta * Sx[IDX(a, b - 1, 0)];
                Sy[IDX(a + 1, b, 0)] = GA[1] * Sy[IDX(a, b, 0)] + a * inv_2zeta * Sy[IDX(a - 1, b, 0)]
                                      + b * inv_2zeta * Sy[IDX(a, b - 1, 0)];
                Sz[IDX(a + 1, b, 0)] = GA[2] * Sz[IDX(a, b, 0)] + a * inv_2zeta * Sz[IDX(a - 1, b, 0)]
                                      + b * inv_2zeta * Sz[IDX(a, b - 1, 0)];
            }
        }
    }

    // Fill remaining (a, b, c) with a >= 1, b >= 1, c >= 1
    if (li >= 1 && lj >= 1 && lk >= 1) {
        for (int b = 1; b <= lj; ++b) {
            for (int c = 1; c <= lk; ++c) {
                Sx[IDX(1, b, c)] = GA[0] * Sx[IDX(0, b, c)] + b * inv_2zeta * Sx[IDX(0, b - 1, c)]
                                  + c * inv_2zeta * Sx[IDX(0, b, c - 1)];
                Sy[IDX(1, b, c)] = GA[1] * Sy[IDX(0, b, c)] + b * inv_2zeta * Sy[IDX(0, b - 1, c)]
                                  + c * inv_2zeta * Sy[IDX(0, b, c - 1)];
                Sz[IDX(1, b, c)] = GA[2] * Sz[IDX(0, b, c)] + b * inv_2zeta * Sz[IDX(0, b - 1, c)]
                                  + c * inv_2zeta * Sz[IDX(0, b, c - 1)];
            }
        }

        for (int a = 1; a < li; ++a) {
            for (int b = 1; b <= lj; ++b) {
                for (int c = 1; c <= lk; ++c) {
                    Sx[IDX(a + 1, b, c)] = GA[0] * Sx[IDX(a, b, c)]
                                          + a * inv_2zeta * Sx[IDX(a - 1, b, c)]
                                          + b * inv_2zeta * Sx[IDX(a, b - 1, c)]
                                          + c * inv_2zeta * Sx[IDX(a, b, c - 1)];
                    Sy[IDX(a + 1, b, c)] = GA[1] * Sy[IDX(a, b, c)]
                                          + a * inv_2zeta * Sy[IDX(a - 1, b, c)]
                                          + b * inv_2zeta * Sy[IDX(a, b - 1, c)]
                                          + c * inv_2zeta * Sy[IDX(a, b, c - 1)];
                    Sz[IDX(a + 1, b, c)] = GA[2] * Sz[IDX(a, b, c)]
                                          + a * inv_2zeta * Sz[IDX(a - 1, b, c)]
                                          + b * inv_2zeta * Sz[IDX(a, b - 1, c)]
                                          + c * inv_2zeta * Sz[IDX(a, b, c - 1)];
                }
            }
        }
    }

    #undef IDX
}

/*
 * Main kernel for computing 3-center overlap integrals.
 * Template on MAX_L_TOTAL for buffer sizing, runtime angular momentum.
 */
template <int MAX_L_TOTAL>
__global__ void GINTfill_int3c_overlap_kernel_general(
    double* __restrict__ output,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int stride_j,
    const int stride_ij,
    const int ao_offsets_i,
    const int ao_offsets_j,
    const double* __restrict__ aux_coords,
    const double* __restrict__ aux_exponents
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) return;

    // Cartesian dimensions
    const int ncart_i = (i_l + 1) * (i_l + 2) / 2;
    const int ncart_j = (j_l + 1) * (j_l + 2) / 2;
    const int ncart_k = (k_l + 1) * (k_l + 2) / 2;

    // Get basis pair info from cache
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int ish = c_bpcache.bas_pair2bra[bas_ij];
    const int jsh = c_bpcache.bas_pair2ket[bas_ij];

    // AO indices
    const int ao_i = c_bpcache.ao_loc[ish] - ao_offsets_i;
    const int ao_j = c_bpcache.ao_loc[jsh] - ao_offsets_j;

    // Basis center coordinates
    const double* bas_x = c_bpcache.bas_coords;
    const double* bas_y = bas_x + c_bpcache.nbas;
    const double* bas_z = bas_y + c_bpcache.nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];
    const double Bx = bas_x[jsh];
    const double By = bas_y[jsh];
    const double Bz = bas_z[jsh];

    // Surface Gaussian center and exponent
    const double Cx = aux_coords[task_grid * 3 + 0];
    const double Cy = aux_coords[task_grid * 3 + 1];
    const double Cz = aux_coords[task_grid * 3 + 2];
    const double gamma = aux_exponents[task_grid];

    // Note: aux_norm is NOT applied here because the Python wrapper should pass
    // coefficients that match what libcint expects. The fakemol in CPU GOSTSHYP
    // uses ang_norm[l] as the raw contraction coefficient in _env, and libcint
    // reads it directly. Our kernel computes the raw integral and relies on
    // Python to scale appropriately.
    // TODO: Consider passing aux_coefficients array if needed.

    // Recursion buffers - size determined by MAX_L_TOTAL
    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE];
    double Sy[BUF_SIZE];
    double Sz[BUF_SIZE];

    // Use c_idx and c_l_locs for Cartesian component lookup
    const int i_loc = c_l_locs[i_l];
    const int j_loc = c_l_locs[j_l];
    const int k_loc = c_l_locs[k_l];

    // Local accumulation buffer - accumulate across primitives before atomicAdd
    // Max size: 15 (g-type) * 15 (g-type) * 10 (f-type aux) = 2250
    // For typical cases (up to d-type aux): 15 * 15 * 6 = 1350
    constexpr int MAX_LOCAL_BUF = 2250;
    double local_output[MAX_LOCAL_BUF];
    const int local_size = ncart_i * ncart_j * ncart_k;

    // Initialize local buffer to zero
    for (int idx = 0; idx < local_size; ++idx) {
        local_output[idx] = 0.0;
    }

    // Loop over primitives and accumulate to local buffer
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        const double alpha = c_bpcache.a1[ij];
        const double beta = c_bpcache.a2[ij];
        const double coeff_ij = c_bpcache.e12[ij];
        const double aij = alpha + beta;

        // Combined exponent for all three centers
        const double zeta = aij + gamma;
        const double inv_zeta = 1.0 / zeta;

        // Product center P = (alpha*A + beta*B) / aij
        const double Px = (alpha * Ax + beta * Bx) / aij;
        const double Py = (alpha * Ay + beta * By) / aij;
        const double Pz = (alpha * Az + beta * Bz) / aij;

        // PC^2 for exponential factor - compute early for screening
        const double PCx = Px - Cx, PCy = Py - Cy, PCz = Pz - Cz;
        const double PC2 = PCx * PCx + PCy * PCy + PCz * PCz;

        // Early prefactor screening - compute prefactor before recursion setup
        // Note: coeff_ij = c_bpcache.e12[ij] already contains:
        //   norm * ci * cj * exp(-dist_ij * alpha * beta / aij)
        const double pi_over_zeta = M_PI * inv_zeta;
        const double prefactor = sqrt(pi_over_zeta) * pi_over_zeta
                               * exp(-aij * gamma * inv_zeta * PC2)
                               * coeff_ij;

        if (fabs(prefactor) < 1e-20) continue;

        // Now compute quantities needed only for non-negligible contributions
        const double inv_2zeta = 0.5 * inv_zeta;

        // Combined center G = (aij*P + gamma*C) / zeta
        const double Gx = (aij * Px + gamma * Cx) * inv_zeta;
        const double Gy = (aij * Py + gamma * Cy) * inv_zeta;
        const double Gz = (aij * Pz + gamma * Cz) * inv_zeta;

        // Displacement vectors for recursion
        const double GA[3] = {Gx - Ax, Gy - Ay, Gz - Az};
        const double GB[3] = {Gx - Bx, Gy - By, Gz - Bz};
        const double GC[3] = {Gx - Cx, Gy - Cy, Gz - Cz};

        // Compute recursion
        compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
            i_l, j_l, k_l, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);

        // Accumulate to LOCAL buffer (not global memory)
        #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))

        for (int iK = 0; iK < ncart_k; ++iK) {
            const int k_cart = k_loc + iK;
            const int kx = c_idx[k_cart];
            const int ky = c_idx[k_cart + TOT_NF];
            const int kz = c_idx[k_cart + 2 * TOT_NF];

            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int j_cart = j_loc + iJ;
                const int jx = c_idx[j_cart];
                const int jy = c_idx[j_cart + TOT_NF];
                const int jz = c_idx[j_cart + 2 * TOT_NF];

                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int i_cart = i_loc + iI;
                    const int ix = c_idx[i_cart];
                    const int iy = c_idx[i_cart + TOT_NF];
                    const int iz = c_idx[i_cart + 2 * TOT_NF];

                    const double val = prefactor
                        * Sx[S_IDX(ix, jx, kx)]
                        * Sy[S_IDX(iy, jy, ky)]
                        * Sz[S_IDX(iz, jz, kz)];

                    // Accumulate to local buffer
                    const int local_idx = iK * ncart_j * ncart_i + iJ * ncart_i + iI;
                    local_output[local_idx] += val;
                }
            }
        }

        #undef S_IDX
    }

    // Write accumulated results to global memory with single atomicAdd per element
    for (int iK = 0; iK < ncart_k; ++iK) {
        for (int iJ = 0; iJ < ncart_j; ++iJ) {
            for (int iI = 0; iI < ncart_i; ++iI) {
                const int local_idx = iK * ncart_j * ncart_i + iJ * ncart_i + iI;
                const double val = local_output[local_idx];
                if (fabs(val) > 1e-20) {
                    // Output index: output[grid * ncart_k + iK, j, i]
                    const int out_idx = (ao_i + iI) + (ao_j + iJ) * stride_j
                                      + (task_grid * ncart_k + iK) * stride_ij;
                    atomicAdd(&output[out_idx], val);
                }
            }
        }
    }
}

/*
 * Density-contracted kernel: computes sum_ij D_ij * S_ijk -> forces[k]
 */
template <int MAX_L_TOTAL>
__global__ void GINTfill_int3c_overlap_density_contracted_kernel_general(
    double* __restrict__ forces,
    const double* __restrict__ dm,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int nao,
    const double* __restrict__ aux_coords,
    const double* __restrict__ aux_exponents
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) return;

    const int ncart_i = (i_l + 1) * (i_l + 2) / 2;
    const int ncart_j = (j_l + 1) * (j_l + 2) / 2;
    const int ncart_k = (k_l + 1) * (k_l + 2) / 2;

    // Get basis pair info
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int ish = c_bpcache.bas_pair2bra[bas_ij];
    const int jsh = c_bpcache.bas_pair2ket[bas_ij];

    // Use global AO indices for density matrix access (not local offsets)
    const int ao_i = c_bpcache.ao_loc[ish];
    const int ao_j = c_bpcache.ao_loc[jsh];

    // Basis coordinates
    const double* bas_x = c_bpcache.bas_coords;
    const double* bas_y = bas_x + c_bpcache.nbas;
    const double* bas_z = bas_y + c_bpcache.nbas;
    const double Ax = bas_x[ish], Ay = bas_y[ish], Az = bas_z[ish];
    const double Bx = bas_x[jsh], By = bas_y[jsh], Bz = bas_z[jsh];

    // Aux Gaussian
    const double Cx = aux_coords[task_grid * 3 + 0];
    const double Cy = aux_coords[task_grid * 3 + 1];
    const double Cz = aux_coords[task_grid * 3 + 2];
    const double gamma = aux_exponents[task_grid];

    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE];
    double Sy[BUF_SIZE];
    double Sz[BUF_SIZE];

    const int i_loc = c_l_locs[i_l];
    const int j_loc = c_l_locs[j_l];
    const int k_loc = c_l_locs[k_l];

    double result[15] = {0.0};  // Max ncart_k for f-orbitals is 10, use 15 for safety

    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        const double alpha = c_bpcache.a1[ij];
        const double beta = c_bpcache.a2[ij];
        const double coeff_ij = c_bpcache.e12[ij];
        const double aij = alpha + beta;

        const double zeta = aij + gamma;
        const double inv_zeta = 1.0 / zeta;

        const double Px = (alpha * Ax + beta * Bx) / aij;
        const double Py = (alpha * Ay + beta * By) / aij;
        const double Pz = (alpha * Az + beta * Bz) / aij;

        // PC^2 for exponential factor - compute early for screening
        const double PCx = Px - Cx, PCy = Py - Cy, PCz = Pz - Cz;
        const double PC2 = PCx * PCx + PCy * PCy + PCz * PCz;

        // Early prefactor screening
        const double pi_over_zeta = M_PI * inv_zeta;
        const double prefactor = sqrt(pi_over_zeta) * pi_over_zeta
                               * exp(-aij * gamma * inv_zeta * PC2)
                               * coeff_ij;

        if (fabs(prefactor) < 1e-20) continue;

        // Quantities needed only for non-negligible contributions
        const double inv_2zeta = 0.5 * inv_zeta;

        const double Gx = (aij * Px + gamma * Cx) * inv_zeta;
        const double Gy = (aij * Py + gamma * Cy) * inv_zeta;
        const double Gz = (aij * Pz + gamma * Cz) * inv_zeta;

        const double GA[3] = {Gx - Ax, Gy - Ay, Gz - Az};
        const double GB[3] = {Gx - Bx, Gy - By, Gz - Bz};
        const double GC[3] = {Gx - Cx, Gy - Cy, Gz - Cz};

        compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
            i_l, j_l, k_l, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);

        // c_idx layout: [all x][all y][all z] with TOT_NF elements each
        #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))

        for (int iK = 0; iK < ncart_k; ++iK) {
            const int k_cart = k_loc + iK;
            const int kx = c_idx[k_cart];
            const int ky = c_idx[k_cart + TOT_NF];
            const int kz = c_idx[k_cart + 2 * TOT_NF];

            double sum_ij = 0.0;
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int j_cart = j_loc + iJ;
                const int jx = c_idx[j_cart];
                const int jy = c_idx[j_cart + TOT_NF];
                const int jz = c_idx[j_cart + 2 * TOT_NF];

                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int i_cart = i_loc + iI;
                    const int ix = c_idx[i_cart];
                    const int iy = c_idx[i_cart + TOT_NF];
                    const int iz = c_idx[i_cart + 2 * TOT_NF];

                    const double s_val = Sx[S_IDX(ix, jx, kx)]
                                       * Sy[S_IDX(iy, jy, ky)]
                                       * Sz[S_IDX(iz, jz, kz)];

                    // dm is in row-major: dm[i, j] = dm[i * nao + j]
                    const int ii = ao_i + iI;
                    const int jj = ao_j + iJ;
                    const int dm_idx = ii * nao + jj;
                    double dm_contrib = dm[dm_idx] * s_val;

                    // For off-diagonal shell pairs (ish != jsh), also include
                    // the transposed contribution D[j,i] * S_jik
                    // Since S is symmetric, this is D[j,i] * S_ijk
                    if (ish != jsh) {
                        const int dm_idx_T = jj * nao + ii;
                        dm_contrib += dm[dm_idx_T] * s_val;
                    }

                    sum_ij += dm_contrib;
                }
            }
            result[iK] += prefactor * sum_ij;
        }

        #undef S_IDX
    }

    // Write results with atomic adds
    for (int iK = 0; iK < ncart_k; ++iK) {
        atomicAdd(&forces[task_grid * ncart_k + iK], result[iK]);
    }
}

/*
 * Amplitude-contracted kernel: computes sum_k amplitude_k * S_ijk -> fock[i,j]
 * Uses grid-striding to process multiple grids per thread, reducing atomic contention.
 */
template <int MAX_L_TOTAL>
__global__ void GINTfill_int3c_overlap_amplitude_contracted_kernel_general(
    double* __restrict__ fock,
    const double* __restrict__ amplitudes,
    const BasisProdOffsets offsets,
    const int i_l, const int j_l, const int k_l,
    const int nprim_ij,
    const int nao,
    const double* __restrict__ aux_coords,
    const double* __restrict__ aux_exponents
) {
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;

    if (task_ij >= ntasks_ij) return;

    const int ncart_i = (i_l + 1) * (i_l + 2) / 2;
    const int ncart_j = (j_l + 1) * (j_l + 2) / 2;
    const int ncart_k = (k_l + 1) * (k_l + 2) / 2;

    // Get basis pair info
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int ish = c_bpcache.bas_pair2bra[bas_ij];
    const int jsh = c_bpcache.bas_pair2ket[bas_ij];

    // Use global AO indices for Fock matrix access (not local offsets)
    const int ao_i = c_bpcache.ao_loc[ish];
    const int ao_j = c_bpcache.ao_loc[jsh];

    // Basis coordinates
    const double* bas_x = c_bpcache.bas_coords;
    const double* bas_y = bas_x + c_bpcache.nbas;
    const double* bas_z = bas_y + c_bpcache.nbas;
    const double Ax = bas_x[ish], Ay = bas_y[ish], Az = bas_z[ish];
    const double Bx = bas_x[jsh], By = bas_y[jsh], Bz = bas_z[jsh];

    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE];
    double Sy[BUF_SIZE];
    double Sz[BUF_SIZE];

    const int i_loc = c_l_locs[i_l];
    const int j_loc = c_l_locs[j_l];
    const int k_loc = c_l_locs[k_l];

    // Accumulate Fock contributions across ALL grids (max 15x15 for g-orbitals)
    double fock_ij[225] = {0.0};

    // Grid-striding loop: each thread processes multiple grid points
    const int grid_stride = gridDim.y * blockDim.y;
    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
         task_grid < ngrids;
         task_grid += grid_stride) {

        // Aux Gaussian for this grid point
        const double Cx = aux_coords[task_grid * 3 + 0];
        const double Cy = aux_coords[task_grid * 3 + 1];
        const double Cz = aux_coords[task_grid * 3 + 2];
        const double gamma = aux_exponents[task_grid];

        // Load amplitudes for this grid point
        double amp_k[15];
        for (int iK = 0; iK < ncart_k; ++iK) {
            amp_k[iK] = amplitudes[task_grid * ncart_k + iK];
        }

        // Loop over primitives for this grid point
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            const double alpha = c_bpcache.a1[ij];
            const double beta = c_bpcache.a2[ij];
            const double coeff_ij = c_bpcache.e12[ij];
            const double aij = alpha + beta;

            const double zeta = aij + gamma;
            const double inv_zeta = 1.0 / zeta;

            const double Px = (alpha * Ax + beta * Bx) / aij;
            const double Py = (alpha * Ay + beta * By) / aij;
            const double Pz = (alpha * Az + beta * Bz) / aij;

            // PC^2 for exponential factor - compute early for screening
            const double PCx = Px - Cx, PCy = Py - Cy, PCz = Pz - Cz;
            const double PC2 = PCx * PCx + PCy * PCy + PCz * PCz;

            // Early prefactor screening
            const double pi_over_zeta = M_PI * inv_zeta;
            const double prefactor = sqrt(pi_over_zeta) * pi_over_zeta
                                   * exp(-aij * gamma * inv_zeta * PC2)
                                   * coeff_ij;

            if (fabs(prefactor) < 1e-20) continue;

            // Quantities needed only for non-negligible contributions
            const double inv_2zeta = 0.5 * inv_zeta;

            const double Gx = (aij * Px + gamma * Cx) * inv_zeta;
            const double Gy = (aij * Py + gamma * Cy) * inv_zeta;
            const double Gz = (aij * Pz + gamma * Cz) * inv_zeta;

            const double GA[3] = {Gx - Ax, Gy - Ay, Gz - Az};
            const double GB[3] = {Gx - Bx, Gy - By, Gz - Bz};
            const double GC[3] = {Gx - Cx, Gy - Cy, Gz - Cz};

            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l, j_l, k_l, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);

            // c_idx layout: [all x][all y][all z] with TOT_NF elements each
            #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))

            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int j_cart = j_loc + iJ;
                const int jx = c_idx[j_cart];
                const int jy = c_idx[j_cart + TOT_NF];
                const int jz = c_idx[j_cart + 2 * TOT_NF];

                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int i_cart = i_loc + iI;
                    const int ix = c_idx[i_cart];
                    const int iy = c_idx[i_cart + TOT_NF];
                    const int iz = c_idx[i_cart + 2 * TOT_NF];

                    double val = 0.0;
                    for (int iK = 0; iK < ncart_k; ++iK) {
                        const int k_cart = k_loc + iK;
                        const int kx = c_idx[k_cart];
                        const int ky = c_idx[k_cart + TOT_NF];
                        const int kz = c_idx[k_cart + 2 * TOT_NF];

                        const double s_val = Sx[S_IDX(ix, jx, kx)]
                                           * Sy[S_IDX(iy, jy, ky)]
                                           * Sz[S_IDX(iz, jz, kz)];
                        val += amp_k[iK] * s_val;
                    }
                    fock_ij[iJ * ncart_i + iI] += prefactor * val;
                }
            }

            #undef S_IDX
        }
    }  // end grid-striding loop

    // Write to Fock matrix with atomic adds (once per thread, after all grids)
    // For off-diagonal shell pairs (ish != jsh), also write to transposed position
    for (int iJ = 0; iJ < ncart_j; ++iJ) {
        for (int iI = 0; iI < ncart_i; ++iI) {
            const double fval = fock_ij[iJ * ncart_i + iI];
            if (fabs(fval) < 1e-20) continue;  // Skip negligible values
            const int ii = ao_i + iI;
            const int jj = ao_j + iJ;
            const int fock_idx = ii * nao + jj;
            atomicAdd(&fock[fock_idx], fval);

            // For off-diagonal shell pairs, also fill in transposed element
            if (ish != jsh) {
                const int fock_idx_T = jj * nao + ii;
                atomicAdd(&fock[fock_idx_T], fval);
            }
        }
    }
}
