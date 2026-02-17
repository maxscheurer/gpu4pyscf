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
 * 3-center overlap integral ip1 (derivative w.r.t. orbital center) kernels.
 *
 * Produces fock[x, i, j] = sum_k a_k * d/dA_i S_ijk
 * where d/dA_i is the derivative w.r.t. the center of orbital i.
 *
 * With aosym (ish >= jsh), the kernel computes:
 *   - Bra derivative: d/dA_ish -> fock[ao_i, ao_j]
 *   - Ket derivative: d/dA_jsh -> fock[ao_j, ao_i]  (for off-diagonal pairs)
 *
 * The bra derivative formula is:
 *   dS/dA_x = 2*alpha * S[ix+1,...] - ix * S[ix-1,...]
 * The ket derivative formula is:
 *   dS/dB_x = 2*beta  * S[...,jx+1,...] - jx * S[...,jx-1,...]
 *
 * The recursion is computed at (li+1, lj+1, lk) to have access to both.
 */

#pragma once

#include <cuda_runtime.h>
#include "cint2e.cuh"
#include "gint.h"

// Include the recursion function from g3c_overlap.cu
// (g3c_overlap.cu is #pragma once so safe to include)
#include "g3c_overlap.cu"

/*
 * Amplitude-contracted ip1 kernel.
 *
 * For each shell pair (ish, jsh) with ish >= jsh:
 *   - Computes bra derivative (d/dA_ish) -> fock[ao_i, ao_j]
 *   - For ish != jsh: also computes ket derivative (d/dA_jsh) -> fock[ao_j, ao_i]
 *   - For ish == jsh: diagonal loop naturally covers both orderings
 *
 * Template on MAX_L_TOTAL = (li+1) + (lj+1) + lk = li + lj + lk + 2.
 */
template <int MAX_L_TOTAL>
__global__ void GINTfill_int3c_overlap_ip1_amplitude_contracted_kernel_general(
    double* __restrict__ fock_x,
    double* __restrict__ fock_y,
    double* __restrict__ fock_z,
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
    const bool off_diagonal = (ish != jsh);

    // Use global AO indices for Fock matrix access
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

    // Bra derivative: fock[ao_i, ao_j] (ncart_i * ncart_j entries)
    double fock_ij_x[225] = {0.0};
    double fock_ij_y[225] = {0.0};
    double fock_ij_z[225] = {0.0};

    // Ket derivative: fock[ao_j, ao_i] (ncart_j * ncart_i entries) — only for off-diagonal
    double fock_ji_x[225] = {0.0};
    double fock_ji_y[225] = {0.0};
    double fock_ji_z[225] = {0.0};

    // Grid-striding loop
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

        // Loop over primitives
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

            const double PCx = Px - Cx, PCy = Py - Cy, PCz = Pz - Cz;
            const double PC2 = PCx * PCx + PCy * PCy + PCz * PCz;

            const double pi_over_zeta = M_PI * inv_zeta;
            const double prefactor = sqrt(pi_over_zeta) * pi_over_zeta
                                   * exp(-aij * gamma * inv_zeta * PC2)
                                   * coeff_ij;

            if (fabs(prefactor) < PRIMITIVE_OVERLAP_CUTOFF) continue;

            const double inv_2zeta = 0.5 * inv_zeta;

            const double Gx = (aij * Px + gamma * Cx) * inv_zeta;
            const double Gy = (aij * Py + gamma * Cy) * inv_zeta;
            const double Gz = (aij * Pz + gamma * Cz) * inv_zeta;

            const double GA[3] = {Gx - Ax, Gy - Ay, Gz - Az};
            const double GB[3] = {Gx - Bx, Gy - By, Gz - Bz};
            const double GC[3] = {Gx - Cx, Gy - Cy, Gz - Cz};

            // Compute recursion at (li+1, lj+1, lk) to access both
            // S[ix+1,...] (bra derivative) and S[...,jx+1,...] (ket derivative)
            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l + 1, j_l + 1, k_l, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);

            #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))

            const double two_alpha = 2.0 * alpha;
            const double two_beta = 2.0 * beta;

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

                    double bra_x = 0.0, bra_y = 0.0, bra_z = 0.0;
                    double ket_x = 0.0, ket_y = 0.0, ket_z = 0.0;

                    for (int iK = 0; iK < ncart_k; ++iK) {
                        const int k_cart = k_loc + iK;
                        const int kx = c_idx[k_cart];
                        const int ky = c_idx[k_cart + TOT_NF];
                        const int kz = c_idx[k_cart + 2 * TOT_NF];

                        const double Sy_val = Sy[S_IDX(iy, jy, ky)];
                        const double Sz_val = Sz[S_IDX(iz, jz, kz)];
                        const double Sx_val = Sx[S_IDX(ix, jx, kx)];

                        // === Bra derivative: d/dA_x = 2*alpha * S[ix+1,...] - ix * S[ix-1,...] ===
                        double dsx_bra = two_alpha * Sx[S_IDX(ix + 1, jx, kx)];
                        if (ix > 0) dsx_bra -= ix * Sx[S_IDX(ix - 1, jx, kx)];
                        bra_x += amp_k[iK] * dsx_bra * Sy_val * Sz_val;

                        double dsy_bra = two_alpha * Sy[S_IDX(iy + 1, jy, ky)];
                        if (iy > 0) dsy_bra -= iy * Sy[S_IDX(iy - 1, jy, ky)];
                        bra_y += amp_k[iK] * Sx_val * dsy_bra * Sz_val;

                        double dsz_bra = two_alpha * Sz[S_IDX(iz + 1, jz, kz)];
                        if (iz > 0) dsz_bra -= iz * Sz[S_IDX(iz - 1, jz, kz)];
                        bra_z += amp_k[iK] * Sx_val * Sy_val * dsz_bra;

                        // === Ket derivative: d/dB_x = 2*beta * S[ix,jx+1,...] - jx * S[ix,jx-1,...] ===
                        if (off_diagonal) {
                            double dsx_ket = two_beta * Sx[S_IDX(ix, jx + 1, kx)];
                            if (jx > 0) dsx_ket -= jx * Sx[S_IDX(ix, jx - 1, kx)];
                            ket_x += amp_k[iK] * dsx_ket * Sy_val * Sz_val;

                            double dsy_ket = two_beta * Sy[S_IDX(iy, jy + 1, ky)];
                            if (jy > 0) dsy_ket -= jy * Sy[S_IDX(iy, jy - 1, ky)];
                            ket_y += amp_k[iK] * Sx_val * dsy_ket * Sz_val;

                            double dsz_ket = two_beta * Sz[S_IDX(iz, jz + 1, kz)];
                            if (jz > 0) dsz_ket -= jz * Sz[S_IDX(iz, jz - 1, kz)];
                            ket_z += amp_k[iK] * Sx_val * Sy_val * dsz_ket;
                        }
                    }

                    // Bra derivative -> fock[ao_i, ao_j] (layout: [iJ * ncart_i + iI])
                    const int bra_idx = iJ * ncart_i + iI;
                    fock_ij_x[bra_idx] += prefactor * bra_x;
                    fock_ij_y[bra_idx] += prefactor * bra_y;
                    fock_ij_z[bra_idx] += prefactor * bra_z;

                    // Ket derivative -> fock[ao_j, ao_i] (layout: [iI * ncart_j + iJ])
                    if (off_diagonal) {
                        const int ket_idx = iI * ncart_j + iJ;
                        fock_ji_x[ket_idx] += prefactor * ket_x;
                        fock_ji_y[ket_idx] += prefactor * ket_y;
                        fock_ji_z[ket_idx] += prefactor * ket_z;
                    }
                }
            }

            #undef S_IDX
        }
    }  // end grid-striding loop

    // Write bra derivatives to fock[ao_i, ao_j]
    for (int iJ = 0; iJ < ncart_j; ++iJ) {
        for (int iI = 0; iI < ncart_i; ++iI) {
            const int bra_idx = iJ * ncart_i + iI;
            const int ii = ao_i + iI;
            const int jj = ao_j + iJ;
            const int fock_idx = ii * nao + jj;

            double fval;
            fval = fock_ij_x[bra_idx];
            if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                atomicAdd(&fock_x[fock_idx], fval);

            fval = fock_ij_y[bra_idx];
            if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                atomicAdd(&fock_y[fock_idx], fval);

            fval = fock_ij_z[bra_idx];
            if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                atomicAdd(&fock_z[fock_idx], fval);
        }
    }

    // Write ket derivatives to fock[ao_j, ao_i] (only for off-diagonal pairs)
    if (off_diagonal) {
        for (int iI = 0; iI < ncart_i; ++iI) {
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int ket_idx = iI * ncart_j + iJ;
                const int ii = ao_i + iI;
                const int jj = ao_j + iJ;
                const int fock_idx = jj * nao + ii;  // transposed: fock[ao_j, ao_i]

                double fval;
                fval = fock_ji_x[ket_idx];
                if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                    atomicAdd(&fock_x[fock_idx], fval);

                fval = fock_ji_y[ket_idx];
                if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                    atomicAdd(&fock_y[fock_idx], fval);

                fval = fock_ji_z[ket_idx];
                if (fabs(fval) >= PRIMITIVE_OVERLAP_CUTOFF)
                    atomicAdd(&fock_z[fock_idx], fval);
            }
        }
    }
}
