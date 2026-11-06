/*
 * Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
 * Screened 3-center overlap kernels specialized at compile time on the AO
 * angular momenta (LI, LJ). Recursion buffers are sized exactly
 * (LA+1)(LB+1)(LC+1) and all loops are unrolled so that they live in
 * registers. Same arithmetic and screening as g3c_overlap_screened.cu.
 * Included from nr_fill_ao_int3c_overlap.cu after g3c_overlap_screened.cu.
 */

// PySCF Cartesian ordering: lx = l..0, ly = l-lx..0, lz = l-lx-ly
__host__ __device__ constexpr int ovlp_cart_comp(int l, int n, int axis)
{
    int idx = 0;
    for (int lx = l; lx >= 0; --lx) {
        for (int ly = l - lx; ly >= 0; --ly) {
            if (idx == n) return axis == 0 ? lx : (axis == 1 ? ly : l - lx - ly);
            ++idx;
        }
    }
    return 0;
}

/*
 * 1D Obara-Saika recursion for the 3-center overlap:
 *   S(a+1,b,c) = GA S(a,b,c) + 1/(2z) (a S(a-1,b,c) + b S(a,b-1,c) + c S(a,b,c-1))
 * and analogous increments in b (GB) and c (GC).
 * Layout: S[(a * (LB+1) + b) * (LC+1) + c].
 */
template <int LA, int LB, int LC>
__device__ __forceinline__ void ovlp_os1d(const double ga, const double gb,
                                          const double gc, const double i2z,
                                          double* __restrict__ S)
{
    constexpr int NB = LB + 1, NC = LC + 1;
    #define T_IDX(a, b, c) (((a) * NB + (b)) * NC + (c))
    S[T_IDX(0, 0, 0)] = 1.0;
    #pragma unroll
    for (int c = 0; c < LC; ++c) {
        double v = gc * S[T_IDX(0, 0, c)];
        if (c > 0) v += c * i2z * S[T_IDX(0, 0, c - 1)];
        S[T_IDX(0, 0, c + 1)] = v;
    }
    #pragma unroll
    for (int b = 0; b < LB; ++b) {
        #pragma unroll
        for (int c = 0; c <= LC; ++c) {
            double v = gb * S[T_IDX(0, b, c)];
            if (b > 0) v += b * i2z * S[T_IDX(0, b - 1, c)];
            if (c > 0) v += c * i2z * S[T_IDX(0, b, c - 1)];
            S[T_IDX(0, b + 1, c)] = v;
        }
    }
    #pragma unroll
    for (int a = 0; a < LA; ++a) {
        #pragma unroll
        for (int b = 0; b <= LB; ++b) {
            #pragma unroll
            for (int c = 0; c <= LC; ++c) {
                double v = ga * S[T_IDX(a, b, c)];
                if (a > 0) v += a * i2z * S[T_IDX(a - 1, b, c)];
                if (b > 0) v += b * i2z * S[T_IDX(a, b - 1, c)];
                if (c > 0) v += c * i2z * S[T_IDX(a, b, c - 1)];
                S[T_IDX(a + 1, b, c)] = v;
            }
        }
    }
    #undef T_IDX
}

#define OVLP_T_PAIR_SETUP                                                    \
    const int lane = threadIdx.x & 31;                                        \
    const int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;            \
    if (task >= ntasks_ij) return;                                            \
    const int bas_ij = bas_ij0 + task;                                        \
    const int prim_ij = prim_ij0 + task * nprim_ij;                           \
    const int ish = c_bpcache.bas_pair2bra[bas_ij];                           \
    const int jsh = c_bpcache.bas_pair2ket[bas_ij];                           \
    const int ao_i = c_bpcache.ao_loc[ish];                                   \
    const int ao_j = c_bpcache.ao_loc[jsh];                                   \
    const double* bas_x = c_bpcache.bas_coords;                               \
    const double* bas_y = bas_x + c_bpcache.nbas;                             \
    const double* bas_z = bas_y + c_bpcache.nbas;                             \
    const double Ax = bas_x[ish], Ay = bas_y[ish], Az = bas_z[ish];           \
    const double Bx = bas_x[jsh], By = bas_y[jsh], Bz = bas_z[jsh];           \
    constexpr int NI = (LI + 1) * (LI + 2) / 2;                               \
    constexpr int NJ = (LJ + 1) * (LJ + 2) / 2;                               \
    const int t0 = pair_ptr[bas_ij];                                          \
    const int t1 = pair_ptr[bas_ij + 1];

/*
 * Density contraction, aux l = 0..KMAX (see GINTovlp_density_screened_kernel).
 */
template <int LI, int LJ, int KMAX>
__global__ void GINTovlp_density_screened_t_kernel(
    double* __restrict__ out,
    const double* __restrict__ dm, const int nao,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_T_PAIR_SETUP
    if (t0 == t1) return;
    constexpr int NCOMP = (KMAX + 1) * (KMAX + 2) * (KMAX + 3) / 6;
    constexpr int NB = LJ + 1, NC = KMAX + 1;
    constexpr int SBUF = (LI + 1) * NB * NC;

    double dmc[NI * NJ];
    #pragma unroll
    for (int iJ = 0; iJ < NJ; ++iJ) {
        #pragma unroll
        for (int iI = 0; iI < NI; ++iI) {
            double d = dm[(ao_i + iI) * nao + ao_j + iJ];
            if (ish != jsh) d += dm[(ao_j + iJ) * nao + ao_i + iI];
            dmc[iJ * NI + iI] = d;
        }
    }

    for (int t = t0; t < t1; ++t) {
        const int grid = pair_tiles[t] * OVLP_TILE + lane;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        double acc[NCOMP];
        #pragma unroll
        for (int c = 0; c < NCOMP; ++c) acc[c] = 0.0;
        bool touched = false;

        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            touched = true;
            double Sx[SBUF], Sy[SBUF], Sz[SBUF];
            ovlp_os1d<LI, LJ, KMAX>(GA[0], GB[0], GC[0], inv_2zeta, Sx);
            ovlp_os1d<LI, LJ, KMAX>(GA[1], GB[1], GC[1], inv_2zeta, Sy);
            ovlp_os1d<LI, LJ, KMAX>(GA[2], GB[2], GC[2], inv_2zeta, Sz);
            #pragma unroll
            for (int iJ = 0; iJ < NJ; ++iJ) {
                const int jx = ovlp_cart_comp(LJ, iJ, 0);
                const int jy = ovlp_cart_comp(LJ, iJ, 1);
                const int jz = ovlp_cart_comp(LJ, iJ, 2);
                #pragma unroll
                for (int iI = 0; iI < NI; ++iI) {
                    const int ix = ovlp_cart_comp(LI, iI, 0);
                    const int iy = ovlp_cart_comp(LI, iI, 1);
                    const int iz = ovlp_cart_comp(LI, iI, 2);
                    const double d = prefactor * dmc[iJ * NI + iI];
                    int c = 0;
                    #pragma unroll
                    for (int l = 0; l <= KMAX; ++l) {
                        #pragma unroll
                        for (int n = 0; n < (l + 1) * (l + 2) / 2; ++n, ++c) {
                            const int kx = ovlp_cart_comp(l, n, 0);
                            const int ky = ovlp_cart_comp(l, n, 1);
                            const int kz = ovlp_cart_comp(l, n, 2);
                            acc[c] += d * Sx[(ix * NB + jx) * NC + kx]
                                        * Sy[(iy * NB + jy) * NC + ky]
                                        * Sz[(iz * NB + jz) * NC + kz];
                        }
                    }
                }
            }
        }
        if (touched) {
            #pragma unroll
            for (int c = 0; c < NCOMP; ++c) atomicAdd(&out[grid * NCOMP + c], acc[c]);
        }
    }
}

template <int LI, int LJ>
__global__ void GINTovlp_fock_sp_screened_t_kernel(
    double* __restrict__ fock, const int nao,
    const double* __restrict__ amp_s, const double* __restrict__ amp_p,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_T_PAIR_SETUP
    if (t0 == t1) return;
    constexpr int NB = LJ + 1, NC = 2;
    constexpr int SBUF = (LI + 1) * NB * NC;
    double fock_ij[NI * NJ];
    #pragma unroll
    for (int n = 0; n < NI * NJ; ++n) fock_ij[n] = 0.0;

    for (int t = t0; t < t1; ++t) {
        const int grid = pair_tiles[t] * OVLP_TILE + lane;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        const double a_s = amp_s[grid];
        const double a_p0 = amp_p[grid * 3 + 0];
        const double a_p1 = amp_p[grid * 3 + 1];
        const double a_p2 = amp_p[grid * 3 + 2];
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            double Sx[SBUF], Sy[SBUF], Sz[SBUF];
            ovlp_os1d<LI, LJ, 1>(GA[0], GB[0], GC[0], inv_2zeta, Sx);
            ovlp_os1d<LI, LJ, 1>(GA[1], GB[1], GC[1], inv_2zeta, Sy);
            ovlp_os1d<LI, LJ, 1>(GA[2], GB[2], GC[2], inv_2zeta, Sz);
            #pragma unroll
            for (int iJ = 0; iJ < NJ; ++iJ) {
                const int jx = ovlp_cart_comp(LJ, iJ, 0);
                const int jy = ovlp_cart_comp(LJ, iJ, 1);
                const int jz = ovlp_cart_comp(LJ, iJ, 2);
                #pragma unroll
                for (int iI = 0; iI < NI; ++iI) {
                    const int ix = ovlp_cart_comp(LI, iI, 0);
                    const int iy = ovlp_cart_comp(LI, iI, 1);
                    const int iz = ovlp_cart_comp(LI, iI, 2);
                    const double sx0 = Sx[(ix * NB + jx) * NC];
                    const double sy0 = Sy[(iy * NB + jy) * NC];
                    const double sz0 = Sz[(iz * NB + jz) * NC];
                    const double val = a_s * sx0 * sy0 * sz0
                        + a_p0 * Sx[(ix * NB + jx) * NC + 1] * sy0 * sz0
                        + a_p1 * sx0 * Sy[(iy * NB + jy) * NC + 1] * sz0
                        + a_p2 * sx0 * sy0 * Sz[(iz * NB + jz) * NC + 1];
                    fock_ij[iJ * NI + iI] += prefactor * val;
                }
            }
        }
    }

    #pragma unroll
    for (int iJ = 0; iJ < NJ; ++iJ) {
        #pragma unroll
        for (int iI = 0; iI < NI; ++iI) {
            const double fval = warp_sum(fock_ij[iJ * NI + iI]);
            if (lane != 0 || fabs(fval) < c_overlap_cutoff) continue;
            const int ii = ao_i + iI;
            const int jj = ao_j + iJ;
            atomicAdd(&fock[ii * nao + jj], fval);
            if (ish != jsh) atomicAdd(&fock[jj * nao + ii], fval);
        }
    }
}

template <int LI, int LJ>
__global__ void GINTovlp_ip1_sp_atoms_screened_t_kernel(
    double* __restrict__ grad,
    const double* __restrict__ dsym, const int nao,
    const double* __restrict__ amp_s, const double* __restrict__ amp_p,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_T_PAIR_SETUP
    if (t0 == t1) return;
    const bool off_diagonal = (ish != jsh);
    constexpr int NB = LJ + 2, NC = 2;
    constexpr int SBUF = (LI + 2) * NB * NC;
    double dd[NI * NJ];
    #pragma unroll
    for (int iJ = 0; iJ < NJ; ++iJ) {
        #pragma unroll
        for (int iI = 0; iI < NI; ++iI)
            dd[iJ * NI + iI] = dsym[(ao_i + iI) * nao + ao_j + iJ];
    }
    double gA[3] = {0.0, 0.0, 0.0};
    double gB[3] = {0.0, 0.0, 0.0};

    for (int t = t0; t < t1; ++t) {
        const int grid = pair_tiles[t] * OVLP_TILE + lane;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        const double amp[4] = {amp_s[grid], amp_p[grid * 3 + 0],
                               amp_p[grid * 3 + 1], amp_p[grid * 3 + 2]};
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            double Sx[SBUF], Sy[SBUF], Sz[SBUF];
            ovlp_os1d<LI + 1, LJ + 1, 1>(GA[0], GB[0], GC[0], inv_2zeta, Sx);
            ovlp_os1d<LI + 1, LJ + 1, 1>(GA[1], GB[1], GC[1], inv_2zeta, Sy);
            ovlp_os1d<LI + 1, LJ + 1, 1>(GA[2], GB[2], GC[2], inv_2zeta, Sz);
            const double two_alpha = 2.0 * alpha;
            const double two_beta = 2.0 * beta;
            double bA[3] = {0.0, 0.0, 0.0};
            double bB[3] = {0.0, 0.0, 0.0};
            #define SI(S, a, b, c) S[((a) * NB + (b)) * NC + (c)]
            #pragma unroll
            for (int iJ = 0; iJ < NJ; ++iJ) {
                const int jx = ovlp_cart_comp(LJ, iJ, 0);
                const int jy = ovlp_cart_comp(LJ, iJ, 1);
                const int jz = ovlp_cart_comp(LJ, iJ, 2);
                #pragma unroll
                for (int iI = 0; iI < NI; ++iI) {
                    const int ix = ovlp_cart_comp(LI, iI, 0);
                    const int iy = ovlp_cart_comp(LI, iI, 1);
                    const int iz = ovlp_cart_comp(LI, iI, 2);
                    const double d = dd[iJ * NI + iI];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int kx = (k == 1), ky = (k == 2), kz = (k == 3);
                        const double sx = SI(Sx, ix, jx, kx);
                        const double sy = SI(Sy, iy, jy, ky);
                        const double sz = SI(Sz, iz, jz, kz);
                        const double w = d * amp[k];
                        double dx = two_alpha * SI(Sx, ix + 1, jx, kx);
                        if (ix > 0) dx -= ix * SI(Sx, ix - 1, jx, kx);
                        double dy = two_alpha * SI(Sy, iy + 1, jy, ky);
                        if (iy > 0) dy -= iy * SI(Sy, iy - 1, jy, ky);
                        double dz = two_alpha * SI(Sz, iz + 1, jz, kz);
                        if (iz > 0) dz -= iz * SI(Sz, iz - 1, jz, kz);
                        bA[0] += w * dx * sy * sz;
                        bA[1] += w * sx * dy * sz;
                        bA[2] += w * sx * sy * dz;
                        if (off_diagonal) {
                            double ex = two_beta * SI(Sx, ix, jx + 1, kx);
                            if (jx > 0) ex -= jx * SI(Sx, ix, jx - 1, kx);
                            double ey = two_beta * SI(Sy, iy, jy + 1, ky);
                            if (jy > 0) ey -= jy * SI(Sy, iy, jy - 1, ky);
                            double ez = two_beta * SI(Sz, iz, jz + 1, kz);
                            if (jz > 0) ez -= jz * SI(Sz, iz, jz - 1, kz);
                            bB[0] += w * ex * sy * sz;
                            bB[1] += w * sx * ey * sz;
                            bB[2] += w * sx * sy * ez;
                        }
                    }
                }
            }
            #undef SI
            #pragma unroll
            for (int x = 0; x < 3; ++x) {
                gA[x] += prefactor * bA[x];
                gB[x] += prefactor * bB[x];
            }
        }
    }

    const int iatm = c_bpcache.bas_atm[ish];
    const int jatm = c_bpcache.bas_atm[jsh];
    #pragma unroll
    for (int x = 0; x < 3; ++x) {
        const double va = warp_sum(gA[x]);
        const double vb = warp_sum(gB[x]);
        if (lane == 0) {
            atomicAdd(&grad[iatm * 3 + x], va);
            if (off_diagonal) atomicAdd(&grad[jatm * 3 + x], vb);
        }
    }
}
