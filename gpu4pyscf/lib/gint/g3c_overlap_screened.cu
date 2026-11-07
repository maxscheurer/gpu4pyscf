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
 * Spatially screened 3-center overlap kernels for GOSTSHYP.
 *
 * Surface Gaussians are sorted spatially (Morton order) and grouped into tiles
 * of OVLP_TILE consecutive points. For every shell pair a list of tiles is built
 * with a rigorous upper bound of the per-primitive prefactor used by the
 * unscreened kernels,
 *     pref = (g/z)^{3/2} |e12| exp(-a g / z |P-C|^2),   z = a + g,
 * so a tile is dropped only if every primitive of the pair would be skipped for
 * every point of the tile by the existing `fabs(prefactor) < cutoff` test.
 *
 * Compute kernels map one warp to one shell pair; each lane handles one point of
 * the current tile. Included from nr_fill_ao_int3c_overlap.cu (shares constants).
 */

#ifndef OVLP_TILE
#define OVLP_TILE 1
#endif
#define OVLP_TPW (32 / OVLP_TILE)  // tiles processed concurrently per warp
#define OVLP_WARPS_PER_BLOCK 4

/*
 * Thread per shell pair: count (fill == 0) or write (fill == 1) the tiles that
 * may contain a non-negligible primitive prefactor. Two levels: superblocks of
 * `tiles_per_super` consecutive tiles are first tested with a pair-level bound
 * (min exponent, max |e12|, bounding sphere of the primitive centers P), then
 * individual tiles with the per-primitive bound.
 */
__device__ __forceinline__ double ovlp_bound(double aij, double e, double d2,
                                             double gmin, double gmax)
{
    const double mu = aij * gmin / (aij + gmin);
    double f = gmax / (aij + gmax);
    f = f * sqrt(f);
    return f * e * exp(-mu * d2);
}

__global__ void GINTovlp_screen_kernel(
    int* __restrict__ counts,          // [n_bas_pairs] mode 0: #blocks, mode 1: #tiles
    int* __restrict__ blk_out,         // mode 1: passing block ids  [nblk_pass]
    unsigned* __restrict__ mask_out,   // mode 1: tile hit mask per passing block
    const int* __restrict__ blk_ptr,   // mode 1: [n_bas_pairs+1] offsets of passing blocks
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const double* __restrict__ tile_sphere,  // [ntiles, 4]: cx, cy, cz, R
    const double* __restrict__ tile_gamma,   // [ntiles, 2]: gmin, gmax
    const int ntiles,
    const double* __restrict__ super_sphere, // [nsuper, 4]
    const double* __restrict__ super_gamma,  // [nsuper, 2]
    const int tiles_per_super,
    const double* __restrict__ block_sphere, // [nblock, 4]
    const double* __restrict__ block_gamma,  // [nblock, 2]
    const int tiles_per_block,               // tiles_per_super % tiles_per_block == 0
    const double thresh, const int fill)
{
    // one warp per shell pair; lanes test superblocks / tiles in parallel
    const int lane = threadIdx.x & 31;
    const int task = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (task >= ntasks_ij) return;
    const int bas_ij = bas_ij0 + task;
    const int p0 = prim_ij0 + task * nprim_ij;

    // pair-level aggregate over primitives
    double amin = 1e300, emax = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    for (int ij = p0; ij < p0 + nprim_ij; ++ij) {
        amin = fmin(amin, c_bpcache.a12[ij]);
        emax = fmax(emax, fabs(c_bpcache.e12[ij]));
        px += c_bpcache.x12[ij]; py += c_bpcache.y12[ij]; pz += c_bpcache.z12[ij];
    }
    px /= nprim_ij; py /= nprim_ij; pz /= nprim_ij;
    double rp = 0.0;
    for (int ij = p0; ij < p0 + nprim_ij; ++ij) {
        const double dx = c_bpcache.x12[ij] - px;
        const double dy = c_bpcache.y12[ij] - py;
        const double dz = c_bpcache.z12[ij] - pz;
        rp = fmax(rp, sqrt(dx * dx + dy * dy + dz * dz));
    }
    rp = rp * (1.0 + 1e-12) + 1e-12;

    const unsigned lt_mask = (1u << lane) - 1u;
    int n = 0;   // mode 0: passing blocks; mode 1: kept tiles
    int k = 0;   // mode 1: passing blocks written
    const int out0 = fill ? blk_ptr[bas_ij] : 0;
    const int nsuper = (ntiles + tiles_per_super - 1) / tiles_per_super;
    for (int sb0 = 0; sb0 < nsuper; sb0 += 32) {
        const int sb = sb0 + lane;
        bool pass = false;
        if (sb < nsuper) {
            const double dx = px - super_sphere[sb * 4 + 0];
            const double dy = py - super_sphere[sb * 4 + 1];
            const double dz = pz - super_sphere[sb * 4 + 2];
            double d = sqrt(dx * dx + dy * dy + dz * dz) - rp - super_sphere[sb * 4 + 3];
            d = d > 0.0 ? d : 0.0;
            pass = ovlp_bound(amin, emax, d * d, super_gamma[sb * 2 + 0],
                              super_gamma[sb * 2 + 1]) >= thresh;
        }
        unsigned smask = __ballot_sync(0xffffffffu, pass);
        while (smask) {
            const int s = sb0 + __ffs(smask) - 1;
            smask &= smask - 1;
            // middle level: blocks of tiles_per_block tiles inside superblock s
            const int blocks_per_super = tiles_per_super / tiles_per_block;
            const int nblock = (ntiles + tiles_per_block - 1) / tiles_per_block;
            const int bb = s * blocks_per_super;
            const int be = min(nblock, bb + blocks_per_super);
            for (int b0 = bb; b0 < be; b0 += 32) {
            const int bk = b0 + lane;
            bool bpass = false;
            if (bk < be) {
                const double dx = px - block_sphere[bk * 4 + 0];
                const double dy = py - block_sphere[bk * 4 + 1];
                const double dz = pz - block_sphere[bk * 4 + 2];
                double d = sqrt(dx * dx + dy * dy + dz * dz) - rp - block_sphere[bk * 4 + 3];
                d = d > 0.0 ? d : 0.0;
                bpass = ovlp_bound(amin, emax, d * d, block_gamma[bk * 2 + 0],
                                   block_gamma[bk * 2 + 1]) >= thresh;
            }
            unsigned bmask = __ballot_sync(0xffffffffu, bpass);
            while (bmask) {
            const int blk = b0 + __ffs(bmask) - 1;
            bmask &= bmask - 1;
            if (!fill) { ++n; continue; }
            // tiles_per_block <= 32: one ballot covers the whole block
            const int tb = blk * tiles_per_block;
            const int te = min(ntiles, tb + tiles_per_block);
            {
                const int t = tb + lane;
                bool keep = false;
                if (t < te) {
                    const double cx = tile_sphere[t * 4 + 0];
                    const double cy = tile_sphere[t * 4 + 1];
                    const double cz = tile_sphere[t * 4 + 2];
                    const double R  = tile_sphere[t * 4 + 3];
                    const double gmin = tile_gamma[t * 2 + 0];
                    const double gmax = tile_gamma[t * 2 + 1];
                    for (int ij = p0; ij < p0 + nprim_ij; ++ij) {
                        const double dx = c_bpcache.x12[ij] - cx;
                        const double dy = c_bpcache.y12[ij] - cy;
                        const double dz = c_bpcache.z12[ij] - cz;
                        double d = sqrt(dx * dx + dy * dy + dz * dz) - R;
                        d = d > 0.0 ? d : 0.0;
                        if (ovlp_bound(c_bpcache.a12[ij], fabs(c_bpcache.e12[ij]),
                                       d * d, gmin, gmax) >= thresh) {
                            keep = true; break;
                        }
                    }
                }
                const unsigned kmask = __ballot_sync(0xffffffffu, keep);
                if (lane == 0) {
                    blk_out[out0 + k] = blk;
                    mask_out[out0 + k] = kmask;
                }
                ++k;
                n += __popc(kmask);
            }
            }  // blocks
            }
        }
    }
    if (lane == 0) counts[bas_ij] = n;
}

/*
 * Expand (block, hit mask) records into the dense per-pair tile CSR.
 * One warp per pair.
 */
__global__ void GINTovlp_expand_kernel(
    int* __restrict__ tiles_out, const int* __restrict__ pair_ptr,
    const int* __restrict__ blk, const unsigned* __restrict__ mask,
    const int* __restrict__ blk_ptr, const int tiles_per_block, const int npairs)
{
    const int lane = threadIdx.x & 31;
    const int p = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (p >= npairs) return;
    const unsigned lt_mask = (1u << lane) - 1u;
    int n = pair_ptr[p];
    for (int k = blk_ptr[p]; k < blk_ptr[p + 1]; ++k) {
        const unsigned m = mask[k];
        if ((m >> lane) & 1u)
            tiles_out[n + __popc(m & lt_mask)] = blk[k] * tiles_per_block + lane;
        n += __popc(m);
    }
}

// Shared prologue of the per-primitive work (identical arithmetic to the
// unscreened kernels so screening decisions match bit-for-bit).
#define OVLP_PRIM_SETUP                                                      \
    const double alpha = c_bpcache.a1[ij];                                    \
    const double beta = c_bpcache.a2[ij];                                     \
    const double coeff_ij = c_bpcache.e12[ij];                                \
    const double aij = alpha + beta;                                          \
    const double zeta = aij + gamma;                                          \
    const double inv_zeta = 1.0 / zeta;                                       \
    const double Px = (alpha * Ax + beta * Bx) / aij;                         \
    const double Py = (alpha * Ay + beta * By) / aij;                         \
    const double Pz = (alpha * Az + beta * Bz) / aij;                         \
    const double PCx = Px - Cx, PCy = Py - Cy, PCz = Pz - Cz;                 \
    const double PC2 = PCx * PCx + PCy * PCy + PCz * PCz;                     \
    const double gamma_over_zeta = gamma * inv_zeta;                          \
    const double prefactor = sqrt(gamma_over_zeta) * gamma_over_zeta         \
                           * exp(-aij * gamma * inv_zeta * PC2) * coeff_ij;   \
    if (fabs(prefactor) < c_overlap_cutoff) continue;                         \
    const double inv_2zeta = 0.5 * inv_zeta;                                  \
    const double Gx = (aij * Px + gamma * Cx) * inv_zeta;                     \
    const double Gy = (aij * Py + gamma * Cy) * inv_zeta;                     \
    const double Gz = (aij * Pz + gamma * Cz) * inv_zeta;                     \
    const double GA[3] = {Gx - Ax, Gy - Ay, Gz - Az};                         \
    const double GB[3] = {Gx - Bx, Gy - By, Gz - Bz};                         \
    const double GC[3] = {Gx - Cx, Gy - Cy, Gz - Cz};

#define OVLP_PAIR_SETUP                                                      \
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
    const int ncart_i = (i_l + 1) * (i_l + 2) / 2;                            \
    const int ncart_j = (j_l + 1) * (j_l + 2) / 2;                            \
    const int i_loc = c_l_locs[i_l];                                          \
    const int j_loc = c_l_locs[j_l];                                          \
    /* dense mode (pair_tiles == NULL): every point for every pair */         \
    const int t0 = pair_tiles ? pair_ptr[bas_ij] : 0;                         \
    const int t1 = pair_tiles ? pair_ptr[bas_ij + 1]                          \
                              : (ngrids + OVLP_TILE - 1) / OVLP_TILE;

/*
 * Density contraction for all aux Cartesian components with l = 0..kmax:
 *   out[g, c] += sum_ij D_ij S_ij,(g,c)
 * Component order: l=0 (1), l=1 (3), l=2 (6), l=3 (10), each in c_idx order.
 */
template <int MAX_L_TOTAL>
__global__ void GINTovlp_density_screened_kernel(
    double* __restrict__ out, const int ncomp,
    const double* __restrict__ dm, const int nao,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int i_l, const int j_l, const int kmax,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_PAIR_SETUP
    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE], Sy[BUF_SIZE], Sz[BUF_SIZE];
    #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))

    for (int t = t0 + lane / OVLP_TILE; t < t1; t += OVLP_TPW) {
        const int grid = (pair_tiles ? pair_tiles[t] : t) * OVLP_TILE + lane % OVLP_TILE;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        double acc[20];
        for (int c = 0; c < ncomp; ++c) acc[c] = 0.0;
        bool touched = false;

        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            touched = true;
            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l, j_l, kmax, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int jx = c_idx[j_loc + iJ];
                const int jy = c_idx[j_loc + iJ + TOT_NF];
                const int jz = c_idx[j_loc + iJ + 2 * TOT_NF];
                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int ix = c_idx[i_loc + iI];
                    const int iy = c_idx[i_loc + iI + TOT_NF];
                    const int iz = c_idx[i_loc + iI + 2 * TOT_NF];
                    const int ii = ao_i + iI;
                    const int jj = ao_j + iJ;
                    double d = dm[ii * nao + jj];
                    if (ish != jsh) d += dm[jj * nao + ii];
                    d *= prefactor;
                    int c = 0;
                    for (int l = 0; l <= kmax; ++l) {
                        const int k_loc = c_l_locs[l];
                        const int nk = (l + 1) * (l + 2) / 2;
                        for (int iK = 0; iK < nk; ++iK, ++c) {
                            const int kx = c_idx[k_loc + iK];
                            const int ky = c_idx[k_loc + iK + TOT_NF];
                            const int kz = c_idx[k_loc + iK + 2 * TOT_NF];
                            acc[c] += d * Sx[S_IDX(ix, jx, kx)]
                                        * Sy[S_IDX(iy, jy, ky)]
                                        * Sz[S_IDX(iz, jz, kz)];
                        }
                    }
                }
            }
        }
        if (touched) {
            for (int c = 0; c < ncomp; ++c) atomicAdd(&out[c * ngrids + grid], acc[c]);
        }
    }
    #undef S_IDX
}

/*
 * Gradient density contractions, reduced to the 11 per-point quantities the
 * GOSTSHYP gradient uses (aux l = 0..3, contracted with the surface normal n):
 *   0: s   1-3: p_x,p_y,p_z   4-6: sum_q d_xq n_q (x,y,z)   7: tr d
 *   8-10: f_xxx+f_xyy+f_xzz, f_xxy+f_yyy+f_yzz, f_xxz+f_yyz+f_zzz
 * out is component-major [11, ngrids].
 */
#define OVLP_GRAD_NCOMP 11
__device__ __forceinline__ void ovlp_grad_terms(
    const double x0, const double x1, const double x2, const double x3,
    const double y0, const double y1, const double y2, const double y3,
    const double z0, const double z1, const double z2, const double z3,
    const double nx, const double ny, const double nz, const double w,
    double* __restrict__ acc)
{
    acc[0] += w * x0 * y0 * z0;
    acc[1] += w * x1 * y0 * z0;
    acc[2] += w * x0 * y1 * z0;
    acc[3] += w * x0 * y0 * z1;
    const double dxx = x2 * y0 * z0, dyy = x0 * y2 * z0, dzz = x0 * y0 * z2;
    const double dxy = x1 * y1 * z0, dxz = x1 * y0 * z1, dyz = x0 * y1 * z1;
    acc[4] += w * (dxx * nx + dxy * ny + dxz * nz);
    acc[5] += w * (dxy * nx + dyy * ny + dyz * nz);
    acc[6] += w * (dxz * nx + dyz * ny + dzz * nz);
    acc[7] += w * (dxx + dyy + dzz);
    acc[8]  += w * (x3 * y0 * z0 + x1 * y2 * z0 + x1 * y0 * z2);
    acc[9]  += w * (x2 * y1 * z0 + x0 * y3 * z0 + x0 * y1 * z2);
    acc[10] += w * (x2 * y0 * z1 + x0 * y2 * z1 + x0 * y0 * z3);
}

template <int MAX_L_TOTAL>
__global__ void GINTovlp_grad_density_screened_kernel(
    double* __restrict__ out,
    const double* __restrict__ dm, const int nao,
    const double* __restrict__ normals,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int i_l, const int j_l,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_PAIR_SETUP
    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE], Sy[BUF_SIZE], Sz[BUF_SIZE];
    #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))
    for (int t = t0 + lane / OVLP_TILE; t < t1; t += OVLP_TPW) {
        const int grid = (pair_tiles ? pair_tiles[t] : t) * OVLP_TILE + lane % OVLP_TILE;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        const double nx = normals[grid * 3 + 0];
        const double ny = normals[grid * 3 + 1];
        const double nz = normals[grid * 3 + 2];
        double acc[OVLP_GRAD_NCOMP];
        for (int c = 0; c < OVLP_GRAD_NCOMP; ++c) acc[c] = 0.0;
        bool touched = false;
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            touched = true;
            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l, j_l, 3, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int jx = c_idx[j_loc + iJ];
                const int jy = c_idx[j_loc + iJ + TOT_NF];
                const int jz = c_idx[j_loc + iJ + 2 * TOT_NF];
                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int ix = c_idx[i_loc + iI];
                    const int iy = c_idx[i_loc + iI + TOT_NF];
                    const int iz = c_idx[i_loc + iI + 2 * TOT_NF];
                    const int ii = ao_i + iI;
                    const int jj = ao_j + iJ;
                    double d = dm[ii * nao + jj];
                    if (ish != jsh) d += dm[jj * nao + ii];
                    ovlp_grad_terms(
                        Sx[S_IDX(ix, jx, 0)], Sx[S_IDX(ix, jx, 1)], Sx[S_IDX(ix, jx, 2)], Sx[S_IDX(ix, jx, 3)],
                        Sy[S_IDX(iy, jy, 0)], Sy[S_IDX(iy, jy, 1)], Sy[S_IDX(iy, jy, 2)], Sy[S_IDX(iy, jy, 3)],
                        Sz[S_IDX(iz, jz, 0)], Sz[S_IDX(iz, jz, 1)], Sz[S_IDX(iz, jz, 2)], Sz[S_IDX(iz, jz, 3)],
                        nx, ny, nz, prefactor * d, acc);
                }
            }
        }
        if (touched) {
            for (int c = 0; c < OVLP_GRAD_NCOMP; ++c)
                atomicAdd(&out[c * ngrids + grid], acc[c]);
        }
    }
    #undef S_IDX
}

__device__ __forceinline__ double warp_sum(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/*
 * Fused s+p amplitude contraction:
 *   fock[i,j] += sum_g (amp_s[g] S_ij0,g + sum_k amp_p[g,k] S_ij1k,g)
 */
template <int MAX_L_TOTAL>
__global__ void GINTovlp_fock_sp_screened_kernel(
    double* __restrict__ fock, const int nao,
    const double* __restrict__ amp_s, const double* __restrict__ amp_p,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int i_l, const int j_l,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_PAIR_SETUP
    if (t0 == t1) return;  // whole warp
    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE], Sy[BUF_SIZE], Sz[BUF_SIZE];
    #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))
    double fock_ij[225];
    const int nij = ncart_i * ncart_j;
    for (int n = 0; n < nij; ++n) fock_ij[n] = 0.0;

    for (int t = t0 + lane / OVLP_TILE; t < t1; t += OVLP_TPW) {
        const int grid = (pair_tiles ? pair_tiles[t] : t) * OVLP_TILE + lane % OVLP_TILE;
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
            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l, j_l, 1, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int jx = c_idx[j_loc + iJ];
                const int jy = c_idx[j_loc + iJ + TOT_NF];
                const int jz = c_idx[j_loc + iJ + 2 * TOT_NF];
                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int ix = c_idx[i_loc + iI];
                    const int iy = c_idx[i_loc + iI + TOT_NF];
                    const int iz = c_idx[i_loc + iI + 2 * TOT_NF];
                    const double sx0 = Sx[S_IDX(ix, jx, 0)];
                    const double sy0 = Sy[S_IDX(iy, jy, 0)];
                    const double sz0 = Sz[S_IDX(iz, jz, 0)];
                    const double val = a_s * sx0 * sy0 * sz0
                        + a_p0 * Sx[S_IDX(ix, jx, 1)] * sy0 * sz0
                        + a_p1 * sx0 * Sy[S_IDX(iy, jy, 1)] * sz0
                        + a_p2 * sx0 * sy0 * Sz[S_IDX(iz, jz, 1)];
                    fock_ij[iJ * ncart_i + iI] += prefactor * val;
                }
            }
        }
    }
    #undef S_IDX

    for (int iJ = 0; iJ < ncart_j; ++iJ) {
        for (int iI = 0; iI < ncart_i; ++iI) {
            const double fval = warp_sum(fock_ij[iJ * ncart_i + iI]);
            if (lane != 0 || fabs(fval) < c_overlap_cutoff) continue;
            const int ii = ao_i + iI;
            const int jj = ao_j + iJ;
            atomicAdd(&fock[ii * nao + jj], fval);
            if (ish != jsh) atomicAdd(&fock[jj * nao + ii], fval);
        }
    }
}

/*
 * AO-center derivative of the s+p amplitude-contracted overlap, contracted with
 * the symmetrized density Dsym = D + D^T and summed per atom:
 *   grad[atom(i), x] += sum_g sum_k amp_k,g sum_ij Dsym_ij dS_ij,(g,k)/dA_x
 *   grad[atom(j), x] += ...                                       dB_x  (i != j shells)
 * (+d/dA convention; the Python caller negates.)
 */
template <int MAX_L_TOTAL>
__global__ void GINTovlp_ip1_sp_atoms_screened_kernel(
    double* __restrict__ grad,   // [natm, 3]
    const double* __restrict__ dsym, const int nao,
    const double* __restrict__ amp_s, const double* __restrict__ amp_p,
    const int bas_ij0, const int prim_ij0, const int ntasks_ij, const int nprim_ij,
    const int i_l, const int j_l,
    const int* __restrict__ pair_ptr, const int* __restrict__ pair_tiles,
    const double* __restrict__ coords, const double* __restrict__ exps,
    const int ngrids)
{
    OVLP_PAIR_SETUP
    if (t0 == t1) return;
    const bool off_diagonal = (ish != jsh);
    constexpr int STRIDE = MAX_L_TOTAL + 2;
    constexpr int BUF_SIZE = STRIDE * STRIDE * STRIDE;
    double Sx[BUF_SIZE], Sy[BUF_SIZE], Sz[BUF_SIZE];
    #define S_IDX(a, b, c) ((a) * STRIDE * STRIDE + (b) * STRIDE + (c))
    double gA[3] = {0.0, 0.0, 0.0};
    double gB[3] = {0.0, 0.0, 0.0};

    for (int t = t0 + lane / OVLP_TILE; t < t1; t += OVLP_TPW) {
        const int grid = (pair_tiles ? pair_tiles[t] : t) * OVLP_TILE + lane % OVLP_TILE;
        if (grid >= ngrids) continue;
        const double Cx = coords[grid * 3 + 0];
        const double Cy = coords[grid * 3 + 1];
        const double Cz = coords[grid * 3 + 2];
        const double gamma = exps[grid];
        const double amp[4] = {amp_s[grid], amp_p[grid * 3 + 0],
                               amp_p[grid * 3 + 1], amp_p[grid * 3 + 2]};
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
            OVLP_PRIM_SETUP
            compute_3c_overlap_recursion_general<MAX_L_TOTAL>(
                i_l + 1, j_l + 1, 1, GA, GB, GC, inv_2zeta, Sx, Sy, Sz);
            const double two_alpha = 2.0 * alpha;
            const double two_beta = 2.0 * beta;
            double bA[3] = {0.0, 0.0, 0.0};
            double bB[3] = {0.0, 0.0, 0.0};
            for (int iJ = 0; iJ < ncart_j; ++iJ) {
                const int jx = c_idx[j_loc + iJ];
                const int jy = c_idx[j_loc + iJ + TOT_NF];
                const int jz = c_idx[j_loc + iJ + 2 * TOT_NF];
                for (int iI = 0; iI < ncart_i; ++iI) {
                    const int ix = c_idx[i_loc + iI];
                    const int iy = c_idx[i_loc + iI + TOT_NF];
                    const int iz = c_idx[i_loc + iI + 2 * TOT_NF];
                    const double d = dsym[(ao_i + iI) * nao + (ao_j + iJ)]
                                   + dsym[(ao_j + iJ) * nao + (ao_i + iI)];  // D + D^T
                    for (int k = 0; k < 4; ++k) {
                        const int kx = (k == 1), ky = (k == 2), kz = (k == 3);
                        const double sx = Sx[S_IDX(ix, jx, kx)];
                        const double sy = Sy[S_IDX(iy, jy, ky)];
                        const double sz = Sz[S_IDX(iz, jz, kz)];
                        const double w = d * amp[k];
                        double dx = two_alpha * Sx[S_IDX(ix + 1, jx, kx)];
                        if (ix > 0) dx -= ix * Sx[S_IDX(ix - 1, jx, kx)];
                        double dy = two_alpha * Sy[S_IDX(iy + 1, jy, ky)];
                        if (iy > 0) dy -= iy * Sy[S_IDX(iy - 1, jy, ky)];
                        double dz = two_alpha * Sz[S_IDX(iz + 1, jz, kz)];
                        if (iz > 0) dz -= iz * Sz[S_IDX(iz - 1, jz, kz)];
                        bA[0] += w * dx * sy * sz;
                        bA[1] += w * sx * dy * sz;
                        bA[2] += w * sx * sy * dz;
                        if (off_diagonal) {
                            double ex = two_beta * Sx[S_IDX(ix, jx + 1, kx)];
                            if (jx > 0) ex -= jx * Sx[S_IDX(ix, jx - 1, kx)];
                            double ey = two_beta * Sy[S_IDX(iy, jy + 1, ky)];
                            if (jy > 0) ey -= jy * Sy[S_IDX(iy, jy - 1, ky)];
                            double ez = two_beta * Sz[S_IDX(iz, jz + 1, kz)];
                            if (jz > 0) ez -= jz * Sz[S_IDX(iz, jz - 1, kz)];
                            bB[0] += w * ex * sy * sz;
                            bB[1] += w * sx * ey * sz;
                            bB[2] += w * sx * sy * ez;
                        }
                    }
                }
            }
            for (int x = 0; x < 3; ++x) {
                gA[x] += prefactor * bA[x];
                gB[x] += prefactor * bB[x];
            }
        }
    }
    #undef S_IDX

    const int iatm = c_bpcache.bas_atm[ish];
    const int jatm = c_bpcache.bas_atm[jsh];
    for (int x = 0; x < 3; ++x) {
        const double va = warp_sum(gA[x]);
        const double vb = warp_sum(gB[x]);
        if (lane == 0) {
            atomicAdd(&grad[iatm * 3 + x], va);
            if (off_diagonal) atomicAdd(&grad[jatm * 3 + x], vb);
        }
    }
}
