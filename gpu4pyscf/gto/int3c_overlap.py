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
Spatially screened 3-center overlap integrals for the GOSTSHYP model.

    S_ij,g = integral chi_i(r) chi_j(r) G_g(r) dr,
    G_g(r) = (gamma_g / pi)^{3/2} x^a y^b z^c exp(-gamma_g |r|^2),  r -> r - C_g

where chi_i, chi_j are AO basis functions and G_g are Cartesian surface
Gaussians of aux angular momentum a + b + c = 0..3 whose s component is
normalized to one. The tensor is never materialized: the
kernels contract it with a density matrix or with per-point amplitudes.

OverlapScreen builds, for the shell pairs of a VHFOpt-like object (see
gpu4pyscf.solvent.gostshyp._OverlapPairOpt), lists of Morton-sorted point
tiles that can contribute above the cutoff, and drives the CUDA kernels in
lib/gint/g3c_overlap_screened*.cu. In dense mode (cutoff <= 0, forced, or
lists too large for GPU memory) the same kernels run over all points.
"""

import ctypes
import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import load_library, get_avail_mem

libgint = load_library('libgint')

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

    # Fraction of free GPU memory the per-pair lists may use before falling
    # back to dense mode (all points for every pair, no lists).
    MAX_MEM_FRACTION = 0.25

    def __init__(self, intopt, coords, exponents, cutoff, force_dense=False):
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

        self.force_dense = force_dense
        self.dense = True
        self.pair_ptr = self.pair_tiles = None
        self.nnz = None
        if force_dense or not cutoff > 0:
            # no screening possible (cutoff <= 0) or requested: dense mode
            return

        assert OVLP_TILES_PER_BLOCK <= 32
        npairs = int(intopt.bas_pairs_locs[-1])
        thresh = 0.5 * cutoff
        libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(cutoff))
        stream = cp.cuda.get_current_stream()
        null = ctypes.c_void_p(0)

        def screen(fill, counts, blk, mask, blk_ptr):
            for cp_ij_id in range(len(intopt.log_qs)):
                err = libgint.GINTovlp_screen(
                    ctypes.c_void_p(stream.ptr), intopt.bpcache, ctypes.c_int(cp_ij_id),
                    ctypes.c_void_p(counts.data.ptr),
                    ctypes.c_void_p(blk.data.ptr) if fill else null,
                    ctypes.c_void_p(mask.data.ptr) if fill else null,
                    ctypes.c_void_p(blk_ptr.data.ptr) if fill else null,
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

        # 1) count passing (pair, block) records with the cheap block-level
        # bounds, 2) evaluate the exact tile test once, storing a hit mask per
        # passing block, 3) expand the masks into the per-pair tile CSR.
        nblk = cp.zeros(npairs, dtype=np.int32)
        screen(0, nblk, None, None, None)
        blk_ptr = cp.zeros(npairs + 1, dtype=np.int32)
        cp.cumsum(nblk, out=blk_ptr[1:])
        nrec = max(int(blk_ptr[-1]), 1)
        budget = self.MAX_MEM_FRACTION * get_avail_mem()
        if nrec * 8 > budget:
            return  # dense mode
        blk = cp.empty(nrec, dtype=np.int32)
        mask = cp.empty(nrec, dtype=np.uint32)
        counts = cp.zeros(npairs, dtype=np.int32)
        screen(1, counts, blk, mask, blk_ptr)
        ptr = cp.zeros(npairs + 1, dtype=np.int32)
        cp.cumsum(counts, out=ptr[1:])
        nnz = int(ptr[-1])
        if nnz * 4 > budget:
            return  # dense mode
        tiles = cp.empty(max(nnz, 1), dtype=np.int32)
        err = libgint.GINTovlp_expand(
            ctypes.c_void_p(stream.ptr), ctypes.c_void_p(tiles.data.ptr),
            ctypes.c_void_p(ptr.data.ptr), ctypes.c_void_p(blk.data.ptr),
            ctypes.c_void_p(mask.data.ptr), ctypes.c_void_p(blk_ptr.data.ptr),
            ctypes.c_int(OVLP_TILES_PER_BLOCK), ctypes.c_int(npairs))
        if err != 0:
            raise RuntimeError('GINTovlp_expand failed')
        del blk, mask
        self.pair_ptr = ptr
        self.pair_tiles = tiles
        self.nnz = nnz
        self.dense = False

    def matches(self, intopt, coords, cutoff, force_dense=False):
        return (intopt is self.intopt and cutoff == self.cutoff
                and coords.shape[0] == self.ngrids
                and force_dense == self.force_dense)

    def _launch(self, fn, *args):
        intopt = self.intopt
        libgint.GINTset_int3c_overlap_constants(intopt.bpcache, ctypes.c_double(self.cutoff))
        stream = cp.cuda.get_current_stream()
        if self.dense:
            lists = (ctypes.c_void_p(0), ctypes.c_void_p(0))
        else:
            lists = (ctypes.c_void_p(self.pair_ptr.data.ptr),
                     ctypes.c_void_p(self.pair_tiles.data.ptr))
        tail = (*lists,
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

    def grad_density(self, mol, dm, normals):
        """Per-point density contractions used by the GOSTSHYP gradient.

        Returns [ngrids, 11]: s, p (3), sum_q d_xq n_q (3), tr d,
        f-traces (f_xxx+f_xyy+f_xzz, f_xxy+f_yyy+f_yzz, f_xxz+f_yyz+f_zzz).
        """
        dm_c = self._sorted_cart_dm(mol, dm)
        nao_cart = dm_c.shape[0]
        n_s = cp.ascontiguousarray(
            cp.asarray(normals, dtype=np.float64).reshape(-1, 3)[self.perm])
        out = cp.zeros((11, self.ngrids))
        self._launch(libgint.GINTovlp_grad_density_screened,
                     ctypes.c_void_p(out.data.ptr),
                     ctypes.c_void_p(dm_c.data.ptr), ctypes.c_int(nao_cart),
                     ctypes.c_void_p(n_s.data.ptr))
        res = cp.empty((self.ngrids, 11))
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
        """AO-center (ip1) derivatives contracted with D + D^T, summed per atom.

        Returns [natm, 3]:
            sum_{ij in atom A} sum_g (D_ij + D_ji)
                (amp_s[g] nabla_i S_ij,g(s) + sum_c amp_p[g, c] nabla_i S_ij,g(p_c))
        with nabla_i = -d/dA_i (PySCF int3c1e_ip1 convention).
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
