# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Screened 3-center overlap contractions (gpu4pyscf.gto.int3c_overlap) against
PySCF/libcint int3c1e and int3c1e_ip1 integrals with the surface Gaussians as
a fake molecule (pyscf.solvent.gostshyp.fakemol_for_gaussian).
"""

import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.solvent.gostshyp import fakemol_for_gaussian
from gpu4pyscf.gto.int3c_overlap import OverlapScreen
from gpu4pyscf.solvent.gostshyp import _OverlapPairOpt
from gpu4pyscf.solvent.pcm import gen_surface, modified_Bondi

WATER_DIMER = '''
O   0.0000  0.0000  0.1173
H   0.0000  0.7572 -0.4692
H   0.0000 -0.7572 -0.4692
O   4.5000  0.0000  0.1173
H   4.5000  0.7572 -0.4692
H   4.5000 -0.7572 -0.4692
'''


def setUpModule():
    global mols
    mols = {
        # s, p, d AO shells (templated kernels for li, lj <= 2)
        'dimer_cart': gto.M(atom=WATER_DIMER, basis='cc-pvdz', cart=True, verbose=0),
        'dimer_sph': gto.M(atom=WATER_DIMER, basis='cc-pvdz', cart=False, verbose=0),
        # f AO shells on F (generic kernels)
        'hf_tz_sph': gto.M(atom='H 0 0 0; F 0 0 0.92', basis='cc-pvtz', cart=False, verbose=0),
    }


def tearDownModule():
    global mols
    del mols


def surface_gaussians(mol, ng=50):
    """GOSTSHYP-like surface Gaussians: SWIG points, gamma = pi ln2 / area."""
    surf = gen_surface(mol, ng=ng, rad=1.2 * modified_Bondi)
    coords = cp.asnumpy(surf['grid_coords'])
    exps = np.pi * np.log(2.0) / cp.asnumpy(surf['area'])
    normals = coords - mol.atom_coords()[np.repeat(
        np.arange(mol.natm), np.diff(np.asarray(surf['gslice_by_atom']), axis=1).ravel())]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return coords, exps, normals


def ref_int3c(mol, coords, exps, l, intor='int3c1e'):
    """S[(x,) i, j, g, c] for normalized Cartesian aux Gaussians of angular momentum l.

    The aux functions are (gamma/pi)^{3/2} x^a y^b z^c exp(-gamma r^2), the
    convention of the GPU kernels; AO indices follow mol.cart.
    """
    molc = mol.copy()
    molc.cart = True
    fake = fakemol_for_gaussian(coords, exps, l=l, cart=True,
                                coeffs=(exps / np.pi) ** 1.5)
    sup = molc + fake
    nbas = molc.nbas
    comp = 3 if intor.endswith('_ip1') else 1
    v = sup.intor(intor, comp=comp, shls_slice=(0, nbas, 0, nbas, nbas, sup.nbas))
    if not mol.cart:
        c2s = mol.cart2sph_coeff()
        v = np.einsum('...pqk,pi,qj->...ijk', v, c2s, c2s, optimize=True)
    ncart = (l + 1) * (l + 2) // 2
    return v.reshape(v.shape[:-1] + (len(exps), ncart))


class KnownValues(unittest.TestCase):

    def _screen(self, mol, coords, exps, cutoff, force_dense):
        intopt = _OverlapPairOpt(mol, cutoff)
        intopt.build(1e-20, aosym=True, group_size=None)
        scr = OverlapScreen(intopt, cp.asarray(coords), cp.asarray(exps),
                            cutoff, force_dense=force_dense)
        self.assertEqual(scr.dense, force_dense or cutoff <= 0)
        if not scr.dense:
            # screening must actually drop (pair, tile) blocks for this test
            npairs = int(intopt.bas_pairs_locs[-1])
            self.assertLess(scr.nnz, npairs * scr.ntiles)
        return scr

    def _check(self, key, cutoff, force_dense):
        mol = mols[key]
        coords, exps, normals = surface_gaussians(mol)
        ngrids = len(exps)
        scr = self._screen(mol, coords, exps, cutoff, force_dense)

        rng = np.random.default_rng(7)
        nao = mol.nao
        dm = rng.standard_normal((nao, nao)) * 0.1   # not symmetric on purpose
        amp_s = rng.standard_normal(ngrids)
        amp_p = rng.standard_normal((ngrids, 3))

        S = [ref_int3c(mol, coords, exps, l) for l in range(4)]
        # sum_ij D_ij S_ij,(g,c), components ordered l = 0..3, PySCF Cartesian order
        rho = [np.einsum('ij,ijgc->gc', dm, s) for s in S]

        def close(a, b, msg):
            b = np.asarray(b)
            a = cp.asnumpy(a)
            self.assertEqual(a.shape, b.shape, msg)
            tol = 1e-11 * max(1.0, abs(b).max())
            self.assertLess(abs(a - b).max(), tol, msg)

        for kmax in (0, 1, 3):
            close(scr.density(mol, cp.asarray(dm), kmax=kmax),
                  np.hstack(rho[:kmax + 1]), f'{key} density kmax={kmax}')

        # grad_density: s, p, sum_q d_xq n_q, tr d, f traces
        d = rho[2]   # xx xy xz yy yz zz
        dmat = d[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
        f = rho[3]   # xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
        ref = np.hstack([
            rho[0], rho[1], np.einsum('gxq,gq->gx', dmat, normals),
            (d[:, 0] + d[:, 3] + d[:, 5])[:, None],
            np.stack([f[:, 0] + f[:, 3] + f[:, 5],
                      f[:, 1] + f[:, 6] + f[:, 8],
                      f[:, 2] + f[:, 7] + f[:, 9]], axis=1)])
        close(scr.grad_density(mol, cp.asarray(dm), cp.asarray(normals)), ref,
              f'{key} grad_density')

        # fock_sp: sum_g amp_s S(s) + sum_gc amp_p S(p_c)
        ref = (np.einsum('ijg,g->ij', S[0][..., 0], amp_s)
               + np.einsum('ijgc,gc->ij', S[1], amp_p))
        close(scr.fock_sp(mol, cp.asarray(amp_s), cp.asarray(amp_p)), ref,
              f'{key} fock_sp')

        # ip1_sp_atoms: <nabla i|j g> (int3c1e_ip1) contracted with D + D^T per atom
        ip_s = ref_int3c(mol, coords, exps, 0, 'int3c1e_ip1')[..., 0]
        ip_p = ref_int3c(mol, coords, exps, 1, 'int3c1e_ip1')
        op = (np.einsum('xijg,g->xij', ip_s, amp_s)
              + np.einsum('xijgc,gc->xij', ip_p, amp_p))
        per_ao = np.einsum('xij,ij->ix', op, dm + dm.T)
        aoslice = mol.aoslice_by_atom()
        ref = np.array([per_ao[p0:p1].sum(axis=0) for p0, p1 in aoslice[:, 2:]])
        close(scr.ip1_sp_atoms(mol, cp.asarray(dm), cp.asarray(amp_s),
                               cp.asarray(amp_p)), ref, f'{key} ip1_sp_atoms')

    def test_screened_cart(self):
        self._check('dimer_cart', 1e-14, False)

    def test_screened_sph(self):
        self._check('dimer_sph', 1e-14, False)

    def test_forced_dense(self):
        self._check('dimer_sph', 1e-14, True)

    def test_zero_cutoff_dense(self):
        self._check('dimer_cart', 0.0, False)

    def test_f_shells_screened(self):
        self._check('hf_tz_sph', 1e-14, False)

    def test_f_shells_zero_cutoff(self):
        self._check('hf_tz_sph', 0.0, False)


if __name__ == "__main__":
    print("Full Tests for screened 3-center overlap integrals")
    unittest.main()
