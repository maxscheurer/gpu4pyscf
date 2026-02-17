"""Stress test: GOSTSHYP SCF + gradient on C60."""
import sys
import time
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.solvent.gostshyp import GOSTSHYP
from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients

xyz_file = sys.argv[1] if len(sys.argv) > 1 else 'c60.xyz'
basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-svp'

mol = gto.M(atom=xyz_file, basis=basis, unit='Angstrom', verbose=5)
print(f'\n{"="*70}')
print(f'  {xyz_file} / {basis} / sph')
print(f'  Atoms: {mol.natm}, NAO: {mol.nao}')
print(f'{"="*70}')

mempool = cp.get_default_memory_pool()

# RHF + GOSTSHYP SCF
mf = scf.RHF(mol).density_fit("def2-universal-jkfit")
mf = mf.GOSTSHYP()

mf.max_cycle = 100
mf.conv_tol = 1e-8

print('\nStarting SCF...')
t0 = time.perf_counter()
e = mf.kernel()
t_scf = time.perf_counter() - t0
sol = mf.with_solvent
print(f'\nSCF energy: {e:.10f}')
print(f'SCF time:        {t_scf:.2f} s')
print(f'  GOSTSHYP wall: {sol._t_wall:.3f} s ({sol._n_kernel} calls, '
      f'{sol._t_wall/t_scf*100:.1f}% of SCF)')
print(f'  GOSTSHYP GPU:  {sol._t_gpu_ms:.1f} ms')
print(f'Converged:       {mf.converged}')
print(f'GPU mem:         {mempool.used_bytes() / 1e6:.1f} MB')

# Gradient
print('\nComputing GOSTSHYP gradient...')
dm = mf.make_rdm1()
ngrids = sol.n_gaussian
print(f'Surface grid points: {ngrids}')

t0 = time.perf_counter()
grad = GOSTSHYPGradients(sol).kernel(dm)
t_grad = time.perf_counter() - t0
print(f'Gradient time: {t_grad:.2f} s')
print(f'GPU mem:       {mempool.used_bytes() / 1e6:.1f} MB')
print(f'Max |grad|:    {np.max(np.abs(grad)):.6e}')
print(f'Sum (Newton 3): {np.sum(grad, axis=0)}')

print(f'\n{"="*70}')
print(f'  SUMMARY: {xyz_file} / {basis}')
print(f'  NAO={mol.nao}, ngrids={ngrids}')
print(f'  SCF:      {t_scf:.2f} s')
print(f'  Gradient: {t_grad:.2f} s')
print(f'  Peak GPU: {mempool.total_bytes() / 1e6:.1f} MB')
print(f'{"="*70}')
