"""
Benchmark GOSTSHYP with vdW/OCC on Amylose chains.

PBE/def2-SV(P), 50 GPa, vdW/OCC cavity (default).
Parametrized over Amylose{1,2,4,8,16,32,48,64}.xyz.

Run:
    pytest test_amylose_benchmark.py -v -s

Summary table and CSV are written at session end.
"""
import os
import time
import numpy as np
import cupy as cp
import pytest
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.solvent.gostshyp import GOSTSHYP, gostshyp_for_scf

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BASIS_FILE = os.path.join(BENCH_DIR, 'def2sv_p_.dat')

# Ordered by size so smaller systems run first
AMYLOSE_XYZS = [
    'Amylose1.xyz',   #  24 atoms
    'Amylose2.xyz',   #  45 atoms
    'Amylose4.xyz',   #  87 atoms
    'Amylose8.xyz',   # 171 atoms
    'Amylose16.xyz',  # 339 atoms
    'Amylose32.xyz',  # 675 atoms
    'Amylose48.xyz',  # 1011 atoms
    'Amylose64.xyz',  # 1347 atoms
]

# Module-level list to collect results from each parametrized test
_results = []


def _label(xyz):
    return xyz.replace('.xyz', '')


@pytest.fixture(params=AMYLOSE_XYZS, ids=[_label(x) for x in AMYLOSE_XYZS])
def xyz_file(request):
    return os.path.join(BENCH_DIR, request.param)


def test_amylose(xyz_file):
    """Run PBE/def2-SV(P) + GOSTSHYP(50 GPa, vdW/OCC) SCF + gradient."""
    label = _label(os.path.basename(xyz_file))
    row = {'system': label}
    overlap_cutoff = 1e-11

    mol = gto.M(atom=xyz_file, basis=BASIS_FILE, unit='Ang', verbose=4)
    row['natom'] = mol.natm
    row['nao'] = mol.nao
    row['overlap_cutoff'] = overlap_cutoff

    # --- SCF ---
    mf = dft.RKS(mol)#.density_fit("def2-universal-jfit")
    mf.xc = 'pbe'
    mf.conv_tol = 1e-7
    mf.max_cycle = 200
    mf.grids.level = 3

    gost = GOSTSHYP(mol, {'pressure_mpa': 50_000, 'overlap_cutoff': overlap_cutoff })
    mf = gostshyp_for_scf(mf, gost)

    t0 = time.perf_counter()
    e_tot = mf.kernel()
    t_scf = time.perf_counter() - t0

    row['converged'] = bool(mf.converged)
    row['e_tot'] = float(e_tot)
    row['e_gostshyp'] = float(gost.e)
    row['ngrids'] = int(gost.n_gaussian)
    row['t_scf'] = round(t_scf, 1)
    row['t_gost_scf'] = round(gost._t_wall, 1)
    row['%_gost_scf'] = round(100.0 * gost._t_wall / t_scf, 1) if t_scf > 0 else 0.0

    # Force statistics
    forces_np = cp.asnumpy(gost.forces)
    row['n_neg_F'] = int(np.sum(forces_np <= 0))
    row['min_F'] = float(np.min(forces_np))

    # Area statistics
    areas_np = cp.asnumpy(gost.areas)
    row['min_A'] = float(np.min(areas_np))
    row['total_A'] = float(np.sum(areas_np))

    assert mf.converged, f'{label}: SCF did not converge'

    # --- Gradient ---
    t0 = time.perf_counter()
    grad_obj = mf.nuc_grad_method()
    grad_obj.verbose = 0
    de = grad_obj.kernel()
    t_grad = time.perf_counter() - t0

    row['t_grad'] = round(t_grad, 1)
    row['t_gost_grad'] = round(gost._grad_t_wall, 1)
    row['%_gost_grad'] = round(100.0 * gost._grad_t_wall / t_grad, 1) if t_grad > 0 else 0.0
    row['max_grad'] = float(np.max(np.abs(de)))
    row['rms_grad'] = float(np.sqrt(np.mean(de**2)))
    row['max_grad_gost'] = float(np.max(np.abs(grad_obj.de_solvent)))

    _results.append(row)
