#!/usr/bin/env python
"""Profile MOIST and GPU costs for a DROP GOSTSHYP cavity.

Examples
--------
conda run -n overlap_cuda python benchmarks/gostshyp/bench_drop.py
conda run -n overlap_cuda python benchmarks/gostshyp/bench_drop.py \
    --xyz benchmarks/gostshyp/Amylose8.xyz --skip-gradient
"""

import argparse
import json
import os
import time

import cupy as cp
from pyscf import gto, scf

from gpu4pyscf.solvent.gostshyp import GOSTSHYP


HERE = os.path.dirname(os.path.abspath(__file__))


def timed(function):
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    result = function()
    cp.cuda.Device().synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz', default=os.path.join(HERE, 'Amylose1.xyz'))
    parser.add_argument('--basis', default=os.path.join(HERE, 'def2sv_p_.dat'))
    parser.add_argument('--npoints', type=int, default=110)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--skip-gradient', action='store_true')
    args = parser.parse_args()

    mol = gto.M(atom=args.xyz, basis=args.basis, unit='Ang', verbose=0)
    dm = scf.RHF(mol).get_init_guess()
    gost = GOSTSHYP(mol, options={
        'cavity': 'drop',
        'npoints': args.npoints,
    })

    _, build_wall = timed(gost.build)
    kernel_times = []
    for _ in range(args.repeats):
        _, elapsed = timed(lambda: gost.kernel(dm))
        kernel_times.append(elapsed)

    result = {
        'system': os.path.basename(args.xyz),
        'natom': mol.natm,
        'nao': mol.nao,
        'ngrids': gost.n_gaussian,
        'build_wall_s': build_wall,
        'moist_build_s': gost._t_moist_build,
        'kernel_median_s': float(sorted(kernel_times)[len(kernel_times) // 2]),
        'kernel_calls': gost._n_kernel,
        'moist_gradient_s': 0.0,
    }

    if not args.skip_gradient:
        _, first_gradient = timed(lambda: gost.grad(dm))
        moist_after_first = gost._t_moist_grad
        _, cached_gradient = timed(lambda: gost.grad(dm))
        result.update({
            'gradient_first_s': first_gradient,
            'gradient_cached_s': cached_gradient,
            'moist_gradient_s': moist_after_first,
            'moist_gradient_recomputed': gost._t_moist_grad != moist_after_first,
            'moist_fraction_first_gradient': (
                moist_after_first / first_gradient if first_gradient else 0.0),
        })

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
