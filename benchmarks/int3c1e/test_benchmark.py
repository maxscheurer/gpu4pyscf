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
Performance benchmarks for 3-center overlap integrals.

These benchmarks measure GPU vs CPU performance with proper timing
(CUDA events for GPU, excluding memory transfers where possible).

Results are collected using pytest-harvest and saved to CSV.
"""

import pytest
import numpy as np

from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.gto.int3c1e import VHFOpt

from .molecules import create_hydrogen_grid, create_water_box
from pyscf.solvent.gostshyp import GOSTSHYP as GOSTSHYP_CPU

from .utils import (
    compute_int3c_overlap_cpu,
    gpu_timer_events,
    cpu_timer,
    generate_test_grids,
)


def format_time(t):
    """Format time in human-readable units."""
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    elif t < 1:
        return f"{t*1e3:.2f}ms"
    else:
        return f"{t:.3f}s"


def print_benchmark_result(test_name, molecule, nao, ngrids, basis, cart,
                           gpu_time, cpu_time, extra_info=""):
    """Print a nicely formatted benchmark result."""
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    cart_str = "cart" if cart else "sph"

    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    print(f"  Molecule: {molecule:10s}  Basis: {basis:12s}  Mode: {cart_str}")
    print(f"  NAO: {nao:4d}            Grids: {ngrids:4d}")
    if extra_info:
        print(f"  {extra_info}")
    print(f"  " + "-"*66)
    print(f"  GPU time:  {format_time(gpu_time):>10s}")
    print(f"  CPU time:  {format_time(cpu_time):>10s}")
    print(f"  Speedup:   {speedup:>10.2f}x  {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")
    print(f"{'='*70}")


# =============================================================================
# Full Integral Benchmarks
# =============================================================================

@pytest.mark.parametrize("basis", ['def2-svp', 'def2-tzvp', 'def2-tzvpp'])
@pytest.mark.parametrize("cart", [True, False], ids=['cart', 'sph'])
def test_benchmark_full_integral_water(basis, cart, results_bag):
    """Benchmark full 3-center overlap integral on water box."""
    mol = create_water_box(2, 2, 2, basis=basis, cart=cart)
    nao = mol.nao
    ngrids = 1000
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap(
            mol, aux_coords, aux_exponents, aux_l=0, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    cpu_result, cpu_time, cpu_std = cpu_timer(
        lambda: compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart),
        n_warmup=2, n_runs=5
    )

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Full Integral (get_int3c_overlap)",
        "8xH2O", nao, ngrids, basis, cart, gpu_time, cpu_time
    )

    results_bag['test_type'] = 'full_integral'
    results_bag['molecule'] = '8xH2O'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = basis
    results_bag['cart'] = cart
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


@pytest.mark.parametrize("grid_size", [(3, 3, 3), (4, 4, 4)],
                         ids=['27H', '64H'])
def test_benchmark_full_integral_hydrogen(grid_size, results_bag):
    """Benchmark full integral on hydrogen grid with def2-tzvp."""
    Nx, Ny, Nz = grid_size
    mol = create_hydrogen_grid(Nx, Ny, Nz, basis='def2-tzvp', cart=False)
    nao = mol.nao
    natoms = Nx * Ny * Nz
    ngrids = 1000
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap(
            mol, aux_coords, aux_exponents, aux_l=0, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    cpu_result, cpu_time, cpu_std = cpu_timer(
        lambda: compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart),
        n_warmup=2, n_runs=5
    )

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Full Integral (get_int3c_overlap)",
        f"H{natoms}", nao, ngrids, 'def2-tzvp', False, gpu_time, cpu_time
    )

    results_bag['test_type'] = 'full_integral'
    results_bag['molecule'] = f'H{natoms}'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = 'def2-tzvp'
    results_bag['cart'] = False
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


# =============================================================================
# Density-Contracted Benchmarks
# =============================================================================

@pytest.mark.parametrize("basis", ['def2-svp', 'def2-tzvp', 'def2-tzvpp', 'def2-qzvp'])
@pytest.mark.parametrize("cart", [True, False], ids=['cart', 'sph'])
def test_benchmark_density_contracted_water(basis, cart, results_bag):
    """Benchmark density-contracted integral on water box."""
    mol = create_water_box(2, 2, 2, basis=basis, cart=cart)
    nao = mol.nao
    ngrids = 100
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)

    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = (dm + dm.T) / 2

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap_density_contracted(
            mol, aux_coords, aux_exponents, aux_l=0, dm=dm, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    def cpu_density_contracted():
        int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart)
        return np.einsum('ij,gkij->gk', dm, int3c_cpu)

    cpu_result, cpu_time, cpu_std = cpu_timer(cpu_density_contracted, n_warmup=2, n_runs=5)

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Density-Contracted (D_ij * S_ijk -> F_k)",
        "8xH2O", nao, ngrids, basis, cart, gpu_time, cpu_time,
        "Operation: sum_ij dm[i,j] * int3c[g,k,i,j]"
    )

    results_bag['test_type'] = 'density_contracted'
    results_bag['molecule'] = '8xH2O'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = basis
    results_bag['cart'] = cart
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


@pytest.mark.parametrize("grid_size", [(3, 3, 3), (4, 4, 4)],
                         ids=['27H', '64H'])
def test_benchmark_density_contracted_hydrogen(grid_size, results_bag):
    """Benchmark density-contracted integral on hydrogen grid."""
    Nx, Ny, Nz = grid_size
    mol = create_hydrogen_grid(Nx, Ny, Nz, basis='def2-tzvp', cart=False)
    nao = mol.nao
    natoms = Nx * Ny * Nz
    ngrids = 100
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)

    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = (dm + dm.T) / 2

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap_density_contracted(
            mol, aux_coords, aux_exponents, aux_l=0, dm=dm, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    def cpu_density_contracted():
        int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart)
        return np.einsum('ij,gkij->gk', dm, int3c_cpu)

    cpu_result, cpu_time, cpu_std = cpu_timer(cpu_density_contracted, n_warmup=2, n_runs=5)

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Density-Contracted (D_ij * S_ijk -> F_k)",
        f"H{natoms}", nao, ngrids, 'def2-tzvp', False, gpu_time, cpu_time,
        "Operation: sum_ij dm[i,j] * int3c[g,k,i,j]"
    )

    results_bag['test_type'] = 'density_contracted'
    results_bag['molecule'] = f'H{natoms}'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = 'def2-tzvp'
    results_bag['cart'] = False
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


# =============================================================================
# Amplitude-Contracted Benchmarks
# =============================================================================

@pytest.mark.parametrize("basis", ['def2-svp', 'def2-tzvp', 'def2-tzvpp', 'def2-qzvp'])
@pytest.mark.parametrize("cart", [True, False], ids=['cart', 'sph'])
def test_benchmark_amplitude_contracted_water(basis, cart, results_bag):
    """Benchmark amplitude-contracted integral on water box."""
    mol = create_water_box(2, 2, 2, basis=basis, cart=cart)
    nao = mol.nao
    ngrids = 1000
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)
    ncart_aux = 1

    np.random.seed(123)
    amplitudes = np.random.randn(ngrids, ncart_aux)

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap_amplitude_contracted(
            mol, aux_coords, aux_exponents, aux_l=0, amplitudes=amplitudes, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    def cpu_amplitude_contracted():
        int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart)
        return np.einsum('gk,gkij->ij', amplitudes, int3c_cpu)

    cpu_result, cpu_time, cpu_std = cpu_timer(cpu_amplitude_contracted, n_warmup=2, n_runs=5)

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Amplitude-Contracted (A_k * S_ijk -> F_ij)",
        "8xH2O", nao, ngrids, basis, cart, gpu_time, cpu_time,
        "Operation: sum_gk amp[g,k] * int3c[g,k,i,j]"
    )

    results_bag['test_type'] = 'amplitude_contracted'
    results_bag['molecule'] = '8xH2O'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = basis
    results_bag['cart'] = cart
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


@pytest.mark.parametrize("grid_size", [(3, 3, 3), (4, 4, 4)],
                         ids=['27H', '64H'])
def test_benchmark_amplitude_contracted_hydrogen(grid_size, results_bag):
    """Benchmark amplitude-contracted integral on hydrogen grid."""
    Nx, Ny, Nz = grid_size
    mol = create_hydrogen_grid(Nx, Ny, Nz, basis='def2-tzvp', cart=False)
    nao = mol.nao
    natoms = Nx * Ny * Nz
    ngrids = 100
    aux_coords, aux_exponents = generate_test_grids(ngrids=ngrids, seed=42)
    ncart_aux = 1

    np.random.seed(123)
    amplitudes = np.random.randn(ngrids, ncart_aux)

    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: int3c_overlap.get_int3c_overlap_amplitude_contracted(
            mol, aux_coords, aux_exponents, aux_l=0, amplitudes=amplitudes, intopt=intopt),
        n_warmup=3, n_runs=10
    )

    def cpu_amplitude_contracted():
        int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=mol.cart)
        return np.einsum('gk,gkij->ij', amplitudes, int3c_cpu)

    cpu_result, cpu_time, cpu_std = cpu_timer(cpu_amplitude_contracted, n_warmup=2, n_runs=5)

    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-9, rtol=1e-9)

    print_benchmark_result(
        "Amplitude-Contracted (A_k * S_ijk -> F_ij)",
        f"H{natoms}", nao, ngrids, 'def2-tzvp', False, gpu_time, cpu_time,
        "Operation: sum_gk amp[g,k] * int3c[g,k,i,j]"
    )

    results_bag['test_type'] = 'amplitude_contracted'
    results_bag['molecule'] = f'H{natoms}'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = 'def2-tzvp'
    results_bag['cart'] = False
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


# =============================================================================
# GOSTSHYP Kernel Benchmarks
# =============================================================================

def create_biphenyl(basis='def2-tzvp', cart=True):
    """
    Create biphenyl molecule (C12H10) for GOSTSHYP benchmarks.

    Biphenyl is a mid-size organic molecule with ~264 basis functions
    in def2-TZVP, representative of realistic molecular applications.
    """
    from pyscf import gto

    # Biphenyl geometry (Angstrom)
    atom = '''
    C   0.000000   1.400000   0.000000
    C   1.212436   0.700000   0.000000
    C   1.212436  -0.700000   0.000000
    C   0.000000  -1.400000   0.000000
    C  -1.212436  -0.700000   0.000000
    C  -1.212436   0.700000   0.000000
    H   0.000000   2.490000   0.000000
    H   2.156069   1.245000   0.000000
    H   2.156069  -1.245000   0.000000
    H   0.000000  -2.490000   0.000000
    H  -2.156069  -1.245000   0.000000
    H  -2.156069   1.245000   0.000000
    C   0.000000   1.400000   4.200000
    C   1.212436   0.700000   4.200000
    C   1.212436  -0.700000   4.200000
    C   0.000000  -1.400000   4.200000
    C  -1.212436  -0.700000   4.200000
    C  -1.212436   0.700000   4.200000
    H   0.000000   2.490000   4.200000
    H   2.156069   1.245000   4.200000
    H   2.156069  -1.245000   4.200000
    H   0.000000  -2.490000   4.200000
    H  -2.156069  -1.245000   4.200000
    H  -2.156069   1.245000   4.200000
    '''

    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', cart=cart, verbose=0)
    return mol


def gostshyp_kernel_cpu_reference(mol, dm, pressure_mpa=50000, npoints=110,
                                   scaling_factor=1.2, cavity='vdw/occ'):
    """
    CPU reference implementation of GOSTSHYP kernel using pyscf-forge.

    Returns (energy, fock) tuple.
    """
    cpu = GOSTSHYP_CPU(mol, options={
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
        'cavity': cavity,
        'direct': True,
    })
    cpu.build()
    energy, fock = cpu.kernel(dm)
    return energy, fock


@pytest.mark.parametrize("basis", ['def2-svp', 'def2-tzvp'])
@pytest.mark.parametrize("cart", [True, False], ids=['cart', 'sph'])
def test_benchmark_gostshyp_biphenyl(basis, cart, results_bag):
    """
    Benchmark GPU vs CPU GOSTSHYP kernel on biphenyl molecule.

    This tests the full GOSTSHYP kernel including:
    - Forces computation via density-contracted p-type integrals
    - Fock term 1 via amplitude-contracted s-type integrals
    - gtilde_expval via density-contracted s-type integrals
    - Fock term 2 via amplitude-contracted p-type integrals
    """
    from pyscf import scf as cpu_scf
    from gpu4pyscf.solvent.gostshyp import GOSTSHYP

    mol = create_biphenyl(basis=basis, cart=cart)
    nao = mol.nao

    # Use converged SCF DM to ensure all forces are positive
    # (avoids differences in negative-amplitude handling between GPU and CPU)
    mf = cpu_scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    # Build GPU GOSTSHYP
    gostshyp_gpu = GOSTSHYP(mol)
    gostshyp_gpu.build()
    ngrids = gostshyp_gpu.n_gaussian

    # GPU timing
    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: gostshyp_gpu.kernel(dm),
        n_warmup=3, n_runs=10
    )

    # CPU timing (integral-direct mode)
    cpu_result, cpu_time, cpu_std = cpu_timer(
        lambda: gostshyp_kernel_cpu_reference(mol, dm),
        n_warmup=2, n_runs=5
    )

    # Verify correctness
    energy_gpu, fock_gpu = gpu_result
    energy_cpu, fock_cpu = cpu_result
    # Convert CuPy to numpy if needed
    if hasattr(fock_gpu, 'get'):
        fock_gpu = fock_gpu.get()
    # Use 1e-7 atol/rtol for benchmarks - this accounts for:
    # 1. Different accumulation order between GPU and CPU
    # 2. Near-zero elements where GPU applies tighter cutoffs than CPU einsum
    np.testing.assert_allclose(energy_gpu, energy_cpu, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(fock_gpu, fock_cpu, atol=1e-7, rtol=1e-7)

    print_benchmark_result(
        "GOSTSHYP kernel",
        "biphenyl", nao, ngrids, basis, cart, gpu_time, cpu_time,
        "Full kernel: 2x density-contracted + 2x amplitude-contracted"
    )

    results_bag['test_type'] = 'gostshyp'
    results_bag['molecule'] = 'biphenyl'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = basis
    results_bag['cart'] = cart
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


# =============================================================================
# GOSTSHYP Gradient Benchmarks
# =============================================================================

def gostshyp_gradient_cpu_reference(mol, dm, pressure_mpa=50000, npoints=110,
                                     scaling_factor=1.2, cavity='vdw/occ'):
    """
    CPU reference implementation of GOSTSHYP gradient using pyscf-forge.

    Returns the total gradient (natm, 3).
    """
    cpu = GOSTSHYP_CPU(mol, options={
        'pressure_mpa': pressure_mpa,
        'npoints': npoints,
        'scaling_factor': scaling_factor,
        'cavity': cavity,
        'direct': False,
    })
    cpu.build()
    cpu.kernel(dm)
    return cpu.grad(dm)


@pytest.mark.parametrize("basis", ['def2-svp', 'def2-tzvp'])
@pytest.mark.parametrize("cart", [True, False], ids=['cart', 'sph'])
def test_benchmark_gostshyp_gradient_biphenyl(basis, cart, results_bag):
    """
    Benchmark GPU vs CPU GOSTSHYP gradient on biphenyl molecule.

    This tests the full GOSTSHYP gradient including:
    - Area derivatives (dE1)
    - Gtilde operator gradient with ip1 s-type + ip2 + d-type width (dE2)
    - Force operator gradient with ip1 p-type + ip2 + f-type width (dE3)
    """
    from pyscf import scf as cpu_scf
    from gpu4pyscf.solvent.gostshyp import GOSTSHYP
    from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients

    mol = create_biphenyl(basis=basis, cart=cart)
    nao = mol.nao

    # Use converged SCF DM to ensure all forces are positive
    mf = cpu_scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    dm = mf.make_rdm1()

    # Build GPU GOSTSHYP and run energy first (gradient needs cached values)
    gostshyp_gpu = GOSTSHYP(mol)
    gostshyp_gpu.build()
    gostshyp_gpu.kernel(dm)
    ngrids = gostshyp_gpu.n_gaussian

    grad_obj = GOSTSHYPGradients(gostshyp_gpu)

    # GPU timing
    gpu_result, gpu_time, gpu_std = gpu_timer_events(
        lambda: grad_obj.kernel(dm),
        n_warmup=3, n_runs=10
    )

    # CPU timing
    cpu_result, cpu_time, cpu_std = cpu_timer(
        lambda: gostshyp_gradient_cpu_reference(mol, dm),
        n_warmup=2, n_runs=5
    )

    # Verify correctness — use 1e-5 atol since gradient values are O(100) for
    # larger bases, and accumulation order differences grow with system size
    np.testing.assert_allclose(gpu_result, cpu_result, atol=1e-5, rtol=1e-5)

    print_benchmark_result(
        "GOSTSHYP gradient",
        "biphenyl", nao, ngrids, basis, cart, gpu_time, cpu_time,
        "Full gradient: dE1 (area) + dE2 (gtilde ip1/ip2/d) + dE3 (force ip1/ip2/f)"
    )

    results_bag['test_type'] = 'gostshyp_gradient'
    results_bag['molecule'] = 'biphenyl'
    results_bag['nao'] = nao
    results_bag['ngrids'] = ngrids
    results_bag['basis'] = basis
    results_bag['cart'] = cart
    results_bag['gpu_time'] = gpu_time
    results_bag['cpu_time'] = cpu_time
    results_bag['speedup'] = cpu_time / gpu_time if gpu_time > 0 else float('inf')


# =============================================================================
# Synthesis - Runs last and prints summary table
# =============================================================================

def test_zzz_synthesis(fixture_store):
    """
    Synthesize benchmark results into a summary table.
    """
    import os
    import pandas as pd

    results_bag_data = fixture_store.get('results_bag', {})
    if not results_bag_data:
        pytest.skip("No benchmark results to synthesize")

    rows = []
    for test_id, bag in results_bag_data.items():
        if bag:
            row = bag.copy()
            row['test_id'] = test_id
            rows.append(row)

    if not rows:
        pytest.skip("No benchmark results with data")

    df = pd.DataFrame(rows)

    # Save to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'int3c1e_benchmark_results.csv')
    df.to_csv(output_path, index=False)

    # Print beautiful summary
    print("\n")
    print("=" * 90)
    print("                      3-CENTER OVERLAP INTEGRAL BENCHMARK SUMMARY")
    print("=" * 90)

    # Group by test type
    for test_type in ['full_integral', 'density_contracted', 'amplitude_contracted',
                       'gostshyp', 'gostshyp_gradient']:
        subset = df[df['test_type'] == test_type] if 'test_type' in df.columns else df
        if len(subset) == 0:
            continue

        type_name = {
            'full_integral': 'FULL INTEGRAL (get_int3c_overlap)',
            'density_contracted': 'DENSITY-CONTRACTED (D_ij * S_ijk -> F_k)',
            'amplitude_contracted': 'AMPLITUDE-CONTRACTED (A_k * S_ijk -> F_ij)',
            'gostshyp': 'GOSTSHYP KERNEL (full pressure model)',
            'gostshyp_gradient': 'GOSTSHYP GRADIENT (dE1 + dE2 + dE3)'
        }.get(test_type, test_type.upper())

        print(f"\n{type_name}")
        print("-" * 90)
        print(f"{'Molecule':<10} {'Basis':<12} {'Mode':<5} {'NAO':>6} {'Grids':>6} "
              f"{'GPU':>10} {'CPU':>10} {'Speedup':>10}")
        print("-" * 90)

        for _, row in subset.iterrows():
            cart_str = "cart" if row.get('cart', True) else "sph"
            gpu_str = f"{row['gpu_time']*1000:.2f}ms"
            cpu_str = f"{row['cpu_time']*1000:.2f}ms"
            speedup = row.get('speedup', row['cpu_time']/row['gpu_time'])
            speedup_str = f"{speedup:.2f}x"

            print(f"{row['molecule']:<10} {row.get('basis', 'N/A'):<12} {cart_str:<5} "
                  f"{row['nao']:>6} {row['ngrids']:>6} {gpu_str:>10} {cpu_str:>10} {speedup_str:>10}")

    print("\n" + "=" * 90)
    print(f"Results saved to: {output_path}")
    print("=" * 90 + "\n")
