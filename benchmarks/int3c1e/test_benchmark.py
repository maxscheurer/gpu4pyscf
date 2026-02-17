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


def gostshyp_kernel_cpu_reference(mol, dm, pressure_mpa=50000, npoints=110, scaling_factor=1.2):
    """
    CPU reference implementation of GOSTSHYP kernel using integral-direct
    (streaming/chunked) algorithm. Based on the reference implementation in
    /dfs/is/home/m341593/Projects/overlap_cuda/gostshyp/gostshyp.py (direct=True).

    Integrals are computed on-the-fly in chunks rather than materializing
    the full [nao, nao, ngrids] tensor.
    """
    from pyscf import lib
    from pyscf.solvent.pcm import gen_surface, modified_Bondi
    from .utils import fakemol_for_gaussian

    pressure_au = pressure_mpa * 3.3989309735473356e-08

    # Build surface
    radii = scaling_factor * modified_Bondi
    surface_dict = gen_surface(mol, ng=npoints, rad=radii)
    areas = np.asarray(surface_dict['area'])
    grid_coords = np.asarray(surface_dict['grid_coords'])
    atom_coords = mol.atom_coords()
    gslice_by_atom = surface_dict['gslice_by_atom']

    # Compute atom index and surface normals
    atom_idx = np.zeros(len(areas), dtype=int)
    for ia, (p0, p1) in enumerate(gslice_by_atom):
        atom_idx[p0:p1] = ia

    ref_coords = atom_coords[atom_idx]
    dr = grid_coords - ref_coords
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    surface_normals = dr / dr_norm

    # Compute widths
    widths = np.pi * np.log(2) / areas
    nao = mol.nao
    n_gaussian = len(areas)

    # Build fakemols for full surface (integrals computed via shell slicing)
    gmol = fakemol_for_gaussian(grid_coords, widths, l=0, cart=mol.cart)
    gmol_p = fakemol_for_gaussian(grid_coords, widths, l=1, cart=mol.cart,
                                   coeffs=2.0 * widths)
    supermol = mol + gmol
    supermol_p = mol + gmol_p

    # Determine chunking based on memory
    max_memreq = 5 * n_gaussian * nao**2 * 8.0 / 1e6  # MB
    max_memory = max(2000, mol.max_memory * 0.9 - lib.current_memory()[0])
    n_chunks = 1
    if max_memreq >= max_memory:
        n_chunks = int(max_memreq // max_memory + 1)

    shells = np.arange(n_gaussian)
    chunks = np.array_split(shells, n_chunks)

    energy = 0.0
    fock = np.zeros_like(dm)

    for shell_slice in chunks:
        off1, off2 = int(shell_slice[0]), len(shell_slice)
        slices = (0, mol.nbas, 0, mol.nbas,
                  mol.nbas + off1, mol.nbas + off1 + off2)

        overlap3_s = supermol.intor("int3c1e", shls_slice=slices, aosym="s2")
        overlap3_s = lib.unpack_tril(overlap3_s, axis=0)

        overlap3_p = supermol_p.intor("int3c1e", shls_slice=slices, aosym="s2")
        overlap3_p = lib.unpack_tril(overlap3_p, axis=0).reshape(nao, nao, -1, 3)

        force_operators = np.einsum(
            'ijgc,gc->ijg', overlap3_p,
            surface_normals[shell_slice], optimize=True)
        forces = np.einsum('ij,ijg->g', dm, force_operators, optimize=True)

        amplitudes = pressure_au * areas[shell_slice] / forces

        f1 = np.einsum('g,ijg->ij', amplitudes, overlap3_s, optimize=True)

        gtilde_expval = np.einsum('ij,ijg->g', dm, overlap3_s, optimize=True)
        response_coeff = -pressure_au * areas[shell_slice] * gtilde_expval / (forces ** 2)
        f2 = np.einsum('g,ijg->ij', response_coeff, force_operators, optimize=True)

        fock += f1 + f2
        energy += np.vdot(f1, dm)

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
    from gpu4pyscf.solvent.gostshyp import GOSTSHYP

    mol = create_biphenyl(basis=basis, cart=cart)
    nao = mol.nao

    # Create test density matrix (random symmetric)
    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = (dm + dm.T) / 2

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
                                     scaling_factor=1.2):
    """
    CPU reference implementation of GOSTSHYP gradient using PySCF's libcint.

    Returns the total gradient (natm, 3).
    """
    from pyscf import lib
    from pyscf.solvent.pcm import gen_surface, modified_Bondi
    from pyscf.solvent.grad.pcm import get_dF_dA
    from .utils import fakemol_for_gaussian

    pressure_au = pressure_mpa * 3.3989309735473356e-08
    radii = scaling_factor * modified_Bondi
    surface_dict = gen_surface(mol, ng=npoints, rad=radii)
    areas = np.asarray(surface_dict['area'])
    grid_coords = np.asarray(surface_dict['grid_coords'])
    atom_coords = mol.atom_coords()
    gslice_by_atom = surface_dict['gslice_by_atom']

    atom_idx = np.zeros(len(areas), dtype=int)
    for ia, (p0, p1) in enumerate(gslice_by_atom):
        atom_idx[p0:p1] = ia

    ref_coords = atom_coords[atom_idx]
    dr = grid_coords - ref_coords
    dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
    surface_normals = dr / dr_norm
    widths = np.pi * np.log(2) / areas
    nao = mol.nao
    n_gaussian = len(areas)

    # Energy quantities
    gmol = fakemol_for_gaussian(grid_coords, widths, l=0, cart=mol.cart)
    gmol_p = fakemol_for_gaussian(grid_coords, widths, l=1, cart=mol.cart,
                                   coeffs=2.0 * widths)
    supermol = mol + gmol
    supermol_p = mol + gmol_p
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)

    overlap3_s = supermol.intor("int3c1e", shls_slice=slices, aosym="s1")
    overlap3_p = supermol_p.intor("int3c1e", shls_slice=slices, aosym="s1")
    overlap3_p = overlap3_p.reshape(nao, nao, n_gaussian, 3)
    force_operators = np.einsum('ijgc,gc->ijg', overlap3_p, surface_normals,
                                optimize=True)
    forces = np.einsum('ij,ijg->g', dm, force_operators, optimize=True)
    gtilde_expval = np.einsum('ij,ijg->g', dm, overlap3_s, optimize=True)
    amplitudes = pressure_au * areas / forces

    # Area derivatives
    surface_dict_np = {
        'grid_coords': grid_coords, 'area': areas,
        'gslice_by_atom': gslice_by_atom,
        'R_vdw': np.asarray(surface_dict['R_vdw']),
        'switch_fun': np.asarray(surface_dict['switch_fun']),
        'R_in_J': np.asarray(surface_dict['R_in_J']),
        'R_sw_J': np.asarray(surface_dict['R_sw_J']),
        'atom_coords': np.asarray(surface_dict['atom_coords']),
    }
    _, dareas = get_dF_dA(surface_dict_np)
    dareas = dareas.transpose(1, 2, 0)

    wgrad_prefs = -np.pi * np.log(2) / (areas ** 2)

    # dE1
    dE1 = pressure_au * np.einsum('acg,g->ac', dareas, gtilde_expval / forces,
                                   optimize=True)

    # dE2
    dPQ = supermol.intor("int3c1e_ip1", shls_slice=slices)
    dPQ = np.einsum('xijn,n->xij', dPQ, amplitudes, optimize=True)
    slices_g = (mol.nbas, mol.nbas + gmol.nbas, 0, mol.nbas, 0, mol.nbas)
    dG = supermol.intor("int3c1e_ip1", shls_slice=slices_g)
    aoslice = mol.aoslice_by_atom()
    dgtilde_braket = np.einsum('xij,ij->ix', dPQ, dm, optimize=True)
    dgtilde_braket += np.einsum('xij,ji->ix', dPQ, dm, optimize=True)
    dgtilde_gaussian = np.einsum('xnij,n,ij->nx', dG, amplitudes, dm,
                                  optimize=True)
    gtilde_operator_grad = np.asarray(
        [np.sum(dgtilde_braket[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
    np.add.at(gtilde_operator_grad, atom_idx, dgtilde_gaussian)
    gtilde_operator_grad *= -1.0

    gmol_d = fakemol_for_gaussian(grid_coords, widths, l=2,
                                   coeffs=wgrad_prefs * amplitudes, cart=True)
    supermol_d = mol + gmol_d
    supermol_d.cart = True
    slices_d = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_d.nbas)
    nao_cart = mol.nao_nr(cart=True)
    overlap3d = supermol_d.intor("int3c1e", shls_slice=slices_d).reshape(
        nao_cart, nao_cart, -1, 6)
    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized="sp")
        overlap3d = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3d, c2s,
                              optimize=True)
    diagd = overlap3d[:, :, :, 0] + overlap3d[:, :, :, 3] + overlap3d[:, :, :, 5]
    imd = np.einsum('ijg,ij->g', diagd, dm, optimize=True)
    dE_d = -1.0 * np.einsum('acg,g->ac', dareas, imd, optimize=True)
    dE2 = gtilde_operator_grad + dE_d

    # dE3
    coeffs = -2.0 * pressure_au * areas * gtilde_expval * widths / (forces * forces)
    gmol_p2 = fakemol_for_gaussian(grid_coords, widths, l=1, coeffs=coeffs)
    supermol_p2 = mol + gmol_p2
    slices_p2 = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_p2.nbas)
    dpq = supermol_p2.intor("int3c1e_ip1", shls_slice=slices_p2).reshape(
        3, nao, nao, -1, 3)
    dpq[:, :, :] *= surface_normals
    slices_g2 = (mol.nbas, mol.nbas + gmol_p2.nbas, 0, mol.nbas, 0, mol.nbas)
    dG2 = supermol_p2.intor("int3c1e_ip1", shls_slice=slices_g2).reshape(
        3, -1, 3, nao, nao)
    dpq_ix = np.einsum('xijnp,ij->ix', dpq, dm, optimize=True)
    dpq_ix += np.einsum('xijnp,ji->ix', dpq, dm, optimize=True)
    dG2 = np.einsum('xnpij,np->xnij', dG2, surface_normals, optimize=True)
    dG2 = np.einsum('xnij,ij->nx', dG2, dm, optimize=True)
    force_operator_grad = np.asarray(
        [np.sum(dpq_ix[p0:p1], axis=0) for p0, p1 in aoslice[:, 2:]])
    np.add.at(force_operator_grad, atom_idx, dG2)
    force_operator_grad *= -1.0

    f_coeffs_val = -2.0 * widths * wgrad_prefs
    gmol_f = fakemol_for_gaussian(grid_coords, widths, l=3, coeffs=f_coeffs_val,
                                   cart=True)
    supermol_f = mol + gmol_f
    supermol_f.cart = True
    slices_f = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol_f.nbas)
    overlap3f = supermol_f.intor("int3c1e", shls_slice=slices_f).reshape(
        nao_cart, nao_cart, -1, 10)
    if not mol.cart:
        c2s = mol.cart2sph_coeff(normalized="sp")
        overlap3f = np.einsum('ij,jkgd,kl->ilgd', c2s.T, overlap3f, c2s,
                              optimize=True)
    xf = overlap3f[:, :, :, 0] + overlap3f[:, :, :, 3] + overlap3f[:, :, :, 5]
    yf = overlap3f[:, :, :, 1] + overlap3f[:, :, :, 6] + overlap3f[:, :, :, 8]
    zf = overlap3f[:, :, :, 2] + overlap3f[:, :, :, 7] + overlap3f[:, :, :, 9]
    dx = np.einsum('ijg,ij->g', xf, dm, optimize=True)
    dy = np.einsum('ijg,ij->g', yf, dm, optimize=True)
    dz = np.einsum('ijg,ij->g', zf, dm, optimize=True)
    dr_f = np.vstack((dx, dy, dz)).T
    rf2 = 1.0 / (forces * forces)
    dr_f *= surface_normals
    dFdR = dareas * wgrad_prefs * forces / widths
    dFdR += np.einsum('gc,axg->axg', dr_f, dareas, optimize=True)
    width_grad_ftype = -pressure_au * np.einsum(
        'g,g,axg,g->ax', areas, gtilde_expval, dFdR, rf2, optimize=True)
    dE3 = force_operator_grad + width_grad_ftype

    return dE1 + dE2 + dE3


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
    from gpu4pyscf.solvent.gostshyp import GOSTSHYP
    from gpu4pyscf.solvent.grad.gostshyp import Gradients as GOSTSHYPGradients

    mol = create_biphenyl(basis=basis, cart=cart)
    nao = mol.nao

    # Create test density matrix (random symmetric)
    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = (dm + dm.T) / 2

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
