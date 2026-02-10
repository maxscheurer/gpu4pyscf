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
Correctness tests for 3-center overlap integrals.

These tests validate that GPU results match CPU reference implementations
within numerical tolerance.
"""

import pytest
import numpy as np
from pyscf import gto

from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.gto.int3c1e import VHFOpt

from .utils import compute_int3c_overlap_cpu, generate_test_grids


def make_test_molecule(basis='sto-3g', cart=True):
    """Create a small test molecule (water)."""
    mol = gto.M(
        atom='''
            O  0.0000  0.0000  0.1173
            H  0.0000  0.7572 -0.4692
            H  0.0000 -0.7572 -0.4692
        ''',
        basis=basis,
        unit='angstrom',
        cart=cart
    )
    return mol


def make_h2_molecule(cart=True):
    """Create H2 molecule for contracted tests (known to work with GPU kernels)."""
    mol = gto.M(
        atom='H 0 0 0; H 0.74 0 0',
        basis='sto-3g',
        unit='angstrom',
        cart=cart
    )
    return mol


@pytest.mark.parametrize("cart", [True, False], ids=['cartesian', 'spherical'])
@pytest.mark.parametrize("basis", ['sto-3g', '6-31g', 'def2-svp', 'def2-qzvp'])
def test_int3c_overlap_correctness(cart, basis):
    """Test that GPU 3-center overlap integrals match CPU reference."""
    mol = make_test_molecule(basis=basis, cart=cart)
    nao = mol.nao
    aux_cart = cart  # aux_cart must match mol.cart

    # Small number of grid points for correctness test
    aux_coords, aux_exponents = generate_test_grids(ngrids=10, seed=42)

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU integrals
    int3c_gpu = int3c_overlap.get_int3c_overlap(
        mol, aux_coords, aux_exponents, aux_l=0, intopt=intopt, aux_cart=aux_cart)

    # Compute CPU reference
    int3c_cpu = compute_int3c_overlap_cpu(
        mol, aux_coords, aux_exponents, aux_l=0, aux_cart=aux_cart)

    # Compare
    np.testing.assert_allclose(
        int3c_gpu, int3c_cpu, atol=1e-10, rtol=1e-10,
        err_msg=f"GPU/CPU mismatch for basis={basis}, cart={cart}"
    )


@pytest.mark.parametrize("cart", [True, False], ids=['cartesian', 'spherical'])
@pytest.mark.parametrize("aux_l", [0, 1, 2, 3], ids=['s-type', 'p-type', 'd-type', 'f-type'])
def test_int3c_overlap_angular_momentum(cart, aux_l):
    """Test 3-center overlap with different auxiliary angular momenta.

    Tests all combinations of:
    - AO/aux basis: Cartesian vs spherical (aux_cart matches mol.cart)
    - Auxiliary angular momentum: s, p, d, f
    """
    mol = make_test_molecule(basis='def2-tzvp', cart=cart)
    aux_cart = cart  # aux_cart must match mol.cart

    aux_coords, aux_exponents = generate_test_grids(ngrids=5, seed=123)

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU integrals
    int3c_gpu = int3c_overlap.get_int3c_overlap(
        mol, aux_coords, aux_exponents, aux_l=aux_l, intopt=intopt, aux_cart=aux_cart)

    # Compute CPU reference
    int3c_cpu = compute_int3c_overlap_cpu(
        mol, aux_coords, aux_exponents, aux_l=aux_l, aux_cart=aux_cart)

    np.testing.assert_allclose(
        int3c_gpu, int3c_cpu, atol=1e-10, rtol=1e-10,
        err_msg=f"GPU/CPU mismatch for aux_l={aux_l}, cart={cart}"
    )


@pytest.mark.parametrize("cart", [True, False], ids=['cartesian', 'spherical'])
@pytest.mark.parametrize("basis", ['sto-3g', '6-31g'], ids=['single-shell', 'multi-shell'])
def test_int3c_overlap_density_contracted(cart, basis):
    """Test density-contracted 3-center overlap.

    Tests with both single-shell (sto-3g) and multi-shell (6-31g) bases
    to verify correct AO indexing in the GPU kernel.
    Note: Density-contracted kernel always uses Cartesian auxiliaries.
    """
    mol = make_test_molecule(basis=basis, cart=cart)
    nao = mol.nao
    # Density-contracted uses Cartesian aux, so CPU reference must use cart=True mol
    mol_cart = mol.copy()
    mol_cart.cart = True

    aux_coords, aux_exponents = generate_test_grids(ngrids=5, seed=42)

    # Random symmetric density matrix
    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = dm + dm.T

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU contracted (returns CuPy array)
    forces_gpu = int3c_overlap.get_int3c_overlap_density_contracted(
        mol, aux_coords, aux_exponents, aux_l=0, dm=dm, intopt=intopt)

    # Compute CPU reference using Cartesian aux (aux_cart=True)
    # The kernel contracts with dm internally, so we compute full integral and contract
    int3c_cpu = compute_int3c_overlap_cpu(
        mol_cart, aux_coords, aux_exponents, aux_l=0, aux_cart=True)
    # Transform dm to Cartesian basis if mol uses spherical
    if not mol.cart:
        c2s = mol.cart2sph_coeff()
        dm_cart = c2s @ dm @ c2s.T
    else:
        dm_cart = dm
    forces_cpu = np.einsum('ij,gkij->gk', dm_cart, int3c_cpu)

    np.testing.assert_allclose(
        forces_gpu.get(), forces_cpu, atol=1e-9, rtol=1e-9,
        err_msg=f"Density-contracted GPU/CPU mismatch for cart={cart}, basis={basis}"
    )


@pytest.mark.parametrize("cart", [True, False], ids=['cartesian', 'spherical'])
@pytest.mark.parametrize("basis", ['sto-3g', '6-31g'], ids=['single-shell', 'multi-shell'])
def test_int3c_overlap_amplitude_contracted(cart, basis):
    """Test amplitude-contracted 3-center overlap.

    Tests with both single-shell (sto-3g) and multi-shell (6-31g) bases
    to verify correct AO indexing in the GPU kernel.
    Note: Amplitude-contracted kernel always uses Cartesian auxiliaries.
    """
    mol = make_test_molecule(basis=basis, cart=cart)
    nao = mol.nao
    # Amplitude-contracted uses Cartesian aux, so CPU reference must use cart=True mol
    mol_cart = mol.copy()
    mol_cart.cart = True
    nao_cart = mol_cart.nao

    aux_coords, aux_exponents = generate_test_grids(ngrids=5, seed=123)
    ngrids = len(aux_coords)
    ncart_aux = 1  # s-type

    # Random amplitudes
    np.random.seed(123)
    amplitudes = np.random.randn(ngrids, ncart_aux)

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU contracted (returns CuPy array)
    fock_gpu = int3c_overlap.get_int3c_overlap_amplitude_contracted(
        mol, aux_coords, aux_exponents, aux_l=0, amplitudes=amplitudes, intopt=intopt)

    # Compute CPU reference using Cartesian aux (aux_cart=True)
    int3c_cpu = compute_int3c_overlap_cpu(
        mol_cart, aux_coords, aux_exponents, aux_l=0, aux_cart=True)
    fock_cpu_cart = np.einsum('gk,gkij->ij', amplitudes, int3c_cpu)
    # Transform Fock from Cartesian to spherical if mol uses spherical
    if not mol.cart:
        c2s = mol.cart2sph_coeff()
        fock_cpu = c2s.T @ fock_cpu_cart @ c2s
    else:
        fock_cpu = fock_cpu_cart

    np.testing.assert_allclose(
        fock_gpu.get(), fock_cpu, atol=1e-9, rtol=1e-9,
        err_msg=f"Amplitude-contracted GPU/CPU mismatch for cart={cart}, basis={basis}"
    )


def test_int3c_overlap_convenience_api():
    """Test the convenience API that builds VHFOpt internally."""
    mol = make_test_molecule(basis='sto-3g', cart=True)

    aux_coords, aux_exponents = generate_test_grids(ngrids=5, seed=42)

    # Use convenience API (builds VHFOpt internally)
    int3c_gpu = int3c_overlap.int3c_overlap(
        mol, aux_coords, aux_exponents, aux_l=0)

    # Compute CPU reference
    int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0)

    np.testing.assert_allclose(
        int3c_gpu, int3c_cpu, atol=1e-10, rtol=1e-10,
        err_msg="Convenience API GPU/CPU mismatch"
    )
