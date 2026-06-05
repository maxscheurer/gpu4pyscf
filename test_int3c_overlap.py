#!/usr/bin/env python
"""
Test script for 3-center overlap integral implementation.

This script tests the GPU implementation of 3-center overlap integrals
against a CPU reference using PySCF's existing integral machinery.
"""

import numpy as np
from pyscf import gto, scf, lib
from pyscf.solvent.gostshyp import fakemol_for_gaussian
from gpu4pyscf.gto import int3c_overlap
from gpu4pyscf.gto.int3c1e import VHFOpt

def make_test_molecule(name='h2o'):
    """Create test molecules."""
    if name == 'h2':
        mol = gto.M(
            atom='H 0 0 0; H 0.74 0 0',
            basis='sto-3g',
            unit='angstrom'
        )
    elif name == 'h2o':
        mol = gto.M(
            atom='''
                O  0.0000  0.0000  0.1173
                H  0.0000  0.7572 -0.4692
                H  0.0000 -0.7572 -0.4692
            ''',
            basis='6-31g',
            unit='angstrom'
        )
    elif name == 'hf':
        mol = gto.M(
            atom='H 0 0 0; F 1 0 0',
            basis='cc-pvdz',
            unit='angstrom'
        )
    else:
        raise ValueError(f"Unknown molecule: {name}")
    return mol


def compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0):
    """
    Compute 3-center overlap integrals using PySCF's int3c1e with supermol approach.

    Includes N_j = (γ/π)^{3/2} normalization to match the GPU kernel convention.
    """
    nao = mol.nao
    ngrids = len(aux_coords)
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2

    # GPU kernel uses (gamma/pi)^{3/2} prefactor in the integral
    N_j = (aux_exponents / np.pi) ** 1.5

    # Build fake molecule for auxiliary Gaussians
    gmol = fakemol_for_gaussian(aux_coords, aux_exponents, l=aux_l, cart=mol.cart, coeffs=N_j)

    # Create supermolecule and compute integrals
    supermol = mol + gmol
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)
    int3c = supermol.intor("int3c1e", shls_slice=slices, aosym="s1")
    # Shape: (nao, nao, ngrids * ncart_aux)

    # Reshape to (ngrids, ncart_aux, nao, nao)
    int3c = int3c.reshape(nao, nao, ngrids, ncart_aux).transpose(2, 3, 0, 1)

    return int3c


def test_basic_integral():
    """Test basic 3-center overlap integral computation."""
    print("Test: Basic 3-center overlap integral")
    print("-" * 50)

    mol = make_test_molecule('h2')
    nao = mol.nao
    print(f"Molecule: H2, basis: sto-3g, nao: {nao}")

    # Create a few surface points
    aux_coords = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    aux_exponents = np.array([1.0, 1.0, 1.0])
    ngrids = len(aux_coords)
    print(f"Number of grid points: {ngrids}")

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU integrals
    print("Computing GPU integrals...")
    int3c_gpu = int3c_overlap.get_int3c_overlap(
        mol, aux_coords, aux_exponents, aux_l=0, intopt=intopt)
    print(f"GPU result shape: {int3c_gpu.shape}")

    # Compute CPU reference
    print("Computing CPU reference...")
    int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0)
    print(f"CPU result shape: {int3c_cpu.shape}")

    # Compare
    max_diff = np.max(np.abs(int3c_gpu - int3c_cpu))
    print(f"Max difference: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("PASS: GPU and CPU results match!")
        return True
    else:
        print("FAIL: Results don't match")
        print("GPU sample:\n", int3c_gpu[0, 0])
        print("CPU sample:\n", int3c_cpu[0, 0])
        return False


def test_density_contracted():
    """Test density-contracted 3-center overlap."""
    print("\nTest: Density-contracted 3-center overlap")
    print("-" * 50)

    mol = make_test_molecule('h2')
    nao = mol.nao

    # Create surface points
    aux_coords = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    aux_exponents = np.array([0.5, 0.5])
    ngrids = len(aux_coords)

    # Random density matrix
    np.random.seed(42)
    dm = np.random.randn(nao, nao)
    dm = dm + dm.T  # symmetrize

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU contracted
    print("Computing GPU density-contracted...")
    forces_gpu = int3c_overlap.get_int3c_overlap_density_contracted(
        mol, aux_coords, aux_exponents, aux_l=0, dm=dm, intopt=intopt)
    forces_gpu = forces_gpu.get()  # Convert CuPy to NumPy for comparison
    print(f"GPU forces shape: {forces_gpu.shape}")

    # Compute CPU reference by contracting full tensor
    print("Computing CPU reference...")
    int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0)
    forces_cpu = np.einsum('ij,gkij->gk', dm, int3c_cpu)
    print(f"CPU forces shape: {forces_cpu.shape}")

    # Compare
    max_diff = np.max(np.abs(forces_gpu - forces_cpu))
    print(f"Max difference: {max_diff:.2e}")
    print(f"GPU forces: {forces_gpu.flatten()}")
    print(f"CPU forces: {forces_cpu.flatten()}")

    if max_diff < 1e-8:
        print("PASS: GPU and CPU density-contracted results match!")
        return True
    else:
        print("FAIL: Results don't match")
        return False


def test_amplitude_contracted():
    """Test amplitude-contracted 3-center overlap."""
    print("\nTest: Amplitude-contracted 3-center overlap")
    print("-" * 50)

    mol = make_test_molecule('h2')
    nao = mol.nao

    # Create surface points
    aux_coords = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    aux_exponents = np.array([0.5, 0.5])
    ngrids = len(aux_coords)
    ncart_aux = 1  # s-type

    # Random amplitudes
    np.random.seed(123)
    amplitudes = np.random.randn(ngrids, ncart_aux)

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU contracted
    print("Computing GPU amplitude-contracted...")
    fock_gpu = int3c_overlap.get_int3c_overlap_amplitude_contracted(
        mol, aux_coords, aux_exponents, aux_l=0, amplitudes=amplitudes, intopt=intopt)
    fock_gpu = fock_gpu.get()  # Convert CuPy to NumPy for comparison
    print(f"GPU fock shape: {fock_gpu.shape}")

    # Compute CPU reference
    print("Computing CPU reference...")
    int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0)
    fock_cpu = np.einsum('gk,gkij->ij', amplitudes, int3c_cpu)
    print(f"CPU fock shape: {fock_cpu.shape}")

    # Compare
    max_diff = np.max(np.abs(fock_gpu - fock_cpu))
    print(f"Max difference: {max_diff:.2e}")
    print(f"GPU fock:\n{fock_gpu}")
    print(f"CPU fock:\n{fock_cpu}")

    if max_diff < 1e-8:
        print("PASS: GPU and CPU amplitude-contracted results match!")
        return True
    else:
        print("FAIL: Results don't match")
        return False


def test_p_type_auxiliary():
    """Test with p-type auxiliary functions (needed for forces)."""
    print("\nTest: p-type auxiliary functions")
    print("-" * 50)

    mol = make_test_molecule('h2')
    nao = mol.nao

    # Create surface points
    aux_coords = np.array([
        [0.0, 0.0, 1.0],
    ])
    aux_exponents = np.array([0.5])
    ngrids = len(aux_coords)

    # Build VHFOpt
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=True)

    # Compute GPU integrals with p-type auxiliary
    print("Computing GPU integrals with p-type aux...")
    try:
        int3c_gpu = int3c_overlap.get_int3c_overlap(
            mol, aux_coords, aux_exponents, aux_l=1, intopt=intopt)
        print(f"GPU result shape: {int3c_gpu.shape}")

        # Compute CPU reference
        print("Computing CPU reference...")
        int3c_cpu = compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=1)
        print(f"CPU result shape: {int3c_cpu.shape}")

        # Compare
        max_diff = np.max(np.abs(int3c_gpu - int3c_cpu))
        print(f"Max difference: {max_diff:.2e}")

        if max_diff < 1e-10:
            print("PASS: p-type auxiliary GPU and CPU results match!")
            return True
        else:
            print("FAIL: Results don't match")
            return False
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing 3-center overlap integral implementation")
    print("=" * 60)

    results = []

    try:
        results.append(('basic', test_basic_integral()))
    except Exception as e:
        print(f"FAIL: basic test raised exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(('basic', False))

    try:
        results.append(('density_contracted', test_density_contracted()))
    except Exception as e:
        print(f"FAIL: density_contracted test raised exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(('density_contracted', False))

    try:
        results.append(('amplitude_contracted', test_amplitude_contracted()))
    except Exception as e:
        print(f"FAIL: amplitude_contracted test raised exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(('amplitude_contracted', False))

    try:
        results.append(('p_type_aux', test_p_type_auxiliary()))
    except Exception as e:
        print(f"FAIL: p_type_aux test raised exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(('p_type_aux', False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")
