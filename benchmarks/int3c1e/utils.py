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
Utility functions for int3c1e benchmarks.

This module provides timing utilities and CPU reference implementations
for validating GPU 3-center overlap integral results.
"""

import time
import numpy as np
import cupy as cp
from pyscf import gto


def fakemol_for_gaussian(coords, exponents, l=0, cart=True, coeffs=None):
    """
    Create a fake molecule for auxiliary Gaussians (from CPU GOSTSHYP reference).

    Parameters
    ----------
    coords : ndarray of shape (ngrids, 3)
        Coordinates of Gaussian centers
    exponents : ndarray of shape (ngrids,)
        Gaussian exponents
    l : int, optional
        Angular momentum (default: 0)
    cart : bool, optional
        Use Cartesian basis functions (default: True)
    coeffs : ndarray, optional
        Contraction coefficients (default: ones)

    Returns
    -------
    fakemol : pyscf.gto.Mole
        Fake molecule object representing auxiliary Gaussians
    """
    nbas = coords.shape[0]
    if coeffs is None:
        coeffs = np.ones_like(exponents)
    angmom = np.zeros_like(exponents)
    angmom[:] = l

    ang_norm = {
        0: 2.0 * np.sqrt(np.pi),
        1: 2.0 * np.sqrt(np.pi / 3),
        2: 1.0,
        3: 1.0,
    }

    fakeatm = np.zeros((nbas, gto.mole.ATM_SLOTS), dtype=np.int32)
    fakebas = np.zeros((nbas, gto.mole.BAS_SLOTS), dtype=np.int32)
    fakeenv = [0] * gto.mole.PTR_ENV_START
    ptr = gto.mole.PTR_ENV_START
    fakeatm[:, gto.mole.PTR_COORD] = np.arange(ptr, ptr + nbas * 3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas * 3
    fakebas[:, gto.mole.ATOM_OF] = np.arange(nbas)
    fakebas[:, gto.mole.ANG_OF] = angmom
    fakebas[:, gto.mole.NPRIM_OF] = 1
    fakebas[:, gto.mole.NCTR_OF] = 1
    fakebas[:, gto.mole.PTR_EXP] = ptr + np.arange(nbas) * 2
    fakebas[:, gto.mole.PTR_COEFF] = ptr + np.arange(nbas) * 2 + 1
    # angular normalization
    coeff = ang_norm[l] * coeffs
    fakeenv.append(np.vstack((exponents, coeff)).T.ravel())

    fakemol = gto.Mole()
    fakemol.cart = cart
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = np.hstack(fakeenv)
    fakemol._built = True
    return fakemol


def compute_int3c_overlap_cpu(mol, aux_coords, aux_exponents, aux_l=0, aux_cart=True):
    """
    Compute 3-center overlap integrals using PySCF's int3c1e with supermol approach.

    This matches the CPU GOSTSHYP reference implementation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular object
    aux_coords : ndarray of shape (ngrids, 3)
        Coordinates of auxiliary Gaussian centers
    aux_exponents : ndarray of shape (ngrids,)
        Exponents of auxiliary Gaussians
    aux_l : int, optional
        Angular momentum of auxiliary functions (default: 0)
    aux_cart : bool, optional
        If True (default), auxiliary functions are Cartesian.
        If False, auxiliary functions are spherical harmonics.
        Must match mol.cart setting.

    Returns
    -------
    int3c : ndarray of shape (ngrids, naux, nao, nao)
        3-center overlap integrals.
        naux = (aux_l+1)*(aux_l+2)//2 if aux_cart else 2*aux_l+1
    """
    nao = mol.nao
    ngrids = len(aux_coords)
    ncart_aux = (aux_l + 1) * (aux_l + 2) // 2
    nsph_aux = 2 * aux_l + 1
    naux = ncart_aux if aux_cart else nsph_aux

    # aux_cart must match mol.cart for PySCF's int3c1e
    assert aux_cart == mol.cart, "aux_cart must match mol.cart"

    # Build fake molecule for auxiliary Gaussians
    gmol = fakemol_for_gaussian(aux_coords, aux_exponents, l=aux_l, cart=aux_cart)

    # Create supermolecule and compute integrals
    supermol = mol + gmol
    slices = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + gmol.nbas)
    int3c = supermol.intor("int3c1e", shls_slice=slices, aosym="s1")
    # Shape: (nao, nao, ngrids * naux)

    # Reshape to (ngrids, naux, nao, nao)
    int3c = int3c.reshape(nao, nao, ngrids, naux).transpose(2, 3, 0, 1)

    return int3c


def gpu_timer_events(fn, n_warmup=3, n_runs=10):
    """
    Time a GPU function using CUDA events for accurate GPU timing.

    This measures pure GPU kernel time, excluding host-device memory transfer.
    Results are converted to numpy for comparison after timing.

    Parameters
    ----------
    fn : callable
        Function to time (should return the result, CuPy or NumPy array)
    n_warmup : int, optional
        Number of warmup runs (default: 3)
    n_runs : int, optional
        Number of timed runs (default: 10)

    Returns
    -------
    result : numpy.ndarray
        Result from the function (converted to numpy if needed)
    mean_time : float
        Mean execution time in seconds
    std_time : float
        Standard deviation of execution time in seconds
    """
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    times = []

    for i in range(n_warmup + n_runs):
        start_event.record()
        result = fn()
        # Synchronize to ensure kernel completes before recording end time
        cp.cuda.Device().synchronize()
        end_event.record()
        end_event.synchronize()
        if i >= n_warmup:
            # get_elapsed_time returns milliseconds
            times.append(cp.cuda.get_elapsed_time(start_event, end_event) / 1000)

    # Convert to numpy for comparison (not timed)
    if hasattr(result, 'get'):
        result = result.get()

    return result, np.mean(times), np.std(times)


def cpu_timer(fn, n_warmup=2, n_runs=5):
    """
    Time a CPU function using perf_counter.

    Parameters
    ----------
    fn : callable
        Function to time (should return the result)
    n_warmup : int, optional
        Number of warmup runs (default: 2)
    n_runs : int, optional
        Number of timed runs (default: 5)

    Returns
    -------
    result : array
        Result from the function
    mean_time : float
        Mean execution time in seconds
    std_time : float
        Standard deviation of execution time in seconds
    """
    times = []

    for i in range(n_warmup + n_runs):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        if i >= n_warmup:
            times.append(end - start)

    return result, np.mean(times), np.std(times)


def generate_test_grids(ngrids=50, seed=42, extent=5.0):
    """
    Generate random auxiliary coordinates and exponents for testing.

    Parameters
    ----------
    ngrids : int, optional
        Number of grid points (default: 50)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    extent : float, optional
        Spatial extent of random coordinates (default: 5.0)

    Returns
    -------
    aux_coords : ndarray of shape (ngrids, 3)
        Random coordinates
    aux_exponents : ndarray of shape (ngrids,)
        Random exponents (uniform in [0.5, 2.0])
    """
    np.random.seed(seed)
    aux_coords = np.random.uniform(-extent, extent, (ngrids, 3))
    aux_exponents = np.random.uniform(0.5, 2.0, ngrids)
    return aux_coords, aux_exponents
