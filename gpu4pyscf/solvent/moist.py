# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

"""Optional MOIST DROPSvdW cavity adapter.

MOIST owns cavity construction and nuclear derivatives on the host.  This
module keeps that dependency optional and normalizes its arrays before they
cross the NumPy/CuPy boundary in :mod:`gpu4pyscf.solvent.gostshyp`.
"""

import numpy as np

try:
    from moist import CavityDROPSvdW, Structure
    HAS_MOIST = True
except ImportError:
    CavityDROPSvdW = None
    Structure = None
    HAS_MOIST = False


def _require_moist():
    if not HAS_MOIST:
        raise ImportError(
            'MOIST is required for cavity="drop". Install the optional '
            'dependency with gpu4pyscf[moist].')


def build_drop_cavity(mol, nleb=110, **kwargs):
    """Build a MOIST DROPSvdW cavity for a PySCF molecule."""
    _require_moist()
    numbers = np.ascontiguousarray(mol.atom_charges(), dtype=np.int32)
    positions = np.ascontiguousarray(mol.atom_coords(), dtype=np.float64)
    cavity = CavityDROPSvdW(nleb=nleb, **kwargs)
    cavity.update(Structure(numbers, positions))
    return cavity


def get_surface_data(cavity):
    """Return ``(coordinates, areas, owners)`` in GPU4PySCF layouts.

    Coordinates have shape ``(ngrid, 3)`` and areas and zero-based owners have
    shape ``(ngrid,)``.  The arrays are C-contiguous NumPy arrays.
    """
    coords = np.ascontiguousarray(cavity.xyz.T, dtype=np.float64)
    areas = np.ascontiguousarray(cavity.a, dtype=np.float64)
    owners = np.ascontiguousarray(cavity.owner, dtype=np.int32)
    return coords, areas, owners


def get_anchor_gradient(cavity):
    """Compute DROP area and grid-coordinate nuclear derivatives.

    Returns
    -------
    dareas : ndarray, shape (natm, 3, ngrid)
        ``d(area[g]) / d(R[A, alpha])``.
    dcoords : ndarray, shape (3, 3, natm, ngrid)
        ``d(grid[g, x]) / d(R[A, alpha])``, indexed as ``x, alpha, A, g``.
    """
    cavity.compute_anchor_gradient()
    grad = cavity.get_anchor_gradient()
    dareas = np.ascontiguousarray(
        grad.a_i1_rA.transpose(1, 0, 2), dtype=np.float64)
    dcoords = np.ascontiguousarray(grad.xyz1_rA, dtype=np.float64)
    return dareas, dcoords
