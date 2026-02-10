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
Molecular system generation utilities for benchmarks.

This module provides functions for creating test molecular systems
such as hydrogen grids and water boxes.
"""

import numpy as np
from pyscf import gto


def create_hydrogen_grid(Nx, Ny, Nz, dist=0.5, basis="cc-pvqz", unit="Bohr", cart=True):
    """
    Create a 3D grid of hydrogen atoms.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of hydrogen atoms in each direction
    dist : float, optional
        Distance between adjacent atoms (default: 0.5 Bohr)
    basis : str, optional
        Basis set name (default: 'cc-pvqz')
    unit : str, optional
        Coordinate unit (default: 'Bohr')
    cart : bool, optional
        Use Cartesian basis functions (default: True)

    Returns
    -------
    mol : pyscf.gto.Mole
        PySCF molecule object
    """
    N = Nx * Ny * Nz
    atom = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                atom.append(["H", (i * dist, j * dist, k * dist)])

    return gto.M(atom=atom, basis=basis, unit=unit, cart=cart, spin=N % 2)


def create_water_box(Nx, Ny, Nz, spacing=5.0, basis="def2-tzvpp", unit="Bohr", cart=True, seed=42):
    """
    Create a 3D grid of water molecules with random rotations.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of water molecules in each direction
    spacing : float, optional
        Distance between molecule centers (default: 5.0 Bohr)
    basis : str, optional
        Basis set name (default: 'def2-tzvpp')
    unit : str, optional
        Coordinate unit (default: 'Bohr')
    cart : bool, optional
        Use Cartesian basis functions (default: True)
    seed : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    mol : pyscf.gto.Mole
        PySCF molecule object
    """
    np.random.seed(seed)
    atom = []

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                center = np.array([i * spacing, j * spacing, k * spacing])

                # Random rotation
                theta_x = np.random.uniform(0, 2 * np.pi)
                theta_y = np.random.uniform(0, 2 * np.pi)
                theta_z = np.random.uniform(0, 2 * np.pi)

                Rx = np.array([[1, 0, 0],
                               [0, np.cos(theta_x), -np.sin(theta_x)],
                               [0, np.sin(theta_x), np.cos(theta_x)]])
                Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                               [0, 1, 0],
                               [-np.sin(theta_y), 0, np.cos(theta_y)]])
                Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                               [np.sin(theta_z), np.cos(theta_z), 0],
                               [0, 0, 1]])
                R = Rz @ Ry @ Rx

                # Water geometry (Bohr)
                O_local = np.array([0.0, 0.0, 0.0])
                H1_local = np.array([0.0, 1.43, 1.11])
                H2_local = np.array([0.0, -1.43, 1.11])

                atom.append(["O", tuple(center + R @ O_local)])
                atom.append(["H", tuple(center + R @ H1_local)])
                atom.append(["H", tuple(center + R @ H2_local)])

    return gto.M(atom=atom, basis=basis, unit=unit, cart=cart, spin=0)
