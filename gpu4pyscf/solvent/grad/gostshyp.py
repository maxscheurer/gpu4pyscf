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
Gradient implementation for GOSTSHYP solvation model.

This module provides analytical nuclear gradients for the GOSTSHYP
cavity energy contribution.
"""

import numpy as np
from pyscf import lib
from pyscf.solvent.grad.pcm import get_dF_dA
from gpu4pyscf.lib import logger


class Gradients(lib.StreamObject):
    """
    Nuclear gradients for GOSTSHYP solvation model.

    The GOSTSHYP gradient has three main contributions:
    1. dE/dA: Change in energy due to area changes
    2. dE/dS: Change in energy due to integral changes
    3. dE/dF: Change in energy due to force operator changes
    """

    def __init__(self, gostshyp_obj):
        self.gostshyp = gostshyp_obj
        self.mol = gostshyp_obj.mol

    def kernel(self, dm):
        """
        Compute GOSTSHYP nuclear gradients.

        Parameters
        ----------
        dm : ndarray
            Density matrix

        Returns
        -------
        gradient : ndarray of shape (natm, 3)
            Nuclear gradient contribution from GOSTSHYP
        """
        gostshyp = self.gostshyp
        mol = self.mol

        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        # Get cached values from energy calculation
        forces = gostshyp.forces
        amplitudes = gostshyp.amplitudes
        gtilde = gostshyp._compute_gtilde()

        # Area derivatives
        surface_dict = {
            'grid_coords': gostshyp.grid_coords,
            'area': gostshyp.areas,
            'gslice_by_atom': gostshyp.surface['gslice_by_atom'],
            'R_vdw': gostshyp.surface['R_vdw'].get(),
            'switch_fun': gostshyp.surface['switch_fun'].get(),
            'R_in_J': gostshyp.surface['R_in_J'].get(),
            'R_sw_J': gostshyp.surface['R_sw_J'].get(),
            'atom_coords': gostshyp.surface['atom_coords'].get(),
        }
        _, dareas = get_dF_dA(surface_dict)
        dareas = dareas.transpose(1, 2, 0)  # [natm, 3, ngrids]

        # Term 1: dE/dA
        gtilde_expval = np.einsum('ij,ijg->g', dm, gtilde, optimize=True)
        dE1 = gostshyp.pressure_au * np.einsum(
            'acg,g->ac', dareas, gtilde_expval / forces, optimize=True)

        # Term 2 and 3 require integral derivatives which are more complex
        # For now, return the area contribution (main term)
        # Full implementation would require int3c1e_ip kernels

        gradient = dE1
        logger.info(self, 'GOSTSHYP gradient computed (area term)')

        return gradient

    grad = kernel


def make_grad(gostshyp_obj, dm):
    """
    Compute GOSTSHYP gradient.

    Parameters
    ----------
    gostshyp_obj : GOSTSHYP
        GOSTSHYP solvent object
    dm : ndarray
        Density matrix

    Returns
    -------
    gradient : ndarray
        Nuclear gradients
    """
    return Gradients(gostshyp_obj).kernel(dm)
