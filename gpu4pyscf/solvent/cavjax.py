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

"""Optional CavJAX boundary for GPU GOSTSHYP.

The external dependency is imported only when :class:`CavJAXBackend` is
constructed. CavJAX uses Bohr, Bohr squared, and outward normals; GOSTSHYP
stores inward normals.
"""

import importlib
from collections.abc import Mapping

import numpy as np

_INSTALL_ERROR = (
    "The CavJAX GOSTSHYP backend requires the optional CavJAX dependency. "
    "Install it with `pip install 'gpu4pyscf[cavjax]'`."
)


def _as_finite_array(value, shape, name):
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'CavJAX returned invalid {name}') from error
    if array.shape != shape:
        raise ValueError(
            f'CavJAX returned {name} with shape {array.shape}; expected {shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'CavJAX returned non-finite {name}')
    return array


class CavJAXBackend:
    """Retained CPU CavJAX object with contiguous NumPy inputs and outputs."""

    def __init__(self, atomic_numbers, options=None):
        if options is None:
            options = {}
        if not isinstance(options, Mapping):
            raise TypeError('cavjax_kwargs must be a mapping')

        try:
            cavjax = importlib.import_module('cavjax')
            jax = importlib.import_module('jax')
            jnp = importlib.import_module('jax.numpy')
        except ImportError as error:
            raise ImportError(_INSTALL_ERROR) from error

        cpu_devices = jax.devices('cpu')
        if not cpu_devices:
            raise RuntimeError('CavJAX requires an available JAX CPU device')

        self._jax = jax
        self._jnp = jnp
        self._cavjax = cavjax
        self._device = cpu_devices[0]
        self._natm = len(atomic_numbers)
        try:
            with jax.default_device(self._device):
                self._model = cavjax.MolecularCavity(
                    np.asarray(atomic_numbers, dtype=np.int64), **dict(options))
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid CavJAX configuration: {error}') from error

    @property
    def n_points(self):
        return self._model.n_points

    @property
    def n_shape_directions(self):
        return self._model.n_shape_directions

    def build(self, positions):
        """Build a surface and return points, areas, and inward normals."""
        positions = _as_finite_array(
            positions, (self._natm, 3), 'positions')
        with self._jax.default_device(self._device):
            surface = self._model.build(positions)

        q = self.n_points
        points = _as_finite_array(surface.points, (q, 3), 'points')
        areas = _as_finite_array(surface.areas, (q,), 'areas')
        outward = _as_finite_array(surface.normals, (q, 3), 'normals')
        if np.any(areas <= 0):
            raise ValueError('CavJAX returned non-positive surface areas')
        norms = np.linalg.norm(outward, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-10):
            raise ValueError('CavJAX returned non-unit surface normals')
        return points, areas, np.ascontiguousarray(-outward)

    def response(self, positions, point_cotangent, area_cotangent,
                 inward_normal_cotangent):
        """Return the nuclear response to GOSTSHYP surface cotangents."""
        positions = _as_finite_array(
            positions, (self._natm, 3), 'positions')
        q = self.n_points
        point_cotangent = _as_finite_array(
            point_cotangent, (q, 3), 'point cotangent')
        area_cotangent = _as_finite_array(
            area_cotangent, (q,), 'area cotangent')
        inward_normal_cotangent = _as_finite_array(
            inward_normal_cotangent, (q, 3), 'normal cotangent')

        # n_inward = -n_outward, hence dE/dn_outward = -dE/dn_inward.
        with self._jax.default_device(self._device):
            cotangents = self._cavjax.SurfaceCotangent(
                points=self._jnp.asarray(point_cotangent),
                areas=self._jnp.asarray(area_cotangent),
                normals=self._jnp.asarray(-inward_normal_cotangent),
            )
            response = self._model.response(positions, cotangents)
        return _as_finite_array(response, (self._natm, 3), 'response')
