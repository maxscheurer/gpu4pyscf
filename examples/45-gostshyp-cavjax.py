#!/usr/bin/env python
"""GPU GOSTSHYP with the optional CavJAX global cavity.

Install the optional backend with::

    pip install 'gpu4pyscf[cavjax]'

CavJAX runs on a JAX CPU device and requires float64 support. Set
``JAX_ENABLE_X64=1`` before starting Python. Surface positions are passed in
Bohr, areas are Bohr**2, and CavJAX's outward normals are converted to the
inward convention used by GOSTSHYP.
"""

from pyscf import gto

from gpu4pyscf import scf
from gpu4pyscf.solvent import _attach_solvent
from gpu4pyscf.solvent.gostshyp import GOSTSHYP

mol = gto.M(
    atom='H 0 0 0; H 0 0 1.4; H 0 0 5; H 0 0 6.4',
    unit='Bohr',
    basis='def2-svp',
)

pressure = GOSTSHYP(mol, options={
    'cavity': 'cavjax',
    'pressure_mpa': 50_000,
    'cavjax_kwargs': {
        # This is a global natm * points_per_atom budget, not an atom-centered
        # Lebedev grid. CavJAX resolves a fixed topology at construction.
        'points_per_atom': 100,
        'radius_scale': 1.0,
        'radius_offset': 0.0,
        'target_shape_directions': 162,
        'support_temperature': 0.2834589187,
        'wall_temperature': 0.1511780900,
        'root_steps': 48,
    },
})
mf = _attach_solvent._for_scf(scf.RHF(mol), pressure)
mf.kernel()
gradient = mf.Gradients().kernel()

print('GOSTSHYP gradient (Hartree/Bohr):')
print(gradient)
print('CavJAX build:', pressure._t_cavjax_build, 's')
print('Gradient integrals:', pressure._grad_t_integrals, 's')
print('Cotangent assembly:', pressure._grad_t_cotangent, 's')
print('CavJAX VJP:', pressure._grad_t_cavjax_vjp, 's')

# Geometry scanners call reset() and reuse the same fixed-topology CavJAX
# object as long as atomic numbers and their order are unchanged. Construct a
# new GOSTSHYP object if the composition/order changes. CavJAX is integral-
# direct only; full-grid cached GOSTSHYP operators are intentionally rejected.
