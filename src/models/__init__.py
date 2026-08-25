"""romda.models — the model layer.

The modelling core (``Model``, ``HistoryTracker``, the integrators and the
physical models) lives in the standalone ``dynamodels`` package and is
re-exported here; this subpackage keeps the data-driven models (ESN, POD-ESN,
LinearModel).

The ``sys.modules`` aliases below keep previously saved pickles loadable:
stock pickle stores fully-qualified class paths such as
``romda.models.physical.rijke.Rijke``, which now resolve to the dynamodels
modules. (Pickles written *after* the split store ``dynamodels.*`` paths and
need dynamodels installed to load — which romda guarantees by dependency.)
"""
import sys

import dynamodels
from dynamodels import (
    ConstantIntegrator,
    DiscreteIntegrator,
    HistoryTracker,
    Integrator,
    IVPIntegrator,
    Model,
    physical,
)

# bind as package attributes too, so `import romda.models.model` followed by
# attribute access (romda.models.model.Model) works, not just `from ... import`
model = dynamodels.model
history = dynamodels.history
integrator = dynamodels.integrator

sys.modules[f'{__name__}.model'] = dynamodels.model
sys.modules[f'{__name__}.history'] = dynamodels.history
sys.modules[f'{__name__}.integrator'] = dynamodels.integrator
sys.modules[f'{__name__}.physical'] = dynamodels.physical
for _mod in ('annular', 'kuramoto_sivashinsky', 'lorenz63', 'lorenz96', 'rijke', 'van_der_pol'):
    sys.modules[f'{__name__}.physical.{_mod}'] = getattr(dynamodels.physical, _mod)

from . import data_driven  # noqa: E402  (needs the aliases in place)

__all__ = [
    "Model",
    "HistoryTracker",
    "Integrator",
    "IVPIntegrator",
    "DiscreteIntegrator",
    "ConstantIntegrator",
    "physical",
    "data_driven",
]
