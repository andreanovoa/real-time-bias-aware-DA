# Data-driven models

## Summary

**File:** `src/models/data_driven/`. All use `DiscreteIntegrator`. `ESN_model` mixes in
`EchoStateNetwork` — the reservoir core from the external
[`echostatenetwork`](https://andreanovoa.github.io/EchoStateNetwork/) package (re-exported by
`romda.models.data_driven` for convenience; see its own documentation for the reservoir API) —
and `POD_ESN` mixes in `ESN_model` and [`POD`](models_data_driven_projectors.md).

| Class | Key parameters | Notes |
|---|---|---|
| `ESN_model` | `N_units`, `rho`, `sigma_in`, `N_wash` | Inherits `EchoStateNetwork`; requires training `data` at init |
| `POD_ESN` | `N_modes`, `sensor_locations` | Inherits `ESN_model` + `POD`; sensor placement via QR |
| `LinearModel` | `F` (transition matrix), `Q_noise` | $\boldsymbol{\psi}_{t+1} = \mathbf{F}\boldsymbol{\psi}_t + \boldsymbol{\eta}_t$ |

## Model pages

- [Echo State Network model](models_data_driven_esn.md)
- [POD-ESN model](models_data_driven_pod_esn.md)
- [Linear model](models_data_driven_linear.md)
- [Projectors and modal decompositions](models_data_driven_projectors.md)

The `Projector` hierarchy (`POD`/`SPOD`) and the standalone decomposition functions
(`pod_utils.py`) live in the `autoencoders/` subpackage. Import everything from
`romda.models.data_driven` (or its `autoencoders` subpackage).

## Model pages

- [Echo State Network model](models_data_driven_esn.md)
- [POD-ESN model](models_data_driven_pod_esn.md)
- [Linear model](models_data_driven_linear.md)
- [Projectors and modal decompositions](models_data_driven_projectors.md)
