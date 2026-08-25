# Data-driven models

## Summary

**File:** `src/models/data_driven/`. All use `DiscreteIntegrator`. `ESN_model` mixes in
[`EchoStateNetwork`](tools.md); `POD_ESN` mixes in `ESN_model` and [`POD`](tools.md). The
`Projector` hierarchy itself (`POD`/`SPOD`) lives in this package's `autoencoders/` subpackage.

| Class | Key parameters | Notes |
|---|---|---|
| `ESN_model` | `N_units`, `rho`, `sigma_in`, `N_wash` | Inherits `EchoStateNetwork`; requires training `data` at init |
| `POD_ESN` | `N_modes`, `sensor_locations` | Inherits `ESN_model` + `POD`; sensor placement via QR |
| `LinearModel` | `F` (transition matrix), `Q_noise` | $\boldsymbol{\psi}_{t+1} = \mathbf{F}\boldsymbol{\psi}_t + \boldsymbol{\eta}_t$ |

------

::: romda.models.data_driven.esn.ESN_model

::: romda.models.data_driven.pod_esn.POD_ESN

<figure markdown>
  ![POD-ESN data assimilation pipeline](../figs/DA/POD-ESN_DA.png){ width="700" }
  <figcaption>State estimation via data assimilation on a POD-ESN reduced-order model.</figcaption>
</figure>

<figure markdown>
  ![POD-ESN state and parameter estimation](../figs/DA/POD-ESN-SPE.png){ width="700" }
  <figcaption>Joint state and parameter estimation on the POD-ESN.</figcaption>
</figure>

::: romda.models.data_driven.linear_model.LinearModel
