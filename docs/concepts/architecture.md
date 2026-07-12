# Architecture

`romda` is organised around four cooperating abstractions: **models** forecast the
physics, the **ensemble** manages uncertainty, **filters** assimilate the data, and
**bias estimators** account for what the model cannot represent.

```
             Observations (truth / experimental data)
                              │  y_obs, t_obs
                              ▼
   ┌────────────────────── Ensemble ────────────────────────┐
   │                                                        │
   │   Model (m members)          Bias (estimator)          │
   │   ├─ physical: VdP, Rijke,   ├─ ESN_bias               │
   │   │  Annular, Lorenz, KS     ├─ ConstantBias           │
   │   └─ data-driven: ESN_model, ├─ DriftLinearBias        │
   │      POD_ESN                 └─ NoBias                 │
   │        │  forecast_step()          │                   │
   │        ▼                           ▼                   │
   │   ┌─────────────── analysis_step ────────────────┐     │
   │   │  Filter: EnKF │ EnSRKF │ rBA_EnKF (γ-reg.)   │     │
   │   └───────────────────────────────────────────────┘    │
   └────────────────────────────────────────────────────────┘
```

## Models — `romda.models`

Every forecast model derives from [`Model`][romda.models.model.Model], which provides:

- a **state history** ([`HistoryTracker`][romda.models.history.HistoryTracker]) with
  pre-allocated storage: `hist` has shape `(Nt, N, m)` — time × state × ensemble members;
- a **time-integration strategy** ([`Integrator`][romda.models.integrator.Integrator]):
  `IVPIntegrator` for continuous ODE models (SciPy `solve_ivp`, parallelized over the
  ensemble), `DiscreteIntegrator` for map-based models (ESN, ETDRK4), and
  `ConstantIntegrator` for frozen states;
- the **observation operator** `M` mapping the (augmented) state to the observables.

Physical models implement `time_derivative(t, psi, **params)`; discrete models
implement `time_step(Nt)`. Data-driven models
([`ESN_model`][romda.models.data_driven.esn.ESN_model],
[`POD_ESN`][romda.models.data_driven.pod_esn.POD_ESN]) train themselves at
construction from the provided data.

## Ensemble — `romda.ensemble`

[`Ensemble`][romda.ensemble.Ensemble] wraps a model with `m` perturbed copies of the
state and, optionally, of selected parameters (`est_alpha`, sampled from `std_alpha`).
Its two main methods mirror the two halves of sequential data assimilation:

- **`forecast_step(t_end)`** — advances the model ensemble *and* the bias estimator to
  the next observation time;
- **`analysis_step(d, Cdd)`** — applies the configured filter to the augmented state
  `[φ; α; y]`, enforces parameter bounds (rejection + inflation), stores the analysis,
  and updates the bias estimator with the analysis innovation `d − ⟨y^a⟩`.

Two counters control the start-up transient: `num_DA_blind` (bias-blind EnKF analyses
before activating the bias-aware filter) and `num_SE_only` (state-estimation-only
analyses before parameter estimation starts).

## Filters — `romda.data_assimilation`

Filters are callables constructed with the observation operator. All share the
interface `Aa = filter(Af, d, Cdd, ...)` on the augmented ensemble:

| Filter | Call signature | Bias-aware |
| --- | --- | --- |
| [`EnKF`][romda.data_assimilation.EnKF] | `(Af, d, Cdd)` | no |
| [`EnSRKF`][romda.data_assimilation.EnSRKF] | `(Af, d, Cdd)` | no |
| [`rBA_EnKF`][romda.data_assimilation.rBA_EnKF] | `(Af, d, Cdd, Cbb, b, J)` | yes (γ) |

## Bias estimators — `romda.bias_estimators`

A bias estimator derives from [`Bias`][romda.bias_estimators.bias.Bias] and provides:

- a **forecast** of the bias between analyses (`time_integrate`);
- the **Jacobian** of the bias w.r.t. the observables (`state_derivative`), used by the
  r-EnKF;
- an **update rule** from the analysis innovation (`update_state_from_innovation`),
  optionally Bayesian (an internal EnSRKF on the bias state).

The estimator's ensemble size `N_ens` may differ from the model's `m`; the mean bias is
then used in the analysis (the bias is defined on the ensemble mean).

## Decompositions — `romda.tools`

[`POD`][romda.tools.autoencoders.POD] and [`SPOD`][romda.tools.autoencoders.SPOD] share
a common `Projector` interface (`fit`, `encode`, `decode`, `reconstruct`, `score`) and
handle NaN-masked snapshot data (e.g. solid bodies in a flow). They are pure
dimensionality-reduction tools; combined with the
[`EchoStateNetwork`][romda.tools.esn_core.EchoStateNetwork] reservoir they form the
[`POD_ESN`][romda.models.data_driven.pod_esn.POD_ESN] reduced-order model.

!!! note "Under development"
    The `AE` and `CAE` autoencoder projectors in `romda.tools.autoencoders` are stubs,
    and `scripts/mains/` still targets a legacy API. The core pipeline documented here
    is stable and covered by the test suite.
