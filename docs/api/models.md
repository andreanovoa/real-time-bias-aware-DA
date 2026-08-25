# Models

## Summary

**Package:** [`dynamodels`](https://github.com/andreanovoa/dynamodels) (re-exported as `romda.models`)

Base class for all dynamical systems. Every model stores its time history via `HistoryTracker` and delegates time integration to an `Integrator` instance.

**Key attributes:** `psi0`, `dt`, `alpha` (param dict), `M` (obs matrix), `Nphi`, `Nq`, `Na`

**Key concrete methods:** `time_integrate()`, `init_ensemble()`, `get_observables()`, `get_observable_hist()`, `reset_model()`, `copy()`

Subclasses must implement:
- `obs_labels` — property; labels for observable dimensions
- `time_derivative(t, psi, **params)` — static method (continuous models), **or**
- `time_step(Nt)` — discrete-map stepping (discrete models)

```
Model
├── Lorenz63
├── Lorenz96
├── VdP
├── Annular
├── KS
├── Rijke
├── ESN_model (+ EchoStateNetwork)
│   └── POD_ESN (+ POD)
└── LinearModel
```

### Model API builders

#### HistoryTracker — `dynamodels.history`

Mixin class used by both `Model` and `Bias`. Provides a pre-allocated buffer that grows on demand, avoiding repeated `np.concatenate` calls.

| Property / Method | Description |
|---|---|
| `hist`, `hist_t` | Valid portion of the state / time buffer |
| `current_state`, `current_time` | Latest entry in the buffer |
| `update_history(state, t)` | Append, reset, or overwrite last states |

#### Integrator — `dynamodels.integrator`

Strategy pattern: `Model` holds one `Integrator` instance and calls `integrator.advance()`. The integrator dispatches to `advance_single` or `advance_ensemble` depending on the ensemble size.

| Class | Use case | Mechanism |
|---|---|---|
| `IVPIntegrator` | Continuous ODEs | `scipy.integrate.solve_ivp`; multiprocessing pool for ensembles |
| `DiscreteIntegrator` | Discrete maps (ESN, KS, linear) | Calls `model.time_step()`; interpolates if `dt_output ≠ dt_step` |
| `ConstantIntegrator` | Constant-bias placeholder | Returns `ψ(t) = ψ(0)` |

```
Integrator
├── IVPIntegrator
├── DiscreteIntegrator
└── ConstantIntegrator
```

### Available Models

The concrete subclasses, with figures, live on their own pages:
[Physical models](models_physical.md) (`VdP`, `Lorenz63`, `Lorenz96`, `KS`, `Rijke`, `Annular`)
and [Data-driven models](models_data_driven.md) (`ESN_model`, `POD_ESN`, `LinearModel`).

------

::: dynamodels.model.Model

::: dynamodels.history.HistoryTracker

::: dynamodels.integrator.Integrator

::: dynamodels.integrator.IVPIntegrator

::: dynamodels.integrator.DiscreteIntegrator

::: dynamodels.integrator.ConstantIntegrator
