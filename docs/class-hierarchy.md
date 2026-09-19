# Class hierarchy

How the four core class families — models, estimators, bias estimators and observations —
relate, before diving into the [API reference](api/index.md).

## Models

#### Physical models — [`dynamodels.physical`](https://andreanovoa.github.io/dynamodels/)

All use `IVPIntegrator` (scipy `solve_ivp`) except KS.

| Class | Dim | Key parameters | Integrator |
|---|---|---|---|
| `Lorenz63` | 3 | `rho`, `sigma`, `beta` | IVP |
| `Lorenz96` | Nx | `F`, `Nx` | IVP |
| `VdP` | 2 | `beta`, `zeta`, `kappa`, `law`, `omega` | IVP |
| `Annular` | 4 | `omega`, `nu`, `c2beta`, `kappa`, `epsilon` | IVP |
| `KS` | Nx | `nu`, `L`, `Nx` | Discrete (ETDRK4) |
| `Rijke` | 2Nm+Nc | `beta`, `tau`, `C1`, `C2`, `kappa` | IVP |

#### Data-driven models — `src/models/data_driven/`

All use `DiscreteIntegrator`. The ML base classes `EchoStateNetwork` (from the external `echostatenetwork` package) and the `Projector` hierarchy (`POD`/`SPOD`, in `src/models/data_driven/autoencoders/` — see [Data-driven models](api/models_data_driven.md)) are mixed in via multiple inheritance.

| Class | Key parameters | Notes |
|---|---|---|
| `ESN_model` | `N_units`, `rho`, `sigma_in`, `N_wash` | Inherits `EchoStateNetwork`; requires training `data` at init |
| `POD_ESN` | `N_modes`, `sensor_locations` | Inherits `ESN_model` + `POD`; sensor placement via QR |
| `LinearModel` | `F` (transition matrix), `Q_noise` | `ψ_{t+1} = F @ ψ_t + noise` |

---

## Estimator

**File:** `src/estimators/__init__.py`

Abstract base class (ABC) that wraps a `Model` and an optional `Bias` into a full DA loop. Provides the shared `forecast_step()` that advances both the model and the bias in time.

**Key attributes:** `est_phi`, `est_alpha`, `est_bias`, `inflation_factor`, `num_DA_blind`, `num_SE_only`

**Key concrete methods:** `forecast_step()`, `_init_bias()`, `_MA()` (applies measurement operator `M`)

Subclasses must implement:
- `analysis_step(d, Cdd)` — Bayesian update given observation vector `d` and noise covariance `Cdd`

```
Estimator  (ABC)
├── EnsembleEstimator
│   ├── EnKF
│   ├── EnSRKF
│   └── rBA_EnKF
└── DeterministicEstimator
    └── KalmanFilter
```

The relationship between the three main classes — attributes, not subclasses:

```
Estimator instance
  .model  →  Model instance   (used for the forecast)
  .bias   →  Bias instance    (optional; None if bias-unaware)
             .forecaster  →  Model instance   (usually an ESN_model)
```

---

### Available estimators

#### Ensemble estimators — `src/estimators/ensembles.py`

`EnsembleEstimator` is the concrete intermediate base; leaf classes only implement `_analysis_kernel(Af, d, Cdd)`.

**Key attributes:** `m` (ensemble size), `std_phi`, `std_alpha`, `regularization_factor`

| Class | Filter type | Reference |
|---|---|---|
| `EnKF` | Stochastic EnKF (perturbed observations) | Evensen (2003) |
| `EnSRKF` | Deterministic square-root EnKF | Tippett et al. (2003) |
| `rBA_EnKF` | Regularised bias-aware EnKF, weight `γ` | Nóvoa & Magri (2022) |

#### Deterministic estimators — `src/estimators/deterministic.py`

`DeterministicEstimator` owns the state mean `ψ` and covariance `Cpp` directly (no ensemble).

| Class | Notes |
|---|---|
| `KalmanFilter` | Standard linear KF; propagates covariance with Jacobian `F_jac` |


---

## Bias

**File:** `src/bias_estimators/bias.py`

Base class for observation-bias estimators. Mixes in `HistoryTracker`. Wraps a `forecaster` model to produce bias corrections at each assimilation step.

**Key attributes:** `innovation`, `dt`, `forecaster`, `Nq`, `N_dim`, `upsample`, `biased_observations`

Subclasses must implement:
- `init_forecaster(**kwargs)` — build/train the internal forecasting model
- `state_derivative()` — return the Jacobian `J = db/dy` used by bias-aware filters

```
Bias
├── ESN_bias
└── ConstantBias
    └── NoBias
```

| Class | File | Notes |
|---|---|---|
| `ESN_bias` | `src/bias_estimators/esn.py` | Correlation-based training; `biased_observations = True`; the workhorse |
| `ConstantBias` | `src/bias_estimators/constantbias.py` | Fixed bias estimate; `NoBias` subclass for bias-blind DA |

---

### Forecaster

The `forecaster` attribute of a `Bias` instance is a `Model` subclass — typically a data-driven model trained on the residual between observations and model output.

```
Bias instance
  .forecaster  →  Model instance   (usually ESN_model)
```

The forecaster is initialised inside `init_forecaster()` and stepped forward in sync with the main model during `Estimator.forecast_step()`. Its output is the predicted bias `b(t)`, which enters the analysis step as a correction to the observation.

---

## Observations

**File:** `src/observations.py`

Standalone class (no inheritance). Generates or loads ground-truth data, applies a configurable manual bias, adds noise, and exposes observation time indices for the DA loop.

**Key attributes:**

| Attribute | Description |
|---|---|
| `y_true` | Clean biased truth signal `(Nt, Nq, 1)` |
| `y_raw` | Noisy observed signal `(Nt, Nq, 1)` |
| `b_true` | Applied bias `(Nt, Nq, 1)` |
| `t_true` | Full time vector |
| `y_obs`, `t_obs` | Observations at assimilation times |
| `obs_idx` | Indices into `t_true` at which observations are taken |
| `Nt_obs` | Subsampling rate (every `Nt_obs` steps) |

**Noise options** (`noise_type`): `'gauss, add'`, `'gauss, mult'`, coloured noise variants.

**Manual bias options** (`manual_bias`): `'linear'`, `'periodic'`, `'time'`, `'cosine'`, or any callable `f(y_true, t_true) -> (b, name)`.

**Key method:** `plot_truth(case)` — five-panel figure (raw, truth, PDF, PSD, difference).
