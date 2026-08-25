# Stochastic filters (ensemble Kalman filters)

## Summary

**File:** `src/estimators/ensembles.py`. `EnsembleEstimator` is the concrete intermediate base; leaf classes only implement `_analysis_kernel(Af, d, Cdd)`.

**Key attributes:** `m` (ensemble size), `std_phi`, `std_alpha`, `regularization_factor`

| Class | Filter type | Reference |
|---|---|---|
| `EnKF` | Stochastic EnKF (perturbed observations) | Evensen (2003) |
| `EnSRKF` | Deterministic square-root EnKF | Tippett et al. (2003) |
| `rBA_EnKF` | Regularized bias-aware EnKF, weight γ | Nóvoa, Racca & Magri (2023) |

**Covariance inflation** (`src/estimators/inflation.py`, Evensen 2009 Chap. 15): fixed
multiplicative inflation via `inflation_factor`. Applied factors are logged in
`estimator.inflation_history`.

------

::: romda.estimators.EnsembleEstimator

::: romda.estimators.EnKF

::: romda.estimators.EnSRKF

::: romda.estimators.rBA_EnKF
