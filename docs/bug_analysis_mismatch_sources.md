# Analysis: Sources of Mismatch Between `main` and `bias_ens` Tutorials

**Date:** 2026-03-24  
**Tutorials compared:**
- Working (main): `scripts/tutorials/13_bias-aware-DA-intro.ipynb`
- Broken (bias_ens): `scripts/tutorials/1_Introtuction_to_real-time_DA/13_bias-aware-DA-intro.ipynb`

---

## Summary

The two tutorials implement the same bias-aware data assimilation algorithm (rBA-EnKF) but with fundamentally different code architectures. The `main` branch uses a procedural approach (`create_truth`, `create_ensemble`, `dataAssimilation`), while the `bias_ens` branch uses an object-oriented approach (`Observations`, `Ensemble`, `ESN_bias` classes). Several bugs and inconsistencies in the `bias_ens` branch cause the assimilation results to diverge from the working `main` branch.

---

## Critical Bugs (HIGH impact — likely cause of divergent results)

### 1. **Negative regularization factor `gamma = -1`** (tutorial Cell 22)

**File:** `scripts/tutorials/1_Introtuction_to_real-time_DA/13_bias-aware-DA-intro.ipynb`, Cell 22  
**Code:** `filter_ens.filter.gamma = -1.`

The regularization factor `gamma` (called `k` in the main branch) controls the bias penalization term in the rBA-EnKF equations:

```
Cinv = (m-1)*Cdd + (I+J^T)(I+J)*Cqq + gamma * CdWb * J^T*J * Cqq
Aa = Af + K * [(I+J^T)(D-Y) - gamma * CdWb * J^T * b]
```

Setting `gamma = -1` **inverts the regularization**, causing the filter to amplify bias rather than penalize it. The main branch uses `regularization_factor = 1.0` (positive). This is the single most impactful bug.

**Fix:** Change `filter_ens.filter.gamma = -1.` to `filter_ens.filter.gamma = 1.` in the tutorial.

---

### 2. **Regularization term uses `b` instead of `B`** (`rBA_EnKF.__call__`)

**File:** `src/data_assimilation.py`, line ~232  
**Code:** `Aa = Af + np.dot(K, np.dot(Iq + J.T, D - Y) - self.gamma * np.dot(CdWb, np.dot(J.T, b)))`

In the main branch, the last term is `J.T @ B` where `B` has shape `(Nq, m)` — the bias replicated across all ensemble members. In the `bias_ens` branch, it uses `b` which has shape `(Nq, N_ens)`. When `N_ens != m` (the tutorial sets `N_ens=5` but `m=10`), this creates a **shape mismatch** in the matrix multiplication. Even when it doesn't error out (due to numpy broadcasting), the result is mathematically incorrect because the regularization term has wrong dimensions.

**Fix:** Ensure `b` is expanded to shape `(Nq, m)` before the final Aa computation, consistent with how `B` is constructed for the `D - Y` term.

---

### 3. **Observation covariance `Cdd` computed differently**

**Main branch (in `dataAssimilation()`):**
```python
Cdd = np.diag(std_obs * np.ones(Nq)) * np.max(abs(y_obs), axis=0) ** 2
# With std_obs = 0.1
```
This scales the observation covariance by the maximum absolute observation value squared — a standard approach to make the covariance relative to the signal magnitude.

**bias_ens tutorial (Cell 20-22):**
```python
std_d = 0.02 * np.std(y_ens, axis=2)  
Cdd = np.eye(Nq) * (std_d ** 2)
```
This uses 2% of the ensemble standard deviation — a much smaller value that doesn't account for signal magnitude.

**Impact:** A significantly different `Cdd` fundamentally changes the Kalman gain and the balance between observations and forecast, leading to different assimilation behavior.

**Fix:** Use a `Cdd` construction consistent with the main branch, e.g.:
```python
Cdd = np.diag(std_obs * np.ones(Nq)) * np.max(abs(truth.y_obs), axis=0) ** 2
```

---

## Moderate Bugs (MEDIUM impact — contribute to differences)

### 4. **No blind phase before bias-aware activation**

**Main branch:** The assimilation starts with blind EnKF for `num_DA_blind` steps before activating the bias-aware filter:
```python
ensemble.activate_bias_aware = ti >= ensemble.num_DA_blind
if case.activate_bias_aware:
    Aa = rBA_EnKF(...)
else:
    Aa = EnKF(...)  # Fall back to standard EnKF
```

**bias_ens branch:** The filter is always bias-aware when `gamma is not None`:
```python
if self.filter.is_bias_aware:  # True whenever gamma is set
    # Always runs rBA_EnKF
```

**Impact:** Using bias-aware filtering from the very first step (before the bias model has had time to converge) can destabilize the assimilation.

**Fix:** Add logic to the `analysis_step()` method to optionally use a standard EnKF for the first `num_DA_blind` steps, or set `gamma = 0` for the initial steps (which effectively reduces rBA-EnKF to standard EnKF).

---

### 5. **Bias forecast does not receive model observables**

**Main branch (`forecastStep`):**
```python
y = case.get_observable_hist(Nt)
b, t_b = case.bias.time_integrate(t=t, y=y)
```
The ESN bias model receives model observables `y` as input for its forecast. This is critical for the washout/initialization phase.

**bias_ens branch (`forecast_step`):**
```python
b, t_b = pb.time_integrate(**kwargs_local)
# kwargs_local only has Nt, no observables passed
```
And `time_integrate` calls `self.integrator.advance(Nt=Nt)` which runs the ESN in closed-loop without fresh model observables.

**Impact:** The ESN bias model may use stale or incorrect input during forecast, degrading its prediction accuracy over time.

**Fix:** Pass model observables to the bias `time_integrate` call so the ESN can use them if needed.

---

### 6. **ESN bias state includes reservoir — dimension mismatch in regularization**

In the `bias_ens` branch, `ESN_bias.new_innovation_to_state()` (in `src/bias_estimators/esn.py`) appends the reservoir state to the bias state:
```python
state[-esn.N_units:, :] = r_open
```

This makes the bias state dimension `(2*Nq + N_units, N_ens)`, but the `current_bias` property only returns `state[bias_idx, :]` which is `(Nq, N_ens)`. While this is correct for extracting `b`, the reservoir update via a single open-loop step (using only the mean reservoir state `r_mean`) may be inconsistent with how the main branch handles reservoir updates.

**Impact:** Reservoir desynchronization between bias states and the forecaster can accumulate over DA cycles.

---

## Minor Differences (LOW impact — expected tutorial variations)

### 7. **Different initial conditions and parameter ranges**

| Parameter | Main | bias_ens |
|-----------|------|----------|
| `psi0` | `rng.random(2)+5` (~[5.6, 5.2]) | `np.array([2., 1.])` |
| `std_psi` / `std_phi` | 0.3 | 0.1 |
| `alpha zeta` | (40, 80) | (40, 50) |
| `alpha beta` | (50, 80) | (50, 60) |
| `alpha kappa` | (3, 5) | (3, 4) |
| `t_start` | 1.5 | 0.6 |
| `t_stop` | 1.8 | 1.0 |

While these are expected tutorial customizations, the narrower parameter ranges in `bias_ens` could mask parameter estimation issues.

### 8. **Different ESN hyperparameters**

| Parameter | Main | bias_ens |
|-----------|------|----------|
| `upsample` | 3 | 5 |
| `N_wash` | 10 | 5 |
| `t_val` | `t_CR * 1` | `t_CR * 0.5` |
| `N_ens` | 1 (default) | 5 |
| `correlation_based_training` | False | True |

These affect the ESN bias model's accuracy but are tutorial-specific choices.

### 9. **Different RNG usage paths**

Both tutorials use `rng = np.random.default_rng(0)` in their notebook cells and `rng = np.random.default_rng(6)` in `data_assimilation.py`. However, because the code paths differ (different initialization, different number of calls to the RNG), the random number sequences will diverge even with the same seeds.

---

## Code Architecture Issues (not bugs, but design concerns)

### 10. **`rBA_EnKF.__call__` signature mismatch with main**

The main branch's `rBA_EnKF(Af, d, Cdd, Cbb, k, M, b, J)` takes `k` (regularization) and `M` (obs operator) as arguments. The `bias_ens` branch's `rBA_EnKF.__call__(self, Af, d, Cdd, Cbb, b, bd, J)` stores `gamma` and `M` internally. The `bd` (observation bias) parameter is new and handled differently from the main branch's approach of modifying `d` directly.

### 11. **Bias update in analysis uses `new_innovation_to_state` instead of direct assignment**

The main branch directly assigns the innovation to the bias state:
```python
ensemble.bias.update_history(b=ia, update_last_state=True)
```

The `bias_ens` branch uses a method that also performs an ESN open-loop step:
```python
bias_state = self.bias.new_innovation_to_state(innovation)
```

This additional step modifies the reservoir state, which is not done in the main branch.
