# Action Plan: Fixing the `bias_ens` Branch Tutorial

**Date:** 2026-03-24  
**Branch:** `bug_search`  
**Reference:** Comparison of `main` (working) vs `bias_ens` (broken) tutorial `13_bias-aware-DA-intro.ipynb`

---

## Overview

The `bias_ens` branch tutorial produces incorrect data assimilation results due to a combination of bugs in the tutorial notebook and in the core `src/data_assimilation.py` and `src/ensemble.py` files. Below are the prioritized actions to fix the issues, ordered by impact.

---

## Priority 1: Critical Fixes (Must Do)

### Action 1.1 — Fix the regularization factor `gamma` in the tutorial

**File:** `scripts/tutorials/1_Introtuction_to_real-time_DA/13_bias-aware-DA-intro.ipynb` (Cell 22)

**Current (broken):**
```python
filter_ens.filter.gamma = -1.
```

**Fix:**
```python
filter_ens.filter.gamma = 1.
```

**Reason:** A negative `gamma` inverts the regularization term in the rBA-EnKF equations. The cost function penalizes bias with `gamma * ||J * b||^2`, and a negative value transforms the penalty into a reward, amplifying bias instead of correcting it. The main branch uses `regularization_factor = 1.0`.

---

### Action 1.2 — Fix the regularization term shape mismatch in `rBA_EnKF`

**File:** `src/data_assimilation.py`, in `rBA_EnKF.__call__`

**Current (broken):**
```python
Aa = Af + np.dot(K, np.dot(Iq + J.T, D - Y) - self.gamma * np.dot(CdWb, np.dot(J.T, b)))
```

Here `b` has shape `(Nq, N_ens)` which may not match `(Nq, m)` when `N_ens != m`.

**Fix:** Use `B` (the already-expanded bias) instead of `b` in the last term:
```python
Aa = Af + np.dot(K, np.dot(Iq + J.T, D - Y) - self.gamma * np.dot(CdWb, np.dot(J.T, B)))
```

This is consistent with the main branch, where `B = np.repeat(b, Nm, axis=1)` is used.

---

### Action 1.3 — Fix `Cdd` construction in the tutorial

**File:** `scripts/tutorials/1_Introtuction_to_real-time_DA/13_bias-aware-DA-intro.ipynb` (Cells 20-22)

**Current (broken):**
```python
std_d = .02 * np.std(y_ens, axis=2)
Cdd = np.eye(filter_ens.model.Nq) * (std_d**2)
```

**Fix:** Use the same construction as the main branch:
```python
std_obs = 0.1
Cdd = np.diag(std_obs * np.ones(filter_ens.model.Nq)) * np.max(abs(truth.y_obs), axis=0) ** 2
```

**Reason:** The observation covariance must be scaled relative to the signal magnitude for proper Kalman gain computation. The current implementation uses an extremely small `Cdd`, giving excessive weight to observations.

---

## Priority 2: Important Fixes (Should Do)

### Action 2.1 — Add blind DA phase support

**File:** `src/ensemble.py`, in `analysis_step()`

The main branch supports a `num_DA_blind` parameter that causes the first N analysis steps to use standard EnKF (without bias awareness) before switching to rBA-EnKF. This allows the ensemble to stabilize before the bias model influences the analysis.

**Suggested approach:**
```python
# In analysis_step, add a counter for analysis steps
if self.filter.is_bias_aware and self._analysis_count >= self.num_DA_blind:
    # Use bias-aware filter
    ...
else:
    # Temporarily use standard EnKF (set gamma=0 or use EnKF directly)
    filter_args = (Af, d, Cdd)
    Aa = EnKF_fallback(*filter_args)
```

Alternatively, in the tutorial, manually set `gamma = 0` for the first few steps, then set `gamma = 1.`.

---

### Action 2.2 — Pass observables to bias forecast

**File:** `src/ensemble.py`, in `forecast_step()`

**Current:**
```python
b, t_b = pb.time_integrate(**kwargs_local)
```

**Fix:** Pass model observables so the ESN can use them for open-loop correction if needed:
```python
y_obs = pm.get_observable_hist(Nt=psi.shape[0])
b, t_b = pb.time_integrate(y=y_obs, **kwargs_local)
```

This requires updating `Bias.time_integrate()` and the integrator to accept and use observables.

---

## Priority 3: Tutorial Corrections (Nice to Have)

### Action 3.1 — Align tutorial parameters closer to main branch

While the tutorials don't need to be identical, some parameters diverge unnecessarily:

| Parameter | Current (bias_ens) | Suggested |
|-----------|-------------------|-----------|
| `psi0` | `np.array([2., 1.])` | Keep (different init is fine) |
| `std_phi` | 0.1 | Consider 0.3 to match main |
| `alpha ranges` | Narrow (40-50, 50-60, 3-4) | Consider wider (40-80, 50-80, 3-5) |
| `upsample` | 5 | Consider 3 to match main |
| `N_wash` | 5 | Consider 10 to match main |

### Action 3.2 — Remove unused cells

The tutorial has empty code cells (Cells 17, 28, 30) that should be cleaned up or populated with relevant content.

---

## Verification Plan

After applying the fixes above, verify by:

1. **Run the bias_ens tutorial end-to-end** and check that the assimilation converges (parameters approach true values, bias is estimated correctly).

2. **Compare qualitatively with main branch results:** The time series should show the analyzed state tracking the truth, the bias estimate converging to the actual bias, and parameters converging toward their true values.

3. **Check shapes:** Add assertions in the DA loop to verify:
   - `Af.shape == (Nphi + Na + Nq, m)`
   - `b.shape[0] == Nq`
   - `B.shape == (Nq, m)`
   - `Cdd.shape == (Nq, Nq)`

4. **Test with `gamma = 0`** (no bias regularization, equivalent to standard EnKF) to establish a baseline, then increase `gamma` gradually.

---

## Summary of Changes by File

| File | Changes |
|------|---------|
| `src/data_assimilation.py` | Fix `rBA_EnKF.__call__`: use `B` instead of `b` in the regularization term |
| `src/ensemble.py` | (Optional) Add blind DA phase support, pass observables to bias forecast |
| Tutorial notebook (Cell 22) | Fix `gamma = -1` → `gamma = 1.` |
| Tutorial notebook (Cells 20-22) | Fix `Cdd` construction to match main branch approach |
