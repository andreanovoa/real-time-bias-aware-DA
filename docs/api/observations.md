# Observations

## Summary

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

------

::: romda.observations.Observations
