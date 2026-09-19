# Ablation studies

Systematic sensitivity studies of the real-time (bias-aware) data assimilation
framework on the four test cases:

| Case | Script | Truth | Forecast model | Filters |
|---|---|---|---|---|
| Lorenz63 twin | `run_lorenz63.py` | model + noise (+ optional manual bias) | Lorenz63 | EnKF, EnSRKF, rBA-EnKF |
| Lorenz96 twin | `run_lorenz96.py` | model + noise | Lorenz96 (Nx = 40) | EnKF, EnSRKF |
| Annular thermoacoustics | `run_annular.py` | raw azimuthal experiment data ([Zenodo](https://zenodo.org/records/15609832)) | annular low-order model + ESN bias | rBA-EnKF vs EnKF |
| Cylinder wake | `run_cylinder.py` | held-out noisy snapshots (Re = 100) | POD-ESN reduced-order model | EnSRKF |

Every study varies **one factor at a time** around a fixed, verified default
configuration (the defaults of the corresponding `scripts/mains/main_*.py`),
with `--seeds N` repeated runs per grid point (different ensemble
initializations and noise realizations). Metrics are the normalized RMSE of
the ensemble-mean observables against the *unbiased* truth, split into the
assimilation window (`rmse_da`) and the post-assimilation forecast
(`rmse_forecast`), the normalized ensemble spread in both windows, and the
relative parameter-estimation errors (`alpha_error_<name>`) where the truth
parameters are known (see `scripts/mains/common.compute_metrics`).

## Running

```bash
cd scripts/ablations
python run_lorenz63.py --study ensemble_size            # full sweep (5 seeds)
python run_lorenz63.py --study ensemble_size --quick    # 2x2 smoke test
python run_lorenz63.py --study ensemble_size --plot     # re-plot existing results
```

Each (grid point, seed) result is appended to
`results/ablations/<case>_<study>.jsonl` as soon as it finishes, so an
interrupted sweep resumes where it left off (failed runs are recorded with
`status='failed'` and excluded from the aggregation). Plots are saved to
`scripts/ablations/figs/` (versioned in git): box plots over the repeated
seeds with the individual runs overlaid, so diverged cases show up as
outliers instead of being hidden in a mean ± std summary (violins via
``kind='violin'`` in `plot_ablation`). All RMSE/spread metrics are
normalized by the RMS amplitude of the unbiased truth over the assimilation
window.

## Study matrix

### 1. Ensemble size (`--study ensemble_size`, all cases)

*Question*: how many members are needed before sampling error stops dominating,
and does the square-root filter (EnSRKF) beat the stochastic EnKF at small m?

- Lorenz63: m ∈ {5, 10, 20, 50, 100} × {EnKF, EnSRKF}
- Lorenz96: m ∈ {10, 20, 50, 100} × {EnKF, EnSRKF} — note m < Nx = 40 is the
  rank-deficient regime where inflation becomes critical.
- Annular: m ∈ {10, 20, 40, 60, 80} × {rBA-EnKF, EnKF}
- Cylinder: m ∈ {10, 50, 100, 200}

*Expected*: RMSE decreases roughly monotonically with m and saturates;
EnSRKF should tolerate smaller m (no perturbed-observation sampling noise).
Report the smallest m within 10% of the saturated RMSE — that is the
real-time-viable ensemble size.

### 2. Assimilation frequency (`--study assimilation_frequency`, all cases)

*Question*: how sparse can the observations be in time before the filter loses
track of the (chaotic) truth?

- Lorenz63: Nt_obs ∈ {5, 10, 20, 40, 80} time steps (0.08–1.3 Lyapunov times).
- Lorenz96: folded into the combined `observation_sparsity` study (see 6.).
- Annular: Nt_obs ∈ {15, 25, 35, 50, 70} (limit-cycle dynamics — expect much
  weaker sensitivity than the chaotic cases).
- Cylinder: Nt_obs ∈ {10, 20, 30, 50} (fractions of a shedding period).

*Expected*: for the chaotic twins, a sharp breakdown once the interval between
analyses approaches the error-doubling time; for the periodic/limit-cycle
cases, graceful degradation. The breakdown point sets the minimum sensor rate
for real-time deployment.

### 3. Inflation (`--study inflation`, Lorenz63 & Lorenz96)

*Question*: how much multiplicative inflation is needed to prevent filter
divergence, especially in the undersampled Lorenz96 regime (fixed m = 20 < Nx)?

- factor ∈ {1.0, 1.01, 1.02, 1.05, 1.1}

*Expected*: a U-shaped RMSE curve — divergence (spread collapse) without
inflation, noise-fitting with too much. Cross-check `spread_da` against
`rmse_da`: a well-tuned filter has spread ≈ RMSE (consistency).

### 4. Observation noise (`--study observation_noise`, Lorenz63)

*Question*: robustness to measurement noise, and to a *mis-specified*
observation-error covariance (assumed `std_obs` ≠ true `noise_level`).

- noise_level ∈ {0.01, 0.05, 0.1, 0.2} × assumed std_obs ∈ {0.01, 0.05, 0.1}

*Expected*: RMSE scales with the true noise; underestimating Cdd
(std_obs ≪ noise_level) over-trusts the data and can destabilize the filter,
overestimating it slows convergence but is safe.

### 5. Bias regularization (`--study regularization`, Lorenz63 & Annular)

*Question*: sensitivity of the r-EnKF to the bias-penalty weight γ in the
regularized cost function — the central hyperparameter of the bias-aware
method (Nóvoa, Racca & Magri 2023/2024 erratum).

- Lorenz63 (twin, manual linear bias): γ ∈ {0, 1, 2, 3, 4, 5}
- Annular (real data): γ ∈ {0, 1, ..., 10}, plus the bias-blind EnKF baseline
  (`--study ensemble_size` provides it per m).

*Expected*: γ = 0 reduces to a bias-blind update of the augmented state (the
analysis chases the biased observations); moderate γ separates bias from state
error and should minimize `rmse_da` against the *unbiased* truth; large γ
over-weights the bias model and freezes the update. Report the plateau of
near-optimal γ — a wide plateau means the method is easy to tune.

### 6. Case-specific axes

- **Lorenz96 observation sparsity** (`--study observation_sparsity`): 2-D grid
  of temporal sparsity Nt_obs ∈ {5, 10, 20, 40} (0.08–0.7 Lyapunov times) ×
  spatial sparsity obs_every k ∈ {1, 2, 4, 8} (Nq = 40, 20, 10, 5), plotted in
  one figure (x = Nt_obs, one shade/width per k). Localization is not
  implemented, so this quantifies how far the plain (square-root) EnKF can be
  pushed with sparse coverage — and motivates localization if the k ≥ 4 runs
  fail — and shows whether time- and space-sparsity interact.
- **Annular equivalence ratio** (`--study equivalence_ratio`): all four
  operating conditions ER ∈ {0.4875, 0.5125, 0.5375, 0.5625} × {rBA-EnKF,
  EnKF}. Tests that the conclusions (and the trained ESN bias) are not tuned to
  a single operating point.
- **Cylinder sensors** (`--study sensors`): N_sensors ∈ {1, 2, 3, 4, 8}
  QR-pivot point sensors vs the default down-sampled grid. With N_modes = 4,
  observability requires ~4 well-placed sensors; expect a sharp knee there.
- **Cylinder Wout estimation** (`--study wout_estimation`): state-only vs
  joint state + ESN-output-weight (`Wout` singular values) estimation across
  initial uncertainty std_phi ∈ {0.1, 0.5, 1.1}. Tests whether online weight
  updates compensate for the noisy training data.

## Protocol notes

- **Repeats**: 5 seeds for the cheap twin experiments, 3 for annular/cylinder.
  The seed controls the ensemble initialization (`model.seed`); observation
  noise is redrawn each run. Plots show the seed distribution (box plots with
  individual runs overlaid).
- **One factor at a time**: interactions (e.g., m × inflation) are secondary;
  if a study shows unexpected sensitivity, promote the pair to a 2-D grid by
  adding both axes to the study's `axes` dict — the harness takes the
  Cartesian product automatically.
- **Caching**: the annular ESN bias estimator and the cylinder POD-ESN are
  trained once and pickled; sweeps over filter parameters reuse them. Axes
  that change the training data (annular `Nt_obs`) retrain per level.
- **Cost control**: `--quick` truncates every axis to 2 values and cuts the
  seeds — run it first to smoke-test a study end to end before launching the
  full sweep.
