# EDITS — bias-aware DA fixes, tutorial repairs, and test suite

Branch: `dev-f` (created from `editable`).
Scope: make the bias-aware EnKF work with interchangeable bias estimators, fix the bugs
found along the way, repair the broken tutorials (bias + ROM/POD/SPOD), and add a proper
unit/integration test suite.

---

## 1. Bias-aware data assimilation: what was broken and how it was fixed

The core objective — running the r-EnKF with different types of bias estimators — was
blocked by a chain of bugs in the analysis step and in the estimators themselves.

### `src/data_assimilation.py`
- **`rBA_EnKF.__call__` signature mismatch (blocking bug).** The filter expected
  `(Af, d, Cdd, Cbb, b, bd, J)` but `Ensemble.analysis_step` called it with 6 arguments
  (no `bd`), so every bias-aware analysis raised a `TypeError`. The `bd` argument was
  redundant: `analysis_step` already de-biases the observations (`d + E[b - i]`) before
  the call, so the filter now takes `(Af, d, Cdd, Cbb, b, J)` — the same interface as
  the CMAME implementation on `main`.
- `Filter`, `EnKF`, `EnSRKF` now default `gamma=None`, and `rBA_EnKF` defaults
  `gamma=1.0`. Before, `EnSRKF(M=...)` (used internally by the bias Bayesian update)
  crashed with a missing positional argument.
- `EnSRKF`: replaced `len(E) is not m` with `len(E) != m` (identity comparison on ints —
  works only accidentally for small ensembles).

### `src/ensemble.py`
- **The bias estimator was never updated after the analysis (blocking bug).** On `main`,
  after each analysis the bias state is reset to the analysis innovation
  `i^a = d − ⟨y^a⟩`. On `editable` this step was missing entirely, so the bias estimate
  never left its initial condition. `analysis_step` now calls the new
  `update_history_analysis()`, which stores the analysis state and feeds `i^a` to
  `Bias.update_state_from_innovation`.
- **`alpha_limits_matrix` used the limits of *all* model parameters instead of the
  estimated ones** (`alpha_lims.values()` vs `est_alpha`). With e.g. `est_alpha=['zeta']`
  on the VdP model, the zeta samples (~50) were checked against kappa's upper bound (20)
  and *every* analysis was rejected as "non-physical" — silently disabling the DA. It
  also crashed with an `OverflowError` for models with all-integer limits.
- **Parameter-freeze slicing bug.** When `activate_parameter_estimation=False` (or during
  the new `num_SE_only` window), `Af` was cut before the analysis but the history update
  still took `Aa[:Nphi+Na]`, writing observable rows into the parameter slots. The
  forecast parameters are now re-appended explicitly.
- Implemented `num_DA_blind` (bias-blind EnKF fallback for the first k analyses) and
  `num_SE_only` (state-estimation-only window) which existed as attributes but were dead.
- `get_observables` no longer raises `NotImplementedError`: it now supports `bias=None`,
  `NoBias`, and bias correction for both `Nt == 1` and histories.
- `get_observable_hist` passed the **full bias-estimator state** (including ESN reservoir
  and innovation components) to the un-biasing routine; it now extracts only the bias
  components (`get_bias`).
- `_recover_unbiased_solution` and `analysis_step` now handle a bias-ensemble size that
  differs from the model ensemble size (the mean bias is used, consistent with the
  CMAME definition of bias on the ensemble mean).
- Added a validating `bias` setter (`ensemble.bias = estimator` instead of poking
  `ensemble._bias`).
- `forecast_step` advances the bias the same number of output steps as the model and
  no longer computes an unused observable history.

### `src/bias_estimators/bias.py`
- **`initialize_bias_state` was inverted**: it returned an `Nq`-sized state for
  `biased_observations=True` and `2*Nq` otherwise — exactly opposite to `N_dim` — so the
  base-class constructor assertion could never pass. It now returns `zeros((N_dim, N_ens))`.
- `update_state_from_innovation`:
  - removed a dead-end `raise NotImplementedError` that made every non-Bayesian update
    with a forecaster crash;
  - added the member-to-member assignment when the innovation ensemble matches `N_ens`;
  - a single-member bias estimator now takes the *mean* innovation instead of one random
    resample;
  - `np.cov` output is forced 2-D for `Nq=1` in the Bayesian branch.
- Removed the stale duplicate `plot_train_data` (the maintained copy lives in `aux.py`,
  and the package `__init__` now imports it from there).

### `src/bias_estimators/constantbias.py`, `driftlinear.py` — new working estimators
- **`ConstantBias`** was a placeholder that raised `NotImplementedError` and referenced an
  undefined variable. It is now a working persistent-bias estimator (`db/dt = 0`,
  reset to the innovation at each analysis), with optional constant `k` initial value.
- **`NoBias`** (new class): zero-bias placeholder, restoring parity with `main` (the code
  referenced `'NoBias'` by name in several places but no such class existed on `editable`).
- **`DriftLinearBias`** raised `NotImplementedError`, imported `HistoryTracker` from a
  non-existent module, and referenced undefined variables. It now integrates
  `db/dt = v + A b` (Euler) and is verified against analytic solutions in the tests.
- `bias_estimators/__init__.py` exports `Bias, ESN_bias, ConstantBias, NoBias,
  DriftLinearBias, plot_train_data`.

### `src/bias_estimators/esn.py` / `aux.py`
- `force_retrain=True` was silently ignored (the kwarg was absorbed as a class attribute
  before reaching the retrain check).
- `sample_model_states` crashed with a scalar `std_alpha` on a plain (non-ensemble) ROM;
  a scalar is now interpreted as a relative range around the nominal parameter values.

### `src/tools/esn_core.py`
- `dr_di`/`Jacobian` tested `isinstance(dr_di, csr_matrix)`, but `csr.multiply()` returns
  a **COO** matrix, so the dense branch ran and `*` dispatched to matrix multiplication →
  crash (or wrong values) when computing the ESN bias Jacobian in the r-EnKF. Now uses
  `issparse` and converts `dr_di` back to CSR.

### `src/models/model.py`
- **`governing_eqns_params` is a class-level mutable dict** and `set_fixed_params()`
  mutated it in place, leaking fixed parameters across *different model classes* (e.g.,
  after instantiating a `VdP`, a `Lorenz63` forecast crashed with
  `unexpected keyword argument 'law'`). It now builds an instance-level dict.

### `src/observations.py`
- **`y_raw` was only initialized in the no-bias branch of `_set_bias`.** With a manual
  bias and `add_noise=False`, `y_raw` stayed `None` and `np.atleast_3d(None)` produced a
  garbage `(1,1,1)` object array (this is what broke tutorial 22's truth generation).
  `y_raw` now always defaults to the (biased) truth when no raw data is provided.

### `src/models/data_driven/esn.py`
- Removed `from matplotlib.cm import get_cmap` — the function was removed in
  matplotlib 3.9 (which `pyproject.toml` requires), so the whole package failed to import.

### `src/utils.py`
- `get_wake_data` had a stray `"` embedded in the Zenodo download URL, so the wake
  dataset (tutorials 30–32) could never download.

---

## 2. ROM / POD / SPOD fixes

### `src/tools/pod_spod.py`
- `spod_towne` used `get_window` and `gammaincinv` without importing them → instant
  `NameError` on every call (tutorial 30 §9). Imports added and covered by a test.

### `src/tools/autoencoders.py` (POD / SPOD classes)
- **Flat-data ordering now matches the documented convention.** `_to_flat` produced
  fluid-point-major *interleaved* rows (`row = fluid_pos*Nu + var`), while every tutorial
  (and the docstrings) assume variable-block ordering (`Psi[:N_fluid] = u_x`,
  `Psi[N_fluid:] = u_y`). The mode plots in tutorials 03/30 were therefore scrambled.
  `_to_flat` / `_to_physical_grid` now use block ordering (self-consistent inverses).
- New `grid_index_to_flat_rows()` maps raw-grid indices (e.g., sensor locations) to rows
  of `Psi`/`Q_mean`, accounting for the NaN body mask.
- `POD.fit` now also accepts an already-flat `(N_x, N_t)` matrix (the documented
  `POD(n_modes).fit(Q)` API previously failed the 4-D assertion).
- `Projector.score` compared a grid input against a flat reconstruction (shape error /
  NaN); it now scores in the flat zero-mean space.

### `src/models/data_driven/pod_esn.py`
- **Sensor-placement indexing bug (the one flagged in tutorial 32).**
  `get_observables` did `decode(Z, idx=self.sensor_locations)`, indexing the POD basis
  with *raw-grid* indices. Raw-grid indices and `Psi` rows use different orderings and
  the mask offsets them further, so the model "observed" the wrong spatial points and DA
  could not work. `get_observables` now uses the new `sensor_rows` property
  (`grid_index_to_flat_rows(sensor_locations)`); `sensor_locations` keeps the raw-grid
  convention used to sample the data (`X_flat[:, sensor_locations]` in tutorial 32).
  Verified by `tests/test_pod_esn.py::test_observables_match_data_at_sensors`.

---

## 3. Tutorials

| Tutorial | Status before | What was done |
|---|---|---|
| `00/01/02/04` | pass | untouched (verified by full execution) |
| `03_Class_POD` | broken: imported removed `prepare_data`/`energy_fraction`, `POD.plot_spectrum` etc. | ported to the class API (`preprocess_snapshot`, `plotting.pod` functions), fixed the truncated-reconstruction MSE cell (it scored a reconstruction against itself) |
| `05_Class_Bias` | broken: `sample_model_states` crash with scalar `std_alpha` | fixed in library; **runs end-to-end** (incl. ESN training) |
| `10/11/12` | pass | untouched (verified by full execution) |
| `13_bias-aware-DA-intro` | contained `raise NotImplementedError("There is some bug in the Bias-aware codes...")` + stale cells | placeholder removed; DA-loop cell rewritten with the fixed API (`bias` setter, `num_DA_blind`, no `inn_uncertainty`); junk cells (undefined `input_innovation`, hard-coded `/storage0/...` path) replaced; `alpha_distr` → `distribution_alpha`; **runs end-to-end** |
| `20/23` | pass | untouched |
| `21_TADA_Rijke_twin` | broken: old `main` API (`create_ensemble`, `dataAssimilation`, `dt_obs=` kwarg) | ported to `Observations`/`Ensemble`/`EnKF` with an explicit DA loop; **runs end-to-end** |
| `22_TABADA_Rijke_CMAME` | broken: old API + `Observations` y_raw bug + ESN Jacobian sparse bug | ported to new API (`ESN_bias`, `rBA_EnKF`); **runs end-to-end** |
| `24_TA_azimuthal_data` | code OK; fails here only because Zenodo is blocked by this sandbox's proxy | no code change needed (verify on a machine with Zenodo access) |
| `25_TABADA_annular_raw` | broken: old API (`create_ensemble`, `create_bias_model`, `ESN`, dict-style truth) | ported to new API; **not executable here** (Zenodo blocked) — please run locally |
| `30_POD_SPOD_intro` (+ `.py` twin) | broken: no data download, `POD.plot_spectrum` class-method calls, `spod_towne` NameError | added `get_wake_data` download call, switched to `plotting.pod` functions, fixed imports in library; POD/SPOD/`spod_towne` logic covered by synthetic-data tests |
| `31_esn_pod_tutorial` | broken: `case_pod` never defined (empty init cell), removed `project_data_onto_Psi` API, wrong `reconstruct(Phi=...)` orientation, `run_test` kwarg | missing POD-init cell added; ported to `encode`/`decode`/`build_psi` API; truncation loop no longer mutates the original case; **not executable here** (Zenodo blocked) |
| `32_real-time-DA_ESN-POD_cylinder` | contained `raise ValueError("There is a bug in the sensor placement logic...")` | sensor-placement bug fixed in the library (see §2) and verified by tests; the raise removed; **not executable here** (Zenodo blocked) |

Note on the sandbox: outbound requests to zenodo.org are blocked in this environment
(HTTP 403 from the egress proxy), so the data-dependent notebooks (24, 25, 30, 31, 32)
could not be executed here. Their code paths were fixed and the underlying machinery is
covered by synthetic-data tests (`tests/test_pod_spod.py`, `tests/test_pod_esn.py`).

---

## 4. Test suite (`tests/`)

Previously the only test was `test_tutorials.py` (notebook smoke runner, which in quick
mode skips any notebook containing the string "train" — i.e., most of the interesting
ones). There is now a proper suite (88 tests, ~13 s, no external data):

- `tests/test_filters.py` — unit tests for `EnKF`, `EnSRKF`, `rBA_EnKF`: shapes,
  finite output, analysis-moves-toward-observation property, accepted bias shapes,
  unbiased limit of the r-EnKF, bias-corrected analysis property.
- `tests/test_bias_estimators.py` — `ConstantBias`, `NoBias`, `DriftLinearBias`
  (analytic decay/drift solutions), base-class state formatting, history growth,
  innovation updates, Bayesian update.
- `tests/test_models_history.py` — `HistoryTracker` (reset/append/update-last/capacity
  growth), VdP and Lorenz63 forecasts, observation operator shape (also guards the
  `governing_eqns_params` leak regression).
- `tests/test_observations.py` — truth generation, all manual-bias types, the
  y_raw-without-noise regression, noise levels, frozen attributes, provided-data mode.
- `tests/test_ensemble_da.py` — integration: twin experiments on VdP; every
  bias-estimator × filter combination; bias state tracks the analysis innovation;
  `NoBias` ≡ unbiased limit; `num_DA_blind` / `num_SE_only`; parameter-limit rejection
  (also guards the `alpha_limits_matrix` regression).
- `tests/test_pod_spod.py` — POD (exact/randomized) on synthetic flows with NaN masks,
  SPOD (`Nf=0` ≡ POD), `spod_towne` (guards the missing-import regression),
  encode/decode round trips, variable-block ordering regression,
  `grid_index_to_flat_rows` round trip.
- `tests/test_pod_esn.py` — POD_ESN on a synthetic flow: sensors are fluid grid points,
  observables match the data at the sensors (sensor-placement regression), forecast
  and `measure_modes` modes.

Run with: `python -m pytest tests/` (pyproject already points `testpaths` at `tests`).
The notebook runner `test_tutorials.py` still works as before
(`python -m pytest test_tutorials.py`).

---

## 5. Inconsistencies between `main` and `editable`

Found while cross-reading the two branches:

1. **Missing bias update after analysis** — `main`'s `dataAssimilation()` resets the bias
   to the analysis innovation each cycle; `editable` had dropped this step (now restored,
   see §1).
2. **`rBA_EnKF` interface drift** — `main` passes `(Af, d, Cdd, Cbb, k, M, b, J)` and
   de-biases `d` in `analysisStep`; `editable` had added an extra `bd` argument to the
   filter but its caller was never updated. Resolved in favour of `main`'s split of
   responsibilities (caller de-biases `d`).
3. **`NoBias` class** existed on `main` and was still referenced by name on `editable`
   (`Ensemble.get_observables`, `get_observable_hist`) but did not exist — recreated.
4. **`num_DA_blind` / `num_SE_only`** (`main`: `activate_bias_aware`,
   `activate_parameter_estimation` driven by the assimilation loop) were dead attributes
   on `editable` — reimplemented inside `analysis_step`.
5. **`main`'s `rBA_EnKF_CMAME` variant** (J not transposed) has no counterpart on
   `editable`; only the standard r-EnKF was ported. Straightforward to add as another
   `Filter` subclass if needed.
6. **`scripts/mains/*.py`** (`main_Lorenz`, `main_annular`, `main_cylinder`, and
   `default_parameters/*`) still target `main`'s API (`create`, `run`, `plot_results`
   top-level modules) and do not run on `editable`. Not ported (out of the tutorial
   scope) — flagging as remaining work.
7. `main`'s `check_std_too_large` spread check exists on `editable` as
   `has_valid_spread` but is disabled (`return True`); left as-is (explicitly marked
   temporary in the code).
8. `environment.yml` exists only on `main`; `editable` relies on `pyproject.toml`.
   Note: keep `pyts>=0.13` (needed by `plotting/pod.py`; 0.12 pulls an ancient numba).

---

## 6. Other notes

- `.gitignore`: added `.venv/`.
- Generated ESN configs are written to `src/config/esn_configs/` (already gitignored).
- The tutorial notebooks were re-executed end-to-end where possible in a fresh
  Python 3.12 venv with numpy 2.4 / scipy 1.17 / matplotlib 3.10; all non-data-dependent
  tutorials pass: 00, 01, 02, 04, 05, 10, 11, 12, 13, 20, 21, 22, 23
  (03, 24, 25, 30, 31, 32 need the Zenodo datasets, blocked in this environment).

---

## 7. Addendum (doc branch): package rename and documentation site

- **Package renamed to `romda`** (real-time reduced-order modelling and bias-aware DA).
  All modules moved under `src/romda/` and every import in the library, tests,
  tutorials and scripts now uses the `romda.*` namespace
  (e.g. `from romda.ensemble import Ensemble`). `pip install -e .` installs `romda`.
  Verified: 88/88 tests pass and tutorial 12 executes end-to-end after the rename.
- **Documentation site** (MkDocs Material + mkdocstrings) added:
  - `mkdocs.yml` + pages under `docs/` (home, getting started, architecture,
    bias-aware DA theory with MathJax, tutorials index, API reference, publications);
  - `docs/requirements-docs.txt` — minimal environment to build the site;
  - `.github/workflows/docs.yml` — deploys to the `gh-pages` branch on pushes to
    `doc`/`main` (`mkdocs gh-deploy`). Site URL:
    https://andreanovoa.github.io/real-time-bias-aware-DA/
  - To preview locally: `pip install -r docs/requirements-docs.txt && mkdocs serve`.
- README updated for the new name, layout and quick-start example.
- **CMAME erratum**: the r-EnKF equations as published in CMAME (2023) Eqs. (15)–(16)
  contain typos in the Jacobian transposes (see `docs/2023_CMAME_Erratum.pdf`). The
  `rBA_EnKF` implementation follows the corrected erratum equations (1a)–(1b) — now
  stated explicitly in the docstring, the documentation theory page, tutorial 13, and
  pinned by `tests/test_filters.py::test_matches_corrected_erratum_equations`, which
  verifies the implementation matches the corrected form exactly and rejects the
  as-published (un-transposed) form.

---

## 8. Addendum: reconciliation with the user's late `editable` commits

Two commits (`d120367`, `254ba19` — the user's own "bug search") were pushed to
`editable` after this work started, and were merged into `dev-f` and `doc`. Most of
their fixes had been found independently here (identical resolutions: `bd` removal,
bias update after analysis, `est_alpha` limits, bias-only un-biasing, `issparse`,
`force_retrain`). Unique fixes adopted from them:

- **`Filter.observation_operator` truncation fix** — with `M = [0 | I]`, trimming to
  `M[:, :n]` cut off the identity block whenever parameters were frozen and `Na >= Nq`,
  silently zeroing the observation operator. Now the trailing identity columns are kept
  and only the zero block shrinks (regression-tested).
- **Mean-bias semantics** — `Bias.current_bias` / `current_innovations` now return the
  ensemble-mean (shape `(Nq, 1)`), consistent with the CMAME definition of the bias on
  the ensemble mean; `get_bias`/`get_innovations` format the state before averaging.
- `check_valid_file` (utils) now works for dict inputs — the training-data cache
  validation was a silent no-op before.
- `load_bias_training_dataset` validates the cached dataset dimension
  (`expected_Ndim`) so a cache built with different `biased_observations` is rebuilt.
- `sample_model_states` skips parameter sampling for an *empty* `std_alpha` dict.
- Tutorial 05 gained an ESN-Jacobian verification section (analytic vs central finite
  differences), merged and re-verified end-to-end.

Kept from this session where the two diverged: `EnKF`/`EnSRKF` force `gamma=None`
(their version passed gamma through, which would mark plain filters bias-aware when
built via `Ensemble` and crash `analysis_step`), and the analysis innovation is
computed as `d[:, None] - y_a` (their `d - get_observables()` does not broadcast for
`Nq != m`). Their WIP version of tutorial 13 (dangling syntax) was superseded by the
verified one.

---

## 9. Addendum (doc branch): flat layout, docstring sweep, reference audit

- **Package layout flattened**: the modules moved back from `src/romda/` to `src/`,
  with the import name kept as `romda` via the setuptools mapping
  `package-dir = {"romda" = "src"}` (i.e., `src/__init__.py` is `romda/__init__.py`).
  `tests/conftest.py` registers `src` as the `romda` package when the project is not
  pip-installed.
- **Docstring sweep for the documentation site**: all public-API docstrings converted
  to numpy-style sections (google-style `Args:` blocks and free-text `Inputs:` blobs
  removed), equations rewritten in LaTeX (r-EnKF corrected equations, Van der Pol,
  Lorenz 63, Rijke, annular model, POD/SPOD relations; `DriftLinearBias` keeps its
  original plain-text docstring), stray code
  snippets removed from descriptions, and the filter call signatures documented on
  `__call__` (now rendered via the mkdocstrings filters). The site builds with zero
  griffe/mkdocs warnings.
- **References audit**: every API page and the relevant docstrings now cite the
  corresponding publications — CMAME 2023 + the 2024 erratum (r-EnKF, ESN bias),
  JFM 2022 (Rijke, Van der Pol), JFM 2024 (annular digital twin), Evensen 2009
  (EnKF/EnSRKF), Lorenz 1963, Sirovich 1987 / Halko 2011 / Sieber 2016 / Towne 2018
  (POD/SPOD), and Racca & Magri 2021 (ESN validation). The erratum PDF is published
  with the site and linked from the theory page, the API pages, and the publications
  page.

---

## 10. References added — please check

All bibliographic references introduced during the documentation work, with their
location and provenance. **Provenance key:** *(repo)* = copied/reformatted from text
already in the repository (README, module headers, erratum PDF) — low risk;
*(added)* = written from general knowledge — **please verify the volume/page details**.

### Your papers

| Reference | Where added | Provenance |
| --- | --- | --- |
| Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502. DOI: 10.1016/j.cma.2023.116502 | `rBA_EnKF` and `ESN_bias` docstrings; `docs/api/data_assimilation.md`; `docs/api/bias_estimators.md`; `docs/concepts/bias-aware-da.md`; `docs/index.md` | *(repo)* — README + erratum PDF |
| Nóvoa, Racca & Magri (2024). Erratum to the above (corrected Eqs. 15–16) | `rBA_EnKF` docstring; theory page warning box; both API pages above; `docs/publications.md`; PDF published with the site | *(repo)* — `docs/2023_CMAME_Erratum.pdf` |
| Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. *J. Fluid Mech.*, **948, A35**. DOI: 10.1017/jfm.2022.653 | `VdP` and `Rijke` docstrings; `docs/api/models.md` | DOI *(repo)* — README; **volume/article number (948, A35) *(added)* — please check** |
| Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic instabilities. *J. Fluid Mech.*, 1001, A49. DOI: 10.1017/jfm.2024.1052 | `Annular` docstring (was already there, reformatted); `docs/api/models.md`; `docs/api/bias_estimators.md` | *(repo)* — old Annular docstring + README |

### Data assimilation

| Reference | Where added | Provenance |
| --- | --- | --- |
| Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer. | `EnKF` and `EnSRKF` docstrings (EnKF cites Eq. 9.27, as in the old docstring); `docs/api/data_assimilation.md` | *(repo)* — old docstrings mentioned "Evensen (2009)" |

### Models

| Reference | Where added | Provenance |
| --- | --- | --- |
| Lorenz (1963). Deterministic nonperiodic flow. *J. Atmos. Sci.*, **20, 130–141**. | `Lorenz63` docstring; `docs/api/models.md` | *(added)* — standard citation, please confirm page range |

### POD / SPOD (copied from the reference list already in `pod_spod.py`'s module header)

| Reference | Where added | Provenance |
| --- | --- | --- |
| Sirovich (1987). Turbulence and the dynamics of coherent structures. *Quart. Appl. Math.*, XLV(3), 561–590. | `snapshot_pod` docstring; `docs/api/tools.md` | *(repo)* |
| Halko, Martinsson & Tropp (2011). Finding structure with randomness. *SIAM Review*, 53(2), 217–288. | `snapshot_pod_randomized` docstring; `docs/api/tools.md` | *(repo)* |
| Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal decomposition. *J. Fluid Mech.*, 792, 798–828. | `SPOD` class and `spod_sieber` docstrings; `docs/api/tools.md` | *(repo)* |
| Towne, Schmidt & Colonius (2018). Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis. *J. Fluid Mech.*, 847, 821–867. | `spod_towne` docstring; `docs/api/tools.md` | *(repo)* |
| Mendez et al. (2023). *Data-Driven Fluid Mechanics.* Cambridge University Press. | `docs/api/tools.md` (notation conventions) | *(repo)* — `[Mendez 2023]` in the module header |

### Echo state networks

| Reference | Where added | Provenance |
| --- | --- | --- |
| Lukoševičius (2012). A practical guide to applying echo state networks. In *Neural Networks: Tricks of the Trade*, Springer. | `EchoStateNetwork` docstring | *(added)* — standard ESN reference, please confirm you want it |
| Racca & Magri (2021). Robust optimization and validation of echo state networks for learning chaotic dynamics. *Neural Networks*, **142, 252–268**. | `EchoStateNetwork` docstring; `docs/api/tools.md` | *(added)* — cited as the source of the recycled-validation strategy; **please verify volume/pages and that this is the intended reference** |

### Not added anywhere (for completeness)

The tutorials/publications page also links your JFM 2022/2024 legacy repositories, the
arXiv 2025 preprint, the INTER-NOISE 2022 paper and the PhD thesis — those entries
were carried over verbatim from the README, not newly written.
