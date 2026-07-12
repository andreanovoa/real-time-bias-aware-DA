# Getting started

## Installation

Clone the repository and install the package in editable mode (Python ≥ 3.12):

```bash
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd real-time-bias-aware-DA
pip install -e . --use-pep517        # use .[dev] to include pytest
```

To verify the installation, run the test suite:

```bash
python -m pytest tests/              # unit and integration tests (~10 s)
python -m pytest test_tutorials.py   # executes the tutorial notebooks (slow)
```

## A minimal twin experiment

The building blocks of every `romda` experiment are:

1. an [`Observations`][romda.observations.Observations] object — the truth and the
   noisy measurements to assimilate;
2. an [`Ensemble`][romda.ensemble.Ensemble] — the forecast model wrapped with an
   ensemble of perturbed initial conditions and (optionally) uncertain parameters;
3. a data assimilation **filter** — e.g. [`EnKF`][romda.data_assimilation.EnKF] or the
   bias-aware [`rBA_EnKF`][romda.data_assimilation.rBA_EnKF];
4. optionally, a **bias estimator** from
   [`romda.bias_estimators`](api/bias_estimators.md).

```python
import numpy as np
from romda import Ensemble, Observations
from romda.models.physical import VdP
from romda.data_assimilation import rBA_EnKF
from romda.bias_estimators import ConstantBias

# 1. Truth: a Van der Pol oscillator with a manually-added model bias and noise
truth = Observations(model=VdP,
                     t_start=0.6, t_stop=0.8, t_max=1.0, Nt_obs=30,
                     add_noise=True, noise_type='gauss, add', noise_level=0.02,
                     manual_bias='linear')

# 2.-4. Ensemble with parameter estimation, a bias estimator, and the r-EnKF
ensemble = Ensemble(parent_model=VdP(dt=truth.dt),
                    parent_bias=ConstantBias,
                    da_method=rBA_EnKF,
                    regularization_factor=1.0,   # bias regularization γ
                    m=10,                        # ensemble size
                    std_phi=0.1,                 # initial state uncertainty
                    std_alpha=dict(zeta=(40., 60.)))  # uncertain parameter range

# Observation error covariance
std_obs = 0.05
Cdd = np.diag(std_obs * np.ones(truth.y_obs.shape[1])) * np.max(abs(truth.y_obs), axis=0) ** 2

# Sequential assimilation: forecast to each observation time, then analyse
for d, t_d in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=t_d)
    ensemble.analysis_step(d=d, Cdd=Cdd.copy())

ensemble.model.close()   # close the multiprocessing pools

# Visualize the assimilation results
ensemble.visualize_history(truth=truth)
```

Swapping the bias estimator is a one-line change — pass
[`ESN_bias`][romda.bias_estimators.ESN_bias],
[`DriftLinearBias`][romda.bias_estimators.DriftLinearBias] or
[`NoBias`][romda.bias_estimators.NoBias] as `parent_bias`, or set an instance later:

```python
ensemble.bias = my_trained_esn_bias.copy()
```

## Datasets

The thermoacoustic and cylinder-wake tutorials download their datasets from
[Zenodo](https://zenodo.org/) on first use via
`romda.utils.get_annular_data` and
`romda.utils.get_wake_data`; make sure the machine you run them on can
reach `zenodo.org`.

## Next steps

- Work through the [tutorials](tutorials.md) — they build up from Bayesian estimation
  basics to a real-data digital twin of an annular combustor.
- Read the [architecture overview](concepts/architecture.md) to understand the class
  design before extending the package with your own models or bias estimators.
