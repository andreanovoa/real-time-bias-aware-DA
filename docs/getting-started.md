# Getting started

`romda` combines physical and data-driven forecast models with real-time data assimilation.
The shortest route is to install the package, run the Van der Pol example, and then choose a
model or tutorial from the navigation.

## Install

From PyPI:

```bash
pip install romda
```

For a source checkout:

```bash
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd real-time-bias-aware-DA
conda create -n romda python=3.12
conda activate romda
pip install -e ".[dev,notebooks]" --use-pep517
```

## Quick example

```python
import numpy as np
from romda.bias_estimators import ConstantBias
from romda.estimators import rBA_EnKF
from romda.models.physical import VdP
from romda.observations import Observations

truth = Observations(model=VdP, t_start=0.6, t_stop=0.8, Nt_obs=30,
                     add_noise=True, manual_bias='linear')
Cdd = np.diag((0.05 * np.max(abs(truth.y_obs), axis=0)) ** 2)
ensemble = rBA_EnKF(parent_model=VdP(dt=truth.dt), parent_bias=ConstantBias,
                    m=10, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)))

for data, time in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=time)
    ensemble.analysis_step(d=data, Cdd=Cdd)
```

## Run a physical experiment

Physical and thermoacoustic experiment configurations are documented on the
[Experiments](experiments.md) page:

```bash
python -m romda.experiments configs/tai_da/rijke.yml --seed 1
```

## Continue

- [Physical models](api/models_physical.md)
- [Data-driven models](api/models_data_driven.md)
- [Data assimilation estimators](api/estimators.md)
- [Tutorials](tutorials.md)
