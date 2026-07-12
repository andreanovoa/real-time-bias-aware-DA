# romda — real-time reduced-order modelling and bias-aware data assimilation

`romda` is an open-source Python package for real-time data assimilation with
reduced-order models: ensemble Kalman filters (including the regularized bias-aware
EnKF), physical and data-driven forecast models (ESN, POD-ESN), bias estimators, and
POD/SPOD decompositions — with applications to thermoacoustics and fluid flows.

📖 **Documentation:** https://andreanovoa.github.io/real-time-bias-aware-DA/


---

## 🚀 Getting started

1. **Clone the repository**
```
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd real-time-bias-aware-DA
```

2. **Install the package in editable mode**
```
pip install -e . --use-pep517   # use .[dev] if running tests
```

3. (Optional) **Run the test suite**
```
python -m pytest tests/                 # unit and integration tests
python -m pytest test_tutorials.py      # execute the tutorial notebooks
```

Quick example:
```python
from romda import Ensemble, Observations
from romda.models.physical import VdP
from romda.data_assimilation import rBA_EnKF
from romda.bias_estimators import ConstantBias

truth = Observations(model=VdP, t_start=0.6, t_stop=0.8, Nt_obs=30,
                     add_noise=True, manual_bias='linear')

ensemble = Ensemble(parent_model=VdP(dt=truth.dt),
                    parent_bias=ConstantBias,
                    da_method=rBA_EnKF,
                    m=10, std_phi=0.1, std_alpha=dict(zeta=(40., 60.)))

for d, t_d in zip(truth.y_obs, truth.t_obs):
    ensemble.forecast_step(t_end=t_d)
    ensemble.analysis_step(d=d, Cdd=...)
```

Check out the [Tutorials folder](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials), which includes several jupyter notebooks aiming to ease the understanding of the repository.


---


## 🌟 What is available?
   Data assimilation methods [`romda.data_assimilation`](src/romda/data_assimilation.py)
   * EnKF — ensemble Kalman filter
   * EnSRKF — ensemble square-root Kalman filter
   * rBA-EnKF — regularized bias-aware EnKF

   Physical models [`romda.models.physical`](src/romda/models/physical/)
   * Rijke tube model (dimensional with Galerkin projection)
   * Van der Pol oscillator
   * Lorenz 63
   * Lorenz 96
   * Azimuthal thermoacoustics model
   * Kuramoto-Sivashinsky equation

   Data-driven models [`romda.models.data_driven`](src/romda/models/data_driven/)
   * `ESN_model` — Echo State Network as a forecasting model
   * `POD_ESN` — POD dimensionality reduction + ESN forecaster

   Dimensionality-reduction & algorithm tools [`romda.tools`](src/romda/tools/)
   * `POD`, `SPOD` — (Spectral) Proper Orthogonal Decomposition (Sieber 2016, Towne 2018)
   * `EchoStateNetwork` — reservoir computing building block

   Bias estimators [`romda.bias_estimators`](src/romda/bias_estimators/)
   * Echo State Network
   * Constant bias
   * Drift-linear bias
   * No bias (unbiased limit)


---
## 📂 Structure
```
.
├── data/                        # Dataset files
├── docs/                        # Documentation site sources + media used in the notebooks
├── results/                     # Generated results
├── scripts/                     # Runnable files (mains, tutorials)
│   ├── mains/
│   └── tutorials/
├── src/romda/                   # Source code (the romda package)
│   ├── data_assimilation.py     # EnKF, EnSRKF, rBA-EnKF
│   ├── ensemble.py              # Ensemble wrapper for forecasting and DA
│   ├── observations.py          # Truth / observation generation
│   ├── utils.py
│   ├── bias_estimators/         # Bias model classes
│   │   ├── bias.py              # Bias base class
│   │   ├── esn.py               # ESN_bias
│   │   ├── constantbias.py      # ConstantBias, NoBias
│   │   └── driftlinear.py       # DriftLinearBias
│   ├── config/
│   │   └── esn_config.py
│   ├── models/                  # Model layer
│   │   ├── model.py             # Model base class
│   │   ├── history.py           # HistoryTracker
│   │   ├── integrator.py        # IVPIntegrator, DiscreteIntegrator, ...
│   │   ├── physical/            # Physical models
│   │   └── data_driven/         # ESN_model, POD_ESN
│   ├── plotting/                # Visualization helpers
│   └── tools/                   # Building blocks (no forecaster)
│       ├── autoencoders.py      # AE, CAE, POD, SPOD classes
│       ├── pod_spod.py          # POD / SPOD algorithms
│       └── esn_core.py          # EchoStateNetwork reservoir
│
├── tests/                       # Unit and integration test suite
├── test_tutorials.py            # Notebook execution tests
├── mkdocs.yml                   # Documentation site configuration
├── pyproject.toml               # Package setup
└── README.md                    # This file
```


---

## 📚 Main publications from this repository

##### Journal papers

- [x] Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic instabilities. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2024.1052) |  🏷️ [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0). 
- [x] Nóvoa, Racca & Magri (2023). Inferring unknown unknowns. Computer Methods in Applied Mechanics and Engineering. [Published paper](https://doi.org/10.1016/j.cma.2023.116502) | [_Legacy_ repository](https://github.com/MagriLab/rBA-EnKF).
- [x] Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2022.653) | [_Legacy_ repository](https://github.com/MagriLab/Real-time-TA-DA).

##### Conference papers and proceedings
- [x] Nóvoa & Magri (2025). Online model learning with data-assimilated reservoir computers. [Preprint](https://doi.org/10.48550/arXiv.2504.16767) | 🏷️ [v1.1](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.1).
- [x] Nóvoa & Magri (2024). Real-time digital twins of multiphysics and turbulent flows. [Paper](https://web.stanford.edu/group/ctr/ctrsp24/ii11_NOVOA.pdf).
- [x] Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. In_ 51st International Congress and Exposition on Noise Control Engineering. [Paper](https://az659834.vo.msecnd.net/eventsairwesteuprod/production-inconference-public/808b4f8c38f944d188db8a326a98c65c). | [_Legacy_ repository](https://github.com/MagriLab/IN22-Bias-aware-TADA).

##### PhD thesis
- [x] Nóvoa (2024). Real-time data assimilation in nonlinear dynamcal systems. University of Cambridge. [Thesis](https://doi.org/10.17863/CAM.113001). 

<!-- ##### Conference presentations _(incomplete list)_
- **APS-DFD 2024, Salt Lake City:** [Abstract](https://meetings.aps.org/Meeting/DFD24/Session/C02.14) | [Poster](https://github.com/user-attachments/files/17966063/APS-poster-final-version.pdf).
- **APS-DFD 2023, Washington DC:** [Abstract](https://meetings.aps.org/Meeting/DFD23/Session/L30.8).
- **EFMC14 2022, Athens:** [Abstract](https://euromech.org/conferences/proceedings.htm).
- **APS-DFD 2022, Phoenix:** [Abstract](https://meetings.aps.org/Meeting/DFD22/Session/G12.4). -->

--- 
## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request. For questions or collaborations, please reach out to [A. Nóvoa](https://andreanovoa.github.io/).
