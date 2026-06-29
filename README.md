# Real-time bias-aware data assimilation


Open-source repository for advanced data assimilation techniques, focusing on bias correction in real-time thermoacoustic systems and beyond. 


--- 

## 🚀 Getting started

1. **Prerequisites**

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. **Clone the Repository**
```
git clone https://github.com/andreanovoa/real-time-bias-aware-DA
cd yourproject
```

3. Install the package in editable mode 
```
pip install -e . --use-pep517 # use .[dev] if running tests
```

4. (Optional) Run the tests and save the output onto a text file
```
SUBFOLDERS='["0","1"]' pytest -s --log-cli-level=INFO test_tutorials.py # >> test_output.txt 2>&1 
```

Checkout the [Tutorials folder](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials), which includes several jupyter notebooks aiming to ease the understanding of the repository.


---


## 🌟 What is available?
   Data assimilation methods [`data_assimilation`](src/data_assimilation.py)
   * EnKF — ensemble Kalman filter
   * EnSRKF — ensemble square-root Kalman filter
   * rBA-EnKF — regularized bias-aware EnKF

   Physical models [`models/physical`](src/models/physical/)
   * Rijke tube model (dimensional with Galerkin projection)
   * Van der Pol oscillator
   * Lorenz 63
   * Lorenz 96
   * Azimuthal thermoacoustics model
   * Kuramoto-Sivashinsky equation

   Data-driven models [`models/data_driven`](src/models/data_driven/)
   * `ESN_model` — Echo State Network as a forecasting model
   * `POD_ESN` — POD dimensionality reduction + ESN forecaster

   Dimensionality-reduction & algorithm tools [`tools`](src/tools/)
   * `POD`, `SPOD` — (Spectral) Proper Orthogonal Decomposition (Sieber 2016, Towne 2018)
   * `EchoStateNetwork` — reservoir computing building block

   Bias estimators [`bias_estimators`](src/bias_estimators/)
   * Echo State Network
   * Constant bias
   * Drift-linear bias


---
## 📂 Structure
```
.
├── data/                        # Dataset files
├── docs/                        # Documents and media used in the notebooks
├── results/                     # Generated results
├── scripts/                     # Runnable files (mains, tutorials)
│   ├── mains/
│   ├── post_process/
│   └── tutorials/
├── src/                         # Source code
│   ├── data_assimilation.py     # EnKF, EnSRKF, rBA-EnKF
│   ├── ensemble.py
│   ├── bias.py
│   ├── bias_estimators/         # Bias model classes
│   │   ├── esn.py
│   │   ├── constantbias.py
│   │   └── driftlinear.py
│   ├── observations.py
│   ├── utils.py
│   ├── config/
│   │   └── esn_config.py
│   │
│   ├── models/                  # Model layer
│   │   ├── model.py             # Model base class
│   │   ├── history.py           # HistoryTracker mixin
│   │   ├── integrator.py        # IVPIntegrator, DiscreteIntegrator, ...
│   │   ├── physical/            # Physical models
│   │   │   ├── annular.py
│   │   │   ├── kuramoto_sivashinsky.py
│   │   │   ├── lorenz63.py
│   │   │   ├── lorenz96.py
│   │   │   ├── rijke.py
│   │   │   └── van_der_pol.py
│   │   └── data_driven/         # Data-driven models (autoencoder) + forecaster
│   │       ├── esn.py           # ESN_model
│   │       └── pod_esn.py       # POD_ESN
│   │
│   └── tools/                   # Building blocks (no forecaster)
│       ├── autoencoders.py      # AE, CAE, POD, SPOD classes
│       ├── pod_spod.py          # POD, SPOD algorithms needed in autoencoders.py
│       └── esn_core.py          # EchoStateNetwork reservoir
│
├── tests_tutorials.py           # Tutorial unit tests
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
