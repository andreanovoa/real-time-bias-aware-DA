# Tutorials

The tutorial notebooks below build up from the basic classes to full real-data digital
twins, numbered by topic. The source lives in
[`scripts/tutorials`](https://github.com/andreanovoa/real-time-bias-aware-DA/tree/main/scripts/tutorials).
Topics 0 and 1 are rendered inline (with saved outputs); topics 2–4 link out to
GitHub for now. Two arcs run through them: DA concepts (1) applied to physical models
(2), and reduced-order-model concepts (3) combined with DA in real time (4).

## 0. How the repository works

| Notebook | Contents |
| --- | --- |
| [Class Model](tutorials/0_How_to_repo/00_Class_Model.md) | The `Model` base class, integrators and state history. |
| [Class EchoStateNetwork](https://github.com/andreanovoa/EchoStateNetwork/blob/master/tutorials/01_echo_state_network.ipynb) | The reservoir-computing building block (moved to the [echostatenetwork](https://andreanovoa.github.io/EchoStateNetwork/) package). |
| [Class ESN_model](tutorials/0_How_to_repo/02_Class_ESN_model.md) | An ESN as a forecast model. |
| [Class Observations](tutorials/0_How_to_repo/04_Class_Observations.md) | Truth generation, noise and manual biases. |
| [Class Bias](tutorials/0_How_to_repo/05_Class_Bias.md) | Training and running an ESN bias estimator. |
| [Class Estimator](tutorials/0_How_to_repo/06_Class_Estimator.md) | The `Estimator` hierarchy: instantiating a filter and running the forecast step. |

## 1. Introduction to real-time DA

| Notebook | Contents |
| --- | --- |
| [Real-time DA intro](tutorials/1_Introduction_to_real-time_DA/10_real-time-DA_intro.md) | Bayesian estimation (MAP, univariate example) and the classical Kalman filter on a linear model. |
| [Ensemble DA intro](tutorials/1_Introduction_to_real-time_DA/11_ensemble-DA_intro.md) | The EnKF: twin experiment on the Van der Pol model, and Monte Carlo convergence to the KF. |
| [Augmented formulation](tutorials/1_Introduction_to_real-time_DA/13_real-time-DA-augmented-state.md) | Combined state and parameter estimation. |
| [Chaos and DA (Lorenz 63)](tutorials/1_Introduction_to_real-time_DA/15_real-time-DA_Lorenz63.md) | Ensemble DA on a chaotic system. |
| [Model-bias-aware DA](tutorials/1_Introduction_to_real-time_DA/16_Bias-aware_real-time_DA.md) | The r-EnKF with an ESN bias estimator on the Van der Pol model. |

## 2. Real-time DA in thermoacoustics

| Notebook | Contents |
| --- | --- |
| [Rijke LOM](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/20_Rijke_LOM.ipynb) | The Rijke-tube low-order model (Galerkin method). |
| [TADA Rijke twin](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/21_TADA_Rijke_twin.ipynb) | Twin state/parameter estimation with the EnKF. |
| [TABADA Rijke (CMAME)](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/22_TABADA_Rijke_CMAME.ipynb) | Bias-aware DA with an ESN bias estimator. |
| [Azimuthal LOM](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/23_TA_azimuthal_LOM.ipynb) | The annular-combustor low-order model. |
| [Azimuthal data](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/24_TA_azimuthal_data.ipynb) | Exploring the experimental annular data.* |
| [TABADA annular raw](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/25_TABADA_annular_raw.ipynb) | A real-data digital twin of an annular combustor.* |

## 3. Introduction to reduced-order models

| Notebook | Contents |
| --- | --- |
| [POD / SPOD](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/3_Introduction_to_ROMs/31_POD_SPOD.ipynb) | POD, Sieber SPOD and Towne SPOD on the cylinder wake.* |
| [POD-ESN on the cylinder wake](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/3_Introduction_to_ROMs/35_POD_ESN_cylinder.ipynb) | The packaged `POD_ESN` reduced-order model at scale.* |

## 4. Real-time DA on reduced-order models

| Notebook | Contents |
| --- | --- |
| [Real-time DA with POD-ESN](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/4_Real-time_DA_ROMs/44_DA_POD_ESN_cylinder.ipynb) | Assimilating sparse sensors into the POD-ESN wake model.* |


\* These notebooks download their dataset from [Zenodo](https://zenodo.org/) on first use ([annular data](https://zenodo.org/records/15609832), [wake data](https://zenodo.org/records/15623774)).

## Running the tutorials as tests

The notebook runner executes all tutorials headlessly:

```bash
python -m pytest test_tutorials.py                       # all folders
SUBFOLDERS='["0","1"]' python -m pytest test_tutorials.py  # only folders 0 and 1
NB_TEST_QUICK=1 python -m pytest test_tutorials.py       # skip training-heavy notebooks
```
