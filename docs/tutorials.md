# Tutorials

The [`scripts/tutorials`](https://github.com/andreanovoa/real-time-bias-aware-DA/tree/main/scripts/tutorials)
folder contains Jupyter notebooks that build up from the basic classes to full
real-data digital twins. They are numbered by topic:

## 0 — How the repository works

| Notebook | Contents |
| --- | --- |
| [00 — Class Model](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/00_Class_Model.ipynb) | The `Model` base class, integrators and state history |
| [01 — Class EchoStateNetwork](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/01_Class_EchoStateNetwork.ipynb) | The reservoir-computing building block |
| [02 — Class ESN_model](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/02_Class_ESN_model.ipynb) | An ESN as a forecast model |
| [03 — Class POD](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/03_Class_POD.ipynb) | POD and SPOD on the cylinder-wake data* |
| [04 — Class Observations](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/04_Class_Observations.ipynb) | Truth generation, noise and manual biases |
| [05 — Class Bias](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/0_How_to_repo/05_Class_Bias.ipynb) | Training and running an ESN bias estimator |

## 1 — Introduction to real-time data assimilation

| Notebook | Contents |
| --- | --- |
| [10 — Real-time DA intro](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/1_Introtuction_to_real-time_DA/10_real-time-DA_intro.ipynb) | Bayesian estimation and the Kalman filter from scratch |
| [11 — Augmented state](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/1_Introtuction_to_real-time_DA/11_real-time-DA-augmented-state.ipynb) | Combined state and parameter estimation |
| [12 — Lorenz 63](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/1_Introtuction_to_real-time_DA/12_real-time-DA_Lorenz63.ipynb) | Ensemble DA on a chaotic system |
| [13 — Bias-aware DA intro](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/1_Introtuction_to_real-time_DA/13_bias-aware-DA-intro.ipynb) | The r-EnKF with an ESN bias estimator on the Van der Pol model |

## 2 — Real-time DA in thermoacoustics

| Notebook | Contents |
| --- | --- |
| [20 — Rijke LOM](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/20_Rijke_LOM.ipynb) | The Rijke-tube low-order model |
| [21 — TADA Rijke twin](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/21_TADA_Rijke_twin.ipynb) | Twin state/parameter estimation with the EnKF |
| [22 — TABADA Rijke (CMAME)](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/22_TABADA_Rijke_CMAME.ipynb) | Bias-aware DA with an ESN bias estimator |
| [23 — Azimuthal LOM](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/23_TA_azimuthal_LOM.ipynb) | The annular-combustor low-order model |
| [24 — Azimuthal data](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/24_TA_azimuthal_data.ipynb) | Exploring the experimental annular data* |
| [25 — TABADA annular raw](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/2_Real-time_DA_Thermoacoustics/25_TABADA_annular_raw.ipynb) | A real-data digital twin of an annular combustor* |

## 3 — Real-time DA with POD-ESN reduced-order models

| Notebook | Contents |
| --- | --- |
| [30 — POD / SPOD intro](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/3_Real-time-DA_POD-ESN/30_POD_SPOD_intro.ipynb) | POD, Sieber SPOD and Towne SPOD on the cylinder wake* |
| [31 — ESN-POD tutorial](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/3_Real-time-DA_POD-ESN/31_esn_pod_tutorial.ipynb) | Building a POD-ESN reduced-order model* |
| [32 — Real-time DA with POD-ESN](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/3_Real-time-DA_POD-ESN/32_real-time-DA_ESN-POD_cylinder.ipynb) | Assimilating sparse sensors into a POD-ESN model* |

\* These notebooks download their dataset from [Zenodo](https://zenodo.org/) on first
use ([annular data](https://zenodo.org/records/15609832),
[wake data](https://zenodo.org/records/15623774)).

## Running the tutorials as tests

The notebook runner executes all tutorials headlessly:

```bash
python -m pytest test_tutorials.py                       # all folders
SUBFOLDERS='["0","1"]' python -m pytest test_tutorials.py  # only folders 0 and 1
NB_TEST_QUICK=1 python -m pytest test_tutorials.py       # skip training-heavy notebooks
```
