# Bias-aware data assimilation

*This work was developed during the PhD of [A. Nóvoa](https://scholar.google.com/citations?user=X0TjtAgAAAAJ&hl=en).*<br>

## Main publications
- [x] [Real-time thermoacoustic data assimilation](https://doi.org/10.1017/jfm.2022.653). Journal of Fluid Mechanics (2022).
- [x] [Inferring unknown unknowns](https://doi.org/10.1016/j.cma.2023.116502). Computer Methods in Applied Mechanics and Engineering (2023).
- [x] [A real-time digital twin of azimuthal thermoacoustic instabilities](https://arxiv.org/abs/2404.18793). Journal of Fluid Mechanics (_To appear_)


## Getting started
1. Install the requirements [here](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/requirements.txt).
2. Checkout the [Tutorials folder](https://github.com/andreanovoa/real-time-bias-aware-DA/tree/main/tutorials), which includes several jupyter notebooks aiming to ease the understanding of the repository.
   

## What is available?
   [Data assimilation methods](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/DA.py):
   * EnKF - ensemble Kalman filter
   * EnSRKF - ensemble square-root Kalman filter
   * rBA-EnKF - regularized bias-aware EnKF
   
   [Physical models](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/physical_models.py):
   * Rijke tube model (dimensional with Galerkin projection)
   * Van der Pols
   * Lorenz63
   * Azimuthal thermoaocustics model
    
   [Bias estimators](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/bias_models.py):
   * Echo State Network
   * NoBias

