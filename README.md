# Bias-aware data assimilation

*This work was developed during the PhD of [A. Nóvoa](https://scholar.google.com/citations?user=X0TjtAgAAAAJ&hl=en). The thesis is publicly avaiable [here](https://doi.org/10.17863/CAM.113001).*<br> 

## Main publications and presentations

##### Journal papers


- [x] [A real-time digital twin of azimuthal thermoacoustic instabilities](https://arxiv.org/abs/2404.18793). Journal of Fluid Mechanics (_To appear_)
- [x] [Inferring unknown unknowns](https://doi.org/10.1016/j.cma.2023.116502). Computer Methods in Applied Mechanics and Engineering (2023).
   -  The _legacy_ repository used for this paper can be found in [MagriLab/rBA-EnKF](https://github.com/MagriLab/rBA-EnKF)
- [x] [Real-time thermoacoustic data assimilation](https://doi.org/10.1017/jfm.2022.653). Journal of Fluid Mechanics (2022).
   -  The _legacy_ repository used for this paper can be found in [MagriLab/Real-time-TA-DA](https://github.com/MagriLab/Real-time-TA-DA)
##### Conference papers

- [x] [Bias-aware thermoacoustic data assimilation](https://az659834.vo.msecnd.net/eventsairwesteuprod/production-inconference-public/808b4f8c38f944d188db8a326a98c65c). _In_ 51st International Congress and Exposition on Noise Control Engineering (2022).
   -  The _legacy_ repository used for this paper can be found in [MagriLab/IN22-Bias-aware-TADA](https://github.com/MagriLab/IN22-Bias-aware-TADA)
##### Conference presentations _(incomplete)_
- APS-DFD 2024 Salt Lake City [Abstract](https://meetings.aps.org/Meeting/DFD24/Session/C02.14) and [interactive flash talk poster](https://github.com/user-attachments/files/17966063/APS-poster-final-version.pdf).
- APS-DFD 2023 Washington DC [Abstract](https://meetings.aps.org/Meeting/DFD23/Session/L30.8).
- EFMC14 2022 Athens [Abstract](https://euromech.org/conferences/proceedings.htm)
- APS-DFD 2022 Phoenix [Abstract](https://meetings.aps.org/Meeting/DFD22/Session/G12.4).



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

