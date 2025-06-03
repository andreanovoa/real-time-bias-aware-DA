# Real-time bias-aware data assimilation


Open-source repository for advanced data assimilation techniques, focusing on bias correction in real-time thermoacoustic systems and beyond. 


--- 

## 🚀 Getting started
1. Install the requirements [here](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/requirements.txt).
2. Checkout the [Tutorials folder](https://github.com/andreanovoa/real-time-bias-aware-DA/tree/main/tutorials), which includes several jupyter notebooks aiming to ease the understanding of the repository.


## 🌟 What is available?
   Data assimilation methods [`essentials.DA`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/DA.py):
   * EnKF - ensemble Kalman filter
   * EnSRKF - ensemble square-root Kalman filter
   * rBA-EnKF - regularized bias-aware EnKF
   
   Physical models [`essentials.models_physical`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/models_physical.py):
   * Rijke tube model (dimensional with Galerkin projection)
   * Van der Pols
   * Lorenz63
   * Azimuthal thermoaocustics model
    
   Bias estimators[`essentials.bias_models`](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/essentials/bias_models.py):
   * Echo State Network
   * NoBias

 <!---
 --- 

## 📂 Structure
--->


---

## 📚 Main publications and presentations

##### Journal papers and proceedings

- [x] Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic instabilities. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2024.1052) |  🏷️ [v1.0](https://github.com/andreanovoa/real-time-bias-aware-DA/releases/tag/v1.0). 
- [x] Nóvoa, Racca & Magri (2023). Inferring unknown unknowns. Computer Methods in Applied Mechanics and Engineering. [Published paper](https://doi.org/10.1016/j.cma.2023.116502) | [_Legacy_ repository](https://github.com/MagriLab/rBA-EnKF)
- [x] Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. Journal of Fluid Mechanics. [Published paper](https://doi.org/10.1017/jfm.2022.653) | [_Legacy_ repository](https://github.com/MagriLab/Real-time-TA-DA)

##### PhD theses
- [x] Nóvoa (2024). Real-time data assimilation in nonlinear dynamcal systems. University of Cambridge. [Thesis](https://doi.org/10.17863/CAM.113001). 

##### Conference papers
- [x] Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. In_ 51st International Congress and Exposition on Noise Control Engineering. [Paper](https://az659834.vo.msecnd.net/eventsairwesteuprod/production-inconference-public/808b4f8c38f944d188db8a326a98c65c). | [_Legacy_ repository](https://github.com/MagriLab/IN22-Bias-aware-TADA)

##### Conference presentations _(incomplete list)_
- **APS-DFD 2024, Salt Lake City:** [Abstract](https://meetings.aps.org/Meeting/DFD24/Session/C02.14) | [Poster](https://github.com/user-attachments/files/17966063/APS-poster-final-version.pdf)
- **APS-DFD 2023, Washington DC:** [Abstract](https://meetings.aps.org/Meeting/DFD23/Session/L30.8)
- **EFMC14 2022, Athens:** [Abstract](https://euromech.org/conferences/proceedings.htm)
- **APS-DFD 2022, Phoenix:** [Abstract](https://meetings.aps.org/Meeting/DFD22/Session/G12.4)

--- 
## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request. For questions or collaborations, please reach out to [A. Nóvoa](https://scholar.google.com/citations?user=X0TjtAgAAAAJ&hl=en).
