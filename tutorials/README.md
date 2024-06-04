# Intro to data assimilation (DA)  
This folder includes a collection of interactive (Jupyter notebook) tutorials introductory to real-time data assimilation with ensemble Kalman filters. 

***
## Getting started

From the terminal:

1. Clone repository: ```git clone https://github.com/andreanovoa/real-time-bias-aware-DA```
2. Access the repository: ```cd path-to-repo ```
3. Create virtual environment wih anaconda and activate environment:
   ```
   conda create --yes --name da-env python=3.9
   conda activate da-env
   ```
4. Install the requirements of the main repository and the additional requirements for the tutorials:
   ```
   pip install -r requirements.txt
   pip install -r tutorials/requirements.txt
   ```
5. Access the tutorials folder and launch the jupyter notebooks:
   ```
   cd tutorials
   jupyter lab
   ```

****
*The tutorials are brand new - don't hesitate to submit issues or pull requests!*

***
## Where to start?
Each tutorial file name includes a two-digit code, which indicates the category.

### Repo-specific tutorials [0x]
- [x] 00 - Class Model
- [x] 03 - Create truth

[//]: # (- [ ] 01 - Class Bias)


### Data Assimilation tutorials [1x]
* Bias-unaware DA
  1) [x] 10 - Introduction to DA and twin experiment on the Van der Pol model
  2) [x] 11 - Twin experiment on a chaotic attractor of the Lorenz63 model
  
* Bias-unaware DA
  1) [x] 12 - Introduction to bias-aware DA and twin experiment on the Van der Pol oscillator
 
### Echo state network tutorials [2x]
- [x] 21 - ESN adaptability with data assimilation

 
### Data Assimilation on azimuthal thermoacosustics [3x]

* [x] 30 - Low order model of azimuthal thermoacoustics
* [x] 31 - Experimental data visualization 
* [x] 32 - DA on experimental data: ideal scenario
* [x] 32 - DA on experimental data: realistic scenario


[//]: # ()
[//]: # (* Longitudinal thermoacoustics &#40;Rijke tube&#41; )
[//]: # (  * [ ] 32 - Low order model )
[//]: # (  * [ ] 33 - Higher-order model data visualization)

