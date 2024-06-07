# Intro to data assimilation (DA)  
This folder includes a collection of interactive (Jupyter notebook) tutorials to ease the introduction to this repository.


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
*The tutorials are brand new - please don't hesitate to submit issues or pull requests!*

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
  2) [x] 11 - Twin experiment on the Lorenz63
  
* Bias-unaware DA
  1) [x] 12 - Bias-aware DA twin experiment on Van der Pol model
  2) [ ] 13 - Bias-aware DA twin experiment on Rijke tube model
 
[//]: # (### Echo state network tutorials [2x])
[//]: # (- [x] 21 - ESN adaptability with data assimilation)
[//]: # (- [ ] 20 - ESN as model bias estimator and multi-parameter training approach)

### Data Assimilation on longitudinal thermoacoustics [2x]

* [x] 20 - Low order model of longitudinal thermoacoustics: Rijke tube
* [ ] 21 - Higher order model visualization (manually added bias)
* [ ] 22 - Regularized bias-aware DA

 
### Data Assimilation on azimuthal thermoacosustics [3x]

* [x] 30 - Low order model of azimuthal thermoacoustics
* [x] 31 - Experimental data visualization 
* [x] 32 - DA on experimental data: ideal scenario
* [x] 32 - DA on experimental data: realistic scenario
