## Getting started

From the terminal:

1. Clone repository: ```git clone https://github.com/andreanovoa/real-time-bias-aware-DA```
2. Access the repository: ```cd path-to-repo ```
3. Create virtual environment with anaconda/miniconda and activate environment:
   ```
   conda create --yes --name da-env python=3.9
   conda activate da-env
   ```
4. Install the required packages for the repository:
   ```
   pip install -r requirements.txt
   ```
5. Access the tutorials folder and launch the jupyter notebooks:
   ```
   cd tutorials
   jupyter lab
   ```
Note: if the environment ```da-env``` is not listed in jupyter lab run the following in the terminal and restart the notebook:
```
ipython kernel install --user --name=<da-env>
```



***
***

## Structure of the tutorials
Each tutorial file name starts with a two-digit code, which indicates the category. 
If the tutorial is not checked is because it is under development. Keep tuned!

### Repo-specific tutorials [0x]
- [x] 00 - Class Model
- [x] 01 - Class EchoStateNetwork
- [x] 02 - Class ESN_model -- combining 00 and 01
- [x] 03 - POD with Modulo
- [x] 04 - Create truth


### Data Assimilation tutorials [1x]

* [x] 10 - Introduction to real-time DA from a Bayesian perspective. State estimation only.
* [x] 11 - Introduction to combined state and parameter estimation via augmented formulaiton.
* [x] 12 - Twin experiment on the Lorenz63
* [x] 13 - Introduction to bias-aware DA with a twin experiment on Van der Pol model  

 
### Data Assimilation on thermoacoustics [2x]

* Longitudinal thermoacoustics ([Nóvoa et al. 2023](https://doi.org/10.1016/j.cma.2023.116502))
    * [x] 20 - Low order model of longitudinal thermoacoustics: Rijke tube model, Galerkin method
    * [x] 21 - DA twin experiment on the Rijke tube. 
    * [x] 22 - Bias-aware DA twin experiment on Rijke tube model with added bias.

* Azimuthal thermoacoustics  ([Nóvoa et al. 2024](https://doi.org/10.1017/jfm.2024.1052))
    * [x] 23 - Low order model of azimuthal thermoacoustics
    * [x] 24 - Experimental data visualization 
    * [x] 25 - Real-time digital twin of raw experimental data
    <!-- * [ ] 25 - Generalizability of the real-time digital twin  -->


###  POD-ESN  on a cylinder flow [3x] ([Nóvoa & Magri 2025](http://arxiv.org/abs/2504.16767))
* [x] 30 - Introduction to the POD-ESN model and the dataset
* [ ] 31 - Online adaptation of the ESN via data assimilation
 
<!-- ###  Data Assimilation on echo state networks [4x]
* [ ] 40 - ESN-DA on Lorenz63
* [ ] 41 - POD-ESN-DA on cylinder flow
* [ ] 42 - POD-ESN-DA on turbulent wake -->

****
*Please don't hesitate to submit issues or pull requests!*














