# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from Util import createObservations
from Ensemble import createEnsemble
from DA import dataAssimilation
import TAModels
import Bias

# -------------------------------------------------------- #
save_ = False  # Save simulation? If false, plot results
results_folder = 'results/'

# %% =====================================  CREATE OBSERVATIONS ===================================== #
# true_model = 'Truth_wave.mat'
# true_params = {}
# true_model = TAModels.Rijke
# true_params = {'law': 'sqrt', 'beta': 3E6, 'tau': 1.5E-3}
true_model = TAModels.VdP
true_params = {'law': 'tan',
               'beta': 4e6,
               'kappa': 1e6
               }
y_true, t_true, name_truth = createObservations(true_model, true_params, t_max=3.)

# Define the observations
t_start = 1.
t_stop = 2.5
kmeas = 50  # number of time steps between observations

dt_true = t_true[1] - t_true[0]
obs_idx = np.arange(round(t_start / dt_true), round(t_stop / dt_true) + 1, kmeas)
t_obs = t_true[obs_idx]
obs = y_true[obs_idx]

# number of analysis steps without bias in KF or parameter estimation
num_DA_blind = int(0.0 / t_obs[1] - t_obs[0])
num_SE_only = int(0.0 / t_obs[1] - t_obs[0])

# %% ============================== SELECT TA & BIAS MODELS AND FILTER PARAMETERS ============================== #
forecast_model = TAModels.VdP
biasType = Bias.ESN  # Bias.ESN  # Bias.ESN # None

filt = 'EnKF'  # 'EnKFbias' 'EnKF' 'EnSRKF'
k = 1.  # 1E5  # gamma in the equations. Weight of bT Wbb b

# model_params = {'law': 'sqrt', 'dt': dt_true, 'beta': 3E6, 'tau': 1.5E-3}
model_params = {'law': 'cubic',
                'dt': dt_true,
                'beta': 4e6,
                'kappa': 1e6
                }

filter_params = {'m': 10,  # Dictionary of DA parameters
                 'est_p': ['kappa', 'beta'],  # ['beta', 'tau'],  # #'beta', 'tau'
                 'bias': biasType,
                 'std_psi': 0.1,
                 'std_a': 0.1,
                 'est_b': False,
                 'getJ': True,
                 'inflation': 1.01,
                 'num_DA_blind': int(0.0 / t_obs[1] - t_obs[0]),  # num of obs to start accounting for the bias
                 'num_SE_only': int(0.0 / t_obs[1] - t_obs[0])  # num of obs to start parameter estimation
                 }

if biasType is not None:
    if biasType.name == 'ESN':
        # Compute reference bias. Create an ensemble of training data
        rng = np.random.default_rng(0)

        train_params = {'m': 10,
                        'std_a': 0.2,
                        'est_p': filter_params['est_p'],
                        'law': model_params['law'],
                        'beta': model_params['beta'],  # 'beta': model_params['beta'],
                        'kappa': model_params['kappa']  # 'tau': model_params['tau']
                        }

        ref_ens = createEnsemble(forecast_model, train_params, train_params)

        # name_train = './data/Truth_{}_{}_beta{:.1e}_tau{:.1e}_tmax-{:.2}_std{:.2}_m{}'.format(ref_ens.name,
        # ref_ens.law, ref_ens.beta, ref_ens.tau, t_true[-1], ref_ens.std_a, ref_ens.m)
        name_train = './data/Truth_{}_{}_beta{}_kappa{}_tmax-{:.2}_std{:.2}_m{}'.format(ref_ens.name, ref_ens.law,
                                                                                        ref_ens.beta, ref_ens.kappa,
                                                                                        t_true[-1], ref_ens.std_a,
                                                                                        ref_ens.m)
        if os.path.isfile(name_train):
            print('Loading Reference solution(s)')
            with open(name_train, 'rb') as f:
                ref_ens = pickle.load(f)
        else:
            psi, t = ref_ens.timeIntegrate(Nt=len(t_true) - 1)
            ref_ens.updateHistory(psi, t)
            with open(name_train, 'wb') as f:
                pickle.dump(ref_ens, f)

        plt.figure()
        plt.plot(y_true)
        plt.plot(ref_ens.getObservableHist()[0][:, 0])
        plt.show()

        biasData = np.expand_dims(y_true, -1) - ref_ens.getObservableHist()[0]  # [Nt x Nmic x Ntrain]
        biasData = np.append(biasData, ref_ens.hist[:, -len(ref_ens.est_p):], axis=1)

        # provide data for washout before first observation
        i1 = int(np.where(t_true == t_obs[0])[0]) - kmeas
        i0 = i1 - int(0.1 / dt_true)

        # create bias dictionary
        bias_params = {'trainData': biasData[int(1. / dt_true):],  # remove transient - steady state solution
                       'washout_obs': y_true[i0:i1 + 1],
                       'washout_t': t_true[i0:i1 + 1],
                       'train_TAparams': train_params,
                       'filename': name_truth + '_' + name_train.split('Truth_')[-1],
                       'N_wash': 50,
                       'upsample': 5,
                       'N_units': 100
                       }

    elif biasType.name == 'LinearBias':
        bias_params = {'b1': 0.05,
                       'b2': 0.0}
        # # Manually add bias to observations
        # b1 = 0.2
        # b2 = 0.
        # def bias(y):
        #     return b1 * y + b2
        #
        # biasData = bias(y_true)
        # y_true += biasData
        # obs += bias(obs)
    else:
        raise ValueError('Bias model not defined')
    bias_params['k'] = k
    bias_name = biasType.name
else:
    bias_name = 'None'
    bias_params = {}

# ===========================================  INITIALISE ENSEMBLE  ========================================== #
model_params['beta'] *= 2
model_params['kappa'] *= .5
ensemble = createEnsemble(forecast_model, filter_params, model_params, bias_params)

for _ in [1.]:  # np.linspace(0, 2, 21):
    filter_ens = ensemble.copy()  # copy the initialised ensemble
    filter_ens.bias.k = k

    # ======================================  PERFORM DATA ASSIMILATION ====================================== #

    filter_ens = dataAssimilation(filter_ens, obs, t_obs, method=filt)

    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    Nt_extra = 0
    if filter_ens.hist_t[-1] < t_true[-1]:
        Nt_extra = int(min((t_true[-1] - filter_ens.hist_t[-1]), 0.2) / filter_ens.dt) + 1
        psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
        filter_ens.updateHistory(psi, t)
        if filter_ens.bias is not None:
            y = filter_ens.getObservableHist(Nt_extra)[0]
            a = np.mean(filter_ens.hist[-1, -len(filter_ens.est_p):, :], axis=-1)
            b, t_b = filter_ens.bias.timeIntegrate(Nt=Nt_extra, y=[y, a])
            filter_ens.bias.updateHistory(b, t_b)

    # =========================================== SAVE DATA & PLOT =========================================== #
    truth = dict(y=y_true, t=t_true, name=name_truth, t_obs=t_obs, p_obs=obs)
    parameters = dict(kmeas=kmeas, filt=filt, biasType=biasType, forecast_model=forecast_model,
                      true_model=true_model, num_DA=len(t_obs), Nt_extra=Nt_extra)
    if save_:
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        filename = '{}{}_Truth{}_Forecast{}_Bias{}_k{:.2}'.format(results_folder, filt, name_truth,
                                                                  forecast_model.name, bias_name, k)
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
            pickle.dump(truth, f)
            pickle.dump(filter_ens, f)

    exec(open("post_process.py").read(), {'parameters': parameters,
                                          'filter_ens': filter_ens,
                                          'truth': truth})
