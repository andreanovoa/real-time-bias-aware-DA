# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""
import os
import numpy as np
import pylab as plt
import pickle
from Util import createObservations, CR
from Ensemble import createEnsemble
from DA import dataAssimilation
import TAModels
import Bias
import matplotlib as mpl

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

rng = np.random.default_rng(0)

# -------------------------------------------------------- #
save_ = True  # Save simulation? If false, plot results
results_folder = 'results/VdP_100_Notwin_betazeta_uniform_std0.2_B/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)

# %% =====================================  CREATE OBSERVATIONS ===================================== #
true_model = TAModels.VdP
true_params = {'law': 'tan',
               'beta': 80.,     # forcing
               'zeta': 60.,     # damping
               'kappa': 3.4,    # nonlinearity
               'omega': 2 * np.pi * 120.
               }
y_true, t_true, name_truth = createObservations(true_model, true_params, t_max=5.)
# Manually add bias
b_true = np.cos(y_true)
y_true += b_true
name_truth += '_+cosy'

# Define the observations
t_start = 2.5
t_stop = 4.5
kmeas = 25  # number of time steps between observations

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

filt = 'EnKFbias'  # 'EnKFbias' 'EnKF' 'EnSRKF'
ks = np.linspace(0., 70., 71)  # [10.]  # 1E5  # gamma in the equations. Weight of bT Wbb b

model_params = {'law': true_params['law'],
                'dt': dt_true,
                'zeta': true_params['zeta'] * 0.9,
                'beta': true_params['beta'] * 1.1,
                'kappa': true_params['kappa'],
                'omega': true_params['omega']
                }
filter_params = {'m': 10,  # Dictionary of DA parameters
                 'est_p': ['beta', 'zeta'],  # ['beta', 'tau'],  # #'beta', 'tau'
                 'bias': biasType,
                 'std_psi': 0.2,
                 'std_a': 0.2,
                 'est_b': False,
                 'getJ': True,
                 'inflation': 1.01,
                 'num_DA_blind': int(0.0 / t_obs[1] - t_obs[0]),  # num of obs to start accounting for the bias
                 'num_SE_only': int(0.0 / t_obs[1] - t_obs[0])  # num of obs to start parameter estimation
                 }

train_params = model_params.copy()
train_params = {'m': 100,
                'std_a': 0.3,
                'std_psi': 0.3,
                'est_p': ['zeta', 'beta'],
                'alpha_distr': 'uniform',
                }
ESN_params = {'N_wash': 50,
              'upsample': 5,
              'N_units': 200,
              't_train': 1.0,
              'train_TAparams': train_params
              }

if biasType is not None:
    if biasType.name == 'ESN':
        # Compute reference bias. Create an ensemble of training data
        ref_ens = createEnsemble(forecast_model, train_params, train_params)
        name_train = './data/Truth_{}_{}'.format(ref_ens.name, ref_ens.law)
        for k, v in ref_ens.getParameters().items():
            name_train += '_{}{}'.format(k, v)
        name_train += '_tmax-{:.2}_std{:.2}_m{}_{}'.format(t_true[-1], ref_ens.std_a, ref_ens.m, ref_ens.alpha_distr)

        if os.path.isfile(name_train):
            print('Loading Reference solution(s)')
            with open(name_train, 'rb') as f:
                ref_ens = pickle.load(f)
        else:
            psi, t = ref_ens.timeIntegrate(Nt=len(t_true) - 1)
            ref_ens.updateHistory(psi, t)
            with open(name_train, 'wb') as f:
                pickle.dump(ref_ens, f)

        y_ref, lbl = ref_ens.getObservableHist()
        # if not save_:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
        norm = mpl.colors.Normalize(vmin=-5, vmax=y_ref.shape[-1])
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma)
        fig.suptitle('Training data')
        ax[0].plot(t_true, y_true, color='silver', linewidth=6, alpha=.8)
        Nt = int(.2 // dt_true)
        for ii in range(y_ref.shape[-1]):
            C, R = CR(y_true[-Nt:], y_ref[-Nt:, :, ii])
            line = ax[0].plot(t_true, y_ref[:, :, ii], color=cmap.to_rgba(ii))
            ax[1].plot(ii, C, 'o', color=cmap.to_rgba(ii))
            ax[2].plot(ii, R, 'x', color=cmap.to_rgba(ii))
        plt.tight_layout()
        ax[0].legend(['Truth'], bbox_to_anchor=(0., 1.25), loc="upper left")
        ax[0].set(xlabel='$t$', ylabel=lbl, xlim=[t_true[-1] - 0.05, t_true[-1]])
        ax[1].set(xlabel='$l$', ylabel='Correlation')
        ax[2].set(xlabel='$l$', ylabel='RMS error')
        ax[0].plot(t_true, y_true, color='silver', linewidth=6, alpha=.8)
        for ax1, l in zip(ax[:], [.5, 1., 1.]):
            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            ax1.set_aspect(l * (x1 - x0) / (y1 - y0))


        plt.savefig(results_folder + '00training_data.svg', dpi=350)
        plt.show()

        biasData = np.expand_dims(y_true, -1) - y_ref  # [Nt x Nmic x Ntrain]
        biasData = np.append(biasData, ref_ens.hist[:, -len(ref_ens.est_p):], axis=1)
        # provide data for washout before first observation
        i1 = int(np.where(t_true == t_obs[0])[0]) - kmeas
        i0 = i1 - int(0.1 / dt_true)

        # create bias dictionary
        bias_params = ESN_params.copy()
        bias_params['trainData'] = biasData[int(1. / dt_true):]  # remove transient - steady state solution
        bias_params['washout_obs'] = y_true[i0:i1 + 1]
        bias_params['washout_t'] = t_true[i0:i1 + 1]
        bias_params['filename'] = name_truth + '_' + name_train.split('Truth_')[-1]


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
    # bias_params['k'] = k
    bias_name = biasType.name
else:
    bias_name = 'None'
    bias_params = {}

# ===========================================  INITIALISE ENSEMBLE  ========================================== #
# filter_params['est_p'] = ['beta']
ensemble = createEnsemble(forecast_model, filter_params, model_params, bias_params)

for k in ks:
    filter_ens = ensemble.copy()  # copy the initialised ensemble
    if filter_ens.bias is not None:
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
    truth = dict(y=y_true, t=t_true, name=name_truth, t_obs=t_obs, p_obs=obs, b_true=b_true, true_params=true_params)
    parameters = dict(kmeas=kmeas, filt=filt, biasType=biasType, forecast_model=forecast_model,
                      true_model=true_model, num_DA=len(t_obs), Nt_extra=Nt_extra)
    if save_:
        filename = '{}{}_Truth{}_Forecast{}_Bias{}_k{}'.format(results_folder, filt, name_truth,
                                                               forecast_model.name, bias_name, k)
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
            pickle.dump(truth, f)
            pickle.dump(filter_ens, f)
    else:
        exec(open("post_process.py").read(), {'parameters': parameters,
                                              'filter_ens': filter_ens,
                                              'truth': truth})

if len(ks) > 1:
    exec(open("plot_t_analysus.py").read(), {'folder': results_folder})


