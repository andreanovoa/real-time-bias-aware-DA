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


def simulationName(case, tmax):
    name = './data/Truth_{}_{}'.format(case.name, case.law)
    for kk, vv in ref_ens.getParameters().items():
        name += '_{}{:1.2e}'.format(kk, vv)
    name += '_tmax-{:.2}_std{:.2}_m{}_{}'.format(tmax, case.std_a, case.m, case.alpha_distr)
    return name


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

rng = np.random.default_rng(0)
save_ = True  # Save simulation? If false, plot results
results_folder = 'results/VdP_Fig2_1PE/'
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
    os.makedirs(results_folder + 'figs/')

# %% ==========================================  CREATE OBSERVATIONS ========================================== #
true_model = TAModels.VdP
true_params = {'law': 'tan',
               'beta': 80.,  # forcing
               'zeta': 60.,  # damping
               'kappa': 3.4,  # nonlinearity
               'omega': 2 * np.pi * 120.  # frequency
               }
y_true, t_true, name_truth = createObservations(true_model, true_params, t_max=5.)
# # Manually add bias
b_true = np.cos(y_true)
y_true += b_true
name_truth += '_+cosy'

# Define the observations
t_start = 2.5
t_stop = 4.0
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
biasType = Bias.ESN  # Select either: Bias.ESN or None
filt = 'EnKFbias'  # Select from: 'EnKFbias' 'EnKF' 'EnSRKF

ks = np.linspace(0., 70., 36)
Ls = [1, 10, 100]
stds = [0.01, 0.1, 0.25, 0.5]

model_params = true_params.copy()
model_params['beta'] *= 0.8
# model_params['zeta'] *= 1.1
# model_params['kappa'] *= 1.2

filter_params = {'m': 10,  # Dictionary of DA parameters
                 'est_p': ['beta'],  # ['beta', 'tau'],  # #'beta', 'tau'
                 'bias': biasType,
                 'std_psi': 0.25,
                 'std_a': 0.25,
                 'est_b': False,
                 'getJ': True,
                 'inflation': 1.01,
                 'num_DA_blind': int(0.0 / t_obs[1] - t_obs[0]),  # num of obs to start accounting for the bias
                 'num_SE_only': int(0.0 / t_obs[1] - t_obs[0])  # num of obs to start parameter estimation
                 }

if biasType is None:
    bias_name = 'None'
    bias_params = {}
elif biasType.name == 'ESN':
    train_params = {'m': 10,
                    'std_a': 0.3,
                    'std_psi': 0.3,
                    'est_p': ['beta'],
                    'alpha_distr': 'uniform',
                    }
    ESN_params = {'N_wash': 50,
                  'upsample': 5,
                  'N_units': 100,
                  't_train': 1.0,
                  'train_TAparams': train_params
                  }
else:
    raise ValueError('Bias type not defined')

# =================================== CREATE ESNs FOR EACH Ls ============================ #
Ls_bias_params = []
for L in Ls:
    train_params['m'] = L
    ref_ens = createEnsemble(forecast_model, train_params, train_params)
    # Compute reference bias. Create an ensemble of training data
    name_train = simulationName(ref_ens, t_true[-1])

    # Load/Save training data ------------------------------------------------------------------
    if os.path.isfile(name_train):
        print('Loading {} Reference solution(s)'.format(train_params['m']))
        with open(name_train, 'rb') as f:
            ref_ens = pickle.load(f)
    else:
        psi, t = ref_ens.timeIntegrate(Nt=len(t_true) - 1)
        ref_ens.updateHistory(psi, t)
        with open(name_train, 'wb') as f:
            pickle.dump(ref_ens, f)
    y_ref, lbl = ref_ens.getObservableHist()
    # Plot training data ------------------------------------------------------------------------
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
    ax[0].plot(t_true, y_true, color='silver', linewidth=6, alpha=.9)
    for ax1, l in zip(ax[:], [.5, 1., 1.]):
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()
        ax1.set_aspect(l * (x1 - x0) / (y1 - y0))
    plt.savefig(results_folder + 'figs/' + str(L) + '_training_data.svg', dpi=350)
    # plt.show()
    # Add bias data (for washout) to bias dictionary --------------------------------------------------
    biasData = np.expand_dims(y_true, -1) - y_ref  # [Nt x Nmic x Ntrain]
    biasData = np.append(biasData, ref_ens.hist[:, -len(ref_ens.est_p):], axis=1)
    if len(Ls_bias_params) == 0:
        i1 = int(np.where(t_true == t_obs[0])[0]) - kmeas
        i0 = i1 - int(0.1 / dt_true)
        bias_params = ESN_params.copy()
        bias_params['washout_obs'] = y_true[i0:i1 + 1]
        bias_params['washout_t'] = t_true[i0:i1 + 1]
        bias_name = biasType.name
    bias_params['trainData'] = biasData[int(1. / dt_true):]  # remove transient
    bias_params['filename'] = results_folder + name_truth + '_' + name_train.split('Truth_')[-1] + '_bias'

    Ls_bias_params.append(bias_params)

for std in stds:
    filter_params['std_psi'] = std
    filter_params['std_a'] = std
    # ===========================================  INITIALISE ENSEMBLE  ========================================== #
    ensemble = createEnsemble(forecast_model, filter_params, model_params, bias_params)

    for L, bp in zip(Ls, Ls_bias_params):
        flag = True
        for k in ks:
            filter_ens = ensemble.copy()  # copy the initialised ensemble
            filter_ens.bias.trainESN(bp)
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
            truth = dict(y=y_true, t=t_true, name=name_truth, t_obs=t_obs, p_obs=obs,
                         b_true=b_true, true_params=true_params)
            parameters = dict(kmeas=kmeas, filt=filt, biasType=biasType, forecast_model=forecast_model,
                              true_model=true_model, num_DA=len(t_obs), Nt_extra=Nt_extra)
            if save_:
                save_folder = results_folder + 'std' + str(std) + '/L' + str(L) + '/'
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                filename = '{}{}_Truth{}_Forecast{}_Bias{}_L{}_k{}'.format(save_folder, filt, name_truth,
                                                                           forecast_model.name, bias_name, L, k)

                with open(filename, 'wb') as f:
                    pickle.dump(parameters, f)
                    pickle.dump(truth, f)
                    pickle.dump(filter_ens, f)

                if k in [0, 10, 20, 50] and L == 10 and std in [0.1, 0.3]:
                    exec(open("post_process.py").read(), {'parameters': parameters,
                                                          'filter_ens': filter_ens,
                                                          'truth': truth,
                                                          'folder': results_folder + 'figs/',
                                                          'name': str(L) + '_' + str(k) + '_' + str(std) + '_results'})










