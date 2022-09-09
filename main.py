# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""

import os
# os.environ["OMP_NUM_THREADS"]= '1'
# ___________________________________ #
import time
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import pickle
from Util import createObservations
from Ensemble import createEnsemble
from DA import dataAssimilation
import TAModels
import Bias
# ___________________________________ #
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

# %% ============================== SELECT TA & BIAS MODELS AND FILTER ============================== #

true_model = 'Truth_wave.mat'
forecast_model = TAModels.Rijke
biasType = None # Bias.ESN  # Bias.ESN # None
filt = 'EnSRKF'  # 'EnKFbias' 'EnKF' 'EnSRKF'
k = .5  # 1E5  # gamma in the equations. Weight of bT Wbb b
kmeas = 5E-3  # [s]

# %% =====================================  CREATE OBSERVATIONS ===================================== #
y_true, t_true, obs, t_obs = createObservations(true_model, t_min=1., t_max=8., kmeas=kmeas)
# y_true, t_true, obs, t_obs = createObservations(true_model, TA_params={'law': 'tan'}, t_min=1., t_max=8., kmeas=kmeas)
dt_true = t_true[1] - t_true[0]


# Select training values
beta_train = 3E6
tau_train = 1.5E-3
y_ref, t_ref, _, _ = createObservations(forecast_model, TA_params={'law': 'sqrt',
                                                                   'dt': t_true[1] - t_true[0],
                                                                   'beta': beta_train,
                                                                   'tau': tau_train},
                                        t_min=1., t_max=t_true[-1], kmeas=kmeas)

biasData = y_true - y_ref

# plt.subplots(2, 1)
# plt.subplot(2, 1, 1)
# plt.plot(t_true, y_true[:, 0])
# plt.plot(reference.hist_t, y_ref[:, 0])
# plt.subplot(2, 1, 2)
# plt.plot(reference.hist_t, biasData[:, 0])
# plt.draw()


#
# # Manually add bias to observations
# b1 = 0.2
# b2 = 0.
# def bias(y):
#     return b1 * y + b2
#
# biasData = bias(y_true)
# y_true += biasData
# obs += bias(obs)


# %% ======================================  CREATE ENSEMBLE ====================================== #

num_DA = int(0.4 / (t_obs[1] - t_obs[0]))  # number of analysis steps

model_params = {'dt': 1E-4,                 # Dictionary of TA parameters
                'beta': beta_train * .95,
                'tau': tau_train * 1.05
                }
filter_params = {'m': 10,  # Dictionary of DA parameters
                 'est_p': ['beta', 'tau'],  # ['beta', 'tau'], ['kappa', 'nu']
                 'bias': biasType,
                 'std_psi': 0.1,
                 'std_a': 0.1,
                 'est_b': False,
                 'getJ': True
                 }

if biasType is not None:
    if biasType.name == 'ESN':
        # remove transient - steady state solution
        trainData = biasData[int(1. / dt_true):].squeeze()
        # provide data for washout [at least .025s] before first observation
        idx1 = int(np.where(t_true == t_obs[0])[0])
        idx0 = idx1 - int(0.1 / dt_true)
        # create bias dictionary
        bias_params = {'trainData': trainData,
                       'washout_obs': y_true[idx0:idx1 + 1],
                       'washout_t': t_true[idx0:idx1 + 1],
                       'train_TAparams': {'beta': beta_train,
                                          'tau': tau_train},
                       }
    elif biasType.name == 'LinearBias':
        bias_params = {'b1': 0.05,
                       'b2': 0.0}
    else:
        raise ValueError('Bias model not defined')
    bias_params['k'] = k
else:
    bias_params = {}

ensemble = createEnsemble(forecast_model, filter_params, model_params, bias_params)
filter_ens = ensemble.copy()

# %%
# ======================================  PERFORM DATA ASSIMILATION ====================================== #

filter_ens = dataAssimilation(filter_ens, obs[:num_DA], t_obs[:num_DA], method=filt)

# p.close()
# p.join()


# # Integrate further without assimilation (as ensemble mean)
Nt_extra = 0
if filter_ens.hist_t[-1] < t_true[-1]:
    # if truth very long, integrate only .2s longer
    extra_t = min((t_true[-1] - filter_ens.hist_t[-1]), 0.2)
    Nt_extra = int(extra_t / filter_ens.dt + 1)
    # integrate ensemble expectation, i.e. it's mean
    psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
    filter_ens.updateHistory(psi, t)
    if filter_ens.bias is not None:
        y, _ = filter_ens.getObservableHist(Nt_extra)
        b, t_b = filter_ens.bias.timeIntegrate(Nt=Nt_extra, y=y)
        filter_ens.bias.updateHistory(b, t_b)

# save data
try:
    truth_name = true_model.name
except:
    truth_name = 'Rijke_Wave'

filename = 'results/' + str(date.today()) + '_Bias_None'
with open(filename, 'wb') as f:
    parameters = dict(kmeas=kmeas,
                      filt=filt,
                      biasType=biasType,
                      forecast_model=forecast_model,
                      true_model=true_model,
                      num_DA=num_DA,
                      Nt_extra=Nt_extra
                      )
    truth = dict(y=y_true,
                 t=t_true,
                 name=truth_name,
                 t_obs=t_obs,
                 p_obs=obs
                 )

    pickle.dump(parameters, f)
    pickle.dump(truth, f)
    pickle.dump(filter_ens, f)

# # %% ================================ PLOT time series, parameters and RMS ================================ #
#
#
# y_filter, labels = filter_ens.getObservableHist()
# if len(np.shape(y_filter)) < 3:
#     y_filter = np.expand_dims(y_filter, axis=1)
# if len(np.shape(y_true)) < 3:
#     y_true = np.expand_dims(y_true, axis=-1)
#
# hist = filter_ens.hist
#
# mean = np.mean(hist, -1, keepdims=True)
# y_mean = np.mean(y_filter, -1)
#
# t = filter_ens.hist_t
#
# fig, ax = plt.subplots(3, 2, figsize=[15, 10])
#
# if biasType is None:
#     fig.suptitle(filter_ens.name + ' DA with ' + filt)
# else:
#     fig.suptitle(filter_ens.name + ' DA with ' + filt + 'and ' + biasType.name + ' bias')
#
# x_lims = [t_obs[0] - .05, t_obs[num_DA] + .05]
# c = 'royalblue'
#
# ax[0, 0].plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal')
# std = np.std(y_filter[:, 0, :], axis=1)
# ax[0, 0].fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.5, color=c)
#
# if filter_ens.bias is not None:
#     c = 'black'
#     b = filter_ens.bias.hist
#     t_b = filter_ens.bias.hist_t
#
#     # b_up = np.array([np.interp(t, t_b, b[:,i]) for i in range(filter_ens.bias.N_dim)]).transpose()
#     # b_up = np.expand_dims(b_up, -1)
#
#     # spl = splrep(t_b, b)
#     # b_up = splev(t, spl)
#
#     b_up = np.array([splev(t, splrep(t_b, b[:, i])) for i in range(filter_ens.bias.N_dim)]).transpose()
#     b_up = np.expand_dims(b_up, -1)
#
#     y_unbiased = y_filter + b_up
#     y_mean_u = np.mean(y_unbiased, -1)
#
#     ax[0, 0].plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased signal', linewidth=.5)
#     std = np.std(y_unbiased[:, 0, :], axis=1)
#     ax[0, 0].fill_between(t, y_mean_u[:, 0] + std, y_mean_u[:, 0] - std, alpha=0.2, color=c)
#
#
#
#     c = 'darkorchid'
#
#     b_obs = y_filter - y_true[:len(y_filter)]
#
#     b_mean = np.mean(b_obs, -1)
#     ax[2, 0].plot(t, b_mean[:, 0], '--', color=c, label='Observable bias')
#     std = np.std(b[:, 0, :], axis=1)
#     # ax[2, 0].fill_between(t, b_mean[:, 0] + std, b_mean[:, 0] - std, alpha=0.5, color=c)
#     ax[2, 0].set(ylabel='Bias', xlabel='$t$', xlim=x_lims)
#
#     ax[2, 0].plot(t, b_up[:, 0], label='ESN estimation')
#     ax[2, 0].legend()
#     # if filter_ens.est_b:
#     #     lab = ['b1', 'b2']
#     #     l = 0
#     #     c = ['r', 'b']
#     #     for w in [b1, b2]:
#     #         mean_p = mean[:, ii].squeeze()
#     #         std = np.std(hist[:, ii], axis=1)
#     #         ax[2, 1].plot(t, mean_p, color=c[l], label=lab[l])
#     #         ax[2, 1].set(xlabel='$t$', xlim=x_lims)
#     #         ax[2, 1].fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[l])
#     #         ii += 1
#     #         l += 1
#     #     ax[2, 1].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
#
# ax[0, 0].plot(t_true, y_true[:, 0], color='k', alpha=.2, label='Truth', linewidth=5)
# ax[0, 0].set(ylabel=labels[0], xlabel='$t$', xlim=x_lims)
# ax[0, 0].plot(t_obs[:num_DA], obs[:num_DA, 0], '.', color='r', label='Observations')
# ax[0, 0].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
#
# ii = len(filter_ens.psi0)
# c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
# if filter_ens.est_p:
#     for p in filter_ens.est_p:
#         mean_p = mean[:, ii].squeeze() / filter_ens.alpha0[p]
#         std = np.std(hist[:, ii] / filter_ens.alpha0[p], axis=1)
#
#         ax[1, 0].plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label=p + '/' + p + '$^t$')
#
#         ax[1, 0].set(xlabel='$t$', xlim=x_lims)
#         ax[1, 0].fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
#         ii += 1
#     ax[1, 0].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
#     ax[1, 0].plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
#
# Psi = mean - hist
# Psi = Psi[:-Nt_extra]
#
# Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
# RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
# ax[0, 1].plot(t[:-Nt_extra], RMS, color='firebrick')
# ax[0, 1].set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')
#
# if filter_ens.getJ:
#     J = np.array(filter_ens.hist_J)
#     ax[1, 1].plot(t_obs[:num_DA], J, label=['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$', '$\\mathcal{J}_{b}$'])
#     # ax[1, 1].plot(t_obs[:num_DA], np.sum(J, -1), label='$\\mathcal{J}$')
#
#     ax[1, 1].set(ylabel='Cost function $\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
#     ax[1, 1].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
#
# plt.tight_layout()
# plt.draw()
# plt.show()
#
# # ==================== UNCOMMENT TO SAVE FIGURES ============================== #
# # folder = os.getcwd() + "/figs/" + str(date.today()) + "/"
# # os.makedirs(folder, exist_ok=True)
#
# # plt.savefig(folder
# #             + filt + '_estB' + str(filter_ens.est_b)
# #             + "_b1" + str(b1) + "_b2" + str(b2)
# #             + "_PE" + str(len(filter_ens.est_p))
# #             + "_m" + str(filter_ens.m)
# #             + "_kmeas" + str(kmeas) + ".pdf")
