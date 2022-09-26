# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""

from Util import createObservations, bias
from Ensemble import createEnsemble
from DA import dataAssimilation

from datetime import date
import numpy as np
import pylab as plt

import Rijke as Model

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

# __________________________  CREATE OBSERVATIONS __________________________ #

truth, obs, t_obs = createObservations(Model, LOM=True,
                                       suffix='LOM_PE' + str(date.today()),
                                       t_obs=[2., 5.])
# truth.viewHistory()

num_DA = int(1. / (t_obs[1] - t_obs[0]))  # number of analysis steps

# add bias to observations
obs += bias(obs)

# ____________________________  CREATE ENSEMBLE ____________________________ #


model_params = {'dt': 1E-4}  # Dictionary of TA parameters
filter_params = {'m': 5,  # Dictionary of DA parameters
                 'est_p': []}

ensemble = createEnsemble(Model, filter_params, model_params)

# _______________________  PERFORM DATA ASSIMILATION _______________________ #
# TODO - rewrite dataAssimilation to avoid passing the truth. Instead pass the washout as part of observation vector
for filt in ['EnKF']:
    filter_ens = ensemble.copy()
    filter_ens, _ = dataAssimilation(filter_ens, obs[:num_DA],
                                     t_obs[:num_DA], truth, method=filt)

    # Integrate further without assimilation (as ensemble mean)
    if len(filter_ens.hist_t) < len(truth.hist_t):
        Nt = len(truth.hist_t) - len(filter_ens.hist_t)
        psi, t = filter_ens.timeIntegrate(Nt, averaged=True)
        filter_ens.updateHistory(psi, t)

    # _________________ PLOT timeseries, parameters and RMS ________________ #

    mean_t = np.mean(truth.hist, -1, keepdims=True)
    hist = filter_ens.hist.copy()

    mean_t[:, 0] += bias(mean_t[:, 0])

    if filt == 'EnKFbias':
        hist[:, 0] += bias(hist[:, 0])

    mean = np.mean(hist, 2)
    # Cpp     =   filter_ens.getCovariance(mean_true)

    Psi = hist[:, :filter_ens.N - len(filter_ens.est_p)] - mean_t
    Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.)
           for ti in range(len(mean_t))]

    RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]

    psi_t = truth.hist[:, 0]
    t = filter_ens.hist_t

    fig, ax = plt.subplots(3, 1, figsize=[15, 10], tight_layout=True)
    fig.suptitle(filter_ens.name + ' DA with ' + filt)
    x_lims = [t_obs[0] - .05, t_obs[num_DA] + .05]
    c = 'royalblue'

    ax[0].plot(t, mean[:, 0], '--', color=c, label='Filtered signal')
    ax[0].plot(t, mean_t[:, 0], color='k', alpha=.2, label='Truth', linewidth=5)
    ax[0].set(ylabel='$\\eta$', xlabel='$t$', xlim=x_lims)
    std = np.std(hist[:, 0, :], axis=1)
    ax[0].fill_between(t, mean[:, 0] + std, mean[:, 0] - std, alpha=0.2, color=c)
    ax[0].plot(t_obs[:num_DA], obs[:num_DA], '.', color='r', label='Observations')
    ax[0].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

    i = 0
    c = ['g', 'sandybrown', 'mediumpurple']
    if filter_ens.est_p:
        for p in filter_ens.est_p:
            mean_p = mean[:, len(filter_ens.psi0) + i] / filter_ens.alpha0[p]
            std = np.std(hist[:, len(filter_ens.psi0) + i] / filter_ens.alpha0[p], axis=1)

            ax[1].plot(t, mean_p, color=c[i], label='$\\' + p + '/\\' + p + '^t$')
            ax[1].set(xlabel='$t$', xlim=x_lims)
            ax[1].fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[i])
            i += 1
        ax[1].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
        ax[1].plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)

    ax[2].plot(t, RMS, color='firebrick')
    ax[2].set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')
    plt.tight_layout()
    plt.show()
