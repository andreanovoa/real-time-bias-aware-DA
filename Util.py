# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:45:48 2022

@author: an553
"""

import os as os
import numpy as np
import pylab as plt
import pickle
import scipy.io as sio
from compress_pickle import load, dump
from functools import lru_cache
import pathlib

from scipy.interpolate import interp1d
from scipy.signal import find_peaks

rng = np.random.default_rng(6)


@lru_cache(maxsize=10)
def Cheb(Nc, lims=[0, 1], getg=False):  # __________________________________________________
    """ Compute the Chebyshev collocation derivative matrix (D)
        and the Chevyshev grid of (N + 1) points in [ [0,1] ] / [-1,1]
    """
    g = - np.cos(np.pi * np.arange(Nc + 1, dtype=float) / Nc)
    c = np.hstack([2., np.ones(Nc - 1), 2.]) * (-1) ** np.arange(Nc + 1)
    X = np.outer(g, np.ones(Nc + 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(Nc + 1))
    D -= np.diag(D.sum(1))

    # Modify
    if lims[0] == 0:
        g = (g + 1.) / 2.
    if getg:
        return D, g
    else:
        return D


def createObservations(classParams=None):
    if type(classParams) is dict:
        TA_params = classParams.copy()
        classType = TA_params['model']
        if 't_max' in TA_params.keys():
            t_max = TA_params['t_max']
        else:
            t_max = 8.
            print('t_max=8.')
    else:
        raise ValueError('classParams must be dict')

    # Wave case: load .mat file ====================================
    if type(classType) is str:
        try:
            mat = sio.loadmat('data/Truth_wave.mat')
        except:
            raise ValueError('File ' + classType + ' not defined')
        p_obs = mat['p_mic'].transpose()
        t_obs = mat['t_mic'].transpose()
        if len(np.shape(t_obs)) > 1:
            t_obs = np.squeeze(t_obs, axis=-1)

        if t_obs[-1] > t_max:
            idx = np.argmin(abs(t_obs - t_max))
            t_obs, p_obs = [yy[:idx + 1] for yy in [t_obs, p_obs]]
            print('Data too long. Redefine t_max = ', t_max)

        return p_obs, t_obs, 'Wave'

    # ============================================================
    # Add key parameters to filename
    suffix = ''
    key_save = classType.params + ['law']
    for key, val in TA_params.items():
        if key in key_save:
            if type(val) == str:
                suffix += val + '_'
            else:
                suffix += key + '{:.2e}'.format(val) + '_'

    name = os.path.join(os.getcwd() + '/data/')
    os.makedirs(name, exist_ok=True)
    name += 'Truth_{}_{}tmax-{:.2}'.format(classType.name, suffix, t_max)

    # Load or create and save file
    case = classType(TA_params)
    psi, t = case.timeIntegrate(Nt=int(t_max / case.dt))
    case.updateHistory(psi, t)
    case.close()
    with open(name + '.lzma', 'wb') as f:
        dump(case, f)
    # Retrieve observables
    p_obs = case.getObservableHist()
    if len(np.shape(p_obs)) > 2:
        p_obs = np.squeeze(p_obs, axis=-1)

    return p_obs, case.hist_t, name.split('Truth_')[-1]


def RK4(t, q0, func, *kwargs):
    """ 4th order RK for autonomous systems described by func """
    dt = t[1] - t[0]
    N = len(t) - 1
    qhist = [q0]
    for i in range(N):
        k1 = dt * func(dt, q0, kwargs)
        k2 = dt * func(dt, q0 + k1 / 2, kwargs)
        k3 = dt * func(dt, q0 + k2 / 2, kwargs)
        k4 = dt * func(dt, q0 + k3, kwargs)
        q0 = q0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        qhist.append(q0)

    return np.array(qhist)


def plotHistory(ensemble, truth=None):  # _________________________________________________________________________
    """ Function that plots the history of the observables and the
        parameters with a zoomed region in the state.
    """

    def plotwithshade(x, y, c, yl=None):
        mean = np.mean(y, 1)
        std = np.std(y, 1)
        ax[i, j].plot(x, mean, color=c, label=lbl)
        ax[i, j].fill_between(x, mean + std, mean - std, alpha=.2, color=c)
        if yl is True:
            ax[i, j].set(ylabel=yl)
        ax[i, j].set(xlabel='$t_interp$', xlim=[x[0], x[-1]])
        ax[i, j].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

    t = ensemble.hist_t
    t_zoom = min([len(t) - 1, int(0.05 / ensemble.dt)])

    _, ax = plt.subplots(2, 2, figsize=[10, 5])
    # Truth
    if truth is not None:
        y, _ = truth.getObservableHist()
        y = np.squeeze(y)
        ax[0, 0].plot(t, y, color='k', alpha=.2, label='Truth', linewidth=4)
        ax[0, 1].plot(t, y, color='k', alpha=.2, label='Truth', linewidth=4)

    i, j = [0, 0]
    # State evolution
    y, lbl = ensemble.getObservableHist()
    lbl = lbl[0]
    plotwithshade(t, y[0], 'blue', yl=lbl[0])
    i, j = [0, 1]
    plotwithshade(t[-t_zoom:], y[0][-t_zoom:], 'blue')
    # Parameter evolution
    params = ensemble.hist[:, ensemble.N - len(ensemble.est_p):, :]
    c = ['g', 'sandybrown', 'mediumpurple']
    i, j = [1, 0]
    p_j = 0
    for p in ensemble.est_p:
        lbl = '$\\' + p + '/\\' + p + '^t$'
        plotwithshade(t, params[:, p_j] / ensemble.alpha0[p], c[p_j])
        p_j += 1

    plt.tight_layout()
    plt.show()


def interpolate(t_y, y, t_eval, method='cubic', ax=0, bound=False):
    spline = interp1d(t_y, y, kind=method, axis=ax, copy=True, bounds_error=bound, fill_value=0)
    return spline(t_eval)


def getEnvelope(timeseries_x, timeseries_y, rejectCloserThan=0):
    peaks, peak_properties = find_peaks(timeseries_y, distance=200)
    u_p = interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False)
    return u_p


def CR(y_true, y_est):
    # time average of both quantities
    y_tm = np.mean(y_true, 0, keepdims=True)
    y_em = np.mean(y_est, 0, keepdims=True)

    # correlation
    C = np.sum((y_est - y_em) * (y_true - y_tm)) / np.sqrt(np.sum((y_est - y_em) ** 2) * np.sum((y_true - y_tm) ** 2))
    # root-mean square error
    R = np.sqrt(np.sum((y_true - y_est) ** 2) / np.sum(y_true ** 2))
    return C, R

#
# def get_CR_values(results_folder):
#
#     Ls, RBs, RUs, CBs, CUs = [],[],[],[],[]
#     # ==================================================================================================================
#     ii= -1
#     for Ldir in os.listdir(results_folder):
#         Ldir = results_folder + Ldir + '/'
#         if not os.path.isdir(Ldir):
#             continue
#         ii += 1
#         Ls.append(Ldir.split('L')[-1])
#         flag = True
#         ks = []
#
#         L_RB, L_RU, L_CB, L_CU = [], [], [], []
#         for ff in os.listdir(Ldir):
#             if ff.find('_k') == -1:
#                 continue
#             k = float(ff.split('_k')[-1])
#             ks.append(k)
#             with open(Ldir + ff + '.gz', 'rb') as f:
#                 params = load(f)
#                 truth = load(f)
#                 filter_ens = load(f)
#
#             y, t = filter_ens.getObservableHist(), filter_ens.hist_t
#             b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
#             y_truth = truth['y'][:len(y)]
#             y = np.mean(y, -1)
#
#             # Unbiased signal error
#             if filter_ens.bias.name == 'ESN':
#                 y_unbiased = y[::filter_ens.bias.upsample] + b
#                 y_unbiased = interpolate(t_b, y_unbiased, t)
#             else:
#                 y_unbiased = y + np.expand_dims(b, -1)
#
#             if flag:
#                 N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
#                 i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
#                 i1 = np.argmin(abs(t - truth['t_obs'][params['num_DA']-1]))  # end of assimilation
#
#             C, R = CR(y_truth[i1-N_CR:i1], y[i1-N_CR:i1])
#             L_RB.append([R])
#             L_CB.append([C])
#             C, R = CR(y_truth[i1-N_CR:i1], y_unbiased[i1-N_CR:i1])
#             L_RU.append([R])
#             L_CU.append([C])
#
#             flag = False
#         RBs.append(L_RB)
#         RUs.append(L_RU)
#         CBs.append(L_CB)
#         CUs.append(L_CU)
#
#     # true and pre-DA R
#     y_truth_u = y_truth - truth['b_true'][:len(y)]
#     Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
#     Cpre, Rpre = CR(y_truth[i0 - N_CR:i0 + 1:], y[i0 - N_CR:i0 + 1:])
#
#     results = dict(Ls=Ls, ks=ks,
#                    RBs=RBs, RUs=RUs,
#                    Rt=Rt, Rpre=Rpre,
#                    CBs=CBs, CUs=CUs,
#                    Ct=Ct, Cpre=Cpre)
#
#     with open(results_folder + 'CR_data.gz', 'wb') as f:
#         dump(results, f)
