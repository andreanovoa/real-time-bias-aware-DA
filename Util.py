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

from functools import lru_cache

from scipy.interpolate import interp1d
from scipy.signal import find_peaks

rng = np.random.default_rng(6)


#
# data_folder = os.getcwd() + '\\data\\'


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


def createObservations(classType, TA_params=None, t_max=8.):
    if type(classType) is str:
        try:
            mat = sio.loadmat('data/' + classType)
            p_obs = mat['p_obs'].transpose()
            t_obs = mat['t_obs'].transpose()
            if len(np.shape(t_obs)) > 1:
                t_obs = np.squeeze(t_obs, axis=-1)
            if t_obs[-1] < t_max:
                t_max = t_obs[-1]
                print('Data too short. Redefine t_max = ', t_max)
            elif t_obs[-1] > t_max:
                # t_max = t_obs[-1]
                idx = np.argmin(abs(t_obs - t_max))
                t_obs = t_obs[:idx+1]
                p_obs = p_obs[:idx + 1]
                print('Data too long. Redefine t_max = ', t_max)

        except:
            raise ValueError('File ' + classType + ' not defined')
        filename = 'Wave'
    else:
        if TA_params is None:
            suffix = 'law-sqrt'
        else:
            suffix = ''
            key_save = classType.params + ['law']
            for key, val in TA_params.items():
                if key in key_save:
                    if type(val) == str:
                        suffix += val + '_'
                    else:
                        suffix += key + '{:.1e}'.format(val) + '_'

        name = '/data/Truth_{}_{}tmax-{:.2}'.format(classType.name, suffix, t_max)
        name = os.path.join(os.getcwd() + name)

        if os.path.isfile(name):
            print('Loading Truth')
            with open(name, 'rb') as f:
                case = pickle.load(f)
        else:
            case = classType(TA_params)
            psi, t = case.timeIntegrate(Nt=int(t_max / case.dt))
            case.updateHistory(psi, t)

            # print('Creating Truth. Not saving - uncomment lines below')
            with open(name, 'wb') as f:
                pickle.dump(case, f)
        # Retrieve observables
        t_obs = case.hist_t
        p_obs = case.getObservableHist()[0]
        if len(np.shape(p_obs)) > 2:
            p_obs = np.squeeze(p_obs, axis=2)

        filename = name.split('Truth_')[-1]
    return p_obs, t_obs, filename



def RK4(t, q0, func, *kwargs):
    ''' 4th order RK for autonomous systems described by func '''
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

    # plt.figure()
    # plt.plot(timeseries_y)
    # plt.show()
    u_p = interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False)

    return u_p


def CR(y_true, y_est):
    # time average of both quantities
    y_tm = np.mean(y_true, 0, keepdims=True)
    y_em = np.mean(y_est, 0, keepdims=True)
    # correlation
    C = sum((y_est - y_em) * (y_true - y_tm)) / np.sqrt(sum((y_est - y_em) ** 2) * sum((y_true - y_tm) ** 2))
    # root-mean square error
    R = np.sqrt(sum((y_true - y_est) ** 2) / sum(y_true** 2))
    # print(R,C)
    # plt.figure()
    # plt.plot(y_t)
    # plt.plot(y_e)
    # print(y_tm)
    # print(y_em)
    # plt.show()
    return C, R
## Uncomment the lines below to create the bias signal for training ESN
# if __name__ == '__main__':
#     import VdP as TAmodel
#     from datetime import date
#
#     HOM = createObservations(TAmodel, LOM=False, name='HOM_bias', t_max=15.)
#     LOM = createObservations(TAmodel, LOM=True, name='LOM_bias', t_max=15.)
#
#     bias = HOM[0].hist[:, 0, :] - LOM[0].hist[:, 0, :]
#
#     np.savez('bias_VdP' + str(date.today()), bias=bias)
#
#     fig, ax = plt.subplots(2, 1, figsize=[15, 10], tight_layout=True)
#     ax[0].plot(HOM[0].hist_t, HOM[0].hist[:, 0], label='HOM')
#     ax[0].plot(LOM[0].hist_t, LOM[0].hist[:, 0], 'y', label='LOM')
#     ax[0].set(ylabel='$\eta$', xlabel='$t_interp$', xlim=[14., 14.1])
#     ax[0].legend(loc='best')
#     ax[1].plot(LOM[0].hist_t, bias, 'mediumpurple')
#     ax[1].set(ylabel='bias', xlabel='$t_interp$', xlim=[14., 14.1])
