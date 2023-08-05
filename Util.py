# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:45:48 2022

@author: an553
"""
import os
import numpy as np
import pickle
import scipy.io as sio
from functools import lru_cache
import matplotlib as mpl
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d
from scipy.signal import find_peaks

rng = np.random.default_rng(6)


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
            mat = sio.loadmat(classType + '.mat')
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

    with open(name, 'wb') as f:
        pickle.dump(case, f)

    # Retrieve observables
    p_obs = case.getObservableHist()
    if len(np.shape(p_obs)) > 2:
        p_obs = np.squeeze(p_obs, axis=-1)

    return p_obs, case.hist_t, name.split('Truth_')[-1]


def check_valid_ensemble(true_p, filter_p, bias_p, load_ens, load_truth, load_bias):
    reinit = False
    if type(load_bias) is tuple:
        reinit, break_key, break_val, actual_val = True, None, None, None

    # check that true and forecast model parameters
    for key, val in filter_p.items():
        if hasattr(load_ens, key) and getattr(load_ens, key) != val:
            reinit, break_key, break_val, actual_val = True, key, getattr(load_ens, key), val
            break
    for key, val in true_p.items():
        if key in load_truth.keys() and load_truth[key] != val:
            reinit, break_key, break_val, actual_val = True, key, load_truth[key], val
            break

    # check that the data is sufficiently long for assimilation
    if abs(load_truth['t_obs'][-1] - filter_p['t_stop']) > load_truth['dt_obs']:
        reinit, break_key, break_val, actual_val = True, 't_obs', load_truth['t_obs'][-1], filter_p['t_stop']
    if load_truth['dt_obs'] // load_truth['dt'] != filter_p['kmeas']:
        reinit, break_key, break_val, actual_val = True, 't_obs', load_truth['dt_obs'] // load_truth['dt'], filter_p[
            'kmeas']

    if reinit:
        print('Re-init ensemble as loaded {} = {} != {}'.format(break_key, break_val, actual_val))
    return reinit


@lru_cache(maxsize=10)
def Cheb(Nc, lims=(0, 1), getg=False):  # __________________________________________________
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


def interpolate(t_y, y, t_eval, method='cubic', ax=0, bound=False):
    spline = interp1d(t_y, y, kind=method, axis=ax,
                      copy=True, bounds_error=bound, fill_value=0)
    return spline(t_eval)


def getEnvelope(timeseries_x, timeseries_y):
    peaks, peak_properties = find_peaks(timeseries_y, distance=200)
    u_p = interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False)
    return u_p


def save_to_pickle_file(filename, *args):
    with open(filename, 'wb') as f:
        for arg in args:
            pickle.dump(arg, f)


def load_from_pickle_file(filename):
    args = []
    with open(filename, 'rb') as f:
        while True:
            try:
                arg = pickle.load(f)
                args.append(arg)
            except EOFError:
                break
    if len(args) == 1:
        return args[0]
    else:
        return args


def plot_train_data(truth, train_ens, folder):

    y_ref = train_ens.getObservableHist(Nt=len(truth['t']))
    t = train_ens.hist_t[:len(truth['t'])]

    Nt = int(train_ens.t_CR / truth['dt'])
    i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))
    i0_r = np.argmin(abs(train_ens.hist_t - truth['t_obs'][0]))
    t_CR = train_ens.t_CR

    yt = truth['y'][i0_t - Nt:i0_t]
    bt = truth['b'][i0_t - Nt:i0_t]
    yr = y_ref[i0_r - Nt:i0_r]
    tt = train_ens.hist_t[i0_r - Nt:i0_r]

    RS = []
    for ii in range(y_ref.shape[-1]):
        R = CR(yt, yr[:, :, ii])[1]
        RS.append(R)

    # Plot training data -------------------------------------
    fig = plt.figure(figsize=[12, 4.5], layout="constrained")

    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])
    axs_top = subfigs[0].subplots(1, 2)
    axs_bot = subfigs[1].subplots(1, 2)

    true_RMS = CR(yt, yt - bt)[1]
    norm = mpl.colors.Normalize(vmin=true_RMS, vmax=1.5)

    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    cmap.set_clim(true_RMS, 1.5)

    axs_top[0].plot(tt, yt[:, 0], color='silver', linewidth=6, alpha=.8)

    print(bt.shape)
    axs_top[-1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.8)

    xlims = [[truth['t_obs'][0] - t_CR, truth['t_obs'][0]],
             [truth['t_obs'][0], truth['t_obs'][0] + t_CR * 2]]

    for ii in range(y_ref.shape[-1]):
        clr = cmap.to_rgba(RS[ii])
        axs_top[0].plot(tt, yr[:, 0, ii], color=clr)
        norm_bias = (truth['y'][:, 0] - y_ref[:, 0, ii])
        axs_bot[-1].plot(t, norm_bias, color=clr)
        axs_top[-1].plot(t, norm_bias, color=clr)

    axs_top[0].plot(tt, yt[:, 0], color='silver', linewidth=4, alpha=.5)
    axs_top[-1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.5)

    max_y = np.max(abs(yt[:, 0] - bt[:, 0]))

    axs_bot[0].plot(t, truth['b'][:, 0] / max_y * 100, color='silver', linewidth=4, alpha=.5)

    axs_top[0].legend(['Truth'], bbox_to_anchor=(0., 0.25), loc="upper left")
    axs_top[1].legend(['True RMS $={0:.3f}$'.format(true_RMS)], bbox_to_anchor=(0., 0.25), loc="upper left")

    axs_top[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlims[0])
    axs_top[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[0])
    axs_bot[0].set(xlabel='$t$', ylabel='$b$ normalized [\\%]', xlim=xlims[-1])
    axs_bot[-1].set(xlabel='$t$', ylabel='$b$', xlim=xlims[-1])

    clb = fig.colorbar(cmap, ax=axs_bot, orientation='vertical', extend='max')
    clb.ax.set_title('$\\mathrm{RMS}$')
    clb = fig.colorbar(cmap, ax=axs_top, orientation='vertical', extend='max')
    clb.ax.set_title('$\\mathrm{RMS}$')

    os.makedirs(folder, exist_ok=True)

    L = y_ref.shape[-1]
    plt.savefig(folder + 'L{}_training_data.svg'.format(L), dpi=350)
    plt.close()


def CR(y_true, y_est):
    # time average of both quantities
    y_tm = np.mean(y_true, 0, keepdims=True)
    y_em = np.mean(y_est, 0, keepdims=True)

    # correlation
    C = np.sum((y_est - y_em) * (y_true - y_tm)) / np.sqrt(np.sum((y_est - y_em) ** 2) * np.sum((y_true - y_tm) ** 2))
    # root-mean square error
    R = np.sqrt(np.sum((y_true - y_est) ** 2) / np.sum(y_true ** 2))
    return C, R


def get_error_metrics(results_folder):
    print('computing error metrics...')
    out = dict(Ls=[], ks=[])

    L_dirs, k_files = [], []
    LLL = os.listdir(results_folder)
    for Ldir in LLL:
        if os.path.isdir(results_folder + Ldir + '/') and Ldir[0] == 'L':
            L_dirs.append(results_folder + Ldir + '/')
            out['Ls'].append(float(Ldir.split('L')[-1]))

    for ff in os.listdir(L_dirs[0]):
        k = float(ff.split('_k')[-1])
        out['ks'].append(k)
        k_files.append(ff)

    # sort ks and Ls
    idx_ks = np.argsort(np.array(out['ks']))
    out['ks'] = np.array(out['ks'])[idx_ks]
    out['k_files'] = [k_files[i] for i in idx_ks]

    idx = np.argsort(np.array(out['Ls']))
    out['L_dirs'] = [L_dirs[i] for i in idx]
    out['Ls'] = np.array(out['Ls'])[idx]

    # Output quantities
    keys = ['R_biased_DA', 'R_biased_post',
            'C_biased_DA', 'C_biased_post',
            'R_unbiased_DA', 'R_unbiased_post',
            'C_unbiased_DA', 'C_unbiased_post']
    for key in keys:
        out[key] = np.empty([len(out['Ls']), len(out['ks'])])

    print(out['Ls'])
    print(out['ks'])

    from plotResults import post_process_single
    ii = -1
    for Ldir in out['L_dirs']:
        ii += 1
        print('L = ', out['Ls'][ii])
        jj = -1
        for ff in out['k_files']:
            jj += 1
            # Read file
            truth, filter_ens = load_from_pickle_file(Ldir + ff)[1:]
            truth = truth.copy()

            print('\t k = ', out['ks'][jj], '({}, {})'.format(filter_ens.bias.L, filter_ens.bias.k))
            # Compute biased and unbiased signals
            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
            y_mean = np.mean(y, -1)

            # Unbiased signal error
            if hasattr(filter_ens.bias, 'upsample'):
                y_unbiased = y_mean[::filter_ens.bias.upsample] + b
                y_unbiased = interpolate(t_b, y_unbiased, t)
            else:
                y_unbiased = y_mean + b

            # if jj == 0:
            N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
            i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
            i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

            # cut signals to interval of interest
            y_mean, t, y_unbiased = y_mean[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]

            if ii == 0 and jj == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
                y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

                out['C_true'], out['R_true'] = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
                out['C_pre'], out['R_pre'] = CR(y_truth[:N_CR], y_mean[:N_CR])
                out['t_interp'] = t[::N_CR]
                scale = np.max(y_truth, axis=0)
                for key in ['error_biased', 'error_unbiased']:
                    out[key] = np.empty([len(out['Ls']), len(out['ks']), len(out['t_interp']), y_mean.shape[-1]])

            # End of assimilation
            for yy, key in zip([y_mean, y_unbiased], ['_biased_DA', '_unbiased_DA']):
                C, R = CR(y_truth[-N_CR * 2:-N_CR], yy[-N_CR * 2:-N_CR])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # After Assimilaiton
            for yy, key in zip([y_mean, y_unbiased], ['_biased_post', '_unbiased_post']):
                C, R = CR(y_truth[-N_CR:], yy[-N_CR:])
                out['C' + key][ii, jj] = C
                out['R' + key][ii, jj] = R

            # Compute mean errors
            b_obs = y_truth - y_mean
            b_obs_u = y_truth - y_unbiased
            ei, a = -N_CR, -1
            while ei < len(b_obs) - N_CR - 1:
                a += 1
                ei += N_CR
                out['error_biased'][ii, jj, a, :] = np.mean(abs(b_obs[ei:ei + N_CR]), axis=0) / scale
                out['error_unbiased'][ii, jj, a, :] = np.mean(abs(b_obs_u[ei:ei + N_CR]), axis=0) / scale

    save_to_pickle_file(results_folder + 'CR_data', out)
