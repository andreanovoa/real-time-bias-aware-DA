import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
from Util import interpolate, CR, getEnvelope
from scipy.interpolate import CubicSpline, interp1d
import numpy as np

def plotResults(folder, stds, Ls, k_plot=(None,)):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

    if folder[-1] != '/':
        folder += '/'
    for std in stds:
        for L in Ls:
            results_folder = folder + 'std{}/L{}/'.format(std, L)
            files = os.listdir(results_folder)
            if k_plot[0] is not None:
                for file in files:
                    if file[-3:] == '.py' or file[-4] == '.':
                        continue
                    k = float(file.split('_k')[-1])
                    if k in k_plot:
                        with open(results_folder + file, 'rb') as f:
                            parameters = pickle.load(f)
                            truth = pickle.load(f)
                            filter_ens = pickle.load(f)
                        filename = '{}figs/L{}_std{}_k{}_time'.format(folder, L, std, k)
                        # post_process_single(filter_ens, truth, parameters, filename)
                        post_process_single_SE_Zooms(filter_ens, truth, filename)

                        plt.close()
            filename = '{}figs/CR_L{}_std{}_results'.format(folder, L, std)
            post_process_multiple(results_folder, filename)
            plt.close()
    fig2(folder, Ls, stds)
    plt.close()


# ==================================================================================================================
def post_process_single_SE_Zooms(filter_ens, truth, filename=None, figs_folder=None):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

    y_true = truth['y']
    t_true = truth['t']
    t_obs = truth['t_obs']
    obs = truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    # %% ================================ PLOT time series, parameters and RMS ================================ #

    y_filter, labels = filter_ens.getObservableHist(), filter_ens.obsLabels
    if len(np.shape(y_filter)) < 3:
        y_filter = np.expand_dims(y_filter, axis=1)
    if len(np.shape(y_true)) < 3:
        y_true = np.expand_dims(y_true, axis=-1)

    # normalise results
    norm = 1.  # np.max(abs(y_true[:, 0]))
    y_filter /= norm
    y_true /= norm

    hist = filter_ens.hist
    t = filter_ens.hist_t

    mean = np.mean(hist, -1, keepdims=True)
    y_mean = np.mean(y_filter, -1)
    std = np.std(y_filter[:, 0, :], axis=1)

    fig, ax = plt.subplots(1, 3, figsize=[17.5, 3.5], layout="constrained")
    params_ax, zoomPre_ax, zoom_ax = ax[:]
    x_lims = [t_obs[0] - .05, t_obs[-1] + .05]

    c = 'lightgray'
    zoom_ax.plot(t_true, y_true[:, 0], color=c, linewidth=8)
    zoomPre_ax.plot(t_true, y_true[:, 0], color=c, label='Truth', linewidth=8)
    zoomPre_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    zoom_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='black', linewidth=.8)

    zoom_ax.plot(t_obs, obs[:, 0], '.', color='r',  markersize=10)
    zoomPre_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observations', markersize=10)
    if filter_ens.bias is not None:
        c = 'navy'
        b = filter_ens.bias.hist
        b /= norm
        t_b = filter_ens.bias.hist_t
        y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)

        spline = interp1d(t_b, y_unbiased, kind='cubic', axis=0, copy=True, bounds_error=False, fill_value=0)
        y_unbiased = spline(t)

        y_mean_u = np.mean(y_unbiased, -1)

        t_wash = filter_ens.bias.washout_t
        wash = filter_ens.bias.washout_obs
        wash /= norm

        zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)
        washidx = int(t_obs[0] / filter_ens.dt) - filter_ens.bias.N_wash * filter_ens.bias.upsample
        zoom_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased estimate', linewidth=1.5)
        zoomPre_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-', color=c, linewidth=1.5)

    c = 'lightseagreen'
    zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased estimate', linewidth=1.5, alpha=0.9)
    zoomPre_ax.plot(t, y_mean[:, 0], '--', color=c, linewidth=1.5, alpha=0.9)
    zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoomPre_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

    zoomPre_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    zoom_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    y_lims = [min(y_mean[:, 0]) * 1.2,
              max(y_mean[:, 0]) * 1.2]
    # y_lims = [-7.5, 9.5]
    zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - 0.03, t_obs[-1] + 0.02], ylim=y_lims)
    zoomPre_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - 0.03, t_obs[0] + 0.02], ylim=y_lims)

    # PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
    ii = len(filter_ens.psi0)
    c = ['g', 'mediumpurple',  'sandybrown', 'cyan']
    params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
    params_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')

    if num_DA_blind > 0:
        params_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                       label='Start BE')
    if num_SE_only > 0:
        params_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                       label='Start PE')

    if filter_ens.est_p:
        max_p, min_p = -np.infty, np.infty
        for p in filter_ens.est_p:
            superscript = '^\mathrm{init}$'
            reference_p = filter_ens.alpha0

            mean_p = mean[:, ii].squeeze() / reference_p[p]
            std = np.std(hist[:, ii] / reference_p[p], axis=1)

            max_p = max(max_p, max(mean_p))
            min_p = min(min_p, min(mean_p))

            if p in ['C1', 'C2']:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$' + p + '/' + p + superscript)
            else:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$\\' + p + '/\\' + p + superscript)

            params_ax.set(xlabel='$t$ [s]', xlim=x_lims)
            params_ax.fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
            ii += 1
        params_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=3)
        params_ax.plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
        params_ax.set(ylim=[min_p - 0.1, max_p + 0.1])

    # plt.show()
    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
    else:
        plt.show()


def post_process_single(filter_ens, truth, parameters, filename=None):
    filt = filter_ens.filt
    biasType = filter_ens.biasType
    Nt_extra = parameters['Nt_extra']

    y_true = truth['y']
    t_true = truth['t']
    t_obs = truth['t_obs']
    obs = truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    # %% ================================ PLOT time series, parameters and RMS ================================ #

    y_filter, labels = filter_ens.getObservableHist(), filter_ens.obsLabels
    if len(np.shape(y_filter)) < 3:
        y_filter = np.expand_dims(y_filter, axis=1)
    if len(np.shape(y_true)) < 3:
        y_true = np.expand_dims(y_true, axis=-1)

    # normalise results
    norm = 1.  # np.max(abs(y_true[:, 0]))
    y_filter /= norm
    y_true /= norm

    hist = filter_ens.hist
    t = filter_ens.hist_t

    mean = np.mean(hist, -1, keepdims=True)
    y_mean = np.mean(y_filter, -1)
    std = np.std(y_filter[:, 0, :], axis=1)

    fig, ax = plt.subplots(3, 3, figsize=[20, 12], layout="constrained")
    p_ax, zoomPre_ax, zoom_ax = ax[0, :]
    params_ax, bias_ax = ax[1, :2]
    RMS_ax, J_ax, dJ_ax = ax[2, :]

    if biasType is None:
        fig.suptitle(filter_ens.name + ' DA with ' + filt)
    else:
        fig.suptitle(filter_ens.name + ' DA with ' + filt + ' and ' + biasType.name + ' bias')

    x_lims = [t_obs[0] - .05, t_obs[-1] + .05]

    c = 'lightgray'
    p_ax.plot(t_true, y_true[:, 0], color=c, label='Truth', linewidth=4)
    zoom_ax.plot(t_true, y_true[:, 0], color=c, label='Truth', linewidth=8)
    zoomPre_ax.plot(t_true, y_true[:, 0], color=c, label='Truth', linewidth=8)
    p_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    p_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    zoomPre_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    zoom_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='black', linewidth=.8)

    if filter_ens.bias is not None:
        c = 'navy'
        b = filter_ens.bias.hist
        b /= norm
        t_b = filter_ens.bias.hist_t

        if filter_ens.bias.name == 'ESN':
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            spline = interp1d(t_b, y_unbiased, kind='cubic', axis=0, copy=True, bounds_error=False, fill_value=0)
            y_unbiased = spline(t)

            y_mean_u = np.mean(y_unbiased, -1)

            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs
            wash /= norm

            p_ax.plot(t_wash, wash[:, 0], '.', color='r')
            zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)
            washidx = int(t_obs[0] / filter_ens.dt) - filter_ens.bias.N_wash * filter_ens.bias.upsample
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)
            y_mean_u = np.mean(y_unbiased, -1)
            washidx = int(t_obs[0] / filter_ens.dt)

        p_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.2)
        zoom_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.5)
        zoomPre_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-', color=c, label='Unbiased filtered signal',
                        linewidth=1.5)

        # BIAS PLOT
        bias_ax.plot(t_b, b[:, 0], alpha=0.75, label='ESN estimation')
        b_obs = y_true[:len(y_filter)] - y_filter

        b_mean = np.mean(b_obs, -1)
        bias_ax.plot(t, b_mean[:, 0], '--', color='darkorchid', label='Observable bias')
        # std = np.std(b[:, 0, :], axis=1)
        bias_ax.fill_between(t, b_mean[:, 0] + std, b_mean[:, 0] - std, alpha=0.5, color='darkorchid')

        # y_lims = [min(b_mean[:, 0]) - np.mean(std), (max(b_mean[:, 0]) + max(std))]
        bias_ax.legend()
        bias_ax.set(ylabel='Bias', xlabel='$t$')

    c = 'lightseagreen'  # '#021bf9'
    p_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased filtered signal', linewidth=1.)
    zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=1.5, alpha=0.9)
    zoomPre_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=1.5, alpha=0.9)
    p_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoomPre_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

    p_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observation data')
    zoom_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observation data', markersize=10)
    zoomPre_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observation data', markersize=10)

    y_lims = [min(min(y_true[:, 0]), min(y_mean[:, 0])) * 1.5, max(max(y_true[:, 0]), max(y_mean[:, 0])) * 1.5]
    p_ax.set(ylabel="$p'_\mathrm{mic_1}$ [Pa]", xlabel='$t$ [s]', xlim=x_lims, ylim=y_lims)
    p_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)

    y_lims = [min(min(y_true[:, 0]), min(y_mean[:, 0])) * 1.2, max(max(y_true[:, 0]), max(y_mean[:, 0])) * 1.2]

    zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - 0.03, t_obs[-1] + 0.02], ylim=y_lims)
    zoomPre_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - 0.03, t_obs[0] + 0.02], ylim=y_lims)

    # PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
    ii = len(filter_ens.psi0)
    c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
    params_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')

    if num_DA_blind > 0:
        p_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue')
        params_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                       label='Start BE')
    if num_SE_only > 0:
        p_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet')
        params_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                       label='Start PE')

    if filter_ens.est_p:
        max_p, min_p = -np.infty, np.infty
        for p in filter_ens.est_p:
            superscript = '^\mathrm{init}$'
            reference_p = filter_ens.alpha0

            mean_p = mean[:, ii].squeeze() / reference_p[p]
            std = np.std(hist[:, ii] / reference_p[p], axis=1)

            max_p = max(max_p, max(mean_p))
            min_p = min(min_p, min(mean_p))
            if p in ['C1', 'C2']:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$' + p + '/' + p + superscript)
            else:
                params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$\\' + p + '/\\' + p + superscript)

            params_ax.set(xlabel='$t$', xlim=x_lims)
            params_ax.fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
            ii += 1
        params_ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
        params_ax.plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
        params_ax.set(ylim=[min_p - 0.1, max_p + 0.1])

    # PLOT RMS ERROR
    Psi = mean - hist
    Psi = Psi[:-Nt_extra]

    Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
    RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
    RMS_ax.plot(t[:-Nt_extra], RMS, color='firebrick')
    RMS_ax.set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')

    # PLOT COST FUNCTION
    J = np.array(filter_ens.hist_J)
    J_ax.plot(t_obs, J[:, :-1])
    dJ_ax.plot(t_obs, J[:, -1], color='tab:red')

    dJ_ax.set(ylabel='$d\\mathcal{J}/d\\psi$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.set(ylabel='$\\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.legend(['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$',
                 '$\\mathcal{J}_{b}$'], bbox_to_anchor=(1., 1.),
                loc="upper left", ncol=1)

    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
    # else:
    #     plt.show()


# ==================================================================================================================
def post_process_multiple(folder, filename=None):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


    files = os.listdir(folder)
    flag = True
    biases, esn_errors, biases_ESN = [], [], []
    ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []
    # ==================================================================================================================
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    fig.suptitle(folder)
    subfigs = fig.subfigures(2, 1)
    axCRP = subfigs[0].subplots(1, 3)
    mean_ax = subfigs[1].subplots(1, 2)
    for file in files:
        # if file[-3:] == '.py' or file[-4] == '.':
        #     continue
        if file.find('_k') == -1:
            continue
        k = float(file.split('_k')[-1])
        if k > 10: # uncomment these lines to avoid ploting values over 20
            continue
        with open(folder + file, 'rb') as f:
            parameters = pickle.load(f)
            truth = pickle.load(f)
            filter_ens = pickle.load(f)
        # Observable bias
        y_filter, t_filter = filter_ens.getObservableHist(), filter_ens.hist_t
        y_truth = truth['y'][:len(y_filter)]
        b_obs = y_truth - np.mean(y_filter, -1)
        if flag:
            N_CR = int(.1 / filter_ens.dt)  # Length of interval to compute correlation and RMS
            N_mean = int(.1 / filter_ens.dt)  # Length of interval to average mean error
            istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
            istop = np.argmin(abs(t_filter - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation
            t_interp = t_filter[N_mean::N_mean]

        # ESN bias
        b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
        b_ESN = interpolate(t_b, b, t_filter)

        # Ubiased signal error


        if filter_ens.bias.name == 'ESN':
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            y_unbiased = interpolate(t_b, y_unbiased, t_filter)
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)

        b_obs_u = y_truth - np.mean(y_unbiased, -1)


        # compute mean error, time averaged over a kinterval
        bias, bias_esn, esn_err = [], [], []
        for j in range(int(len(b_obs) // N_mean)):
            i = j * N_mean
            mean_bias_obs = np.mean(abs(b_obs[i:i + N_mean]), 0)
            mean_bias_esn = np.mean(abs(b_ESN[i:i + N_mean]), 0)
            # mean_unbiased_error = np.mean(abs(b_obs[i:i + kinterval] - b_ESN[i:i + kinterval]), 0)
            mean_unbiased_error = np.mean(abs(b_obs_u[i:i + N_mean]), 0)

            bias.append(mean_bias_obs)
            bias_esn.append(mean_bias_esn)
            esn_err.append(mean_unbiased_error)

        biases.append(np.array(bias))
        biases_ESN.append(np.array(bias_esn))
        esn_errors.append(np.array(esn_err))
        ks.append(k)

        # PLOT CORRELATION AND RMS ERROR =====================================================================
        t_obs = truth['t_obs'][:parameters['num_DA']]
        y_obs = interpolate(t_filter, y_truth, t_obs)
        y_obs_b = interpolate(t_filter, np.mean(y_filter, -1), t_obs)
        y_obs_u = interpolate(t_filter, np.mean(y_unbiased, -1), t_obs)

        CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
        CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased
        # Correlation
        bias_c = 'tab:red'
        unbias_c = 'tab:blue'
        ms = 4

        axCRP[0].plot(k, CB, 'o', color=bias_c, label='Biased', markersize=ms, alpha=0.6)
        axCRP[0].plot(k, CU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')
        # RMS error
        axCRP[1].plot(k, RB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
        axCRP[1].plot(k, RU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')

        CB, RB = CR(y_obs, y_obs_b)  # biased
        CU, RU = CR(y_obs, y_obs_u)  # unbiased
        # Correlation
        bias_c = 'tab:red'
        unbias_c = 'tab:blue'
        axCRP[0].plot(k, CB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)
        # RMS error
        axCRP[1].plot(k, RB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)

        # Parameters ========================================================================================
        if filter_ens.est_p:
            if flag:
                N_psi = len(filter_ens.psi0)
                # c = ['tab:orange', 'navy', 'darkcyan', 'cyan']
                c = ['g', 'mediumpurple', 'sandybrown', 'r']
                # c = ['navy', 'chocolate', 'mediumturquoise', 'lightseagreen', 'cyan']
                marker = ['x', '+']
                time = ['$(t_\mathrm{end})$', '$(t_\mathrm{start})$']
                alphas = [1., .2]
                superscript = '^\mathrm{init}$'
                reference_p = filter_ens.alpha0
            for jj, p in enumerate(filter_ens.est_p):
                for kk, idx in enumerate([istop, istart]):
                    hist_p = filter_ens.hist[idx - 1, N_psi + jj] / reference_p[p]
                    if p in ['C1', 'C2']:
                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                          fmt=marker[kk], color=c[jj], label='$'+p+'/'+ p + superscript + time[kk],
                                          capsize=ms, markersize=ms)
                    else:
                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                          fmt=marker[kk], color=c[jj], label='$\\'+p+'/\\' + p + superscript + time[kk],
                                          capsize=ms, markersize=ms)

                    # axCRP[2].plot([min(ks)-1, max(ks)+1],
                    #               [truth['true_params'][p]/ reference_p[p], truth['true_params'][p]/ reference_p[p]],
                    #               '--', color=c[jj], linewidth=.8, alpha=.8, label='$\\' + p + '^\mathrm{true}/\\'
                    #               + p + superscript)

        if flag:
            # compute and plot the baseline correlation and MSE
            if len(truth['b_true']) > 1:
                b_truth = truth['b_true'][:len(y_filter)]
                y_truth_u = y_truth - b_truth
                Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
                axCRP[0].plot((-10, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                axCRP[1].plot((-10, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
            # compute C and R before the assimilation (the initial ensemble has some initialisation error)
            Cpre, Rpre = CR(y_truth[istart - N_CR:istart + 1:], np.mean(y_filter, -1)[istart - N_CR:istart + 1:])
            axCRP[0].plot((-10, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
            axCRP[1].plot((-10, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
            for ax1 in axCRP[1:]:
                ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
            flag = False

    # =========================================================================================================
    xlims = [min(ks) - .5, max(ks) + .5]
    axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$')
    axCRP[1].set(ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')
    # axCRP[2].set(ylim=[0.2, 2.], xlim=xlims)

    for ax1 in axCRP[:]:
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()
        ax1.set_aspect((x1 - x0) / (y1 - y0))

    # PLOT MEAN ERROR EVOLUTION ================================================================================
    for mic in [0]:
        scale = np.max(truth['y'][:, mic])
        norm = colors.Normalize(vmin=0, vmax=max(ks))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        for i, metric in enumerate([biases, esn_errors]):  # , biases_ESN]):
            errors = [b[:, mic] / scale for b in metric]
            for err, k in zip(errors, ks):
                mean_ax[i].plot(t_interp, err * 100, color=cmap.to_rgba(k))
            mean_ax[i].set(xlim=[t_filter[istart] - 0.02, t_filter[istop] + 0.05], xlabel='$t$ [s]')

        mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\%]')
        mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\%]')

        for i in range(2):
            x0, x1 = mean_ax[i].get_xlim()
            y0, y1 = mean_ax[i].get_ylim()
            # print( (x1 - x0) / (y1 - y0))
            mean_ax[i].set_aspect(0.5 * (x1 - x0) / (y1 - y0))

    clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
    clb.ax.set_title('$\\gamma$')

    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
    # else:
    #     plt.show()


# ==================================================================================================================
def fig2(folder, Ls, stds, figs_folder):
    plt.rc('font', family='serif', size=12)
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    fig.suptitle(folder)


    subfigs = fig.subfigures(max(len(Ls), 2), max(len(stds), 2), wspace=0.07)

    for si in range(len(stds)):
        for Li in range(len(Ls)):
            ax = subfigs[Li, si].subplots(1, 2)
            subfigs[Li, si].suptitle('std={}, L={}'.format(stds[si], Ls[Li]))
            files_folder = folder + 'std{}/L{}/'.format(stds[si], Ls[Li])

            files = os.listdir(files_folder)
            flag = True
            ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []

            for file in files:
                if file[-3:] == '.py' or file[-4] == '.':
                    continue
                k = float(file.split('_k')[-1])
                with open(files_folder + file, 'rb') as f:
                    parameters = pickle.load(f)
                    truth = pickle.load(f)
                    filter_ens = pickle.load(f)
                # Observable bias
                y_filter, t_filter = filter_ens.getObservableHist(), filter_ens.hist_t
                y_truth = truth['y'][:len(y_filter)]
                b_obs = y_truth - np.mean(y_filter, -1)
                if flag:
                    N_CR = int(.1 / filter_ens.dt)  # Length of interval to compute correlation and RMS
                    istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
                    istop = np.argmin(abs(t_filter - truth['t_obs'][parameters['num_DA'] - 1]))  # end of assimilation

                # ESN bias
                b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t
                b_ESN = interpolate(t_b, b, t_filter)

                # Ubiased signal error
                y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
                y_unbiased = interpolate(t_b, y_unbiased, t_filter)

                ks.append(k)

                # PLOT CORRELATION AND RMS ERROR =====================================================================
                t_obs = truth['t_obs'][:parameters['num_DA']]
                y_obs = interpolate(t_filter, y_truth, t_obs)
                y_obs_b = interpolate(t_filter, np.mean(y_filter, -1), t_obs)
                y_obs_u = interpolate(t_filter, np.mean(y_unbiased, -1), t_obs)

                CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
                CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased
                # Correlation
                bias_c = 'tab:red'
                unbias_c = 'tab:blue'
                # ax[0].plot(k, CB, 'o', color=bias_c, label='Biased', markersize=4)
                # ax[0].plot(k, CU, '*', color=unbias_c, label='Unbiased', markersize=4)
                # # RMS error
                # ax[1].plot(k, RB, 'o', color=bias_c, label='Biased ', markersize=4)
                # ax[1].plot(k, RU, '*', color=unbias_c, label='Unbiased', markersize=4)
                ms = 4
                ax[0].plot(k, CB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
                ax[0].plot(k, CU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')
                ax[1].plot(k, RB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
                ax[1].plot(k, RU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')

                CB, RB = CR(y_obs, y_obs_b)  # biased
                # Correlation
                bias_c = 'tab:red'
                ax[0].plot(k, CB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)
                # RMS error
                ax[1].plot(k, RB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)

                if flag:
                    # compute and plot the baseline correlation and MSE
                    if len(truth['b_true']) > 1:
                        b_truth = truth['b_true'][:len(y_filter)]
                        y_truth_u = y_truth - b_truth
                        Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
                        ax[0].plot((-10, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                        ax[1].plot((-10, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                    # compute C and R before the assimilation (the initial ensemble has some initialisation error)
                    Cpre, Rpre = CR(y_truth[istart - N_CR:istart + 1:],
                                    np.mean(y_filter, -1)[istart - N_CR:istart + 1:])
                    ax[0].plot((-10, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
                    ax[1].plot((-10, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
                    flag = False

            # =========================================================================================================
            xlims = [min(ks) - 0.2, max(ks) + 0.2]
            ax[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$')
            ax[1].set(ylabel='RMS error', ylim=[0., 1.], xlim=xlims, xlabel='$\\gamma$')

            for ax1 in ax:
                x0, x1 = ax1.get_xlim()
                y0, y1 = ax1.get_ylim()
                ax1.set_aspect((x1 - x0) / (y1 - y0))

    plt.savefig(figs_folder + 'Fig2_results_all_small.svg', dpi=350)


def barPlot(k0_U, k0_B, k10_U, k10_B, Ct, Rt, Cpre, Rpre, figs_folder):
    # =========================================================================================================
    barWidth = 0.1
    br1 = np.arange(len(k0_U))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    cols = ['b', 'c', 'r', 'coral']
    labels = ['$\\gamma = 0$, Unbiased', '$\\gamma = 0$, Biased',
              '$\\gamma = 10$, Unbiased', '$\\gamma = 10$, Biased']

    fig, ax = plt.subplots(1, 2, figsize=(15, 4), layout="constrained")
    for data, br, c, lb in zip([k0_U, k0_B, k10_U, k10_B], [br1, br2, br3, br4], cols, labels):
        C = np.array([x[0] for x in data]).squeeze()
        R = np.array([x[1] for x in data]).squeeze()
        ax[0].bar(br, C, color=c, width=barWidth, edgecolor='k', label=lb)
        ax[1].bar(br, R, color=c, width=barWidth, edgecolor='k', label=lb)

    for axi, cr in zip(ax, [(Ct, Cpre), (Rt, Rpre)]):
        axi.axhline(y=cr[0], color='lightgray', linewidth=4, label='Truth')
        axi.axhline(y=cr[1], color='k', linewidth=2, label='Pre-DA')
        axi.set_xticks([r + barWidth for r in range(len(k0_U))],
                       ['$L=1$', '$L=1$ + data augmentation', '$L=10$ + data augmentation'])
    ax[0].set(ylabel='Correlation', ylim=[.85, 1.02])
    ax[1].set(ylabel='RMS error', ylim=[0, 1])
    axi.legend(bbox_to_anchor=(1., 1.), loc="upper left")

    plt.savefig(figs_folder + 'WhyAugment.svg', dpi=350)
    plt.savefig(figs_folder + 'WhyAugment.pdf', dpi=350)

    # plt.show()


if __name__ == '__main__':
    # folder = 'results/VdP_12.07_newArch_3PE_25kmeas/'
    # Ls = [1, 10, 50, 100]
    # stds = [0.01, 0.1, 0.25]
    # ks = np.linspace(0., 50., 51)
    # plotResults(folder, stds, Ls, k_plot=(0.1,))
    # fig2(folder, Ls, stds)
    myfolder = 'results/VdP_12.12_augment/results/'

