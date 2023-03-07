import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
from Util import interpolate, CR, getEnvelope
from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from run import get_error_metrics

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=12)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


def post_process_loopParams(folder, k_plot=(None,)):
    if folder[-1] != '/':
        folder += '/'
    figs_folder = folder + 'figs/'
    if not os.path.isdir(figs_folder):
        os.makedirs(figs_folder)

    std_dirs = os.listdir(folder)
    for item in std_dirs:
        if not os.path.isdir(folder + item) or item[:3] != 'std':
            continue
        print(folder + item)
        std_folder = folder + item + '/'
        std = item.split('std')[-1]

        # Plot contours
        filename = '{}Contour_std{}_results'.format(folder, std)
        plot_Lk_contours(std_folder, filename)

        # Plot CR and means
        filename = '{}CR_std{}_results'.format(folder, std)
        post_process_multiple(std_folder, filename)

        # Plot timeseries
        if k_plot is not None:
            L_dirs = os.listdir(std_folder)
            for L_item in L_dirs:
                L_folder = std_folder + L_item + '/'
                if not os.path.isdir(L_folder) or L_item[0] != 'L':
                    print(L_folder)
                    continue
                L = L_item.split('L')[-1]
                for k_item in os.listdir(L_folder):
                    kval = float(k_item.split('_k')[-1])
                    print(kval)
                    if kval in k_plot:
                        with open(L_folder + k_item, 'rb') as f:
                            params = pickle.load(f)
                            truth = pickle.load(f)
                            filter_ens = pickle.load(f)
                        filename = '{}L{}_std{}_k{}_time'.format(figs_folder, L, std, kval)
                        print(filename)
                        post_process_single_SE_Zooms(filter_ens, truth, filename=filename)
                        filename = '{}L{}_std{}_k{}_J'.format(figs_folder, L, std, kval)
                        post_process_single(filter_ens, truth, params, filename=filename)


def post_process_WhyAugment(results_dir, figs_dir=None):
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    if not os.path.isdir(figs_dir):
        os.makedirs(figs_dir)

    flag = True
    ks, xtags = [], []
    mydirs = []
    for Ldir in sorted(os.listdir(results_dir), key=str.lower):
        if not os.path.isdir(results_dir + Ldir + '/') or len(Ldir.split('_Augment'))== 1:
            continue
        mydirs.append(Ldir)

    barData = [[] for _ in range(len(os.listdir(results_dir + mydirs[0] + '/')) * 2)]

    for Ldir in mydirs:
        values = Ldir.split('_L')[-1]
        print(Ldir.split('_Augment'))
        L, augment = values.split('_Augment')
        if augment == 'True':
            augment = True
        else:
            augment = False
        L = int(L.split('L')[-1])

        xtags.append('$L={}$'.format(L))
        if augment:
            xtags[-1] += '+ data augmentation'

        ii = -2

        k_files = sorted(os.listdir(results_dir + Ldir + '/'))
        for ff in k_files:
            ii += 2
            with open(results_dir + Ldir + '/' + ff, 'rb') as f:
                params = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)


            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            y_t, b_t = truth['y'], truth['b']

            # ESN estimated bias
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

            # Unbiased signal recovered through interpolation
            y_unbiased = y[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            y_unbiased = interpolate(t_b, y_unbiased, t)

            if flag:
                N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
                istop = np.argmin(abs(t - truth['t_obs'][params['num_DA'] - 1]))  # end of assimilation
                istart = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation

                istop_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                istart_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation

                # plt.figure()
                # plt.plot(y_t[istop_t:istop_t + N_CR])
                # plt.plot(y[istop:istop + N_CR, :, 0])
                # plt.show()

                y_t_u = y_t - b_t
                Ct, Rt = CR(y_t[istart_t - N_CR:istart_t + 1:], y_t_u[istart_t - N_CR:istart_t + 1:])
                Cpre, Rpre = CR(y_t[istart_t - N_CR:istart_t + 1:], np.mean(y, -1)[istart - N_CR:istart + 1:])

                ks.append(filter_ens.bias.k)

            # GET CORRELATION AND RMS ERROR =====================================================================
            CB, RB, CU, RU = [np.zeros(y.shape[-1]) for _ in range(4)]
            for mi in range(y.shape[-1]):
                CB[mi], RB[mi] = CR(y_t[istop_t:istop_t + N_CR], y[istop:istop + N_CR, :, mi])  # biased
                CU[mi], RU[mi] = CR(y_t[istop_t:istop_t + N_CR], y_unbiased[istop:istop + N_CR, :, mi])  # unbiased

            barData[ii].append((np.mean(CU), np.mean(RU), np.std(CU), np.std(RU)))
            barData[ii + 1].append((np.mean(CB), np.mean(RB), np.std(CB), np.std(RB)))


            filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_dir, L, augment, filter_ens.bias.k)
            post_process_single_SE_Zooms(filter_ens, truth, filename=filename)

        flag = False

    # --------------------------------------------------------- #
    cols = ['b', 'c', 'r', 'coral', 'g', '#c1fd95', 'k', 'gray', '#a87900', '#fbdd7e']
    labels = []
    for kk in ks:
        labels.append('$\\gamma = {}$, Unbiased'.format(kk))
        labels.append('$\\gamma = {}$, Biased'.format(kk))

    bar_width = 0.1
    bars = [np.arange(len(barData[0]))]
    for _ in range(len(ks)*2):
        bars.append([x + bar_width for x in bars[-1]])

    fig, ax = plt.subplots(1, 2, figsize=(15, 4), layout="constrained")

    for data, br, c, lb in zip(barData, bars, cols, labels):
        C = np.array([x[0] for x in data]).T.squeeze()
        R = np.array([x[1] for x in data]).T.squeeze()
        Cstd = np.array([x[2] for x in data]).T.squeeze()
        Rstd = np.array([x[3] for x in data]).T.squeeze()
        ax[0].bar(br, C, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[0].errorbar(br, C, yerr=Cstd, fmt='o', capsize=2., color='k', markersize=2)
        ax[1].bar(br, R, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[1].errorbar(br, R, yerr=Rstd, fmt='o', capsize=2., color='k', markersize=2)

    for axi, cr in zip(ax, [(Ct, Cpre), (Rt, Rpre)]):
        axi.axhline(y=cr[0], color='lightgray', linewidth=4, label='Truth')
        axi.axhline(y=cr[1], color='k', linewidth=2, label='Pre-DA')
        axi.set_xticks([r + bar_width for r in range(len(data))], xtags)

    ax[0].set(ylabel='Correlation', ylim=[.85, 1.02])
    ax[1].set(ylabel='RMS error', ylim=[0, Rpre*1.5])
    axi.legend(bbox_to_anchor=(1., 1.), loc="upper left")

    # plt.savefig(figs_dir + 'WhyAugment.svg', dpi=350)
    plt.savefig(figs_dir + 'WhyAugment.pdf', dpi=350)
    plt.show()


# ==================================================================================================================
def post_process_single_SE_Zooms(ensemble, true_data, filename=None):
    truth = true_data.copy()
    filter_ens = ensemble.copy()

    t_obs, obs = truth['t_obs'], truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    # %% ================================ PLOT time series, parameters and RMS ================================ #

    y_filter, labels = filter_ens.getObservableHist(), filter_ens.obsLabels
    if len(np.shape(y_filter)) < 3:
        y_filter = np.expand_dims(y_filter, axis=1)
    if len(np.shape(truth['y'])) < 3:
        truth['y'] = np.expand_dims(truth['y'], axis=-1)

    # normalise results

    hist = filter_ens.hist
    t = filter_ens.hist_t

    mean = np.mean(hist, -1, keepdims=True)
    y_mean = np.mean(y_filter, -1)
    std = np.std(y_filter[:, 0, :], axis=1)

    fig, ax = plt.subplots(1, 3, figsize=[17.5, 3.5], layout="constrained")
    params_ax, zoomPre_ax, zoom_ax = ax[:]
    x_lims = [t_obs[0] - .05, t_obs[-1] + .05]

    c = 'lightgray'
    zoom_ax.plot(truth['t'], truth['y'][:, 0], color=c, linewidth=8)
    zoomPre_ax.plot(truth['t'], truth['y'][:, 0], color=c, label='Truth', linewidth=8)
    zoomPre_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='black', linewidth=.8)
    zoom_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='black', linewidth=.8)

    zoom_ax.plot(t_obs, obs[:, 0], '.', color='r', markersize=10)
    zoomPre_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observations', markersize=10)
    if filter_ens.bias is not None:
        c = 'navy'
        b = filter_ens.bias.hist
        t_b = filter_ens.bias.hist_t

        if filter_ens.bias.name == 'ESN':
            y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
            spline = interp1d(t_b, y_unbiased, kind='cubic', axis=0, copy=True, bounds_error=False, fill_value=0)
            y_unbiased = spline(t)

            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs

            zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)
            washidx = np.argmin(abs(t - filter_ens.bias.washout_t[0]))
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)
            washidx = np.argmin(abs(t - filter_ens.bias.hist_t[0]))

        y_mean_u = np.mean(y_unbiased, -1)
        zoom_ax.plot(t, y_mean_u[:, 0], '-', label='Unbiased filtered signal', color=c, linewidth=1.5)
        zoomPre_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-', color=c,  linewidth=1.5)

    c = 'lightseagreen'
    zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased estimate', linewidth=1.5, alpha=0.9)
    zoomPre_ax.plot(t, y_mean[:, 0], '--', color=c, linewidth=1.5, alpha=0.9)
    zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoomPre_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

    zoomPre_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    zoom_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
    y_lims = [min(min(y_mean[:, 0]), min(truth['y'][:, 0])) * 1.2,
              max(max(y_mean[:, 0]), max(truth['y'][:, 0])) * 1.2]
    zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - 0.03, t_obs[-1] + 0.02], ylim=y_lims)
    zoomPre_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - 0.03, t_obs[0] + 0.02], ylim=y_lims)

    # PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
    ii = len(filter_ens.psi0)
    c = ['g', 'mediumpurple', 'sandybrown', 'cyan']
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

    if filename is not None:
        # plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
        plt.close()
    else:
        plt.show()


def post_process_single(filter_ens, truth, params, filename=None):
    filt = filter_ens.filt
    biasType = filter_ens.biasType
    Nt_extra = params['Nt_extra']

    t_obs, obs = truth['t_obs'], truth['p_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    # %% ================================ PLOT time series, parameters and RMS ================================ #

    y_filter, labels = filter_ens.getObservableHist(), filter_ens.obsLabels
    if len(np.shape(y_filter)) < 3:
        y_filter = np.expand_dims(y_filter, axis=1)
    if len(np.shape(truth['y'])) < 3:
        truth['y'] = np.expand_dims(truth['y'], axis=-1)

    # normalise results
    norm = 1.  # np.max(abs(truth['y'][:, 0]))
    y_filter /= norm
    truth['y'] /= norm

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

    x_lims = [t_obs[0] - filter_ens.t_CR, t_obs[-1] + filter_ens.t_CR]

    c = 'lightgray'
    p_ax.plot(truth['t'], truth['y'][:, 0], color=c, label='Truth', linewidth=4)
    zoom_ax.plot(truth['t'], truth['y'][:, 0], color=c, linewidth=8)
    zoomPre_ax.plot(truth['t'], truth['y'][:, 0], color=c, linewidth=8)
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

            t_wash = filter_ens.bias.washout_t
            wash = filter_ens.bias.washout_obs / norm

            p_ax.plot(t_wash, wash[:, 0], '.', color='r')
            zoomPre_ax.plot(t_wash, wash[:, 0], '.', color='r', markersize=10)
            washidx = np.argmin(abs(t - filter_ens.bias.washout_t[0]))
        else:
            y_unbiased = y_filter + np.expand_dims(b, -1)
            washidx = np.argmin(abs(t - filter_ens.bias.hist_t[0]))

        y_mean_u = np.mean(y_unbiased, -1)
        p_ax.plot(t, y_mean_u[:, 0], '-', color=c, linewidth=1.2)
        zoom_ax.plot(t, y_mean_u[:, 0], '-', color=c, linewidth=1.5)
        zoomPre_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-',
                        color=c, label='Unbiased filtered signal', linewidth=1.5)
        # BIAS PLOT
        bias_ax.plot(t_b, b[:, 0], alpha=0.75, label='ESN estimation')
        b_obs = truth['y'][:len(y_filter)] - y_filter

        b_mean = np.mean(b_obs, -1)
        b_lims = [np.min(b_obs), np.max(b_obs)]

        bias_ax.plot(t, b_mean[:, 0], '--', color='darkorchid', label='Bias')
        bias_ax.fill_between(t, b_mean[:, 0] + std, b_mean[:, 0] - std, alpha=0.5, color='darkorchid')

        bias_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)
        bias_ax.set(ylabel='Bias', xlabel='$t$')

    c = 'lightseagreen'  # '#021bf9'
    p_ax.plot(t, y_mean[:, 0], '--', color=c, linewidth=1.)
    zoom_ax.plot(t, y_mean[:, 0], '--', color=c, linewidth=1.5, alpha=0.9)
    zoomPre_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased filtered signal', linewidth=1.5, alpha=0.9)
    p_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
    zoomPre_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

    p_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Observation data')
    zoom_ax.plot(t_obs, obs[:, 0], '.', color='r', markersize=10)
    zoomPre_ax.plot(t_obs, obs[:, 0], '.', color='r', markersize=10)

    y_lims = [min(min(truth['y'][:, 0]), min(y_mean[:, 0])) * 1.05,
              max(max(truth['y'][:, 0]), max(y_mean[:, 0])) * 1.05]
    p_ax.set(ylabel="$p'_\mathrm{mic_1}$ [Pa]", xlabel='$t$ [s]', xlim=x_lims, ylim=y_lims)
    p_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=1)
    zoomPre_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=1)

    zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - filter_ens.t_CR, t_obs[-1] + filter_ens.t_CR],
                ylim=y_lims)
    zoomPre_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - filter_ens.t_CR, t_obs[0] + filter_ens.t_CR],
                   ylim=y_lims)
    bias_ax.set(xlabel='$t$ [s]', xlim=[t_obs[0] - filter_ens.t_CR, t_obs[-1] + filter_ens.t_CR],
                ylim=b_lims)

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
        params_ax.set(ylim=[min_p - filter_ens.t_CR, max_p + filter_ens.t_CR])

    # PLOT RMS ERROR
    Psi = (mean - hist)[:-Nt_extra]

    Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
    RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
    RMS_ax.plot(t[:-Nt_extra], RMS, color='firebrick')
    RMS_ax.set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')

    # PLOT COST FUNCTION
    J = np.array(filter_ens.hist_J).squeeze()
    J_ax.plot(t_obs, J[:, :-1])
    dJ_ax.plot(t_obs, J[:, -1], color='tab:red')

    dJ_ax.set(ylabel='$d\\mathcal{J}/d\\psi$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.set(ylabel='$\\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.legend(['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$',
                 '$\\mathcal{J}_{b}$'], bbox_to_anchor=(1., 1.),
                loc="upper left", ncol=1)

    if filename is not None:
        # plt.savefig(filename + '.svg', dpi=350)
        plt.savefig(filename + '.pdf', dpi=350)
        plt.close()
    else:
        plt.show()


# ==================================================================================================================
def post_process_multiple(folder, filename=None):
    # ==================================================================================================================
    with open(folder + 'CR_data', 'rb') as f:
        out = pickle.load(f)

    for Li in range(len(out['Ls'])):
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        fig.suptitle(folder)
        subfigs = fig.subfigures(2, 1)
        axCRP = subfigs[0].subplots(1, 3)
        mean_ax = subfigs[1].subplots(1, 2)

        # PLOT CORRELATION AND RMS ERROR =====================================================================
        ms = 4
        for lbl, col, fill, alph in zip(['biased', 'unbiased'], ['#20b2aae5', '#000080ff'], ['none', 'none'], [.6, .6]):
            for ax_i, key in enumerate(['C', 'R']):
                for suf, mk in zip(['_DA', '_post'], ['o', 'x']):
                    val = out[key + '_' + lbl + suf][Li]
                    axCRP[ax_i].plot(out['ks'], val, linestyle='none', marker=mk,
                                     color=col, label=lbl + suf, markersize=ms, alpha=alph, fillstyle=fill)

        # Plor true and pre-DA RMS and correlation---------------------------------
        for suffix, alph, lw in zip(['true', 'pre'], [.2, 1.], [5., 1.]):
            for ii, key in enumerate(['C_', 'R_']):
                val = out[key + suffix]
                axCRP[ii].plot((-10, 100), (val, val), '-', color='k', label=suffix, alpha=alph, linewidth=lw)

        xlims = [min(out['ks']) - .5, max(out['ks']) + .5]
        axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$', ylim=[.95 * out['C_pre'], 1.005])
        axCRP[1].set(ylim=[0., 1.2 * out['R_pre']], ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')

        # PLOT MEAN ERRORS ===============================================================================
        for mic in [0]:
            norm = colors.Normalize(vmin=0, vmax=max(out['ks'])+20)
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGn_r)
            for ax, lbl in zip(mean_ax, ['biased', 'unbiased']):
                for ki, kval in enumerate(out['ks']):
                    im = ax.plot(out['t_interp'], out['error_' + lbl][Li, ki, :, mic] * 100,
                            color=cmap.to_rgba(kval), lw=.9)
                ax.set(xlabel='$t$ [s]', xlim=[out['t_interp'][0], out['t_interp'][-1]])

            mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\%]')
            mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\%]')

        clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
        clb.ax.set_title('$\\gamma$')

        # PLOT PARAMETERS AND MEAN EVOLUTION ==========================================================================
        flag = True
        for file_k in os.listdir(out['L_dirs'][Li]):
            with open(out['L_dirs'][Li] + file_k, 'rb') as f:
                _ = pickle.load(f)
                _ = pickle.load(f)
                filter_ens = pickle.load(f)
            k = filter_ens.bias.k
            if filter_ens.est_p:
                if flag:
                    N_psi = len(filter_ens.psi0)
                    c = ['g', 'mediumpurple', 'sandybrown', 'r']
                    marker = ['x', '+']
                    time = ['$(t_\mathrm{end})$', '$(t_\mathrm{start})$']
                    alphas = [1., .2]
                    superscript = '^\mathrm{init}$'
                    reference_p = filter_ens.alpha0
                for pj, p in enumerate(filter_ens.est_p):
                    for kk, idx in enumerate([out['i1'], out['i0']]):
                        hist_p = filter_ens.hist[idx - 1, N_psi + pj] / reference_p[p]
                        if p in ['C1', 'C2']:
                            axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                              fmt=marker[kk], color=c[pj],
                                              label='$' + p + '/' + p + superscript + time[kk],
                                              capsize=2, markersize=ms, linewidth=.5)
                        else:
                            axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=alphas[kk],
                                              fmt=marker[kk], color=c[pj],
                                              label='$\\' + p + '/\\' + p + superscript + time[kk],
                                              capsize=4, markersize=4, linewidth=.8, mew=.8)
                if flag:
                    axCRP[2].legend()
                    for ax1 in axCRP[1:]:
                        ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
                    flag = False

        for ax1 in axCRP[:]:
            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            ax1.set_aspect((x1 - x0) / (y1 - y0))

        # SAVE PLOT ========================================================================================
        if filename is not None:
            # plt.savefig(filename + '.svg', dpi=350)
            plt.savefig(filename + '_L{}.pdf'.format(out['Ls'][Li]), dpi=350)
            plt.close()


# ==================================================================================================================
def fig2(folder, Ls, stds, figs_dir):
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

            for ff in files:
                if ff[-3:] == '.py' or ff[-4] == '.':
                    continue
                k = float(ff.split('_k')[-1])
                with open(files_folder + ff, 'rb') as f:
                    params = pickle.load(f)
                    truth = pickle.load(f)
                    filter_ens = pickle.load(f)
                # Observable bias
                y_filter, t_filter = filter_ens.getObservableHist(), filter_ens.hist_t
                y_truth = truth['y'][:len(y_filter)]
                if flag:
                    N_CR = int(filter_ens.t_CR / filter_ens.dt)  # Length of interval to compute correlation and RMS
                    istart = np.argmin(abs(t_filter - truth['t_obs'][0]))  # start of assimilation
                    istop = np.argmin(abs(t_filter - truth['t_obs'][params['num_DA'] - 1]))  # end of assimilation

                # ESN bias
                b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

                # Ubiased signal error
                y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
                y_unbiased = interpolate(t_b, y_unbiased, t_filter)

                ks.append(k)

                # PLOT CORRELATION AND RMS ERROR =====================================================================
                t_obs = truth['t_obs'][:params['num_DA']]
                y_obs = interpolate(t_filter, y_truth, t_obs)
                y_obs_b = interpolate(t_filter, np.mean(y_filter, -1), t_obs)

                CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
                CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased
                # Correlation
                bias_c = 'tab:red'
                unbias_c = 'tab:blue'
                ms = 4
                ax[0].plot(k, CB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
                ax[0].plot(k, CU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')
                ax[1].plot(k, RB, 'o', color=bias_c, label='Biased ', markersize=ms, alpha=0.6)
                ax[1].plot(k, RU, 'o', markeredgecolor=unbias_c, label='Unbiased', markersize=ms, fillstyle='none')

                CB, RB = CR(y_obs, y_obs_b)  # biased
                ax[0].plot(k, CB, '+', color=bias_c, label='Biased at $t^a$', markersize=ms)
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

    # plt.savefig(figs_dir + 'Fig2_results_all_small.svg', dpi=350)
    plt.savefig(figs_dir + 'Fig2_results_all_small.pdf', dpi=350)
    plt.close()


def plot_Lk_contours(folder, filename='contour'):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    for item in ['Ls', 'ks']:
        if not np.all(np.array(data[item])[:-1] <= np.array(data[item])[1:]):
            get_error_metrics(folder)
            with open(data_file, 'rb') as f:
                data = pickle.load(f)


    R_metrics = [np.log((data['R_biased_DA'] + data['R_unbiased_DA'])),
                 np.log((data['R_biased_post'] + data['R_unbiased_post']))]

    # C_metrics = [(data['C_biased_DA']+data['C_unbiased_DA']),
    #              (data['C_biased_post']+data['C_unbiased_post'])]

    metrics = R_metrics
    lbl = 'log(RMS_b + RMS_u)'  # , 'C_b + C_u']
    cmap = mpl.cm.Purples.copy()

    max_v = min(np.log(2.), max([np.max(metric) for metric in metrics]))
    min_v = min([np.min(metric) for metric in metrics])

    norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
    mylevs = np.linspace(min_v, max_v, 20)
    cmap.set_over('k')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), layout='tight')
    for ax, metric, title in zip(axs.flatten(), metrics, ['during DA', 'post-DA']):
        if max_v == np.log(2.):
            im = ax.contourf(data['ks'], data['Ls'], metric, cmap=cmap, norm=norm, levels=mylevs, extend='max')
        else:
            im = ax.contourf(data['ks'], data['Ls'], metric, cmap=cmap, norm=norm, levels=mylevs)

        fig.colorbar(im, ax=ax, label=lbl)
        ax.set(title=title, xlabel='$\\gamma$', ylabel='$L$')

        idx = np.argmin(metric)
        idx_i = idx // int(len(data['ks']))
        idx_j = idx % int(len(data['ks']))
        ax.plot(data['ks'][idx_j], data['Ls'][idx_i], 'ro')


    plt.savefig(filename + '.pdf', dpi=350)
    plt.close()



XDG_RUNTIME_DIR = 'tmp/'
if __name__ == '__main__':
    # autumcrisp = 'C:/Users/an553/OneDrive - University of Cambridge/PhD/My_papers/2022.11 - Model-error inference/'
    # myfolder = 'results/VdP_final_.3/'
    # figs_folder = myfolder + 'figs/'

    # fff = '/home/an553/Documents/PycharmProjects/Bias-EnKF/results/Rijke_test/m50_ESN200_L10/'
    #
    # filename = '{}results_CR'.format(fff+'figs/')
    # post_process_multiple(fff, filename)
    #
    # for m in [10, 30, 50, 70, 100]:
    #     myfolder = 'results/Rijke_test/m{}/'.format(m)

    myfolder = 'results/VdP_final_.3/'
    loop_folder = myfolder + 'results_loopParams/'

    # if not os.path.isdir(loop_folder):
    #     continue

    my_dirs = os.listdir(loop_folder)
    for std_item in my_dirs:
        if not os.path.isdir(loop_folder + std_item) or std_item[:3] != 'std':
            continue
        print(loop_folder + std_item)
        std_folder = loop_folder + std_item + '/'

        file = '{}Contour_std{}_results'.format(loop_folder, std_item.split('std')[-1])
        plot_Lk_contours(std_folder, file)

        file = '{}CR_std{}_results'.format(loop_folder, std_item.split('std')[-1])
        post_process_multiple(std_folder, file)
