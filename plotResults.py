import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation
from Util import interpolate, CR, getEnvelope, get_error_metrics
from scipy.interpolate import CubicSpline, interp1d, interp2d
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

XDG_RUNTIME_DIR = 'tmp/'

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

# Figures colors
color_true = 'lightgray'
color_unbias = '#000080ff'
color_bias = '#20b2aae5'
color_obs = 'r'
color_b = 'darkorchid'
colors_alpha = ['green', 'sandybrown',
                [0.7, 0.7, 0.87], 'blue', 'red', 'gold', 'deepskyblue']


def post_process_loopParams(results_dir, k_plot=(None,), figs_dir=None, k_max=100.):
    if results_dir[-1] != '/':
        results_dir += '/'
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    os.makedirs(figs_dir, exist_ok=True)

    std_dirs = os.listdir(results_dir)
    for item in std_dirs:
        if not os.path.isdir(results_dir + item) or item[:3] != 'std':
            continue

        std_dir = results_dir + item + '/'
        std = item.split('std')[-1]

        # Plot contours
        filename = '{}Contour_std{}_results'.format(figs_dir, std)
        plot_Lk_contours(std_dir, filename)

        if 'Rijke' not in results_dir:
            # Plot CR and means
            filename = '{}CR_std{}_results'.format(figs_dir, std)
            post_process_multiple(std_dir, filename, k_max=k_max)

            # Plot timeseries
            if k_plot is not None:
                L_dirs = os.listdir(std_dir)
                for L_item in L_dirs:
                    L_folder = std_dir + L_item + '/'
                    if not os.path.isdir(L_folder) or L_item[0] != 'L':
                        continue
                    L = L_item.split('L')[-1]
                    if L == '50':
                        for k_item in os.listdir(L_folder):
                            kval = float(k_item.split('_k')[-1])
                            if kval in k_plot:
                                with open(L_folder + k_item, 'rb') as f:
                                    params = pickle.load(f)
                                    truth = pickle.load(f)
                                    filter_ens = pickle.load(f)
                                filename = '{}L{}_std{}_k{}_J'.format(figs_dir, L, std, kval)
                                post_process_single(filter_ens, truth, params, filename=filename)


def post_process_WhyAugment(results_dir, k_plot=None, J_plot=None, figs_dir=None):
    if figs_dir is None:
        figs_dir = results_dir + 'figs/'
    os.makedirs(figs_dir, exist_ok=True)

    flag = True
    xtags, mydirs = [], []
    for Ldir in sorted(os.listdir(results_dir), key=str.lower):
        if not os.path.isdir(results_dir + Ldir + '/') or len(Ldir.split('_Augment')) == 1:
            continue
        mydirs.append(Ldir)

    k_files = []
    ks = []
    for ff in os.listdir(results_dir + mydirs[0] + '/'):
        k = float(ff.split('_k')[-1])
        if k_plot is not None and k not in k_plot:
            continue
        k_files.append(ff)
        ks.append(k)
    # sort ks and Ls
    idx_ks = np.argsort(np.array(ks))
    ks = np.array(ks)[idx_ks]
    k_files = [k_files[i] for i in idx_ks]

    colmap = mpl.colormaps['viridis'](np.linspace(0., 1., len(ks) * 2))

    barData = [[] for _ in range(len(ks) * 2)]

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
            xtags[-1] += '\n \\& data augment'
        ii = -2
        for ff in k_files:
            with open(results_dir + Ldir + '/' + ff, 'rb') as f:
                params = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)

            ii += 2
            truth = truth.copy()
            # ---------------------------------------------------------
            y, t = filter_ens.getObservableHist(), filter_ens.hist_t
            b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

            # Unbiased signal error
            y_unbiased = recover_unbiased_solution(t_b, b, t, y, upsample=hasattr(filter_ens.bias, 'upsample'))

            N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
            i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
            i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation

            # cut signals to interval of interest
            y, t, y_unbiased = y[i0 - N_CR:i1 + N_CR], t[i0 - N_CR:i1 + N_CR], y_unbiased[i0 - N_CR:i1 + N_CR]
            y_mean = np.mean(y, -1)

            if flag and ii == 0:
                i0_t = np.argmin(abs(truth['t'] - truth['t_obs'][0]))  # start of assimilation
                i1_t = np.argmin(abs(truth['t'] - truth['t_obs'][-1]))  # end of assimilation
                y_truth, t_truth = truth['y'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
                y_truth_b = y_truth - truth['b'][i0_t - N_CR:i1_t + N_CR]

                Ct, Rt = CR(y_truth[-N_CR:], y_truth_b[-N_CR:])
                Cpre, Rpre = CR(y_truth[:N_CR], y_mean[:N_CR:])

            # GET CORRELATION AND RMS ERROR =====================================================================
            CB, RB, CU, RU = [np.zeros(y.shape[-1]) for _ in range(4)]
            for mi in range(y.shape[-1]):
                CB[mi], RB[mi] = CR(y_truth[-N_CR:], y[-N_CR:, :, mi])  # biased
                CU[mi], RU[mi] = CR(y_truth[-N_CR:], y_unbiased[-N_CR:, :, mi])  # unbiased

            barData[ii].append((np.mean(CU), np.mean(RU), np.std(CU), np.std(RU)))
            barData[ii + 1].append((np.mean(CB), np.mean(RB), np.std(CB), np.std(RB)))

            if filter_ens.bias.k in J_plot:
                filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_dir, L, augment, filter_ens.bias.k)
                # print(filename)
                # post_process_single_SE_Zooms(filter_ens, truth, filename=filename)
                post_process_single(filter_ens, truth, params, filename=filename + '_J')

        flag = False

    # --------------------------------------------------------- #

    labels = []
    for kk in ks:
        labels.append('$\\gamma = {}$, U'.format(kk))
        labels.append('$\\gamma = {}$, B'.format(kk))

    bar_width = 0.1
    bars = [np.arange(len(barData[0]))]
    for _ in range(len(ks) * 2):
        bars.append([x + bar_width for x in bars[-1]])

    fig, ax = plt.subplots(1, 2, figsize=(14, 3), layout="constrained")

    for data, br, c, lb in zip(barData, bars, colmap, labels):
        C = np.array([x[0] for x in data]).T.squeeze()
        R = np.array([x[1] for x in data]).T.squeeze()
        Cstd = np.array([x[2] for x in data]).T.squeeze()
        Rstd = np.array([x[3] for x in data]).T.squeeze()
        ax[0].bar(br, C, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[0].errorbar(br, C, yerr=Cstd, fmt='o', capsize=2., color='k', markersize=2)
        ax[1].bar(br, R, color=c, width=bar_width, edgecolor='k', label=lb)
        ax[1].errorbar(br, R, yerr=Rstd, fmt='o', capsize=2., color='k', markersize=2)

    for axi, cr in zip(ax, [(Ct, Cpre), (Rt, Rpre)]):
        axi.axhline(y=cr[0], color=color_true, linewidth=4, label='Truth')
        axi.axhline(y=cr[1], color='k', linewidth=2, label='Pre-DA')
        axi.set_xticks([r + bar_width for r in range(len(data))], xtags)

    ax[0].set(ylabel='Correlation', ylim=[.85, 1.02])
    ax[1].set(ylabel='RMS error', ylim=[0, Rpre * 1.5])
    axi.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=2)

    plt.savefig(figs_dir + 'WhyAugment.svg', dpi=350)
    # plt.savefig(figs_dir + 'WhyAugment.pdf', dpi=350)
    plt.close()


# ==================================================================================================================
def recover_unbiased_solution(t_b, b, t, y, upsample=True):
    if upsample:
        y_unbiased = interpolate(t, y, t_b) + b
        y_unbiased = interpolate(t_b, y_unbiased, t)
    else:
        y_unbiased = y + b
    return y_unbiased


def post_process_single(filter_ens, truth, filename=None, mic=0, reference_p=None, plot_params=False):
    t_obs, obs = truth['t_obs'], truth['y_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    y_filter, t = filter_ens.getObservableHist(), filter_ens.hist_t
    b, t_b = filter_ens.bias.hist, filter_ens.bias.hist_t

    y_filter, b, obs = [yy[:, mic] for yy in [y_filter, b, obs]]
    y_mean = np.mean(y_filter, -1)

    b = interpolate(t_b, b, t)
    t_b = t
    y_unbiased = y_mean + b

    # y_unbiased = recover_unbiased_solution(t_b, b, t, y_mean, upsample=hasattr(filter_ens.bias, 'upsample'))

    # cut signals to interval of interest -----
    N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS

    i0 = np.argmin(abs(t - truth['t_obs'][0]))  # start of assimilation
    i1 = np.argmin(abs(t - truth['t_obs'][-1]))  # end of assimilation
    y_filter, y_mean, t, y_unbiased = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, t, y_unbiased])

    y_truth = interpolate(truth['t'], truth['y'][:, mic], t)

    std = abs(np.std(y_filter[:, :], axis=1))

    # %% PLOT time series ------------------------------------------------------------------------------------------

    norm = np.max(abs(y_truth), axis=0)  # normalizing constant
    if int(norm) == 1:
        ylbls = [["$p(x_\\mathrm{f})$ [Pa]", "$b(x_\\mathrm{f})$ [Pa]"], ['', '']]
    else:
        ylbls = [["$p(x_\\mathrm{f})$ norm.", "$b(x_\\mathrm{f})$ norm."], ['', '']]


    fig1 = plt.figure(figsize=[9, 5.5], layout="constrained")
    subfigs = fig1.subfigures(2, 1, height_ratios=[1, 1.1])
    ax_zoom = subfigs[0].subplots(2, 2, sharex='col', sharey='row')
    ax_all = subfigs[1].subplots(2, 1, sharex='col')

    y_lims = [np.min(y_truth[:N_CR]) * 1.1, np.max(y_truth[:N_CR]) * 1.1] / norm
    x_lims = [[t[0], t[0] + 2 * filter_ens.t_CR],
              [t[-1] - 2 * filter_ens.t_CR, t[-1]],
              [t[0], t[-1]]]

    if filter_ens.bias.name != 'None' and hasattr(filter_ens.bias, 'N_wash'):
        t_wash, wash = filter_ens.bias.wash_time, filter_ens.bias.wash_obs[:, mic]

    b_obs = y_truth - y_mean
    y_lims_b = [np.min(b_obs[:N_CR]) * 1.1, np.max(b_obs[:N_CR]) * 1.1] / norm

    if filter_ens.est_a:
        hist, hist_t = filter_ens.hist, filter_ens.hist_t
        hist_mean = np.mean(hist, -1, keepdims=True)
        mean_p, std_p, labels_p = [], [], []

        max_p, min_p = -np.infty, np.infty

        if reference_p is None:
            twin, reference_p = False, filter_ens.alpha0
            norm_lbl = lambda x: '/' + x + '$^0$'
        else:
            twin = True
            norm_lbl = lambda x: '/' + x + '$^\\mathrm{true}$'

        ii = filter_ens.Nphi
        for p in filter_ens.est_a:
            m = hist_mean[:, ii].squeeze() / reference_p[p]
            s = abs(np.std(hist[:, ii] / reference_p[p], axis=1))
            max_p, min_p = max(max_p, max(m + 2 * s)), min(min_p, min(m - 2 * s))
            labels_p.append(norm_lbl(filter_ens.param_labels[p]))
            mean_p.append(m)
            std_p.append(s)
            ii += 1

    for axs in [ax_zoom[:, 0], ax_zoom[:, 1]]:
        # Observables ---------------------------------------------------------------------
        axs[0].plot(t, y_truth / norm, color=color_true, linewidth=5, label='t')
        axs[0].plot(t, y_unbiased / norm, '-', color=color_unbias, linewidth=1., label='u')
        axs[0].plot(t, y_mean / norm, '--', color=color_bias, linewidth=1., alpha=0.9, label='b')
        axs[0].fill_between(t, (y_mean + std) / norm, (y_mean - std) / norm, alpha=0.2, color='lightseagreen')
        for mi in range(y_filter.shape[1]):
            axs[0].plot(t, y_filter[:, mi] / norm, lw=.5, color='lightseagreen')

        axs[0].plot(t_obs, obs / norm, '.', color=color_obs, markersize=6, label='o')
        axs[0].set(ylim=y_lims)

        # BIAS ---------------------------------------------------------------------
        if 'wash' in locals().keys():
            axs[0].plot(t_wash, wash / norm, '.', color=color_obs, markersize=6)
        axs[1].plot(t, b_obs / norm, color=color_b, label='O', alpha=0.4, linewidth=3)
        axs[1].plot(t_b, b / norm, color=color_b, label=filter_ens.bias.name, linewidth=.8)

    for axs in [ax_all]:
        axs[0].plot(t, b_obs / norm, color=color_b, label='O', alpha=0.4, linewidth=3)
        axs[0].plot(t_b, b / norm, color=color_b, label=filter_ens.bias.name, linewidth=.8)
        axs[0].set(ylabel=ylbls[0][1], xlim=x_lims[-1], ylim=y_lims_b)
        # PARAMS-----------------------
        if filter_ens.est_a:
            for p, m, s, c, lbl in zip(filter_ens.est_a, mean_p, std_p, colors_alpha, labels_p):
                axs[1].plot(hist_t, m, color=c, label=lbl)
                axs[1].set(xlabel='$t$')
                axs[1].fill_between(hist_t, m + s, m - s, alpha=0.2, color=c)

            axs[1].set(xlabel='$t$ [s]', ylabel="", xlim=x_lims[-1], ylim=[min_p, max_p])
            axs[1].plot((t_obs[0], t_obs[0]), (min_p, max_p), '--', color='k', linewidth=.8)  # DA window
            axs[1].plot((t_obs[-1], t_obs[-1]), (min_p, max_p), '--', color='k', linewidth=.8)  # DA window
            if twin:
                axs[1].plot((x_lims[-1][0], x_lims[-1][-1]), (1, 1), '-', color='k', linewidth=.6)  # DA window
            if num_DA_blind > 0:
                axs[1].plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue')
                axs[1].plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue',
                            label='Start BE')
            if num_SE_only > 0:
                axs[1].plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet')
                axs[1].plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet',
                            label='Start PE')
            # axs.legend(loc='best', orientation='horizontal', ncol=3)
            axs[1].legend(loc='upper left', bbox_to_anchor=(1., 1.), ncol=1, fontsize='xx-small')
            axs[1].set(ylabel='')
            axs[0].legend(loc='upper left', bbox_to_anchor=(1., 1.), ncol=1, fontsize='xx-small')

    # axis labels and limits
    for axs, xl, ylbl in zip([ax_zoom[:, 0], ax_zoom[:, 1]], x_lims, ylbls):
        axs[0].set(ylabel=ylbl[0], xlim=xl)
        axs[1].set(ylabel=ylbl[1], xlim=xl, ylim=y_lims_b, xlabel='$t$ [s]')
        for ax_ in axs:
            ax_.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window
            ax_.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8)  # DA window

    for axs_ in [ax_zoom[:, 0], ax_all]:
        for ax_ in axs_:
            ax_.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for ax_ in ax_zoom[:, 1]:
        ax_.legend(loc='upper left', bbox_to_anchor=(1., 1.), ncol=1, fontsize='xx-small')

    if plot_params:
        plot_parameters(filter_ens, truth, filename=filename, reference_p=reference_p)

    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.close()


# ==================================================================================================================
def post_process_multiple(folder, filename=None, k_max=100., L_plot=None, reference_p=None):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        out = pickle.load(f)

    xlims = [min(out['ks']) - .5, min(k_max, max(out['ks'])) + .5]

    if L_plot is not None:
        Li_plot = [np.argmin(abs(out['Ls'] - l)) for l in L_plot]
    else:
        Li_plot = range(len(out['Ls']))

    for Li in Li_plot:
        fig = plt.figure(figsize=(13, 5), layout="constrained")

        subfigs = fig.subfigures(1, 2, width_ratios=[2.45, 2])
        axCRP = subfigs[0].subplots(1, 3)
        mean_ax = subfigs[1].subplots(1, 2)

        # PLOT CORRELATION AND RMS ERROR  -------------------------------------------------------------------
        ms = 4
        for lbl, col in zip(['biased', 'unbiased'], [color_bias, color_unbias]):
            for ax_i, key in enumerate(['C', 'R']):
                for suf, mk in zip(['_DA', '_post'], ['o', 'x']):
                    val = out[key + '_' + lbl + suf][Li]
                    axCRP[ax_i].plot(out['ks'], val, linestyle='none', marker=mk, color=col,
                                     label=lbl[0] + suf, markersize=ms, alpha=.6, fillstyle='none')

        # Plor true and pre-DA RMS and correlation---------------------------------
        for suffix, alph, lw in zip(['true', 'pre'], [.2, 1.], [5., 1.]):
            for ii, key in enumerate(['C_', 'R_']):
                val = out[key + suffix]
                axCRP[ii].plot((-10, 100), (val, val), '-', color='k', label=suffix, alpha=alph, linewidth=lw)

        axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$', ylim=[.95 * out['C_pre'], 1.005])
        axCRP[1].set(ylim=[0., 1.5 * out['R_pre']], ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')
        axCRP[2].set(xlim=xlims, xlabel='$\\gamma$')

        # PLOT MEAN ERRORS --------------------------------------------------------------------------------------
        for mic in [0]:
            norm = colors.Normalize(vmin=0, vmax=min(k_max, max(out['ks'])) * 1.25)
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlGn_r)
            for ax, lbl in zip(mean_ax, ['biased', 'unbiased']):
                for ki, kval in enumerate(out['ks']):
                    if kval <= k_max:
                        ax.plot(out['t_interp'], out['error_' + lbl][Li, ki, :, mic] * 100,
                                color=cmap.to_rgba(kval), lw=.9)
                ax.set(xlabel='$t$ [s]', xlim=[out['t_interp'][0], out['t_interp'][-1]])

            mean_ax[0].set(ylim=[0, 60], ylabel='Biased signal error [\\%]')
            mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\\%]')
            mean_ax[1].set_yticks([0., 3., 6., 9.])

        clb = fig.colorbar(cmap, ax=mean_ax.ravel().tolist(), orientation='horizontal', shrink=0.5)
        clb.set_ticks(np.linspace(min(out['ks']), min(k_max, max(out['ks'])), 5))

        # PLOT PARAMETERS AND MEAN EVOLUTION -------------------------------------------------------------------
        flag = True
        for file_k in os.listdir(out['L_dirs'][Li]):
            with open(out['L_dirs'][Li] + file_k, 'rb') as f:
                _ = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
            k = filter_ens.bias.k
            if k > k_max:
                continue
            if filter_ens.est_a:
                if flag:
                    N_psi = filter_ens.Nphi
                    i0 = np.argmin(abs(filter_ens.hist_t - truth['t_obs'][0]))  # start of assimilation
                    i1 = np.argmin(abs(filter_ens.hist_t - truth['t_obs'][-1]))  # end of assimilation
                    lbl0, lbl1 = [], []
                    ii = filter_ens.Nphi - 1
                    for p in filter_ens.est_a:
                        ii += 1
                        if p in ['C1', 'C2']:
                            lbl0.append('$' + p)
                            lbl1.append('/' + p + '$^0$')
                        else:
                            lbl0.append('$\\' + p)
                            lbl1.append('/' + p + '$^0$')

                if reference_p is None:
                    reference_p = filter_ens.alpha0

                for pj, p in enumerate(filter_ens.est_a):
                    for idx, a, tt, mk in zip([-1, 0], [1., .2],
                                              ['(t_\\mathrm{end})', '^0'], ['x', '+']):
                        hist_p = filter_ens.hist[idx, N_psi + pj] / reference_p[p]
                        lbl = lbl0[pj] + tt + lbl1[pj]

                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=a, mew=.8, fmt=mk,
                                          color=colors_alpha[pj], label=lbl, capsize=4, markersize=4, linewidth=.8)
                if flag:
                    axCRP[2].legend()
                    for ax1 in axCRP[1:]:
                        ax1.legend(loc='best', bbox_to_anchor=(0., 1., 1., 1.), ncol=2, fontsize='xx-small')
                    flag = False

        for ax1 in axCRP[:]:
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.set_aspect(1. / ax1.get_data_ratio())

        t_obs = truth['t_obs']
        for ax1 in mean_ax[:]:
            ax1.set_aspect(0.8 / ax1.get_data_ratio())
            ax1.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='k', linewidth=.8, alpha=.5)  # DA window
            ax1.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='k', linewidth=.8, alpha=.5)  # DA window

        # SAVE PLOT -------------------------------------------------------------------
        if filename is not None:
            plt.savefig(filename + '_L{}.svg'.format(out['Ls'][Li]), dpi=350)
            plt.close()


def post_process_pdf(filter_ens, truth, params, filename=None, reference_p=None, normalize=True):
    if not filter_ens.est_a:
        raise ValueError('no parameters to infer')

    fig1 = plt.figure(figsize=[7.5, 3 * filter_ens.Na], layout="constrained")
    ax_all = fig1.subplots(filter_ens.Na, 1)
    if filter_ens.Na == 1:
        ax_all = [ax_all]

    hist, hist_t = filter_ens.hist, filter_ens.hist_t
    t_obs, dt_obs = truth['t_obs'], truth['t_obs'][1] - truth['t_obs'][0]

    idx_t = [np.argmin(abs(hist_t - t_o)) for t_o in t_obs]
    idx_t = np.array(idx_t)

    x_lims = [t_obs[0] - dt_obs, t_obs[-1] + dt_obs]
    hist_alpha, labels_p = [], []

    max_p, min_p = -np.infty, np.infty

    if normalize:
        if reference_p is None:
            reference_p = filter_ens.alpha0
            norm_lbl = lambda x: x + '/' + x + '$^0$'
            twin = False
        else:
            norm_lbl = lambda x: '/{' + x + '}^\mathrm{true}'
            twin = True
            k_twin = 1.
    else:
        k_twin = reference_p.copy()
        reference_p = dict()
        for p in filter_ens.est_a:
            reference_p[p] = 1.
            norm_lbl = lambda x: ''
        twin = True

    # REJECTED ANALYSIS ---------------------------------------------------------------------
    if hasattr(filter_ens, 'rejected_analysis'):
        lbl = ['rejected posterior', 'prior', 'likelihood']
        for rejection in filter_ens.rejected_analysis:
            for t_r, reject_posterior, prior, correction in rejection:
                ii = 0
                for p in filter_ens.est_a:
                    a = reject_posterior[ii] / reference_p[p]
                    plot_violins(ax_all[ii], [a], [t_r], color='r', widths=dt_obs / 2, label=lbl[0])
                    a = prior[ii] / reference_p[p]
                    plot_violins(ax_all[ii], [a], [t_r], color='y', widths=dt_obs / 2, label=lbl[1])
                    if correction is not None:
                        rng = np.random.default_rng(6)
                        a_c = rng.multivariate_normal(correction[0], correction[1], filter_ens.m)
                        print(a_c.shape)
                        plot_violins(ax_all[ii], a_c, [t_r], color='grey', widths=dt_obs / 2, label=lbl[-1])

                    ii += 1
            lbl = [None, None]
    # PARAMS ---------------------------------------------------------------------
    ii = filter_ens.Nphi
    for p in filter_ens.est_a:
        m = [hist[ti, ii] / reference_p[p] for ti in idx_t]
        max_p, min_p = max(max_p, np.max(m)), min(min_p, np.min(m))
        if p not in ['C1', 'C2']:
            p = '\\' + p
        labels_p.append('$' + p + norm_lbl(p) + '$')
        hist_alpha.append(m)
        ii += 1

    for ax, p, a, c, lbl in zip(ax_all, filter_ens.est_a, hist_alpha, colors_alpha, labels_p):
        plot_violins(ax, a, t_obs, widths=dt_obs / 2, color=c, label='analysis posterior')
        alpha_lims = [filter_ens.param_lims[p][0] / reference_p[p],
                      filter_ens.param_lims[p][1] / reference_p[p]]
        for lim in alpha_lims:
            ax.plot([hist_t[0], hist_t[-1]], [lim, lim], '--', color=c, lw=0.8)
        if twin:
            ax.plot((x_lims[0], x_lims[-1]), (k_twin[p], k_twin[p]), '-', color='k', linewidth=.6)  # DA window
        ax.set(xlabel='$t$ [s]', ylabel=lbl, xlim=x_lims,
               ylim=[alpha_lims[0] - 0.1 * abs(alpha_lims[1]), alpha_lims[1] + 0.1 * abs(alpha_lims[1])])

        ax.legend()
    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.close()


def plot_violins(ax, values, location, widths=None, color='b', label=None):
    if widths is None:
        violins = ax.violinplot(values, positions=location)
    else:
        violins = ax.violinplot(values, positions=location, widths=widths)
    for vp in violins['bodies']:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(.5)
        vp.set_alpha(0.5)
        vert = vp.get_paths()[0].vertices[:, 0]
        vp.get_paths()[0].vertices[:, 0] = np.clip(vert, np.mean(vert), np.inf)
    if label is not None:
        vp.set_label(label)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violins[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(.75)

    # if label is not None:
    #     ax.legend([violins['bodies'][0]], [label])


def plot_parameters(filter_ens, truth, filename=None, reference_p=None, twin=False):


        if len(filter_ens.est_a) < 4:
            fig1 = plt.figure(figsize=[6, 1.5*len(filter_ens.est_a)], layout="constrained")
            axs = fig1.subplots(len(filter_ens.est_a), 1, sharex='col')
            if len(filter_ens.est_a) == 1:
                axs = [axs]
        else:
            fig1 = plt.figure(figsize=[12, 1.5*int(round(len(filter_ens.est_a)/2))], layout="constrained")
            axs = fig1.subplots(int(round(len(filter_ens.est_a)/2)), 2, sharex='all')
            axs = axs.ravel()

        hist, hist_t = filter_ens.hist, filter_ens.hist_t
        hist_mean = np.mean(hist, -1, keepdims=True)

        t_obs = truth['t_obs']

        x_lims = [t_obs[0], t_obs[-1]]
        mean_p, std_p, labels_p = [], [], []

        if reference_p is None:
            norm_lbl = lambda x:  x
            reference_p = {**filter_ens.alpha0}
            for key in filter_ens.est_a:
                reference_p[key] = 1.
        else:
            norm_lbl = lambda x: x + '/' + x + '$^\\mathrm{ref}$'

        ii = filter_ens.Nphi
        for p in filter_ens.est_a:
            m = hist_mean[:, ii].squeeze() / reference_p[p]
            s = abs(np.std(hist[:, ii] / reference_p[p], axis=1))
            labels_p.append(norm_lbl(filter_ens.param_labels[p]))
            mean_p.append(m)
            std_p.append(s)
            ii += 1

        for ax, p, m, s, c, lbl in zip(axs, filter_ens.est_a, mean_p, std_p, colors_alpha, labels_p):
            max_p = np.max(m + abs(s))
            min_p = np.min(m - abs(s))
            ax.plot(hist_t, m, color=c, label=lbl)
            ax.fill_between(hist_t, m + abs(s), m - abs(s), alpha=0.2, color=c)
            if filter_ens.param_lims[p][0] is not None and filter_ens.param_lims[p][1] is not None:
                for lim in [filter_ens.param_lims[p][0] / reference_p[p],
                            filter_ens.param_lims[p][1] / reference_p[p]]:
                    ax.plot([hist_t[0], hist_t[-1]], [lim, lim], '--', color=c, lw=2, alpha=0.5)

            if twin:
                val = reference_p[p]
                min_p, max_p = min(min_p, val)-min(s), max(max_p, val)+max(s)
                ax.plot((hist_t[0], hist_t[-1]), (val, val), '-', color='k', linewidth=.6, label='truth')

            for idx, cl, ll in zip(['num_DA_blind', 'num_SE_only'], ['darkblue', 'darkviolet'], ['BE', 'PE']):
                idx = getattr(filter_ens, idx)
                if idx > 0:
                    ax.plot((t_obs[idx], t_obs[idx]), (min_p, max_p), '-.', color=cl)
                    ax.plot((t_obs[idx], t_obs[idx]), (min_p, max_p), '-.', color=cl, label='Start '+ll)

            ax.plot((t_obs[0], t_obs[0]), (min_p, max_p), '--', color='k', linewidth=.8)  # DA window
            ax.plot((t_obs[-1], t_obs[-1]), (min_p, max_p), '--', color='k', linewidth=.8)  # DA window
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            ax.set(ylabel='', ylim=[min_p, max_p])

        axs[-1].set(xlabel='$t$ [s]', xlim=x_lims)

        if filename is not None:
            plt.savefig(filename + '_params.svg', dpi=350)

# ==================================================================================================================


def plot_Lk_contours(folder, filename='contour'):
    data_file = folder + 'CR_data'
    if not os.path.isfile(data_file):
        get_error_metrics(folder)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    data = data.copy()
    # -------------------------------------------------------------------------------------------------- #
    R_metrics = [data['R_biased_DA'], data['R_unbiased_DA'],
                 data['R_biased_post'], data['R_unbiased_post']]
    min_idx = [np.argmin(metric) for metric in R_metrics]
    all_datas = [metric.flatten() for metric in R_metrics]
    R_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
              '({:.4},{:.4})'.format(all_datas[0][min_idx[1]], all_datas[1][min_idx[1]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[2]], all_datas[3][min_idx[2]]),
              '({:.4},{:.4})'.format(all_datas[2][min_idx[3]], all_datas[3][min_idx[3]])]
    R_metrics = [np.log(metric) for metric in R_metrics]  # make them logs

    R_lbls = ['log(RMS_b)', 'log(RMS_u)', 'log(RMS_b)', 'log(RMS_u)']
    R_titles = ['during DA', 'during DA', 'post-DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    log_metrics = [(data['R_biased_DA'] + data['R_unbiased_DA']),
                   (data['R_biased_post'] + data['R_unbiased_post'])]
    log_metrics = [np.log(metric) for metric in log_metrics]  # make them logs

    min_idx = [np.argmin(metric) for metric in log_metrics]
    log_text = ['({:.4},{:.4})'.format(all_datas[0][min_idx[0]], all_datas[1][min_idx[0]]),
                '({:.4},{:.4})'.format(all_datas[2][min_idx[1]], all_datas[3][min_idx[1]])]
    log_lbls = ['log(RMS_b + RMS_u)', 'log(RMS_b + RMS_u)']
    log_titles = ['during DA', 'post-DA']
    # -------------------------------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    fig.suptitle('true R_b = {:.4}, preDA R_b = {:.4}'.format(np.min(data['R_true']), (np.min(data['R_pre']))))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])
    subfigs[0].set_facecolor('0.85')

    axs_0 = subfigs[0].subplots(2, 2)
    axs_1 = subfigs[1].subplots(2, 1)

    # Original and interpolated axes
    xo, yo = data['ks'], data['Ls']
    xm = np.linspace(min(data['ks']), max((data['ks'])), 100)
    ym = np.linspace(min(data['Ls']), max((data['Ls'])), 100)

    # cols = [[0.88, 1.0, 1.0],
    #         [0.42, 0.35, 0.8],
    #         [.55, 0.0, 0.0]
    #         ]
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("MyCmapName", cols)

    # Select colormap
    cmap = mpl.cm.RdBu_r.copy()
    cmap.set_over('k')
    for axs, metrics, lbls, ttls, txts in zip([axs_0, axs_1], [R_metrics, log_metrics], [R_lbls, log_lbls],
                                              [R_titles, log_titles], [R_text, log_text]):
        max_v = min(0.1, max([np.max(metric) for metric in metrics]))
        min_v = min([np.min(metric) for metric in metrics])
        if lbls[0] == log_lbls[0]:
            # min_v, max_v = np.log(data['R_true']), np.log(data['R_pre'])
            min_v, max_v = -1.5, 0.2
            # min_v, max_v = 0.25, 1.25

        # norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
        norm = mpl.colors.TwoSlopeNorm(vmin=min_v, vcenter=np.log(0.5), vmax=max_v)
        # norm = mpl.colors.TwoSlopeNorm(vmin=np.exp(min_v), vcenter=0.5, vmax=np.exp(max_v))
        mylevs = np.linspace(min_v, max_v, 11)

        # Create subplots ----------------------------
        for ax, metric, titl, lbl, txt in zip(axs.flatten(), metrics, ttls, lbls, txts):
            func = interp2d(xo, yo, metric, kind='linear')
            zm = func(xm, ym)
            # im = ax.contourf(xm, ym, zm, cmap=cmap, norm=norm, extend='both', locator=10)
            im = ax.contourf(xm, ym, zm, levels=mylevs, cmap=cmap, norm=norm, extend='both')

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_aspect(1. / ax.get_data_ratio())

            im.cmap.set_over('k')
            ax.set(title=titl, xlabel='$\\gamma$', ylabel='$L$')

            # find minimum point
            idx = np.argmin(metric)
            idx_j = idx // int(len(data['ks']))
            idx_i = idx % int(len(data['ks']))
            ki, Lj = data['ks'][idx_i], data['Ls'][idx_j]
            ax.plot(ki, Lj, 'ro', markersize=3)
            ax.annotate('({}, {}), {:.4}\n {}'.format(ki, Lj, np.min(metric), txt),
                        xy=(ki, Lj), xytext=(0., 0.), bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 5})

            plt.colorbar(im, ax=ax, label=lbl, shrink=0.5, ticks=mylevs[::2], format=FormatStrFormatter('%.3f'))

    plt.savefig(filename + '.svg', dpi=350)
    plt.close()

    # ================================================================================================================
    # minimum at post-DA is the file of interest. Plot best solution timeseties
    k_file = '{}L{}/{}'.format(folder, int(data['Ls'][idx_j]), data['k_files'][idx_i])
    with open(k_file, 'rb') as f:
        params = pickle.load(f)
        truth = pickle.load(f)
        filter_ens = pickle.load(f)

    post_process_single(filter_ens, truth, params, filename=filename + '_optimal_solution_J')

    filename = filename + '_optimal_solution_CR'
    post_process_multiple(folder, filename, k_max=20., L_plot=[70])


def plot_Rijke_animation(folder, figs_dir):
    files = os.listdir(folder)

    for ff in files:
        try:
            with open(folder + ff, 'rb') as f:
                params = pickle.load(f)
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
        except:
            continue
        os.makedirs(figs_dir, exist_ok=True)
        filename = '{}results_{}_{}_J'.format(figs_dir, filter_ens.filt, filter_ens.bias.name)

        post_process_single(filter_ens, truth, params, filename=filename, mic=0)

        if filter_ens.filt == 'EnKF':
            filter_ens_BB = filter_ens.copy()
            print('ok1')
        elif filter_ens.filt == 'rBA_EnKF':
            filter_ens_BA = filter_ens.copy()
            print('ok2')

    # load truth
    name_truth = truth['name'].split('_{}'.format(truth['true_params']['manual_bias']))[0]
    with open('data/Truth_{}'.format(name_truth), 'rb') as f:
        truth_ens = pickle.load(f)

    # extract history of observables along the tube
    locs = np.linspace(0, filter_ens.L, 20)
    locs_obs = filter_ens.x_mic

    # Bias-aware EnKF sol
    y_BA = filter_ens_BA.getObservableHist(loc=locs)
    y_BA = np.mean(y_BA, -1)
    # Bias-blind EnKF sol
    y_BB = filter_ens_BB.getObservableHist(loc=locs)
    y_BB = np.mean(y_BB, -1)
    # truth
    y_t = truth_ens.getObservableHist(loc=locs).squeeze()
    y_t += .4 * y_t * np.sin((np.expand_dims(truth_ens.hist_t, -1) * np.pi * 2) ** 2)
    y_t = interpolate(truth_ens.hist_t, y_t, filter_ens.hist_t)

    max_v = [np.max(abs(yy)) for yy in [y_t, y_BA, y_BB]]
    max_v = np.max(max_v)

    # -----------------------

    # fig = plt.figure(figsize=[15, 5], layout='constrained')
    # sub_figs = fig.subfigures(2, 1)
    # ax1 = sub_figs[0].subplots(1, 3, gridspec_kw={'width_ratios': [1, 5, 1]})
    # ax1[0].axis('off')
    # ax1[2].axis('off')
    # ax1 = ax1[1]
    # ax2 = sub_figs[1].subplots(1, 2)

    fig1 = plt.figure(figsize=[10, 2], layout='constrained')
    ax1 = fig1.subplots(1, 1)

    fig2 = plt.figure(figsize=[12, 6], layout='constrained')
    ax2 = fig2.subplots(2, 1)

    t0 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][0] - filter_ens.t_CR / 2)))
    t1 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][-1] + filter_ens.t_CR)))

    t_gif = filter_ens.hist_t[t0:t1:5]

    # all pressure points
    y_BA = filter_ens_BA.getObservableHist(loc=locs)
    y_BB = filter_ens_BB.getObservableHist(loc=locs)
    y_t, y_BB, y_BA = [interpolate(filter_ens.hist_t, yy, t_gif) for yy in [y_t, y_BB, y_BA]]
    max_v = np.max(abs(y_t))

    # observation points
    y_BB_obs = filter_ens_BB.getObservableHist()
    y_BA_obs = filter_ens_BA.getObservableHist()

    y_BA_obs = y_BA_obs + interpolate(filter_ens_BA.bias.hist_t, filter_ens_BA.bias.hist, filter_ens_BA.hist_t)

    y_BA_obs = interpolate(filter_ens_BA.bias.hist_t, y_BA_obs, t_gif)
    y_BB_obs = interpolate(filter_ens_BB.hist_t, y_BB_obs, t_gif)

    # parameters
    reference_p = filter_ens_BA.alpha0
    alpha_BA, std_BA, alpha_BB, std_BB = [], [], [], []
    hist_BA, hist_BB = filter_ens_BA.hist, filter_ens_BB.hist
    for pi, p in enumerate(filter_ens.est_a):
        print(pi)
        ii = filter_ens.Nphi + pi
        alpha_BA.append(np.mean(hist_BA[:, ii], -1) / reference_p[p])
        std_BA.append(abs(np.std(hist_BA[:, ii] / reference_p[p], axis=1)))
        alpha_BB.append(np.mean(hist_BB[:, ii], -1) / reference_p[p])
        std_BB.append(abs(np.std(hist_BB[:, ii] / reference_p[p], axis=1)))

    max_p = max([np.max(np.array(a) + np.array(s)) for a, s in zip([alpha_BA, alpha_BB], [std_BA, std_BB])])
    min_p = min([np.min(np.array(a) - np.array(s)) for a, s in zip([alpha_BA, alpha_BB], [std_BA, std_BB])])

    params_legend = []
    for filter_name in ['EnKF', 'BA-EnKF']:
        for p in filter_ens.est_a:
            params_legend.append('$\\' + p + '$ ' + filter_name)

    # timeseries
    y_BA_tt = filter_ens_BA.getObservableHist()[:, 0]
    y_BA_tt_u = y_BA_tt + interpolate(filter_ens_BA.bias.hist_t, filter_ens_BA.bias.hist, filter_ens_BA.hist_t)


    y_BB_tt = filter_ens_BB.getObservableHist()[:, 0]
    y_t_tt = truth['y'][:, 0]
    y_obs_tt = truth['p_obs'][:, 0]
    pressure_legend = ['Truth', 'Data', 'State + bias  BA', 'State est. BA', 'State est.']

    def animate1(ai):
        ax1.clear()
        ax1.set(ylim=[-max_v, max_v], xlim=[0, 1], title='$t={:.4}$'.format(t_gif[ai]),
                xlabel='$x/L$', ylabel="$p'$ [Pa]")
        ax1.plot(locs, y_t[ai], color=color_true, linewidth=3)
        for loc in filter_ens.x_mic:
            ax1.plot([loc, loc], [0.7 * max_v, max_v], '.-', color='black', linewidth=2)
        ax1.plot([filter_ens.x_mic[0], filter_ens.x_mic[0]], [-max_v, max_v], '--',
                 color='firebrick', linewidth=4, alpha=0.2)
        for yy, c, l in zip([y_BA[ai], y_BB[ai]], [color_bias, 'orange'], ['-', '--']):
            y_mean, y_std = np.mean(yy, -1), abs(np.std(yy, -1))
            ax1.plot(locs, y_mean, l, color=c)
            ax1.fill_between(locs, y_mean + y_std, y_mean - y_std, alpha=0.1, color=c)
        # # Plot observables
        # if any(abs(t_gif[ii] - truth['t_obs']) < 1E-6):
        #     jj = np.argmin(abs(t_gif[ai] - truth['t_obs']))
        #     ax1.plot(locs_obs, truth['p_obs'][jj], 'o', color='red', markersize=4,
        #              markerfacecolor=None, markeredgewidth=2)
        #     # for yy, c in zip([y_BA_obs[ai], y_BB_obs[ai]], ['lightseagreen', 'orange']):
        #     #     y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
        #     #     ax1.plot(locs_obs, y_mean, 'x', color=c, markeredgewidth=2)

    def animate2(ai):
        t_g = t_gif[ai]
        # Plot timeseries ------------------------------------------------------------------------
        t11 = np.argmin(abs(filter_ens.hist_t - t_g))
        t00 = np.argmin(abs(filter_ens.hist_t - (t_g - filter_ens.t_CR / 2.)))
        tt_ = filter_ens.hist_t[t00:t11]
        for ax_ in ax2:
            ax_.clear()
            ax_.set(xlim=[tt_[0], tt_[-1] + filter_ens.t_CR * .05], xlabel='$t$ [s]')
        ax2[0].set(ylim=[-max_v, max_v], ylabel="$p'(x/L=0.2)$ [Pa]")
        yy = interpolate(truth['t'], y_t_tt, tt_)
        ax2[0].plot(tt_, yy, color=color_true, linewidth=3)
        ax2[0].plot(truth['t_obs'][0], y_obs_tt[0], 'o', color=color_obs, markersize=3,
                    markerfacecolor=None, markeredgewidth=2)
        yy = interpolate(filter_ens_BA.bias.hist_t, np.mean(y_BA_tt_u, -1), tt_)
        ax2[0].plot(tt_, yy, color=color_unbias, linewidth=1)
        for yy, c, l in zip([y_BA_tt, y_BB_tt], ['lightseagreen', 'orange'], ['-', '--']):
            yy = interpolate(filter_ens.hist_t, yy, tt_)
            y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
            ax2[0].plot(tt_, y_mean, l, color=c)
        ax2[0].legend(pressure_legend, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        for yy, c, l in zip([y_BA_tt, y_BB_tt], [color_bias, 'orange'], ['-', '--']):
            yy = interpolate(filter_ens.hist_t, yy, tt_)
            y_std = abs(np.std(yy, -1))
            ax2[0].fill_between(tt_, y_mean + y_std, y_mean - y_std, alpha=0.1, color=c)
        # # Plot obs data ------------------------------------------------------------------------
        t11_o = np.argmin(abs(truth['t_obs'] - t_g))
        t00_o = np.argmin(abs(truth['t_obs'] - (t_g - filter_ens.t_CR / 2.)))
        ax2[0].plot(truth['t_obs'][t00_o:t11_o], y_obs_tt[t00_o:t11_o], 'o', color=color_obs,
                    markersize=3, markerfacecolor=None, markeredgewidth=2)

        # plot parameters ------------------------------------------------------------------------
        ax2[1].set(ylim=[min_p, max_p], ylabel="")
        for mean_p, std_p, line_type in zip([alpha_BA, alpha_BB], [std_BA, std_BB], ['-', '--']):
            cols = ['mediumpurple', 'orchid']
            for ppi, pp in enumerate(filter_ens.est_a):
                ax2[1].plot(tt_, mean_p[ppi][t00:t11], line_type, color=cols[ppi], label=pp)
        ax2[1].legend(params_legend, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        for mean_p, std_p, line_type in zip([alpha_BB, alpha_BA], [std_BB, std_BA], ['-', '--']):
            cols = ['mediumpurple', 'orchid']
            for ppi, pp in enumerate(filter_ens.est_a):
                ax2[1].fill_between(tt_, mean_p[ppi][t00:t11] + abs(std_p)[ppi][t00:t11],
                                    mean_p[ppi][t00:t11] - abs(std_p)[ppi][t00:t11], alpha=0.2, color=cols[ppi])

    # Create and save animations ------------------------------------------------------------------------
    ani1 = mpl.animation.FuncAnimation(fig1, animate1, frames=len(t_gif), interval=10, repeat=False)
    ani2 = mpl.animation.FuncAnimation(fig2, animate2, frames=len(t_gif), interval=10, repeat=False)
    writergif = mpl.animation.PillowWriter(fps=10)
    ani1.save(figs_dir + 'ani_tube.gif', writer=writergif)
    ani2.save(figs_dir + 'ani_timeseries.gif', writer=writergif)


if __name__ == '__main__':

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
