import pickle
import os
import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import interp2d
from tabulate import tabulate
from src.utils import interpolate, fun_PSD, categorical_cmap, CR, get_error_metrics


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.backends.backend_pdf as plt_pdf
from matplotlib import colors, rc
from matplotlib.animation import FuncAnimation

rc('animation', html='jshtml')




XDG_RUNTIME_DIR = 'tmp/'

# Figures colors
color_true = 'gray'
color_unbias = '#000080ff'
color_bias = '#20b2aae5'
color_obs = 'r'
color_b = 'indigo'
colors_alpha = ['green', 'sandybrown', [0.7, 0.7, 0.87], 'blue', 'red', 'gold', 'deepskyblue']

y_unbias_props = dict(marker='none', linestyle='--', dashes=(10, 1), lw=.5, color=color_unbias)
y_biased_props = dict(marker='none', linestyle='-', lw=.2, color=color_bias, alpha=.9)
y_biased_mean_props = dict(marker='none', linestyle='--', dashes=(2, .5), lw=1, color='teal')

true_noisy_props = dict(marker='none', linestyle='-', lw=1.2, color=color_true, alpha=.3)
true_props = dict(marker='none', linestyle='-', lw=2, color=color_true, alpha=.6)
obs_props = dict(marker='.', linestyle='none', color=color_obs, markersize=5, markeredgecolor='none')
bias_props = dict(marker='none', linestyle='--', dashes=(10, 1), lw=.5, color=color_b)
bias_obs_props = dict(lw=1.5, color='mediumorchid', alpha=0.7)
bias_obs_noisy_props = dict(lw=1.5, color='k', alpha=0.2)


def recover_unbiased_solution(t_b, b, t, y, upsample=True):
    if b.ndim < y.ndim:
        b = np.expand_dims(b, axis=-1)
    elif b.shape[-1] > 1:
        b = np.mean(b, axis=-1, keepdims=True)
    if upsample:
        b = interpolate(t_b, b, t)
    return y + b



def plot_observables(filter_ens, truth, plot_states=True, plot_bias=False, plot_ensemble_members=False,
                    filename=None, reference_y=1., reference_t=1., max_time=None, dims='all'):
    def interpolate_obs(truth, t):
        return interpolate(truth['t'], truth['y_raw'], t), interpolate(truth['t'], truth['y_true'], t)
    
    def normalize(*arrays, factor):
        return [arr / factor for arr in arrays]

    
    t_obs, obs = truth['t_obs'], truth['y_obs']
    y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t
    y_mean = np.mean(y_filter, -1, keepdims=True)
    Nq = filter_ens.Nq
    dims = range(Nq) if dims == 'all' else dims
    
    N_CR = int(filter_ens.t_CR // filter_ens.dt)
    max_time = min(truth['t_obs'][-1] + filter_ens.t_CR, t[-1]) if max_time is None else max_time
    i0, i1 = [np.argmin(abs(t - ttt)) for ttt in [truth['t_obs'][0], max_time]]
    y_filter, y_mean, t = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, t])
    
    if filter_ens.bias is not None:
        b, t_b = filter_ens.bias.get_bias(state=filter_ens.bias.hist), filter_ens.bias.hist_t
        y_unbiased = recover_unbiased_solution(t_b, b, t, y_mean, upsample=hasattr(filter_ens.bias, 'upsample'))
    else:
        b, t_b, y_unbiased = None, None, None
    
    y_raw, y_true = interpolate_obs(truth, t)
    
    t_wash, wash = truth.get('wash_t', 0.), truth.get('wash_obs', np.zeros_like(obs))
    t_label = '$t$ [s]' if reference_t == 1. else '$t/T$'
    t, t_obs, max_time, t_wash = normalize(t, t_obs, max_time, t_wash, factor=reference_t)
    if t_b is not None:
        t_b = t_b / reference_t
    
    y_raw, y_filter, y_mean, obs, y_true = normalize(y_raw, y_filter, y_mean, obs, y_true, factor=reference_y)
    if y_unbiased is not None:
        y_unbiased = y_unbiased / reference_y
    
    margin, max_y, min_y = 0.15 * np.mean(abs(y_raw)), np.max(y_raw), np.min(y_raw)
    y_lims, x_lims = [min_y - margin, max_y + margin], [[t[0], max_time], [t_obs[-1] - filter_ens.t_CR, max_time]]
    
    def plot_series(kk, true, ref, axs, xlim, lbls):
        for qi, ax in enumerate(axs):
            ax.plot(t, true[:, qi], label=lbls[0], **true_props)
            
            m = np.mean(ref[:, qi], axis=-1)
            ax.plot(t, m, **y_biased_mean_props, label=lbls[1])
            if plot_ensemble_members:
                ax.plot(t, ref[:, qi], **y_biased_props)
            else:
                s = np.std(ref[:, qi], axis=-1)
                ax.fill_between(t, m + s, m - s, alpha=0.5, color=y_biased_props['color'])

            ax.plot(t_obs, obs[:, qi], label='data', **obs_props)

            if 'wash_t' in truth.keys():
                ax.plot(t_wash, wash[:, qi], **obs_props)

            plot_DA_window(t_obs, ax, ens=filter_ens)
            ax.set(ylim=y_lims, xlim=xlim)

            if qi == 0:
                ax.legend(loc='lower left', bbox_to_anchor=(0.1, 1.1), ncol=5, fontsize='small')
            if kk == 0:
                ax.set(ylabel=filter_ens.obs_labels[qi]) 
        axs[-1].set(xlabel=t_label) 
    
    if plot_states:
        fig1, fig1_axs = plt.subplots(len(dims), 2, figsize=(8, 1.5 * len(dims)), layout="constrained", sharex='col', sharey='row')
        if Nq == 0:
            fig1_axs = fig1_axs[np.newaxis, :]

        for axx, xl, col in zip(fig1_axs.T, x_lims, range(2)):
            plot_series(kk=col, axs=axx, true=y_raw, ref=y_filter, xlim=xl, lbls=['True', 'Model estimate'])

        if filename:
            plt.savefig(filename + '.svg', dpi=350)
            plt.close()


def plot_ensemble(ensemble, max_modes=None, reference_params=None, nbins=6):
    if max_modes is None:
        max_modes = ensemble.Nphi
    fig, axs = plt.subplots(figsize=(12, 1.5), layout='tight', nrows=1, ncols=max_modes, sharey=True)
    for ax, ph, lbl in zip(axs.ravel(), ensemble.get_current_state[:ensemble.Nphi, ], ensemble.state_labels):
        ax.hist(ph, bins=nbins, color='tab:green')
        ax.set(xlabel=lbl)

    if ensemble.Na > 0:
        fig, axs = plt.subplots(figsize=(12, 1.5), layout='tight', nrows=1, ncols=ensemble.Na, sharey=True)
        if ensemble.Na == 1:
            axs = [axs]
        else:
            axs = axs.ravel()

        reference_alpha = dict()
        for param in ensemble.est_a:
            reference_alpha[param] = 1.
        if type(reference_params) is dict:
            for param, val in reference_params.items():
                reference_alpha[param] = val

        for ax, a, param in zip(axs, ensemble.get_current_state[-ensemble.Na:, ], ensemble.est_a):
            if isinstance(ensemble.std_a, dict):
                xlims = np.array(ensemble.std_a[param]) / reference_alpha[param]
            else:
                _mean = np.mean(a)
                xlims = _mean * np.array([1-ensemble.std_a, 1+ensemble.std_a]) / reference_alpha[param]

            ax.hist(a / reference_alpha[param], bins=np.linspace(*xlims, nbins))
            ax.set(xlabel=ensemble.alpha_labels[param])


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
                                post_process_single(filter_ens, truth, filename=filename)


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
            k = filter_ens.regularization_factor
            if k > k_max:
                continue
            if filter_ens.est_a:
                if flag:
                    N_psi = filter_ens.Nphi

                if reference_p is None:
                    reference_p = filter_ens.alpha0

                for pj, p in enumerate(filter_ens.est_a):
                    p_lbl = filter_ens.alpha_labels[p]
                    for idx, a, tt, mk in zip([-1, 0], [1., .2],
                                              ['$(t_\\mathrm{end})/$', '$^0/$'], ['x', '+']):
                        hist_p = filter_ens.hist[idx, N_psi + pj] / reference_p[p]
                        lbl = p_lbl + tt + p_lbl

                        axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), alpha=a, mew=.8, fmt=mk,
                                          color=colors_alpha[pj], label=lbl, capsize=4, markersize=6, linewidth=.6)
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
            plot_DA_window(t_obs, ax=ax1)

        # SAVE PLOT -------------------------------------------------------------------
        if filename is not None:
            plt.savefig(filename + '_L{}.svg'.format(out['Ls'][Li]), dpi=350)
            plt.close()


def post_process_pdf(filter_ens, truth, params, filename=None, reference_p=None, normalize=True):
    if not filter_ens.est_a:
        raise ValueError('no input_parameters to infer')

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
        labels_p.append('$' + p + norm_lbl(filter_ens.alpha_labels[p]) + '$')
        hist_alpha.append(m)
        ii += 1

    for ax, p, a, c, lbl in zip(ax_all, filter_ens.est_a, hist_alpha, colors_alpha, labels_p):
        plot_violins(ax, a, t_obs, widths=dt_obs / 2, color=c, label='analysis posterior')
        alpha_lims = [filter_ens.alpha_lims[p][0] / reference_p[p],
                      filter_ens.alpha_lims[p][1] / reference_p[p]]
        for lim in alpha_lims:
            ax.plot([hist_t[0], hist_t[-1]], [lim, lim], '--', color=c, lw=0.8)
        if twin:
            ax.plot((x_lims[0], x_lims[-1]), (k_twin[p], k_twin[p]), '-', color='k', linewidth=.6)
        ax.set(xlabel='$t$ [s]', ylabel=lbl, xlim=x_lims,
               ylim=[alpha_lims[0] - 0.1 * abs(alpha_lims[1]), alpha_lims[1] + 0.1 * abs(alpha_lims[1])])

        ax.legend()
    if filename is not None:
        plt.savefig(filename + '.svg', dpi=350)
        plt.close()


def post_process_single(filter_ens, truth,
                        dims='all',
                        plot_ensemble_members=True,
                        plot_bias=True,
                        reference_y=1.,
                        reference_p=None,
                        reference_t=1.,
                        filename=None
                        ):
    plot_timeseries(filter_ens, truth,
                    dims=dims,
                    plot_ensemble_members=plot_ensemble_members,
                    plot_bias=plot_bias,
                    reference_y=reference_y,
                    reference_t=reference_t,
                    )
    plot_parameters(filter_ens, truth, reference_p=reference_p)

    if filename is not None:
        raise NotImplementedError('To do')


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
            y, t = filter_ens.get_observable_hist(), filter_ens.hist_t
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
                y_truth, t_truth = truth['y_raw'][i0_t - N_CR:i1_t + N_CR], truth['t'][i0_t - N_CR:i1_t + N_CR]
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

            if filter_ens.regularization_factor in J_plot:
                filename = '{}WhyAugment_L{}_augment{}_k{}'.format(figs_dir, L, augment,
                                                                   filter_ens.regularization_factor)
                post_process_single(filter_ens, truth, filename=filename + '_J')

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


def plot_annular_model(forecast_params=None, animate=False, anim_name=None):
    from src.models_physical import Annular
    import datetime
    import time

    if forecast_params is None:
        paramsTA = dict(dt=1 / 51.2E3)
    else:
        paramsTA = forecast_params.copy()
    if anim_name is None:
        anim_name = '{}_Annulat_mov_mix_epsilon.gif'.format(datetime.date.today())

    # Non-ensemble case =============================
    t1 = time.time()
    case = Annular(**paramsTA)
    state, t_ = case.time_integrate(int(case.t_transient * 3 / case.dt))
    case.update_history(state, t_)

    print(case.dt)
    print('Elapsed time = ', str(time.time() - t1))

    fig1 = plt.figure(figsize=[12, 3], layout="constrained")
    subfigs = fig1.subfigures(1, 2, width_ratios=[1.2, 1])

    ax = subfigs[0].subplots(1, 2)
    ax[0].set_title(Annular.name)

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    # State evolution
    y, lbl = case.get_observable_hist(), case.obs_labels

    ax[0].scatter(t_h, y[:, 0], c=t_h, label=lbl, cmap='Blues', s=10, marker='.')

    ax[0].set(xlabel='$t$', ylabel=lbl[0])
    i, j = [0, 1]

    if len(lbl) > 1:
        ax[1].scatter(y[:, 0], y[:, 1], c=t_h, s=3, marker='.', cmap='Blues')
        ax[1].set(xlabel=lbl[0], ylabel=lbl[1])
    else:
        ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='green')

    ax[1].set_aspect(1. / ax[1].get_data_ratio())

    if not animate:
        ax2 = subfigs[1].subplots(2, 1)

        y, lbl = case.get_observable_hist(), case.obs_labels
        y = np.mean(y, axis=-1)

        # print(np.min(y, axis=0))
        sorted_id = np.argsort(np.max(abs(y[-1000:]), axis=0))[::-1]
        y = y[:, sorted_id]
        lbl = [lbl[idx] for idx in sorted_id]

        for ax in ax2:
            ax.plot(t_h, y / 1E3)
        ax2[0].set_title('Acoustic Pressure')
        ax2[0].legend(lbl, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        ax2[0].set(xlim=[t_h[0], t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
        ax2[1].set(xlim=[t_h[-1] - case.t_CR, t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
    else:
        ax2 = subfigs[1].subplots(1, 1, subplot_kw={'projection': 'polar'})
        angles = np.linspace(0, 2 * np.pi, 200)  # Angles from 0 to 2π
        y, lbl = case.get_observable_hist(loc=angles), case.obs_labels
        y = np.mean(y, axis=-1)

        radius = [0, 0.5, 1]
        theta, r = np.meshgrid(angles, radius)

        # Remove radial tick labels
        ax2.set(yticklabels=[], theta_zero_location='S', title='Acoustic Pressure',
                theta_direction=1, rgrids=[], thetagrids=[])

        # Add a white concentric circle
        circle_radius = 0.5
        ax2.plot(angles, [circle_radius] * len(angles), color='black', lw=1)

        idx_max = np.argmax(y[:, 0])
        polar_mesh = ax2.pcolormesh(theta, r, [y[idx_max].T] * len(radius), shading='auto', cmap='RdBu')

        start_i = int((t_h[-1] - .03) // case.dt)
        dt_gif = 10
        t_gif = t_h[start_i::dt_gif]
        y_gif = y[start_i::dt_gif]

        def update(frame):
            ax2.fill(angles, [circle_radius] * len(angles), color='white')
            polar_mesh.set_array([y_gif[frame].T] * len(radius))
            ax2.set_title('Acoustic Pressure $t$ = {:.3f}'.format(t_gif[frame]))  # , fontsize='small')#, labelpad=50)

        plt.colorbar(polar_mesh, label='Pressure', shrink=0.75)
        anim = FuncAnimation(fig1, update, frames=len(t_gif))
        anim.save(anim_name, fps=dt_gif * 10)

    plt.show()


def plot_DA_window(t_obs, ax=None, twin=False, ens=None):
    if ax is None:
        ax = plt.gca()
    ax.axvline(x=t_obs[-1], ls='--', color='k', linewidth=.8)
    ax.axvline(x=t_obs[0], ls='--', color='k', linewidth=.8)
    if twin:
        ax.axhline(y=1, ls='--', color='k', linewidth=.6)

    if ens is not None:
        for idx, cl, ll in zip(['num_DA_blind', 'num_SE_only'],
                               ['darkblue', 'darkviolet'], ['BE', 'PE']):
            idx = getattr(ens, idx)
            if idx > 0:
                ax.axvline(x=t_obs[idx], ls='-.', color=cl, label='Start ' + ll)


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
        min_v = min(np.log(0.4), min([np.min(metric) for metric in metrics]))
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

    post_process_single(filter_ens, truth, filename=filename + '_optimal_solution_J')

    filename = filename + '_optimal_solution_CR'
    post_process_multiple(folder, filename, k_max=20., L_plot=[70])


def plot_parameters(ensembles, truth=None, filename=None, reference_p=None, plot_ensemble_members=False):
    if type(ensembles) is not list:
        ensembles = [ensembles]

    filter_ens = ensembles[0]

    if len(filter_ens.est_a) < 4:
        rows = len(filter_ens.est_a)
        fig1, axs = plt.subplots(rows, ncols=1, sharex='col', figsize=(6, 1.5 * rows), layout="constrained")
        if len(filter_ens.est_a) == 1:
            axs = [axs]
    else:
        rows = len(filter_ens.est_a) // 2
        if len(filter_ens.est_a) % 2:
            rows += 1
        fig1, axs = plt.subplots(rows, ncols=2, sharex='all', figsize=(12, 1.5 * rows), layout="constrained")
        axs = axs.ravel()

    if reference_p is not None:
        ref_p = dict((key, reference_p[key]) for key in filter_ens.est_a)
        twin = True
    else:
        ref_p = dict((key, 1.) for key in filter_ens.est_a)
        twin = False

    def norm_lbl(x, suffix=''):
        if reference_p is not None:
            suffix += f'/{x}' + '$^\\mathrm{ref}$'
        return x + suffix

    if truth is not None:
        t_obs = truth['t_obs']
        xlim = [t_obs[0], t_obs[-1]]
    else:
        t_obs = None
        xlim = [filter_ens.hist_t[0], filter_ens.hist_t[-1]]

    cmap = categorical_cmap(len(filter_ens.est_a), len(ensembles), cmap="Set1")
    p_colors = [cmap[ii::len(ensembles)] for ii in range(len(ensembles))]

    for kk, ens, style, pc in zip(range(len(ensembles)), ensembles, ['-', '--'], p_colors):
        hist, hist_t = ens.hist, ens.hist_t
        hist_mean = np.mean(hist, axis=-1, keepdims=True)

        mean_p, std_p, labels_p, hist_p = [], [], [], []

        for ii, p in enumerate(ens.est_a):
            labels_p.append(norm_lbl(ens.alpha_labels[p]))
            mean_p.append(hist_mean[:, ii+ens.Nphi].squeeze() / ref_p[p])
            std_p.append(abs(np.std(hist[:, ii+ens.Nphi] / ref_p[p], axis=1)))
            if plot_ensemble_members:
                hist_p.append(hist[:, ii+ens.Nphi] / ref_p[p])

        for ii, ax, p, m, s, c, lbl in zip(range(len(labels_p)), axs, ens.est_a, mean_p, std_p, pc, labels_p):
            max_p, min_p = np.max(m + 2*abs(s)), np.min(m - 2*abs(s))
            ax.plot(hist_t, m, ls=style, color=c, label=lbl)
            ax.fill_between(hist_t, m + 2*abs(s), m - 2*abs(s), alpha=0.4, color=c)

            if plot_ensemble_members:
                ax.plot(hist_t, hist_p[ii], color=c, alpha=0.3)


            ylims = ax.get_ylim()
            if kk == 0:
                ylim_before = ax.get_ylim()  # Capture limits before lines
                if filter_ens.alpha_lims[p][0] is not None and filter_ens.alpha_lims[p][1] is not None:
                    for lim in [filter_ens.alpha_lims[p][0] / ref_p[p],
                                filter_ens.alpha_lims[p][1] / ref_p[p]]:
                        ax.axhline(y=lim, color=c, lw=2, alpha=0.5)
                ax.set_ylim(ylim_before)  # Reset to original limits


                if t_obs is not None:
                    plot_DA_window(t_obs, ax=ax, ens=filter_ens, twin=twin)
            
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            # ax.set(ylim=[ylims[0]-s/2, ylims[1]+s/2])

            if kk > 0:
                ylims = axs[0].get_ylim()
                min_p, max_p = min([ylims[0], min_p]), max([ylims[1], max_p])
                ax.set(ylabel='', ylim=[min_p, max_p])

        axs[-1].set(xlabel='$t$ [s]', xlim=xlim)

    if filename is not None:
        plt.savefig(filename + '_params.svg', dpi=350)


def plot_RMS_pdf(ensembles, truth, nbins=40):
    if type(ensembles) is not list:
        ensembles = [ensembles]

    fig, axs_all = plt.subplots(nrows=2 * len(ensembles), ncols=4, sharex=True, sharey=True,
                                figsize=(10, 3 * len(ensembles)), layout='constrained')

    t_ref, y_ref, y_raw = [truth[key] for key in ['t', 'y_true', 'y_raw']]
    if y_ref.ndim < 3:
        y_ref = np.expand_dims(y_ref, axis=-1)
    if y_raw.ndim < 3:
        y_raw = np.expand_dims(y_raw, axis=-1)

    R_truth = np.sqrt(np.sum((y_ref - y_raw) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))

    tts = [[truth['t_obs'][0] - ensembles[0].t_CR, truth['t_obs'][0]],
           [truth['t_obs'][0], truth['t_obs'][0] + ensembles[0].t_CR],
           [truth['t_obs'][-1] - ensembles[0].t_CR, truth['t_obs'][-1]],
           [truth['t_obs'][-1], truth['t_obs'][-1] + ensembles[0].t_CR]]

    times = [[np.argmin(abs(t_ref - tt[0])), np.argmin(abs(t_ref - tt[1]))] for tt in tts]

    legs = ['pre-DA', 'DA', 'DA2', 'post_DA']

    max_RMS = 2
    args = dict(bins=nbins, range=(0, max_RMS), density=True, orientation='vertical')
    ii = -2
    for ens in ensembles:
        ii += 2
        y_est = ens.get_observable_hist()
        y_est = interpolate(ens.hist_t, y_est, t_ref)
        y_mean = np.mean(y_est, axis=-1, keepdims=True)

        R = np.sqrt(np.sum((y_ref - y_est) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
        Rm = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))

        if ens.bias.name != 'NoBias':
            b_est = ens.bias.get_bias(ens.bias.hist)

            b_est[abs(b_est) > np.max(abs(y_ref))] *= 0.

            b_est_interp = interpolate(ens.bias.hist_t, b_est, t_ref)
            y_est_u = y_est + b_est_interp
            y_mean_u = np.mean(y_est_u, axis=-1, keepdims=True)
            R_u = np.sqrt(np.sum((y_ref - y_est_u) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
            Rm_u = np.sqrt(np.sum((y_ref - y_mean_u) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
            RR, RR_m = [R, R_u], [Rm, Rm_u]
        else:
            RR, RR_m = [R], [Rm]
            y_est_u = None

        colours = [[color_bias] * ens.m, ['tab:green'] * ens.m]

        for axs_, yy, rr, rrm, c in zip([axs_all[ii], axs_all[ii + 1]], [y_est, y_est_u], RR, RR_m, colours):
            if yy is None:
                break
            axs_[0].set(ylabel=ens.filter)
            if axs_[0] == axs_all[ii + 1, 0]:
                ylabel = '{}+\n{}'.format(ens.filter, ens.bias.name)
                if hasattr(ens, 'regularization_factor'):
                    ylabel += ' k={}'.format(ens.regularization_factor)
                axs_[0].set(ylabel=ylabel)

            kk = 0
            for jjs, leg, ax in zip(times, legs, axs_):
                j0, j1 = jjs
                segment = rr[j0:j1]
                segment[segment > max_RMS] = max_RMS
                ax.hist(rrm[j0:j1], histtype='step', color=c[0], lw=2, **args)
                ax.hist(segment, histtype='stepfilled', alpha=0.1, stacked=False, color=c, **args)
                mean = np.mean(rrm[j0:j1])
                if mean > max_RMS:
                    ax.axvline(max_RMS, c='tab:red', lw=1, ls='--')
                else:
                    ax.axvline(mean, c=c[0], lw=1, ls='--')

                kk += 1
                ax.legend([leg + '_mean', leg + '_j'])

                ax.hist(R_truth[j0:j1], histtype='step', color='k', lw=1.5, **args)
                ax.axvline(np.mean(R_truth[j0:j1]), c='k', lw=1, ls='--')
                if axs_[0] == axs_all[ii + 1, 0]:
                    ax.set(xlabel='RMS error')


def plot_states_PDF(ensembles, truth, nbins=20, window=None):
    if type(ensembles) is not list:
        ensembles = [ensembles]

    Nq = truth['y_true'].shape[1]
    fig, axs_all = plt.subplots(nrows=2 * len(ensembles), ncols=Nq, sharex='col', sharey=True,
                                figsize=(12, 3 * len(ensembles)), layout='constrained')

    if window is None:
        window = (truth['t_obs'][-1], ensembles[0].hist_t[-1])

    j0, j1 = [np.argmin(abs(truth['t'] - tt)) for tt in window]

    t_ref, y_ref_true, y_ref_raw = [truth[key][j0:j1] for key in ['t', 'y_true', 'y_raw']]

    args_1 = dict(orientation='vertical', histtype='step', bins=nbins, density=False)
    args_2 = dict(orientation='vertical', histtype='stepfilled', bins=nbins, density=False)

    ii = -2
    for ens in ensembles:
        ii += 2

        y_est = ens.get_observable_hist()
        y_est = interpolate(ens.hist_t, y_est, t_ref)

        # Plot bias-corrected solutions
        b_est = ens.bias.get_bias(state=ens.bias.hist, mean=False)
        y_mean = np.mean(y_est, axis=-1, keepdims=True)
        if ens.bias.name != 'NoBias':
            b_est_interp = interpolate(ens.bias.hist_t, b_est, t_ref)
            y_est_u = y_est + b_est_interp
            plot_yy = [y_est, y_est_u]
        else:
            plot_yy = [y_est]

        for yy, axs_, c in zip(plot_yy, [axs_all[ii], axs_all[ii + 1]],
                               [[color_bias, 'tab:blue'], ['tab:green', 'darkgreen']]):
            for qi, ax in enumerate(axs_):
                for mi in range(yy.shape[-1]):
                    ax.hist(yy[:, qi, mi], color=c[0], alpha=0.2, **args_2)
                ax.hist(y_ref_true[:, qi], color='k', alpha=.7, lw=2, **args_1)
                ax.hist(y_ref_raw[:, qi], color='tab:red', alpha=0.7, lw=2, **args_1)
                ax.hist(np.mean(yy[:, qi], axis=-1), color=c[1], alpha=1, ls=(0, (6, 1)), lw=1.5, **args_1)
                ax.axvline(np.mean(y_ref_true[:, qi]), color='k', ls='-', lw=2)
                ax.axvline(np.mean(y_ref_raw[:, qi]), color='tab:red', ls='-', lw=2)
                ax.axvline(np.mean(yy[:, qi]), color=c[0], ls=(0, (6, 6)), lw=2)

        axs_all[ii, 0].set(ylabel=ens.filter)
        ylabel = '{}+\n{}'.format(ens.filter, ens.bias.name)
        if hasattr(ens, 'regularization_factor'):
            ylabel += ' k={}'.format(ens.regularization_factor)
        axs_all[ii + 1, 0].set(ylabel=ylabel)

    for ax, lbl in zip(axs_all[-1, :], ens.obs_labels):
        ax.set(xlabel=lbl)



def plot_timeseries(filter_ens, truth, plot_states=True, plot_bias=False, plot_ensemble_members=False,
                    filename=None, reference_y=1., reference_t=1., max_time=None, dims='all'):

    t_obs, obs = truth['t_obs'], truth['y_obs']
    y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t
    y_mean = np.mean(y_filter, -1, keepdims=True)

    Nq = filter_ens.Nq
    if dims == 'all':
        dims = range(Nq)

    # cut signals to interval of interest -----
    N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS

    if max_time is None:
        max_time = min(truth['t_obs'][-1] + filter_ens.t_CR, t[-1])

    i0, i1 = [np.argmin(abs(t - ttt)) for ttt in [truth['t_obs'][0], max_time]]  # start/end of assimilation

    y_filter, y_mean, t = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, t])

    b = filter_ens.bias.get_bias(state=filter_ens.bias.hist)
    t_b = filter_ens.bias.hist_t

    y_unbiased = recover_unbiased_solution(t_b, b, t, y_mean, upsample=hasattr(filter_ens.bias, 'upsample'))

    y_raw = interpolate(truth['t'], truth['y_raw'], t)
    y_true = interpolate(truth['t'], truth['y_true'], t)

    if 'wash_t' in truth.keys():
        t_wash, wash = truth['wash_t'], truth['wash_obs']
    else:
        t_wash = 0.

    if reference_t == 1.:
        t_label = '$t$ [s]'
    else:
        t_label = '$t/T$'
        t, t_b, t_obs, max_time, t_wash = [tt / reference_t for tt in [t, t_b, t_obs, max_time, t_wash]]

    # % PLOT time series ------------------------------------------------------------------------------------------

    y_raw, y_unbiased, y_filter, y_mean, obs, y_true = [yy / reference_y for yy in [y_raw, y_unbiased, y_filter,
                                                                                    y_mean, obs, y_true]]
    margin = 0.15 * np.mean(abs(y_raw), axis=0)
    max_y = np.max(y_raw, axis=0)
    min_y = np.min(y_raw, axis=0)

    x_lims = [[t_obs[0] - .25 * filter_ens.t_CR, t_obs[0] + filter_ens.t_CR],
              [t_obs[-1] - filter_ens.t_CR, max_time],
              [t[0], max_time]]

    if plot_states:
        fig1 = plt.figure(figsize=(10, 5 * len(dims)), layout="constrained")
        sfs = fig1.subfigures(nrows=2, ncols=1)
        ax_all = sfs[0].subplots(len(dims), ncols=1, sharey='row', sharex='col')
        ax_zoom = sfs[1].subplots(len(dims), ncols=2, sharey='row', sharex='col')

        for axi, qi in enumerate(dims):
            y_lims = [min_y[axi] - margin[axi], max_y[axi] + margin[axi]]
            
            if Nq == 1:
                q_axes = [ax_zoom[0], ax_zoom[1], ax_all]
            else:
                q_axes = [ax_zoom[axi, 0], ax_zoom[axi, 1], ax_all[axi]]

            for ax, xl in zip(q_axes, x_lims):

                # Observables ---------------------------------------------------------------------
                ax.plot(t, y_true[:, qi], label='truth', **true_props)
                if filter_ens.bias.name != 'NoBias':
                    ax.plot(t, y_unbiased[:, qi], label='bias-corrected estimate', **y_unbias_props)

                m = np.mean(y_filter[:, qi], axis=-1)
                ax.plot(t, m, **y_biased_mean_props, label='model estimate')
                if plot_ensemble_members:
                    for mi in range(y_filter.shape[-1]):
                        ax.plot(t, y_filter[:, qi, mi], **y_biased_props)
                else:
                    s = np.std(y_filter[:, qi], axis=-1)
                    ax.fill_between(t, m + s, m - s, alpha=0.5, color=y_biased_props['color'])

                ax.plot(t_obs, obs[:, qi], label='data', **obs_props)
                if 'wash_t' in truth.keys():
                    ax.plot(t_wash, wash[:, qi], **obs_props)
                plot_DA_window(t_obs, ax, ens=filter_ens)
                ax.set(ylim=y_lims, xlim=xl)

            ylbl = '$y_{}$'.format(qi)
            if reference_y != 1.:
                ylbl += ' norm.'
            if Nq == 1:
                ax_zoom[0].set(ylabel=ylbl)
                ax_all.set(ylabel=ylbl)
            else:
                ax_zoom[qi, 0].set(ylabel=ylbl)
                ax_all[qi].set(ylabel=ylbl)

        if Nq == 1:
            ax_all.legend(loc='lower left', bbox_to_anchor=(0.1, 1.1), ncol=5, fontsize='small')
            for ax in [ax_zoom[0], ax_zoom[1], ax_all]:
                ax.set(xlabel=t_label)
        else:
            ax_all[0].legend(loc='lower left', bbox_to_anchor=(0.1, 1.1), ncol=5, fontsize='small')
            for ax in [ax_zoom[-1, 0], ax_zoom[-1, 1], ax_all[-1]]:
                ax.set(xlabel=t_label)

        if filename is not None:
            plt.savefig(filename + '.svg', dpi=350)
            plt.close()

    if plot_bias:
        fig1 = plt.figure(figsize=(10, 5 * len(dims)), layout="constrained")
        subfigs = fig1.subfigures(2, 1)
        ax_all = subfigs[0].subplots(len(dims), 1, sharex='col')
        ax_zoom = subfigs[1].subplots(len(dims), 2, sharex='col', sharey='row')

        b_filter = filter_ens.bias.get_bias(state=filter_ens.bias.hist)
        t_b = filter_ens.bias.hist_t

        y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t
        y_mean = np.mean(y_filter, axis=-1)
        y_mean = interpolate(t, y_mean, t_b)

        y_true = interpolate(truth['t'], truth['y_true'], t_b)

        if len(truth['b']) > 1:
            b_true = interpolate(truth['t'], truth['b'], t_b)
        else:
            b_true = np.nan
            
        innovation = y_true - y_mean

        innovation, b_filter, b_true = [yy / reference_y for yy in [innovation, b_filter, b_true]]


        for axi, qi in enumerate(dims):
            y_lims = [min_y[axi] - margin[axi], max_y[axi] + margin[axi]]
            if Nq == 1:
                q_axes = [ax_zoom[0], ax_zoom[1], ax_all]
            else:
                q_axes = [ax_zoom[axi, 0], ax_zoom[axi, 1], ax_all[axi]]

            for ax, xl in zip(q_axes, x_lims):
                # Observables ---------------------------------------------------------------------
                ax.plot(t_b, innovation[:, qi], label='$\\mathbf{d}^\dagger - \\bar{\\mathbf{y}}$',
                        **bias_obs_noisy_props)
                if isinstance(b_true, np.ndarray):
                    ax.plot(t_b, b_true[:, qi], label='manually added bias', **bias_obs_props)
                ax.plot(t_b, b_filter[:, qi], label='ESN prediction', **bias_props)
                plot_DA_window(t_obs, ax, ens=filter_ens)
                ax.set(ylim=y_lims, xlim=xl)

            ylbl = '$b_{}$'.format(qi)
            if reference_y != 1.:
                ylbl += ' norm.'

            if Nq == 1:
                ax_zoom[0].set(ylabel=ylbl)
                for ax in [ax_zoom[0], ax_zoom[1], ax_all]:
                    ax.set(xlabel=t_label)
                ax_all.legend(loc='lower left', bbox_to_anchor=(0., 1.1), ncol=5, fontsize='small')
            else:
                ax_zoom[qi, 0].set(ylabel=ylbl)
                ax_all[0].legend(loc='lower left', bbox_to_anchor=(0., 1.1), ncol=5, fontsize='small')
                for ax in [ax_zoom[-1, 0], ax_zoom[-1, 1], ax_all[-1]]:
                    ax.set(xlabel=t_label)

        if filename is not None:
            plt.savefig(filename + '.svg', dpi=350)
            plt.close()


def plot_attractor(psi_cases, color, figsize=(8, 8), ensemble_mean=True):
    if type(psi_cases) is not list:
        psi_cases, color = [psi_cases], [color]
    fig = plt.figure(figsize=figsize)
    if psi_cases[0].shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.set(xlabel='$x$', ylabel='$y$')
    else:
        ax = fig.add_subplot(111, projection='3d')
        for axs_ax in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axs_ax.pane.fill = False
        ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')

    for psi_, c in zip(psi_cases, color):
        lw, a = 1., 1.
        if psi_.ndim > 2:
            if psi_.shape[2] > 10 or ensemble_mean:
                psi_ = np.mean(psi_, axis=-1)
            elif psi_.shape[2] > 1:
                lw, a = 0.6, 0.3
        if psi_.shape[1] == 2:
            ax.plot(psi_[:, 0], psi_[:, 1], lw=lw, c=c, alpha=a)
        else:
            ax.plot(psi_[:, 0], psi_[:, 1], psi_[:, 2], lw=lw, c=c)


def plot_truth(y_raw, y_true, t, dt, fig_width=10, window=None, b=np.array([0.]),
               plot_time=False, Nq=None, filename=None, f_max=None, y_obs=None, t_obs=None, model=None, **other):
    if Nq is None:
        Nq = y_true.shape[1]
    if t_obs is None:
        t0 = 0
        if window is None:
            t1 = int(model.t_transient // dt)
        else:
            t1 = int(window // dt)
    else:
        t0 = int((t_obs[0]) // dt)
        if window is None:
            if model is not None and not isinstance(model, str):
                t1 = int((t_obs[-1] + model.t_CR) // dt)
                t1 = min(t1, len(t) - 1)
            else:
                t1 = int((t_obs[-1]) // dt)
        else:
            t1 = int((t_obs[0] + window) // dt)

    xlim = [t[t0], t[t1]]

    plot_bias = np.mean(b ** 2) > 0 and isinstance(model, str)

    if plot_bias:
        noise = y_raw - (y_true - b)
    else:
        noise = y_raw - y_true

    max_y = np.max(abs(y_raw[:t1 - t0]))

    fig1 = plt.figure(figsize=(fig_width, 2 * Nq), layout="constrained")
    subfigs = fig1.subfigures(nrows=1, ncols=4, width_ratios=[2, 0.5, 1, 1])
    labels = ['Raw', 'Post-processed', 'Difference']
    y_labels = ['$\\tilde{y}, y$', '', '$(\\tilde{y}-y)$']
    cols = ['tab:blue', 'mediumseagreen', 'tab:purple']

    ax_01 = subfigs[0].subplots(Nq, 2, sharex='all', sharey='all')
    ax_4 = subfigs[-1].subplots(Nq, 1, sharex='all', sharey='all')

    # Plot zoomed timeseries of raw, post-processed and noise
    if Nq == 1:
        axss = [[ax_01[0]], [ax_01[1]], [ax_4]]
    else:
        axss = [ax_01[:, 0], ax_01[:, 1], ax_4]

    dashes = (10, 1)
    for ax, yy, ttl, lbl, c in zip(axss, [y_raw, y_true, noise], labels, y_labels, cols):
        ax[0].set(title=ttl)
        ax[-1].set(xlabel='$t$', xlim=xlim)
        for qi in range(Nq):
            ax[qi].plot(t, yy[:, qi], color=c)
            ax[qi].axhline(np.mean(yy[:, qi]), color=c)
            if ttl[0] == 'R' and y_obs is not None:
                ax[qi].plot(t_obs, y_obs[:, qi], 'ro', ms=3)
            elif ttl[0] == 'P' and plot_bias:
                ax[qi].plot(t, yy[:, qi] - b[:, qi], color='k', dashes=dashes, lw=.5)
                ax[qi].axhline(np.mean(yy[:, qi] - b[:, qi]), color='k', dashes=dashes, lw=.5)
            if len(lbl) > 1:
                ax[qi].set(ylabel=lbl + '$_{}$'.format(qi))

    # Plot probability density src and power spectral densities
    ax_pdf = subfigs[1].subplots(Nq, 1, sharey='all', sharex='all')
    ax_PSD = subfigs[2].subplots(Nq, 1, sharex='all', sharey='all')
    if Nq == 1:
        ax_pdf = [ax_pdf]
        ax_PSD = [ax_PSD]
    binwidth = 0.01 * max_y
    bins = np.arange(-max_y, max_y + binwidth, binwidth)
    for yy, ttl, lbl, c in zip([y_raw, y_true], labels[:2], y_labels[:2], cols[:2]):
        ax_pdf[0].set(title='PDF')
        ax_pdf[-1].set(xlabel='$p$')
        ax_PSD[-1].set(xlabel='$f$')
        for qi in range(Nq):
            ax_pdf[qi].hist(yy[:, qi], bins=bins, density=True, orientation='horizontal',
                            color=c, label=lbl + '$_{}$'.format(qi), histtype='step')
            if Nq == 1:
                ylims = ax_01[qi].get_ylim()
            else:
                ylims = ax_01[qi, 0].get_ylim()
            ax_pdf[qi].set(yticklabels=[], ylim=ylims)
        f, PSD = fun_PSD(dt, yy.squeeze())
        for qi in range(Nq):
            ax_PSD[qi].semilogy(f, PSD[qi], color=c, label=lbl + '$_{}$'.format(qi))
        ax_PSD[0].set(title='PSD', xlim=[0, f_max])
    if plot_bias:

        f, PSD = fun_PSD(dt, (y_true - b).squeeze())
        for qi in range(Nq):
            ax_pdf[qi].hist(y_true[:, qi] - b[:, qi], bins=bins, density=True, orientation='horizontal',
                            color='k', histtype='step', lw=.5)
            ax_PSD[qi].semilogy(f, PSD[qi], color='k', dashes=dashes, lw=.5)

    # Plot full timeseries if requested
    figs2 = []
    if plot_time:
        for yy, name, c in zip([y_raw, y_true], labels[:2], cols[:2]):
            y_true, t = [zz[t0:] for zz in [yy, t]]
            max_y = np.max(abs(y_true))
            fig2 = plt.figure(figsize=(12, 2 * Nq), layout="constrained")
            subfigs = fig2.subfigures(nrows=1, ncols=2, width_ratios=[1, 0.5])
            for sf, xlims in zip(subfigs, [(t[0], t[-1]), (t[-1000], t[-1])]):
                ax = sf.subplots(Nq, 1, sharex='all')
                if Nq == 1:
                    ax = [ax]
                ax[0].set(title=name)
                ax[-1].set(xlabel='$t$', xlim=xlims)
                for qi in range(Nq):
                    ax[qi].plot(t, y_true[:, qi], color=c)
                    ax[qi].set(ylim=[-max_y, max_y])
            figs2.append(fig2)
    # Show or save plots
    if filename is None:
        plt.show()
    else:
        if filename[-len('.pdf'):] != '.pdf':
            filename += '.pdf'
        os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
        pdf_file = plt_pdf.PdfPages(filename)
        pdf_file.savefig(fig1)
        plt.close(fig1)
        for fig in figs2:
            pdf_file.savefig(fig)
            plt.close(fig)
        pdf_file.close()  # Close results pdf


def plot_violins(ax, values, location, color='b', label=None, alpha=0.5, **kwargs):
    violins = ax.violinplot(values, positions=location, **kwargs)

    for vp in violins['bodies']:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(.5)
        vp.set_alpha(alpha)
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


def plot_MSE_evolution(filter_ens,
                       X_test_raw,
                       X_test_true,
                       truth,
                       max_lines=20,
                       reconstructed_data=None,
                       tiks=None
                       ):
    
    """ 
    Plot the evolution of Mean Squared Error (MSE) as the number of POD modes increases.

    Args:
        - case: The POD instance.
        - original_data: The original dataset.
        - N_modes_cases: List of mode counts to evaluate.
        - N_modes_cases_plot: Specific mode counts to plot.
    """



    original_data = interpolate(truth['t'], X_test_raw.T, filter_ens.hist_t).T
    original_data_true = interpolate(truth['t'], X_test_true.T, filter_ens.hist_t).T


    mse_ref = filter_ens.compute_MSE(POD_data=original_data, 
                                    original_data=original_data_true, 
                                    time_evolution=True)

    if reconstructed_data is None:
        single_case=True
        max_lines = min(max_lines, filter_ens.m)
        forecast_coefficients = filter_ens.get_POD_coefficients(Nt=0)
        reconstructed_data = [filter_ens.reconstruct(Phi=forecast_coefficients[..., ii]) for ii in range(max_lines)]
        mean_forecast = filter_ens.reconstruct(Phi=np.mean(forecast_coefficients, axis=-1))
        mse_mean = filter_ens.compute_MSE(POD_data=mean_forecast, 
                                        original_data=original_data, 
                                        time_evolution=True)
        mse_mean_true = filter_ens.compute_MSE(POD_data=mean_forecast, 
                                        original_data=original_data_true, 
                                        time_evolution=True)
    else:
        single_case=False
        max_lines = len(reconstructed_data)
    
    mse_error ,mse_error_true = [], []
    for rd in reconstructed_data:
        mse_error.append(filter_ens.compute_MSE(POD_data=rd, 
                                                original_data=original_data, 
                                                time_evolution=True))

        mse_error_true.append(filter_ens.compute_MSE(POD_data=rd, 
                                                original_data=original_data_true, 
                                                time_evolution=True))

    lbls = [None] * (max_lines - 1)
    label_mse = ['MSE(POD-ESN, Noisy data)', *lbls]
    label_mse_t = ['MSE(POD-ESN, Truth)', *lbls]

    fig, axs = plt.subplots(1, 3, layout='constrained', width_ratios=[10, .1, .1], figsize=(12, 4))
    
    ax = axs[0]
    if single_case:
        ax.plot(filter_ens.hist_t, mse_mean, c='darkorchid', lw=2, dashes=[5,.5], label='MSE($\\bar{u}_\\text{POD-ESN}$, Noisy data)')
        ax.plot(filter_ens.hist_t, mse_mean_true, c='darkgreen', lw=2, dashes=[5,.5], label='MSE($\\bar{u}_\\text{POD-ESN}$, Truth)')
        ax.plot(filter_ens.hist_t, np.array(mse_error).T, lw=1, label=label_mse, c='C4', alpha=0.4)
        ax.plot(filter_ens.hist_t, np.array(mse_error_true).T, lw=1, label=label_mse_t, c='C2', alpha=0.4)
    else:
        def set_colors(cmap_name):
            cmap = plt.cm.get_cmap(cmap_name)
            gradient = np.linspace(.4, 1, len(mse_error))[::-1]
            return [colors.to_hex(cmap(ii)) for ii in gradient]

        colors_mse, colors_mse_t = set_colors('Reds'), set_colors('Blues')

        for mse, mse_t, c, c_t, lb, lb_t in zip(mse_error, mse_error_true, colors_mse, colors_mse_t, label_mse, label_mse_t):
            ax.plot(filter_ens.hist_t, mse, lw=1, c=c, label=lb, dashes=[5,1])  
            ax.plot(filter_ens.hist_t, mse_t, lw=1, c=c_t, label=lb_t)

    ax.plot(filter_ens.hist_t, mse_ref,  c='k', alpha=1, lw=2, zorder=-1, label='MSE(Noisy data, Truth)')

    [ax.axvline(x=to, lw=.2, c='k', zorder=-1) for to in truth['t_obs']]

    ax.set(yscale='log', xlim=[0, None], xlabel='time', ylabel='MSE error')
    ax.legend()

    if not single_case:

        cmap1 = colors.ListedColormap(colors_mse)
        cmap2 = colors.ListedColormap(colors_mse_t)

        for cmap, cax in zip([cmap1, cmap2], axs[1:]):
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, orientation="vertical")
            cb.set_ticks([])

        yticks = np.linspace(0, 1, len(colors_mse) + 1)[:-1] + (0.5 / len(colors_mse))
        if tiks is None:
            tiks =range(1, len(colors_mse) + 1)
        cb.set_ticks(yticks, labels=tiks)


def plot_covariance(case, idx=-1, tixs=None, plot_correlation=False):

    if not isinstance(idx, list):
        idx = [idx]

    all_matrices = []

    for _i in idx:
        Af = case.hist[_i]
        y = case.get_observable_hist()[_i]

        Af = np.vstack((Af, y))
        N, m = Af.shape

        if plot_correlation:
            Cpp = np.corrcoef(Af - np.mean(Af, axis=-1, keepdims=True))
        else:
            Af_ = Af - np.mean(Af, axis=-1, keepdims=True)
            Cpp = np.dot(Af_, Af_.T) / (m - 1)

        all_matrices.append(Cpp)

    # Get global min/max
    global_vmax = max(np.max(abs(Cpp)) for Cpp in all_matrices)
    args = dict(cmap="PuOr", vmin=-global_vmax, vmax=global_vmax)

    if tixs is None:
        tixs = case.state_labels.copy()
        tixs += [case.alpha_labels[key] for key in case.est_a]
        tixs += case.obs_labels.copy()

    if case.Na > 0:
        nrows = 2
    else:
        nrows = 1

    Nphi, Na, Nq = case.Nphi, case.Na, case.Nq
    N = sum([Nphi, Na, Nq])

    for matrix in all_matrices:
        fig, axs = plt.subplots(nrows, 2, figsize=(N//2, N//2*nrows))
        axs = axs.ravel()


        axs[0].matshow(matrix, **args)
        axs[0].set(xticks=np.arange(N), xticklabels=tixs)
        axs[0].set(yticks=np.arange(N), yticklabels=tixs)

        im = axs[1].matshow(matrix[:Nphi, -Nq:], **args)
        axs[1].set(xticks=np.arange(Nq), xticklabels=tixs[-Nq:])
        axs[1].set(yticks=np.arange(Nphi), yticklabels=tixs[:Nphi])

        if nrows == 2:  
            axs[2].matshow(matrix[Nphi:Nphi + Na, -Nq:], **args)
            axs[2].set(xticks=np.arange(Nq), xticklabels=tixs[-Nq:])
            axs[2].set(yticks=np.arange(Na), yticklabels=tixs[Nphi:Nphi + Na])

            axs[3].matshow(matrix[:Nphi, Nphi:Nphi + Na].T, **args)
            axs[3].set(xticks=np.arange(Nphi), xticklabels=tixs[:Nphi])
            axs[3].set(yticks=np.arange(Na), yticklabels=tixs[Nphi:Nphi + Na])
        
        fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.5)

# ==================================================================================================================
def print_parameter_results(ensembles, true_values=None):
    if type(ensembles) is not list:
        ensembles = [ensembles]

    headers = ['']
    truth_row = ['Truth']

    if true_values is None:
        true_values = ensembles[0].get_default_params

    keys = sorted(true_values.keys())
    
    for key in keys:
        headers.append(key)
        truth_row.append('${:.8}$'.format(true_values[key]))

    rows = [truth_row]
    for ensemble in ensembles:
        alpha = ensemble.get_alpha()
        row = ['{} \n w/ {}'.format(ensemble.filter, ensemble.bias.name)]
        for key in headers[1:]:
            vals = [a[key] for a in alpha]

            row.append('${:.8} \n \\pm {:.4}$'.format(np.mean(vals), np.std(vals)))

        rows.append(row)

    print(tabulate(tabular_data=rows, headers=headers))


def plot_train_dataset(clean_data, noisy_data, t, *split_times):


    
    # Visualize the training dataset
    fig = plt.figure(figsize=(12.5, 5), layout='tight')
    sfs = fig.subfigures(1, 2, width_ratios=[1.2, 1])

    axs = sfs[0].subplots(nrows=clean_data.shape[1], ncols=1, sharex='col', sharey='row')

    for axi, ax in enumerate(axs):
        ax.plot(t, clean_data[:, axi], c='k', lw=1.2, label='Truth')
        if noisy_data is not None:
            ax.plot(t, noisy_data[:, axi], c='r', lw=0.4, label='Noisy data')
        for kk, c, lbl in zip(range(1, len(split_times) + 2), ['C2', 'C1', 'C4', 'lightgray'],
                              ['Train', 'Validation', 'Test', 'Not used']):
            ax.axvspan(sum(split_times[:kk]), sum(split_times[:kk + 1]), facecolor=c, alpha=0.3,
                       label=lbl + ' data')
            ax.axvline(x=sum(split_times[:kk]), color='k', lw=0.5, dashes=[10, 5])
        if axi == 0:
            ax.legend(loc='lower center', ncol=10, bbox_to_anchor=(0.5, 1.0))
    axs[-1].set(xlabel='$t/T$', xlim=[0, t[-1]]);

    ax = sfs[1].add_subplot(111, projection='3d')
    _nt = 1000
    ax.plot(clean_data[-_nt:, 0], clean_data[-_nt:, 1], clean_data[-_nt:, 2], '.-',
            c='k', ms=2, lw=.5, label='Truth')
    
    if noisy_data is not None:
        ax.plot(noisy_data[-_nt:, 0], noisy_data[-_nt:, 1], noisy_data[-_nt:, 2], '.',
                c='r', ms=1, label='Noisy data')
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')


def plot_obs_timeseries(*plot_cases, zoom_window=None, add_pdf=False, t_factor=1, dims='all'):
    """
    Plot the time evolution of the observables in a object of class model
    """

    if not isinstance(plot_cases, (list, tuple)):
        plot_cases = [plot_cases]


    # Ensure dims is a list of integers
    if dims == 'all':
        dims = list(range(plot_cases[0].Nq))
    elif isinstance(dims, int):
        dims = [dims]
    elif not isinstance(dims, (list, tuple, np.ndarray)):
        raise ValueError(f"`dims` must be 'all', int, or list of ints. Got: {dims}")


    fig = plt.figure(figsize=(8, 1.5*len(dims)), layout="constrained")
    axs = fig.subplots(len(dims), 2 + add_pdf, sharey='row', sharex='col')
    if len(dims) == 1:
        axs = [axs]

    xlabel = '$t$'
    if t_factor != 1:
        xlabel += '$/T$'

    for plot_case in plot_cases:
        y = plot_case.get_observable_hist()  # history of the model observables
        lbl = plot_case.obs_labels

        t_h = plot_case.hist_t / t_factor
        if zoom_window is None:
            zoom_window = [t_h[-1] - plot_case.t_CR, t_h[-1]]

        for ii, ax in zip(dims, axs):
            [ax[jj].plot(t_h, y[:, ii], lw=0.8) for jj in range(2)]
            ax[0].set(ylabel=lbl[ii])
            if add_pdf:
                ax[2].hist(y[:, ii], alpha=0.5, histtype='stepfilled', bins=20, density=True, orientation='horizontal',
                           stacked=False)

        axs[-1][0].set(xlabel=xlabel, xlim=[t_h[0], t_h[-1]])
        axs[-1][1].set(xlabel=xlabel, xlim=zoom_window)
        if plot_case.ensemble:
            plt.gcf().legend([f'$mi={mi}$' for mi in range(plot_case.m)], loc='center left', bbox_to_anchor=(1.0, .75),
                             ncol=1, frameon=False)


# def plot_parameters(plot_case):
#     """
#     Plot the time evolution of the parameters in a object of class model
#     """

#     colors_alpha = ['g', 'sandybrown', [0.7, 0.7, 0.87], 'b', 'r', 'gold', 'deepskyblue']

#     t_h = plot_case.hist_t
#     t_zoom = int(plot_case.t_CR / plot_case.dt)

#     hist_alpha = plot_case.hist[:, -plot_case.Na:]
#     mean_alpha = np.mean(hist_alpha, axis=-1)
#     std_alpha = np.std(hist_alpha, axis=-1)

#     fig = plt.figure(figsize=(5, 1.5*plot_case.Na), layout="constrained")
#     axs = fig.subplots(plot_case.Na, 1, sharex='col')
#     if isinstance(axs, plt.Axes):
#         axs = [axs]

#     axs[-1].set(xlabel='$t$', xlim=[plot_case.hist_t[0], plot_case.hist_t[-1]]);
#     for ii, ax, p, c in zip(range(len(axs)), axs, plot_case.est_a, colors_alpha):
#         avg, s, all_h = [xx[:, ii] for xx in [mean_alpha, std_alpha, hist_alpha]]
#         ax.fill_between(t_h, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=c, label='2 std')
#         ax.plot(t_h, avg, color=c, label='mean', lw=2)
#         ax.plot(t_h, all_h, color=c, lw=1., alpha=0.3)
#         ax.set(ylabel=plot_case.alpha_labels[p], ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])





if __name__ == '__main__':
    pass
