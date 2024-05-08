import pickle
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy.interpolate import interp2d
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.backends.backend_pdf as plt_pdf
from essentials.Util import interpolate, fun_PSD

XDG_RUNTIME_DIR = 'tmp/'

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', framealpha=0, edgecolor=(0, 0, 1, 0.1))

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
    #     0
    #     # y_unbiased = interpolate(t, y, t_b) + b
    #     # y_unbiased = interpolate(t_b, y_unbiased, t)
    # else:
    y_unbiased = y + b
    return y_unbiased


def plot_ensemble(ensemble, max_modes=None, reference_params=None, nbins=6):
    if max_modes is None:
        max_modes = ensemble.Nphi
    fig, axs = plt.subplots(figsize=(12, 1.5), layout='tight', nrows=1, ncols=max_modes, sharey=True)
    for ax, ph, lbl in zip(axs.ravel(), ensemble.get_current_state[:-ensemble.Na, ], ensemble.state_labels):
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
            xlims = np.array(ensemble.std_a[param]) / reference_alpha[param]
            ax.hist(a / reference_alpha[param], bins=np.linspace(*xlims, nbins))
            ax.set(xlabel=ensemble.params_labels[param])


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
                    p_lbl = filter_ens.params_labels[p]
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
        labels_p.append('$' + p + norm_lbl(filter_ens.params_labels[p]) + '$')
        hist_alpha.append(m)
        ii += 1

    for ax, p, a, c, lbl in zip(ax_all, filter_ens.est_a, hist_alpha, colors_alpha, labels_p):
        plot_violins(ax, a, t_obs, widths=dt_obs / 2, color=c, label='analysis posterior')
        alpha_lims = [filter_ens.params_lims[p][0] / reference_p[p],
                      filter_ens.params_lims[p][1] / reference_p[p]]
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


def post_process_single(filter_ens, truth, filename=None, mic=0, reference_y=1.,
                        reference_p=None, reference_t=1., plot_params=False):
    t_obs, obs = truth['t_obs'], truth['y_obs']

    num_DA_blind = filter_ens.num_DA_blind
    num_SE_only = filter_ens.num_SE_only

    y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t
    b = filter_ens.bias.get_bias(state=filter_ens.bias.hist)
    t_b = filter_ens.bias.hist_t

    y_filter, b, obs = [yy[:, mic] for yy in [y_filter, b, obs]]
    y_mean = np.mean(y_filter, -1)

    # cut signals to interval of interest -----
    N_CR = int(filter_ens.t_CR // filter_ens.dt)  # Length of interval to compute correlation and RMS
    i0, i1 = [np.argmin(abs(t - truth['t_obs'][idx])) for idx in [0, -1]]  # start/end of assimilation

    y_unbiased = recover_unbiased_solution(t_b, b, t, y_mean, upsample=hasattr(filter_ens.bias, 'upsample'))
    y_filter, y_mean, y_unbiased, t = (yy[i0 - N_CR:i1 + N_CR] for yy in [y_filter, y_mean, y_unbiased, t])

    y_raw = interpolate(truth['t'], truth['y_raw'][:, mic], t)

    y_truth_no_noise = interpolate(truth['t'], truth['y_true'][:, mic], t)
    b_obs = y_truth_no_noise - y_mean
    b_obs_noisy = y_raw - y_mean

    if reference_t == 1.:
        t_label = '$t$ [s]'
        if 'wash_t' in truth.keys():
            t_wash, wash = truth['wash_t'], truth['wash_obs'][:, mic]
    else:
        t_label = '$t/T$'
        t, t_b, t_obs = [tt / reference_t for tt in [t, t_b, t_obs]]
        if 'wash_t' in truth.keys():
            t_wash, wash = truth['wash_t'] / reference_t, truth['wash_obs'][:, mic]

    # % PLOT time series ------------------------------------------------------------------------------------------
    # norm = np.max(y_raw, axis=0)  # normalizing constant
    if int(reference_y) == 1:
        ylbls = [["$p(x_\\mathrm{f})$ [Pa]", "$b(x_\\mathrm{f})$ [Pa]"], ['', '']]
    else:
        ylbls = [["$p(x_\\mathrm{f})$ norm.", "$b(x_\\mathrm{f})$ norm."], ['', '']]

    fig1 = plt.figure(figsize=[8, 5], layout="constrained")
    subfigs = fig1.subfigures(2, 1, height_ratios=[1, 1.1])
    ax_zoom = subfigs[0].subplots(2, 2, sharex='col', sharey='row')
    ax_all = subfigs[1].subplots(2, 1, sharex='col')

    margin = 0.5
    max_y = np.max(abs(y_raw[:N_CR]))
    # y_lims = [np.min(y_raw[:N_CR]) - margin, np.max(y_raw[:N_CR]) + margin]
    y_lims = [(-max_y - margin) / reference_y, (max_y + margin) / reference_y]
    x_lims = [[t_obs[0] - .5 * filter_ens.t_CR, t_obs[0] + filter_ens.t_CR],
              [t_obs[-1] - .5 * filter_ens.t_CR, t_obs[-1] + filter_ens.t_CR],
              [t_obs[0] - .5 * filter_ens.t_CR, t[-1]]]

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
            labels_p.append(norm_lbl(filter_ens.params_labels[p]))
            mean_p.append(m)
            std_p.append(s)
            ii += 1

    for axs in [ax_zoom[:, 0], ax_zoom[:, 1]]:
        # Observables ---------------------------------------------------------------------
        axs[0].plot(t, y_raw / reference_y, label='t-N', **true_noisy_props)
        for mi in range(y_filter.shape[1]):
            axs[0].plot(t, y_filter[:, mi] / reference_y, **y_biased_props)

        for yyy, lbl, kwargs in zip([y_truth_no_noise, y_mean, y_unbiased], ['t', 'b', 'u'],
                                    [true_props, y_biased_mean_props, y_unbias_props]):
            axs[0].plot(t, yyy / reference_y, label=lbl, **kwargs)

        axs[0].plot(t_obs, obs / reference_y, **obs_props)
        axs[0].set(ylim=y_lims)

        # BIAS ---------------------------------------------------------------------
        for bbb, lbl, kwargs in zip([b_obs_noisy, b_obs], ['o-N', 'o'], [bias_obs_noisy_props, bias_obs_props]):
            axs[1].plot(t, bbb / reference_y, label=lbl, **kwargs)
        axs[1].plot(t_b, b / reference_y, label=filter_ens.bias.name, **bias_props)
        if 'wash_t' in truth.keys():
            axs[0].plot(t_wash, wash / reference_y, **obs_props)

    for axs in [ax_all]:
        for bbb, lbl, kwargs in zip([b_obs_noisy, b_obs], ['o-N', 'o'], [bias_obs_noisy_props, bias_obs_props]):
            axs[0].plot(t, bbb / reference_y, label=lbl, **kwargs)

        axs[0].plot(t_b, b / reference_y, label=filter_ens.bias.name, **bias_props)
        axs[0].set(ylabel=ylbls[0][1], xlim=x_lims[-1], ylim=y_lims)
        # PARAMS-----------------------
        if filter_ens.est_a:
            for p, m, s, c, lbl in zip(filter_ens.est_a, mean_p, std_p, colors_alpha, labels_p):
                axs[1].plot(hist_t, m, lw=1., color=c, label=lbl)
                axs[1].set(xlabel=t_label)
                axs[1].fill_between(hist_t, m + s, m - s, alpha=0.6, color=c)

            axs[1].set(xlabel=t_label, ylabel="", xlim=x_lims[-1], ylim=[min_p, max_p])

            plot_DA_window(t_obs, axs[1], twin=twin)
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
        axs[1].set(ylabel=ylbl[1], xlim=xl, ylim=y_lims, xlabel=t_label)

        for ax_ in axs:
            plot_DA_window(t_obs, ax=ax_)

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
    from matplotlib.animation import FuncAnimation
    from essentials.physical_models import Annular
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


def plot_DA_window(t_obs, ax=None, twin=False):
    if ax is None:
        ax = plt.gca()
    ax.axvline(x=t_obs[-1], ls='--', color='k', linewidth=.8)
    ax.axvline(x=t_obs[0], ls='--', color='k', linewidth=.8)
    if twin:
        ax.axhline(y=1, ls='--', color='k', linewidth=.6)


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


def plot_parameters(ensembles, truth, filename=None, reference_p=None):
    if type(ensembles) is not list:
        ensembles = [ensembles]

    filter_ens = ensembles[0]

    if len(filter_ens.est_a) < 4:
        fig1 = plt.figure(figsize=(6., 1.5 * len(filter_ens.est_a)), layout="constrained")
        axs = fig1.subplots(len(filter_ens.est_a), 1, sharex='col')
        if len(filter_ens.est_a) == 1:
            axs = [axs]
    else:
        rows = len(filter_ens.est_a) // 2
        if len(filter_ens.est_a) % 2:
            rows += 1

        fig1 = plt.figure(figsize=(12, 1.5 * rows), layout="constrained")
        axs = fig1.subplots(rows, 2, sharex='all')
        axs = axs.ravel()

    ref_p = {**filter_ens.alpha0}
    for key in filter_ens.est_a:
        ref_p[key] = 1.

    if reference_p is None:
        norm_lbl = lambda x: x
    else:
        norm_lbl = lambda x: x + '/' + x + '$^\\mathrm{ref}$'
        for key, val in reference_p.items():
            ref_p[key] = val

    t_obs = truth['t_obs']
    x_lims = [t_obs[0], t_obs[-1]]
    style = ['-', '--']

    def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
        # number of categories(nc) and the number of subcategories(nsc)
        # and returns a colormap with nc * nsc different colors, where for
        # each category there are nsc colors of same hue.

        if nc > plt.get_cmap(cmap).N:
            raise ValueError("Too many categories for colormap.")
        if continuous:
            ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
        else:
            ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
        cols = np.zeros((nc * nsc, 3))
        for i, c in enumerate(ccolors):
            chsv = matplotlib.colors.rgb_to_hsv(c[:3])
            arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
            arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
            arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
            rgb = matplotlib.colors.hsv_to_rgb(arhsv)
            cols[i * nsc:(i + 1) * nsc, :] = rgb
        # return matplotlib.colors.ListedColormap(cols)
        return cols

    colors = []
    cmap = categorical_cmap(len(filter_ens.est_a), len(ensembles), cmap="Set1")
    for ii in range(len(ensembles)):
        colors.append(cmap[ii::len(ensembles)])

    for kk, ens in enumerate(ensembles):
        hist, hist_t = ens.hist, ens.hist_t
        hist_mean = np.mean(hist, -1, keepdims=True)

        mean_p, std_p, labels_p = [], [], []

        ii = ens.Nphi
        for p in ens.est_a:
            m = hist_mean[:, ii].squeeze() / ref_p[p]
            s = abs(np.std(hist[:, ii] / ref_p[p], axis=1))
            labels_p.append(norm_lbl(ens.params_labels[p]))
            mean_p.append(m)
            std_p.append(s)
            ii += 1

        for ax, p, m, s, c, lbl in zip(axs, ens.est_a, mean_p, std_p, colors[kk], labels_p):
            max_p = np.max(m + abs(s))
            min_p = np.min(m - abs(s))
            ax.plot(hist_t, m, ls=style[kk], color=c, label=lbl)
            ax.fill_between(hist_t, m + abs(s), m - abs(s), alpha=0.6, color=c)
            if kk == 0:
                if filter_ens.params_lims[p][0] is not None and filter_ens.params_lims[p][1] is not None:
                    for lim in [filter_ens.params_lims[p][0] / ref_p[p],
                                filter_ens.params_lims[p][1] / ref_p[p]]:
                        ax.plot([hist_t[0], hist_t[-1]], [lim, lim], '--', color=c, lw=2, alpha=0.5)

                for idx, cl, ll in zip(['num_DA_blind', 'num_SE_only'], ['darkblue', 'darkviolet'], ['BE', 'PE']):
                    idx = getattr(filter_ens, idx)
                    if idx > 0:
                        ax.plot((t_obs[idx], t_obs[idx]), (min_p, max_p), '-.', color=cl)
                        ax.plot((t_obs[idx], t_obs[idx]), (min_p, max_p), '-.', color=cl, label='Start ' + ll)

                plot_DA_window(t_obs, ax=ax)

            ax.legend(loc='upper right', fontsize='small', ncol=2)

            if kk > 0:
                ylims = ax.get_ylim()
                min_p = min([ylims[0], min_p])
                max_p = max([ylims[1], max_p])

            ax.set(ylabel='', ylim=[min_p, max_p])

        axs[-1].set(xlabel='$t$ [s]', xlim=x_lims)

    if filename is not None:
        plt.savefig(filename + '_params.svg', dpi=350)


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
    y_BA = filter_ens_BA.get_observable_hist(loc=locs)
    y_BA = np.mean(y_BA, -1)
    # Bias-blind EnKF sol
    y_BB = filter_ens_BB.get_observable_hist(loc=locs)
    y_BB = np.mean(y_BB, -1)
    # truth
    y_t = truth_ens.get_observable_hist(loc=locs).squeeze()
    y_t += .4 * y_t * np.sin((np.expand_dims(truth_ens.hist_t, -1) * np.pi * 2) ** 2)
    y_t = interpolate(truth_ens.hist_t, y_t, filter_ens.hist_t)

    max_v = [np.max(abs(yy)) for yy in [y_t, y_BA, y_BB]]
    max_v = np.max(max_v)

    # -----------------------

    fig1 = plt.figure(figsize=[10, 2], layout='constrained')
    ax1 = fig1.subplots(1, 1)

    fig2 = plt.figure(figsize=[12, 6], layout='constrained')
    ax2 = fig2.subplots(2, 1)

    t0 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][0] - filter_ens.t_CR / 2)))
    t1 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][-1] + filter_ens.t_CR)))

    t_gif = filter_ens.hist_t[t0:t1:5]

    # all pressure points
    y_BA = filter_ens_BA.get_observable_hist(loc=locs)
    y_BB = filter_ens_BB.get_observable_hist(loc=locs)
    y_t, y_BB, y_BA = [interpolate(filter_ens.hist_t, yy, t_gif) for yy in [y_t, y_BB, y_BA]]
    max_v = np.max(abs(y_t))

    # observation points
    y_BB_obs = filter_ens_BB.get_observable_hist()
    y_BA_obs = filter_ens_BA.get_observable_hist()

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
    y_BA_tt = filter_ens_BA.get_observable_hist()[:, 0]
    y_BA_tt_u = y_BA_tt + interpolate(filter_ens_BA.bias.hist_t, filter_ens_BA.bias.hist, filter_ens_BA.hist_t)

    y_BB_tt = filter_ens_BB.get_observable_hist()[:, 0]
    y_t_tt = truth['y_raw'][:, 0]
    y_obs_tt = truth['p_obs'][:, 0]
    pressure_legend = ['Truth', 'Data', 'State + bias  BA', 'State est. BA', 'State est.']

    def animate1(ai):
        ax1.clear()
        ax1.set(ylim=[-max_v, max_v], xlim=[0, 1], title='$t={:.4}$'.format(t_gif[ai]),
                xlabel='$x/L$', ylabel="$p'$ [Pa]")
        ax1.plot(locs, y_t[ai], color=color_true, linewidth=1.5)
        for loc in filter_ens.x_mic:
            ax1.plot([loc, loc], [0.7 * max_v, max_v], '.-', color='black', linewidth=2)
        ax1.plot([filter_ens.x_mic[0], filter_ens.x_mic[0]], [-max_v, max_v], '--',
                 color='firebrick', linewidth=4, alpha=0.2)
        for yy, c, l in zip([y_BA[ai], y_BB[ai]], [color_bias, 'orange'], ['-', '--']):
            y_mean, y_std = np.mean(yy, -1), abs(np.std(yy, -1))
            ax1.plot(locs, y_mean, l, color=c)
            ax1.fill_between(locs, y_mean + y_std, y_mean - y_std, alpha=0.1, color=c)

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
        ax2[0].plot(tt_, yy, color=color_true, linewidth=1.5)
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
                    filename=None, reference_y=1., reference_t=1., max_time=None):
    t_obs, obs = truth['t_obs'], truth['y_obs']

    y_filter, t = filter_ens.get_observable_hist(), filter_ens.hist_t

    y_mean = np.mean(y_filter, -1, keepdims=True)

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
    y_truth_no_noise = interpolate(truth['t'], truth['y_true'], t)

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
    Nq = filter_ens.Nq
    y_raw, y_unbiased, y_filter, y_mean, obs, y_truth_no_noise = [yy / reference_y for yy in
                                                                  [y_raw, y_unbiased, y_filter,
                                                                   y_mean, obs, y_truth_no_noise]]
    margin = 0.15 * np.mean(abs(y_raw))
    max_y = np.max(y_raw)
    min_y = np.min(y_raw)
    y_lims = [min_y - margin, max_y + margin]

    x_lims = [[t_obs[0] - .25 * filter_ens.t_CR, t_obs[0] + filter_ens.t_CR],
              [t_obs[-1] - filter_ens.t_CR, max_time],
              [t[0], max_time]]

    if plot_states:
        fig1 = plt.figure(figsize=(10, 5 * Nq), layout="constrained")
        sfs = fig1.subfigures(nrows=2, ncols=1)
        ax_all = sfs[0].subplots(Nq, ncols=1, sharey='row', sharex='col')
        ax_zoom = sfs[1].subplots(Nq, ncols=2, sharey='row', sharex='col')

        for qi in range(Nq):
            if Nq == 1:
                q_axes = [ax_zoom[0], ax_zoom[1], ax_all]
            else:
                q_axes = [ax_zoom[qi, 0], ax_zoom[qi, 1], ax_all[qi]]

            for ax, xl in zip(q_axes, x_lims):

                # Observables ---------------------------------------------------------------------
                ax.plot(t, y_truth_no_noise[:, qi], label='truth', **true_props)
                if filter_ens.bias.name != 'NoBias':
                    ax.plot(t, y_unbiased[:, qi], label='ubiased', **y_unbias_props)

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
                plot_DA_window(t_obs, ax)
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
        fig1 = plt.figure(figsize=(9, 5.5), layout="constrained")
        subfigs = fig1.subfigures(1, 2, width_ratios=[1.1, 1])
        ax_zoom = subfigs[0].subplots(Nq, 2, sharex='col', sharey='row')
        ax_all = subfigs[1].subplots(Nq, 1, sharex='col')

        b_filter = b
        b_raw = interpolate(t, y_raw - y_mean, t_b)
        b_truth_no_noise = interpolate(t, y_truth_no_noise - y_mean, t_b)
        b_raw, b_filter, b_truth_no_noise = [yy / reference_y for yy in [b_raw, b_filter, b_truth_no_noise]]

        max_y = np.max(abs(y_raw[:-N_CR]))
        y_lims = [-max_y - margin, max_y + margin]
        for qi in range(Nq):

            for ax, xl in zip([ax_zoom[qi, 0], ax_zoom[qi, 1], ax_all[qi]], x_lims):
                # Observables ---------------------------------------------------------------------
                ax.plot(t_b, b_raw[:, qi], label='t', **bias_obs_noisy_props)
                ax.plot(t_b, b_truth_no_noise[:, qi], label='t', **bias_obs_props)
                ax.plot(t_b, b_filter[:, qi], label='u', **bias_props)
                plot_DA_window(t_obs, ax)
                ax.set(ylim=y_lims, xlim=xl)

            ylbl = '$b_{}$'.format(qi)
            if reference_y != 1.:
                ylbl += ' norm.'
            ax_zoom[qi, 0].set(ylabel=ylbl)

        ax_all[0].legend(loc='upper left', bbox_to_anchor=(0., 1.1), ncol=5, fontsize='xx-small')
        for ax in [ax_zoom[-1, 0], ax_zoom[-1, 1], ax_all[-1]]:
            ax.set(xlabel=t_label)

        if filename is not None:
            plt.savefig(filename + '.svg', dpi=350)
            plt.close()



def plot_attractor(psi_cases, color, figsize=(8, 8)):
    if type(psi_cases) is not list:
        psi_cases, color = [psi_cases], [color]
    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(111, projection='3d')
    for axs_ax in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axs_ax.pane.fill = False
    ax3d.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')
    for psi_, c  in zip(psi_cases, color):
        if psi_.ndim > 2:
            psi_ = np.mean(psi_, axis=-1)
        ax3d.plot(psi_[:, 0], psi_[:, 1], psi_[:, 2], lw=1., c=c)

def plot_truth(y_raw, y_true, t, dt,
               plot_time=False, Nq=None, filename=None, f_max=None, y_obs=None, t_obs=None, model=None, **other):
    if Nq is None:
        Nq = y_true.shape[1]
    if t_obs is None:
        t0 = 0
        t1 = int(model.t_transient // dt)
    else:
        t0 = int(t_obs[0] // dt)
        t1 = int((t_obs[-1] + model.t_CR) // dt)

    noise = y_raw - y_true
    max_y = np.max(abs(y_raw[:t1 - t0]))

    fig1 = plt.figure(figsize=(15, 2 * Nq), layout="constrained")
    subfigs = fig1.subfigures(nrows=1, ncols=4, width_ratios=[2, 0.5, 1, 1])
    labels = ['Raw', 'Post-processed', 'Difference']
    y_labels = ['$\\tilde{y}, y$', '', '$(\\tilde{y}-y)$']
    cols = ['tab:blue', 'mediumseagreen', 'tab:purple']

    ax_01 = subfigs[0].subplots(Nq, 2, sharex='all', sharey='all')
    ax_4 = subfigs[-1].subplots(Nq, 1, sharex='all', sharey='all')

    # Plot zoomed timeseries of raw, post-processed and noise
    for ax, yy, ttl, lbl, c in zip([ax_01[:, 0], ax_01[:, 1], ax_4], [y_raw, y_true, noise], labels, y_labels, cols):
        if Nq == 1:
            ax = [ax]
        ax[0].set(title=ttl)
        ax[-1].set(xlabel='$t$', xlim=[t[t0], t[t1]])
        for qi in range(Nq):
            ax[qi].plot(t, yy[:, qi], color=c)
            ax[qi].axhline(np.mean(yy[:, qi]), color=c)
            if ttl[0] == 'R' and y_obs is not None:
                ax[qi].plot(t_obs, y_obs[:, qi], 'ro', ms=3)
            if len(lbl) > 1:
                ax[qi].set(ylabel=lbl + '$_{}$'.format(qi))

    # Plot probability density essentials and power spectral densities
    ax_pdf = subfigs[1].subplots(Nq, 1, sharey='all', sharex='all')
    ax_PSD = subfigs[2].subplots(Nq, 1, sharex='all', sharey='all')
    if Nq == 1:
        ax_pdf = [ax_pdf]
        ax_PSD = [ax_PSD]
    binwidth = 0.01 * max_y
    bins = np.arange(-max_y, max_y + binwidth, binwidth)
    for yy, ttl, lbl, c in zip([y_raw, y_true], labels[:2], y_labels[:2], cols[:2]):
        yy = yy.squeeze()
        ax_pdf[0].set(title='PDF', ylim=[-max_y, max_y])
        ax_pdf[-1].set(xlabel='$p$')
        ax_PSD[-1].set(xlabel='$f$')
        for qi in range(Nq):
            ax_pdf[qi].hist(yy[:, qi], bins=bins, density=True, orientation='horizontal',
                            color=c, label=lbl + '$_{}$'.format(qi), histtype='step')
        f, PSD = fun_PSD(dt, yy.squeeze())
        for qi in range(Nq):
            ax_PSD[qi].semilogy(f, PSD[qi], color=c, label=lbl + '$_{}$'.format(qi))
        ax_PSD[0].set(title='PSD', xlim=[0, f_max])

    # Plot full timeseries if requested
    figs2 = []
    if plot_time:
        for yy, name, c in zip([y_raw, y_true], labels[:2], cols[:2]):
            y_true, t = [zz[t0:] for zz in [yy, t_true]]
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


def plot_covarriance(case):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    idx = -1

    Af = case.hist[idx]
    y = case.get_observable_hist()[idx]

    Af = np.vstack((Af, y))

    N, m = Af.shape
    Nphi, Na, Nq = case.Nphi, case.Na, case.Nq

    tixs = case.state_labels.copy()
    tixs += [case.params_labels[key] for key in case.est_a]
    tixs += ['$p_{}$'.format(ii) for ii in np.arange(Nq)]

    # range_psi = np.max(Af, axis=-1, keepdims=True) - np.min(Af, axis=-1, keepdims=True)
    # Cpp = (Af - np.mean(Af, axis=-1, keepdims=True)) / range_psi
    # Cpp = np.dot(Cpp, Cpp.T) / (m - 1)

    Cpp = np.corrcoef(Af - np.mean(Af, axis=-1, keepdims=True))

    for ii in range(Cpp.shape[0]):
        for jj in range(Cpp.shape[1]):
            if jj < ii:
                Cpp[ii, jj] = 0

    axs = axs.ravel()
    vmax = np.max(abs(Cpp))
    args = dict(cmap="PuOr", vmin=-vmax, vmax=vmax)

    axs[0].matshow(Cpp, **args)
    axs[0].set(xticks=np.arange(N), xticklabels=tixs)
    axs[0].set(yticks=np.arange(N), yticklabels=tixs)

    im = axs[1].matshow(Cpp[:Nphi, -Nq:], **args)
    cbar = plt.colorbar(im, orientation='vertical', shrink=0.5)
    axs[1].set(xticks=np.arange(Nq), xticklabels=tixs[-Nq:])
    axs[1].set(yticks=np.arange(Nphi), yticklabels=tixs[:Nphi])

    axs[2].matshow(Cpp[Nphi:Nphi + Na, -Nq:], **args)
    axs[2].set(xticks=np.arange(Nq), xticklabels=tixs[-Nq:])
    axs[2].set(yticks=np.arange(Na), yticklabels=tixs[Nphi:Nphi + Na])

    axs[3].matshow(Cpp[:Nphi, Nphi:Nphi + Na].T, **args)
    axs[3].set(xticks=np.arange(Nphi), xticklabels=tixs[:Nphi])
    axs[3].set(yticks=np.arange(Na), yticklabels=tixs[Nphi:Nphi + Na])


# ==================================================================================================================
def print_parameter_results(ensembles, true_values):
    from tabulate import tabulate
    if type(ensembles) is not list:
        ensembles = [ensembles]

    headers = ['']
    truth_row = ['Truth']
    for key, val in true_values.items():
        headers.append(key)
        truth_row.append('${:.8}$'.format(val))

    rows = [truth_row]
    for ensemble in ensembles:
        alpha = ensemble.get_alpha()
        row = ['{} \n w/ {}'.format(ensemble.filter, ensemble.bias.name)]
        for key in headers[1:]:
            vals = [a[key] for a in alpha]

            row.append('${:.8} \n \\pm {:.4}$'.format(np.mean(vals), np.std(vals)))

        rows.append(row)

    print(tabulate(tabular_data=rows, headers=headers))


if __name__ == '__main__':
    pass