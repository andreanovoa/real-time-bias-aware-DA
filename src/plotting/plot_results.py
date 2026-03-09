from tabulate import tabulate
import numpy as np
from utils import interpolate 
from .utils_plotting import *


XDG_RUNTIME_DIR = 'tmp/'


def recover_unbiased_solution(t_b, b, t, y, upsample=True):
    if b.ndim < y.ndim:
        b = np.expand_dims(b, axis=-1)
    elif b.shape[-1] > 1:
        b = np.mean(b, axis=-1, keepdims=True)
    if upsample:
        b = interpolate(t_b, b, t)
    return y + b



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



def plot_parameters(ensembles, t_obs = None, filename=None, reference_p=None, plot_ensemble_members=False):
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

    if t_obs is not None:
        xlim = [t_obs[0], t_obs[-1]]
    else:      
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


def plot_violins(ax, values, location, color='b', label=None, alpha=0.5, **kwargs):
    violins = ax.violinplot(values, positions=location, **kwargs)
    vp = None
    for vp in violins['bodies']:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(.5)
        vp.set_alpha(alpha)
        vert = vp.get_paths()[0].vertices[:, 0]
        vp.get_paths()[0].vertices[:, 0] = np.clip(vert, np.mean(vert), np.inf)
    if label is not None and vp is not None:
        vp.set_label(label) 
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violins[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(.75)


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
        true_values = ensembles[0].alpha0

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
                ax[2].hist(y[:, ii], alpha=0.5, histtype='stepfilled', bins=20, density=True, 
                           orientation='horizontal', stacked=False)

        axs[-1][0].set(xlabel=xlabel, xlim=[t_h[0], t_h[-1]])
        axs[-1][1].set(xlabel=xlabel, xlim=zoom_window)
        if plot_case.ensemble:
            plt.gcf().legend([f'$mi={mi}$' for mi in range(plot_case.m)], loc='center left', 
                             bbox_to_anchor=(1.0, .75), ncol=1, frameon=False)


if __name__ == '__main__':
    pass
