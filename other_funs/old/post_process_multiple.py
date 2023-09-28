import os as os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from Util import interpolate, CR, getEnvelope

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

if __name__ == '__main__':
    L = 10
    std = 0.1
    est_p = ['beta', 'zeta', 'kappa']
    kmeas = 25  # number of time steps between observations

    parent_folder = 'results/VdP_11.27_newEnsemble{}PE_{}kmeas/'.format(len(est_p), kmeas)
    folder = parent_folder + 'std{}/L{}/'.format(std, L)
    figs_folder = parent_folder + 'figs/'

    name = 'all_' + str(L) + '_' + str(std) + '_results'
    show_ = True
else:
    folder = folder
    show_ = False

files = os.listdir(folder)
flag = True
biases, esn_errors, biases_ESN = [], [], []
ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []
# ==================================================================================================================
fig = plt.figure(figsize=(19, 7.5), layout="constrained")
fig.suptitle(folder)
subfigs = fig.subfigures(2, 2, wspace=0.07, width_ratios=[3.5, 1])
axCRP = subfigs[0, 0].subplots(1, 3)
axNU = subfigs[0, 1].subplots(1, 1)
for file in files:
    if file[-3:] == '.py' or file[-4] == '.':
        continue
    k = float(file.split('_k')[-1])
    # if k > 20: # uncomment these lines to avoid ploting values over 20
    #     continue
    with open(folder + file, 'rb') as f:
        parameters = pickle.load(f)
        truth = pickle.load(f)
        filter_ens = pickle.load(f)
    # Observable bias
    y_filter, t_filter = filter_ens.getObservableHist()[0], filter_ens.hist_t
    y_truth = truth['y'][:len(y_filter)]
    b_truth = truth['b_true'][:len(y_filter)]
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
    y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)
    y_unbiased = interpolate(t_b, y_unbiased, t_filter)
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
    axCRP[0].plot(k, CB, 'o', color=bias_c, label='Biased')
    axCRP[0].plot(k, CU, '*', color=unbias_c, label='Unbiased')
    # RMS error
    axCRP[1].plot(k, RB, 'o', color=bias_c, label='Biased ')
    axCRP[1].plot(k, RU, '*', color=unbias_c, label='Unbiased')

    CB, RB = CR(y_obs, y_obs_b)  # biased
    CU, RU = CR(y_obs, y_obs_u)  # unbiased
    # Correlation
    bias_c = 'tab:red'
    unbias_c = 'tab:blue'
    axCRP[0].plot(k, CB, '+', color=bias_c, label='Biased at $t^a$')
    # RMS error
    axCRP[1].plot(k, RB, '+', color=bias_c, label='Biased at $t^a$')



    # Parameters ========================================================================================
    if filter_ens.est_p:
        if flag:
            N_psi = len(filter_ens.psi0)
            # c = ['tab:orange', 'navy', 'darkcyan', 'cyan']
            c = ['navy', 'chocolate', 'mediumturquoise', 'lightseagreen', 'cyan']
            marker = ['x', '+']
            time = ['$(t_\mathrm{end})$', '$(t_\mathrm{start})$']
            alphas = [1., .2]
            superscript = '^\mathrm{init}$'
            reference_p = filter_ens.alpha0
        for jj, p in enumerate(filter_ens.est_p):
            for kk, idx in enumerate([istop]):
                hist_p = filter_ens.hist[idx - 1, N_psi + jj] / reference_p[p]
                axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), capsize=6, alpha=alphas[kk],
                                 fmt=marker[kk], color=c[jj], label='$\\' + p + '/\\' + p + superscript + time[kk])
                axCRP[2].plot([min(ks)-1, max(ks)+1],
                              [truth['true_params'][p]/ reference_p[p], truth['true_params'][p]/ reference_p[p]],
                              '--', color=c[jj], linewidth=.8, alpha=.8, label='$\\' + p + '^\mathrm{true}/\\' + p + superscript)
        if 'beta' in filter_ens.est_p and 'zeta' in filter_ens.est_p:
            # compute growth rate
            final_nu = 0.
            for jj, p in enumerate(filter_ens.est_p):
                if p == 'beta':
                    final_nu += 0.5 * np.mean(filter_ens.hist[istop - 1, N_psi + jj]).squeeze()
                elif p == 'zeta':
                    final_nu -= 0.5 * np.mean(filter_ens.hist[istop - 1, N_psi + jj]).squeeze()

            axNU.plot(k, final_nu, '^', color=c[-1])
            if flag:
                final_nu = 0.
                for jj, p in enumerate(filter_ens.est_p):
                    if p == 'beta':
                        final_nu += 0.5 * np.mean(filter_ens.hist[istart - 1, N_psi + jj]).squeeze()
                    elif p == 'zeta':
                        final_nu -= 0.5 * np.mean(filter_ens.hist[istart - 1, N_psi + jj]).squeeze()
                axNU.plot([-10, 100], [final_nu, final_nu], '-', color='k', label='Pre-DA')
                true_nu = 0.5*(truth['true_params']['beta']-truth['true_params']['zeta'])
                axNU.plot([-10, 100], [true_nu, true_nu], '-', color='k', label='Truth', alpha=0.2, linewidth=5.)



    if flag:
        # compute and plot the baseline correlation and MSE
        y_truth_u = y_truth - b_truth
        Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
        axCRP[0].plot((-10, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
        axCRP[1].plot((-10, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
        # compute C and R before the assimilation (the initial ensemble has some initialisation error)
        Cpre, Rpre = CR(y_truth[istart - N_CR:istart+1:], np.mean(y_filter, -1)[istart - N_CR:istart+1:])
        axCRP[0].plot((-10, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
        axCRP[1].plot((-10, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
        for ax1 in axCRP[1:]:
            ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)



    flag = False
    # PLOT SOME INDIVIDUAL TIME SOLUTIONS ================================================================
    # if k in [0.0, 10.0, 30.0]:
    # exec(open("post_process.py").read(), {'parameters': parameters,
    #                                       'filter_ens': filter_ens,
    #                                       'truth': truth})

# WAIT - ISTART/ISTOP IS FOR Y_TRUTH? DOESN'T MAKE SENSE

# =========================================================================================================
xlims = [min(ks) - 1, max(ks) + 1]
axCRP[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$')
axCRP[1].set(ylabel='RMS error', xlim=xlims, xlabel='$\\gamma$')
axCRP[2].set(ylim=[0.6, 1.6], xlim=xlims)
axNU.set(xlim=xlims, xlabel='$\\gamma$', ylabel='$\\nu$')

for ax1 in np.append(axCRP[:], axNU):
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0) / (y1 - y0))
# plt.tight_layout()

# PLOT MEAN ERROR EVOLUTION ================================================================================

ax = subfigs[1, 0].subplots(1, 2)
mean_ax = ax[:]
for mic in [0]:
    scale = np.max(truth['y'][:, mic])
    norm = mpl.colors.Normalize(vmin=0, vmax=max(ks))
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    for i, metric in enumerate([biases, esn_errors]):  # , biases_ESN]):
        errors = [b[:, mic] / scale for b in metric]
        for err, k in zip(errors, ks):
            mean_ax[i].plot(t_interp, err * 100, color=cmap.to_rgba(k))
        mean_ax[i].set(xlim=[t_filter[istart] - 0.02, t_filter[istop] + 0.05], xlabel='$t$')

    mean_ax[0].set(ylim=[0, 50], ylabel='Biased signal error [\%]')
    mean_ax[1].set(ylim=[0, 10], ylabel='Unbiased signal error [\%]')

    for i in range(2):
        x0, x1 = mean_ax[i].get_xlim()
        y0, y1 = mean_ax[i].get_ylim()
        # print( (x1 - x0) / (y1 - y0))
        mean_ax[i].set_aspect(0.5 * (x1 - x0) / (y1 - y0))

clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
clb.ax.set_title('$\\gamma$')

plt.savefig(figs_folder + name + '.svg', dpi=350)
plt.savefig(figs_folder + name + '.pdf', dpi=350)

if show_:
    plt.show()
