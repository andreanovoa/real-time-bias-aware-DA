import os as os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from Util import interpolate, CR, getEnvelope
from Ensemble import createEnsemble

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


parent_folder = 'results/VdP_Fig2_1PE/'
stds = [0.01, 0.1, 0.25]
Ls = [1, 10, 100]

# ==================================================================================================================
fig = plt.figure(figsize=(19, 7.5), layout="constrained")
fig.suptitle(parent_folder)
subfigs = fig.subfigures(len(Ls), len(stds), wspace=0.07)

for si in range(len(stds)):
    for Li in range(len(Ls)):
        ax = subfigs[Li, si].subplots(1, 2)
        subfigs[Li, si].suptitle('std={}, L={}'.format(stds[si], Ls[Li]))
        folder = parent_folder + 'std{}/L{}/'.format(stds[si], Ls[Li])

        files = os.listdir(folder)
        flag = True
        biases, esn_errors, biases_ESN = [], [], []
        ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []


        for file in files:
            if file[-3:] == '.py' or file[-4] == '.':
                continue
            k = float(file.split('_k')[-1])
            with open(folder + file, 'rb') as f:
                parameters = pickle.load(f)
                if flag:  # init the ensemble class with the first file
                    createEnsemble(parameters['forecast_model'])
                truth = pickle.load(f)
                filter_ens = pickle.load(f)
            # Observable bias
            y_filter, t_filter = filter_ens.getObservableHist()[0], filter_ens.hist_t
            y_truth = truth['y'][:len(y_filter)]
            b_truth = truth['b_true'][:len(y_filter)]
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
            ax[0].plot(k, CB, 'o', color=bias_c, label='Biased')
            ax[0].plot(k, CU, '*', color=unbias_c, label='Unbiased')
            # RMS error
            ax[1].plot(k, RB, 'o', color=bias_c, label='Biased ')
            ax[1].plot(k, RU, '*', color=unbias_c, label='Unbiased')

            CB, RB = CR(y_obs, y_obs_b)  # biased
            CU, RU = CR(y_obs, y_obs_u)  # unbiased
            # Correlation
            bias_c = 'tab:red'
            unbias_c = 'tab:blue'


            # # Parameters ========================================================================================
            # if filter_ens.est_p:
            #     if flag:
            #         N_psi = len(filter_ens.psi0)
            #         c = ['tab:orange', 'navy', 'forestgreen', 'cyan']
            #         marker = ['+', 'x']
            #         time = ['$(t_\mathrm{start})$', '$(t_\mathrm{end})$']
            #         alphas = [0.2, 1.0]
            #         superscript = '^\mathrm{init}$'
            #         reference_p = filter_ens.alpha0
            #     for jj, p in enumerate(filter_ens.est_p):
            #         for kk, idx in enumerate([istart, istop]):
            #             hist_p = filter_ens.hist[idx - 1, N_psi + jj] / reference_p[p]
            #             axCRP[2].errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), capsize=6, alpha=alphas[kk],
            #                              fmt=marker[kk], color=c[jj], label='$\\' + p + '/\\' + p + superscript + time[kk])
            #
            #     if 'beta' in filter_ens.est_p and 'zeta' in filter_ens.est_p:
            #         # compute growth rate
            #         final_nu = 0.
            #         for jj, p in enumerate(filter_ens.est_p):
            #             if p == 'beta':
            #                 final_nu += 0.5 * np.mean(filter_ens.hist[istop - 1, N_psi + jj]).squeeze()
            #             elif p == 'zeta':
            #                 final_nu -= 0.5 * np.mean(filter_ens.hist[istop - 1, N_psi + jj]).squeeze()
            #
            #         axNU.plot(k, final_nu, '^', color=c[-1])
            #         if flag:
            #             final_nu = 0.
            #             for jj, p in enumerate(filter_ens.est_p):
            #                 if p == 'beta':
            #                     final_nu += 0.5 * np.mean(filter_ens.hist[istart - 1, N_psi + jj]).squeeze()
            #                 elif p == 'zeta':
            #                     final_nu -= 0.5 * np.mean(filter_ens.hist[istart - 1, N_psi + jj]).squeeze()
            #             axNU.plot([-10, 100], [final_nu, final_nu], '-', color='k', label='Pre-DA')
            #             true_nu = 0.5*(truth['true_params']['beta']-truth['true_params']['zeta'])
            #             axNU.plot([-10, 100], [true_nu, true_nu], '-', color='k', label='Truth', alpha=0.2, linewidth=5.)


            if flag:
                # compute and plot the baseline correlation and MSE
                y_truth_u = y_truth - b_truth
                Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
                ax[0].plot((-10, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                ax[1].plot((-10, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
                # compute C and R before the assimilation (the initial ensemble has some initialisation error)
                Cpre, Rpre = CR(y_truth[istart - N_CR:istart+1:], np.mean(y_filter, -1)[istart - N_CR:istart+1:])
                ax[0].plot((-10, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
                ax[1].plot((-10, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
                # if si == 0:
                #     for ax1 in ax[1:]:
                #         ax1.legend(bbox_to_anchor=(0., 1.), loc="upper left", ncol=4)



            flag = False

        # =========================================================================================================
        xlims = [-1, 70] #[min(ks) - 0.2, max(ks) + 0.2]
        ax[0].set(ylabel='Correlation', xlim=xlims, xlabel='$\\gamma$')
        ax[1].set(ylabel='RMS error', ylim=[0., 0.4], xlim=xlims, xlabel='$\\gamma$')
        # ax[2].set(ylim=[0.6, 1.6], xlim=xlims)

        for ax1 in ax:
            x0, x1 = ax1.get_xlim()
            y0, y1 = ax1.get_ylim()
            ax1.set_aspect((x1 - x0) / (y1 - y0))

plt.savefig(parent_folder + 'results.svg', dpi=350)

plt.show()

