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

folder = 'results/VdP_2PE_11_02/'
files = os.listdir(folder)
flag = True
biases, esn_errors, biases_ESN = [], [], []
ks, CBs, RBs, CUs, RUs, Cpres, Rpres = [], [], [], [], [], [], []

# fig, ax = plt.subplots(2, 3, figsize=(20, 8))

# fig = plt.figure(figsize=(20, 10))
fig = plt.figure(figsize=(15, 7.5), layout="constrained")
fig.suptitle('overall')
subfigs = fig.subfigures(2, 1)
ax = subfigs[0].subplots(1, 3)
axC, axR, axP = ax[:]
for file in files:
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
        N_CR = int(.5 / filter_ens.dt)  # Length of interval to compute correlation and RMS
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
    CB, RB = CR(y_truth[istop - N_CR:istop], np.mean(y_filter, -1)[istop - N_CR:istop])  # biased
    CU, RU = CR(y_truth[istop - N_CR:istop], np.mean(y_unbiased, -1)[istop - N_CR:istop])  # unbiased
    # Correlation
    bias_c = 'tab:red'
    unbias_c = 'tab:blue'
    axC.plot(k, CB, 'o', color=bias_c, label='Biased')
    axC.plot(k, CU, '*', color=unbias_c, label='Unbiased')
    # RMS error
    axR.plot(k, RB, 'o', color=bias_c, label='Biased ')
    axR.plot(k, RU, '*', color=unbias_c, label='Unbiased')
    # Parameters ========================================================================================
    if filter_ens.est_p:
        if flag:
            N_psi = len(filter_ens.psi0)
            c = ['tab:orange', 'navy', 'tab:red', 'forestgreen','cyan']
            marker = ['+', 'x']
            time = ['$(t_\mathrm{start})$', '$(t_\mathrm{end})$']
            alphas = [0.2, 1.0]
            superscript = '^\mathrm{true}$'
            try:
                reference_p = truth['true_params']
            except:
                reference_p = {'law': 'tan', 'nu': 7., 'kappa': 3.4, 'omega': 2 * np.pi * 120.}
        for jj, p in enumerate(filter_ens.est_p):
            for kk, idx in enumerate([istart, istop]):
                hist_p = filter_ens.hist[idx - 1, N_psi + jj] / reference_p[p]
                axP.errorbar(k, np.mean(hist_p).squeeze(), yerr=np.std(hist_p), capsize=6, alpha=alphas[kk],
                             fmt=marker[kk], color=c[jj], label='$\\' + p + '/\\' + p + superscript + time[kk])
    if flag:
        # compute and plot the baseline correlation and MSE
        y_truth_u = y_truth - b_truth
        Ct, Rt = CR(y_truth[-N_CR:], y_truth_u[-N_CR:])
        axC.plot((0, 100), (Ct, Ct), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
        axR.plot((0, 100), (Rt, Rt), '-', color='k', label='Truth', alpha=0.2, linewidth=5.)
        # compute C and R before the assimilation (the initial ensemble has some initialisation error)
        Cpre, Rpre = CR(y_truth[istart - N_CR:istart:], np.mean(y_filter, -1)[istart - N_CR:istart:])
        axC.plot((0, 100), (Cpre, Cpre), '-', color='k', label='Pre-DA')
        axR.plot((0, 100), (Rpre, Rpre), '-', color='k', label='Pre-DA')
        for ax1 in [axR, axP]:
            ax1.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

    flag = False
    # PLOT SOME INDIVIDUAL TIME SOLUTIONS ================================================================
    # if k in [0.0, 10.0, 30.0]:
    #     exec(open("post_process.py").read(), {'parameters': parameters,
    #                                           'filter_ens': filter_ens,
    #                                           'truth': truth})

# WAIT - ISTART/ISTOP IS FOR Y_TRUTH? DOESN'T MAKE SENSE

# =========================================================================================================

for ax1 in [axC, axR, axP]:
    ax1.set(xlabel='$\\gamma$', xlim=[min(ks) - 0.1, max(ks) + 0.1])
axC.set(ylabel='Correlation')
axR.set(ylabel='RMS error')
# plt.tight_layout()

# PLOT MEAN ERROR EVOLUTION ================================================================================

ax = subfigs[1].subplots(1, 2)
mean_ax = ax[:]
for mic in [0]:
    scale = np.max(truth['y'][:, mic])  # np.median(true_env)#np.mean(prop['peak_heights'])
    norm = mpl.colors.Normalize(vmin=min(ks), vmax=max(ks))
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    for i, metric in enumerate([biases, esn_errors]):  # , biases_ESN]):
        errors = [b[:, mic] / scale for b in metric]
        for err, k in zip(errors, ks):
            # if k == 0.0:
            #     mean_ax[i].plot(t_interp, err * 100, color='r', label='k = 0')
            #     mean_ax[i].legend()
            # elif k <= kmax:
            mean_ax[i].plot(t_interp, err * 100, color=cmap.to_rgba(k))
        mean_ax[i].set(xlim=[t_filter[istart] - 0.02, t_filter[istop] + 0.05], xlabel='$t$')

    mean_ax[0].set(ylabel='Biased signal error [\%]')
    mean_ax[1].set(ylim=[0, 20], ylabel='Unbiased signal error [\%]')

    for i in range(2):
        x0, x1 = mean_ax[i].get_xlim()
        y0, y1 = mean_ax[i].get_ylim()
        # print( (x1 - x0) / (y1 - y0))
        mean_ax[i].set_aspect(0.5 * (x1 - x0) / (y1 - y0))

clb = fig.colorbar(cmap, ax=mean_ax[1], orientation='vertical', fraction=0.1)
clb.ax.set_title('$\\gamma$')

plt.show()
