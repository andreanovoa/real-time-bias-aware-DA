import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline, interp1d
from Ensemble import createEnsemble

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


if __name__ == '__main__':
    folder = 'results/VdP_twin_1PE_ESN100_cos/'
    name = folder + 'EnKFbias_TruthVdP_tan_beta8.0e+01_zeta6.0e+01_kappa3.4e+00_omega7.5e+02_tmax-5.0_+cosy_ForecastVdP_BiasESN_k0'
    with open(name, 'rb') as f:
        parameters = pickle.load(f)
        createEnsemble(parameters['forecast_model'])

        truth = pickle.load(f)
        filter_ens = pickle.load(f)

filt = parameters['filt']
biasType = parameters['biasType']
num_DA = parameters['num_DA']
Nt_extra = parameters['Nt_extra']

y_true = truth['y']
t_true = truth['t']
t_obs = truth['t_obs']
obs = truth['p_obs']

num_DA_blind = filter_ens.num_DA_blind
num_SE_only = filter_ens.num_SE_only

# %% ================================ PLOT time series, parameters and RMS ================================ #

y_filter, labels = filter_ens.getObservableHist()
if len(np.shape(y_filter)) < 3:
    y_filter = np.expand_dims(y_filter, axis=1)
if len(np.shape(y_true)) < 3:
    y_true = np.expand_dims(y_true, axis=-1)

hist = filter_ens.hist

mean = np.mean(hist, -1, keepdims=True)
y_mean = np.mean(y_filter, -1)
std = np.std(y_filter[:, 0, :], axis=1)

t = filter_ens.hist_t

fig, ax = plt.subplots(3, 3, figsize=[20, 12], layout="tight")
p_ax = ax[0, 0]
zoom_ax = ax[0, 2]
zoom2_ax = ax[0, 1]
params_ax = ax[1, 0]
bias_ax = ax[1, 1]
RMS_ax = ax[2, 0]
J_ax = ax[2, 1]
dJ_ax = ax[2, 2]

if biasType is None:
    fig.suptitle(filter_ens.name + ' DA with ' + filt)
else:
    fig.suptitle(filter_ens.name + ' DA with ' + filt + ' and ' + biasType.name + ' bias')

x_lims = [t_obs[0] - .05, t_obs[-1] + .05]

p_ax.plot(t_true, y_true[:, 0], color='silver', label='Truth', linewidth=4)
zoom_ax.plot(t_true, y_true[:, 0], color='silver', label='Truth', linewidth=8)
zoom2_ax.plot(t_true, y_true[:, 0], color='silver', label='Truth', linewidth=8)
p_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
p_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')

if filter_ens.bias is not None:
    c = 'black'
    b = filter_ens.bias.hist
    t_b = filter_ens.bias.hist_t

    # spline = CubicSpline(t_b, b, extrapolate=False)

    y_unbiased = y_filter[::filter_ens.bias.upsample] + np.expand_dims(b, -1)

    spline = interp1d(t_b, y_unbiased, kind='cubic', axis=0, copy=True, bounds_error=False, fill_value=0)
    y_unbiased = spline(t)

    y_mean_u = np.mean(y_unbiased, -1)

    t_wash = filter_ens.bias.washout_t
    wash = filter_ens.bias.washout_obs

    try:
        p_ax.plot(t_wash, wash[:, 0], '.', color='r')
    except:
        pass

    washidx = int(t_obs[0]/filter_ens.dt) - filter_ens.bias.N_wash * filter_ens.bias.upsample
    p_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.2)
    zoom_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.5)
    zoom2_ax.plot(t[washidx:], y_mean_u[washidx:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.5)

    # BIAS PLOT
    bias_ax.plot(t_b, b[:, 0], alpha=0.75, label='ESN estimation')
    b_obs = y_true[:len(y_filter)] - y_filter

    b_mean = np.mean(b_obs, -1)
    bias_ax.plot(t, b_mean[:, 0], '--', color='darkorchid', label='Observable bias')
    # std = np.std(b[:, 0, :], axis=1)
    bias_ax.fill_between(t, b_mean[:, 0] + std, b_mean[:, 0] - std, alpha=0.5, color='darkorchid')

    y_lims = [min(b_mean[:, 0]) - np.mean(std), (max(b_mean[:, 0]) + max(std))]

    bias_ax.legend()
    bias_ax.set(ylabel='Bias', xlabel='$t$', xlim=x_lims, ylim=y_lims)

c = '#021bf9'
p_ax.plot(t, y_mean[:, 0], '--', color=c, label='Biased filtered signal', linewidth=1.)
zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=1.5, alpha=0.9)
zoom2_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=1.5, alpha=0.9)
p_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
zoom2_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)

p_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Assimilation step')
zoom_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Assimilation step', markersize=10)
zoom2_ax.plot(t_obs, obs[:, 0], '.', color='r', label='Assimilation step', markersize=10)

y_lims = [min(y_mean[:, 0]) - np.mean(std) * 1.1, (max(y_mean[:, 0]) + max(std)) * 1.5]
p_ax.set(ylabel="$p'_\mathrm{mic_1}$ [Pa]", xlabel='$t$ [s]', xlim=x_lims, ylim=y_lims)
p_ax.legend(bbox_to_anchor=(0., 1.), loc="lower left", ncol=2)

y_lims = [min(min(y_true[:, 0]), min(y_mean[:, 0])) * 1.1, max(max(y_true[:, 0]), max(y_mean[:, 0])) * 1.1]

zoom_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[-1] - 0.03, t_obs[-1]+0.02], ylim=y_lims)
zoom2_ax.set(ylabel="$\\eta$", xlabel='$t$ [s]', xlim=[t_obs[0] - 0.03, t_obs[0]+0.02], ylim=y_lims)

# zoom_ax.tick_params(labelsize=24)
# zoom_ax.tick_params('Y axis', fontsize = 20)

# PLOT PARAMETER CONVERGENCE-------------------------------------------------------------
ii = len(filter_ens.psi0)
c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
params_ax.plot((t_obs[-1], t_obs[-1]), (-1E6, 1E6), '--', color='dimgray')

if num_DA_blind > 0:
    p_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue')
    params_ax.plot((t_obs[num_DA_blind], t_obs[num_DA_blind]), (-1E6, 1E6), '-.', color='darkblue', label='Start BE')
if num_SE_only > 0:
    p_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet')
    params_ax.plot((t_obs[num_SE_only], t_obs[num_SE_only]), (-1E6, 1E6), '-.', color='darkviolet', label='Start PE')



if filter_ens.est_p:
    max_p, min_p = -1000, 1000
    for p in filter_ens.est_p:
        superscript = '^\mathrm{init}$'
        # reference_p = truth['true_params']
        reference_p = filter_ens.alpha0

        mean_p = mean[:, ii].squeeze() / reference_p[p]
        std = np.std(hist[:, ii] / reference_p[p], axis=1)

        max_p = max(max_p, max(mean_p))
        min_p = min(min_p, min(mean_p))

        params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$\\' + p + '/\\' + p + superscript)

        params_ax.set(xlabel='$t$', xlim=x_lims)
        params_ax.fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
        ii += 1
    params_ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
    params_ax.plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
    params_ax.set(ylim=[min_p-0.1, max_p+0.1])


# PLOT RMS ERROR
Psi = mean - hist
Psi = Psi[:-Nt_extra]

Cpp = [np.dot(Psi[ti], Psi[ti].T) / (filter_ens.m - 1.) for ti in range(len(Psi))]
RMS = [np.sqrt(np.trace(Cpp[i])) for i in range(len(Cpp))]
RMS_ax.plot(t[:-Nt_extra], RMS, color='firebrick')
RMS_ax.set(ylabel='RMS error', xlabel='$t$', xlim=x_lims, yscale='log')

# PLOT COST FUNCTION
if filter_ens.getJ:
    J = np.array(filter_ens.hist_J)
    J_ax.plot(t_obs, J[:, :-1])
    dJ_ax.plot(t_obs, J[:, -1], color='tab:red')
    # ax[1, 1].plot(t_obs[:num_DA], np.sum(J, -1), label='$\\mathcal{J}$')

    dJ_ax.set(ylabel='$d\\mathcal{J}/d\\psi$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.set(ylabel='$\\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.legend(['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$',
                 '$\\mathcal{J}_{b}$'], bbox_to_anchor=(1., 1.),
                loc="upper left", ncol=1)

# plt.tight_layout()

dt = truth['t'][1] - truth['t'][0]
start_idx = int((t_obs[-1] - 0.2) // dt)
end_idx = min(len(y_mean), int((t_obs[-1] + 0.2) // dt))

# plt.figure()
# plt.plot(t_true[start_idx:end_idx], y_true[start_idx:end_idx, 0], color='darkgray', linewidth=5)
# plt.plot(t[start_idx:end_idx], y_mean[start_idx:end_idx, 0], color='royalblue', linewidth=1.5)
# try:
#     plt.plot(t[start_idx:end_idx], y_mean_u[start_idx:end_idx, 0], color='k', linewidth=0.5)
# except:
#     pass
# plt.tight_layout()

plt.savefig(folder + name + '.svg', dpi=350)
plt.savefig(folder + name + '.pdf', dpi=350)

# if __name__ == '__main__':
# plt.show()


