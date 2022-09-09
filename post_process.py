import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import splev, splrep
from Ensemble import createEnsemble

from scipy.io import savemat
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

name = 'results/2022-09-06_Rijke_Wave_Rijke_sqrt'
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


# %% ================================ PLOT time series, parameters and RMS ================================ #


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

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

fig, ax = plt.subplots(3, 2, figsize=[20, 12])
p_ax = ax[0, 0]
zoom_ax = ax[0, 1]
params_ax = ax[1, 0]
RMS_ax = ax[1, 1]
bias_ax = ax[2, 0]
J_ax = ax[2, 1]

print('b', filter_ens.bias.b)
print('r', filter_ens.bias.r)
if biasType is None:
    fig.suptitle(filter_ens.name + ' DA with ' + filt)
else:
    fig.suptitle(filter_ens.name + ' DA with ' + filt + ' and ' + biasType.name + ' bias')

x_lims = [t_obs[0] - .05, t_obs[num_DA] + .05]

p_ax.plot(t_true, y_true[:, 0], color='silver', label='Truth', linewidth=4)
zoom_ax.plot(t_true, y_true[:, 0], color='silver', label='Truth', linewidth=6)
p_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
p_ax.plot((t_obs[num_DA], t_obs[num_DA]), (-1E6, 1E6), '--', color='dimgray')
if filter_ens.bias is not None:
    c = 'black'
    b = filter_ens.bias.hist
    t_b = filter_ens.bias.hist_t

    b_up = np.array([splev(t, splrep(t_b, b[:, i])) for i in range(filter_ens.bias.N_dim)]).transpose()
    b_up = np.expand_dims(b_up, -1)

    y_unbiased = y_filter + b_up
    y_mean_u = np.mean(y_unbiased, -1)

    t_wash = filter_ens.bias.washout_t
    wash = filter_ens.bias.washout_obs

    p_ax.plot(t_wash, wash[:, 0], '.', color='r')

    p_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=.5)
    zoom_ax.plot(t, y_mean_u[:, 0], '-', color=c, label='Unbiased filtered signal', linewidth=1.)


    # BIAS PLOT
    bias_ax.plot(t, b_up[:, 0], alpha=0.75, label='ESN estimation')
    b_obs = y_true[:len(y_filter)] - y_filter

    b_mean = np.mean(b_obs, -1)
    bias_ax.plot(t, b_mean[:, 0], '--', color='darkorchid', label='Observable bias')
    # std = np.std(b[:, 0, :], axis=1)
    bias_ax.fill_between(t, b_mean[:, 0] + std, b_mean[:, 0] - std, alpha=0.5, color='darkorchid')

    y_lims = [min(b_mean[:, 0]) - np.mean(std), (max(b_mean[:, 0]) + max(std))]

    bias_ax.legend()
    bias_ax.set(ylabel='Bias', xlabel='$t$', xlim=x_lims, ylim=y_lims)

c = 'royalblue'
p_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=1.)
zoom_ax.plot(t, y_mean[:, 0], '--', color=c, label='Filtered signal', linewidth=2.)
p_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
zoom_ax.fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.2, color=c)
p_ax.plot(t_obs[:num_DA], obs[:num_DA, 0], '.', color='r', label='Observations data')
zoom_ax.plot(t_obs[:num_DA], obs[:num_DA, 0], '.', color='r', label='Assimilation step', markersize=10)

y_lims = [min(y_mean[:, 0]) - np.mean(std)*1.1, (max(y_mean[:, 0]) + max(std)) * 1.5]
p_ax.set(ylabel="$p'_\mathrm{mic_1}$ [Pa]", xlabel='$t$ [s]', xlim=x_lims, ylim=y_lims)
p_ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

y_lims = [min(y_true[:, 0])*1.1, max(y_true[:, 0])* 1.1]

zoom_ax.set(xlim=[t_obs[num_DA]-0.05, t_obs[num_DA]], ylim=y_lims)

zoom_ax.tick_params(labelsize = 24)
# zoom_ax.tick_params('Y axis', fontsize = 20)

# PLOT PARAMETER CONVERGENCE
ii = len(filter_ens.psi0)
c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
params_ax.plot((t_obs[0], t_obs[0]), (-1E6, 1E6), '--', color='dimgray')
params_ax.plot((t_obs[num_DA], t_obs[num_DA]), (-1E6, 1E6), '--', color='dimgray')

if filter_ens.est_p:
    for p in filter_ens.est_p:
        if filter_ens.bias is None:
            reference_p = filter_ens.alpha0[p]
            superscript = '^\mathrm{true}$'
        else:
            reference_p = filter_ens.bias.train_TAparams[p]
            superscript = '^\mathrm{train}$'

        mean_p = mean[:, ii].squeeze() / reference_p
        std = np.std(hist[:, ii] / reference_p, axis=1)

        params_ax.plot(t, mean_p, color=c[ii - len(filter_ens.psi0)], label='$\\' + p + '/\\' + p + superscript)

        params_ax.set(xlabel='$t$', xlim=x_lims)
        params_ax.fill_between(t, mean_p + std, mean_p - std, alpha=0.2, color=c[ii - len(filter_ens.psi0)])
        ii += 1
    params_ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
    params_ax.plot(t[1:], t[1:] / t[1:], '-', color='k', linewidth=.5)
params_ax.set(ylim=[0.2, 1.5])
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
    J_ax.plot(t_obs[:num_DA], J, label=['$\\mathcal{J}_{\\psi}$', '$\\mathcal{J}_{d}$', '$\\mathcal{J}_{b}$'])
    # ax[1, 1].plot(t_obs[:num_DA], np.sum(J, -1), label='$\\mathcal{J}$')

    J_ax.set(ylabel='Cost function $\mathcal{J}$', xlabel='$t$', xlim=x_lims, yscale='log')
    J_ax.legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

plt.tight_layout()

plt.draw()
#### Save Results for matlab postprocessing

dt = truth['t'][1]-truth['t'][0]
start_idx = int((t_obs[num_DA] - 0.2) // dt)
end_idx = min(len(y_mean), int((t_obs[num_DA] + 0.2) // dt))

plt.figure()
plt.plot(t_true[start_idx:end_idx], y_true[start_idx:end_idx, 0], color='darkgray', linewidth=5)
plt.plot(t[start_idx:end_idx], y_mean[start_idx:end_idx, 0], color='royalblue', linewidth=1.5)
try:
    plt.plot(t[start_idx:end_idx], y_mean_u[start_idx:end_idx, 0], color='k', linewidth=0.5)
except: pass

plt.draw()
with open(name+'.mat', 'wb') as f:  # need 'wb' in Python3
    savemat(f, {"p_true": y_true[start_idx:end_idx, 0].transpose()})
    savemat(f, {"p_bias": y_mean[start_idx:end_idx, 0]})
    try:
        savemat(f, {"p_unbias": y_mean_u[start_idx:end_idx, 0]})
    except:
        savemat(f, {"p_unbias": False})
    savemat(f, {"dt": dt})
    savemat(f, {'t': t[start_idx:end_idx]})

with open('ESN_data.mat', 'wb') as f:
    savemat(f, {"r": filter_ens.bias.r.transpose()})
    savemat(f, {"b": filter_ens.bias.b.transpose()})
    savemat(f, {'Win': filter_ens.bias.Win[0]})
    savemat(f, {'Wout': filter_ens.bias.Wout[0]})
    savemat(f, {'W': filter_ens.bias.W[0]})
    savemat(f, {'norm': filter_ens.bias.norm.transpose()})
    savemat(f, {'sigma_in': filter_ens.bias.sigma_in})
    savemat(f, {'rho': filter_ens.bias.rho})

# plt.show()
# ==================== UNCOMMENT TO SAVE FIGURES ============================== #
# folder = os.getcwd() + "/figs/" + str(date.today()) + "/"
# os.makedirs(folder, exist_ok=True)

# plt.savefig(folder
#             + filt + '_estB' + str(filter_ens.est_b)
#             + "_b1" + str(b1) + "_b2" + str(b2)
#             + "_PE" + str(len(filter_ens.est_p))
#             + "_m" + str(filter_ens.m)
#             + "_kmeas" + str(kmeas) + ".pdf")
