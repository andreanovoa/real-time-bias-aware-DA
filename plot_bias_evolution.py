import os as os
from Ensemble import createEnsemble
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


def getEnvelope(timeseries_x, timeseries_y, rejectCloserThan=0):
    peaks, peak_properties = find_peaks(timeseries_y, distance=200, height=5)

    u_p = interp1d(timeseries_x[peaks], timeseries_y[peaks], bounds_error=False)

    return u_p, peak_properties


folder = 'results/2022-09-22_small-std/'
# folder = 'results/2022-09-23_big-std/'
# folder = 'results/2022-09-23_small-std_no-wash/'
# folder = 'results/2022-09-23_big-std_no-wash/'

if __name__ == '__main__':

    # file = 'results/2022-09-21/EnKFbias_TruthWave_ForecastRijke_BiasESN_k1.4'
    flag = True
    biases = []
    esn_errors = []
    biases_ESN = []
    ks = []

    env_err, env_esn, env_obs = ([], [], [])
    for file in os.listdir(folder):
        k = float(file.split('_k')[-1])
        with open(folder + file, 'rb') as f:
            parameters = pickle.load(f)
            if flag:
                createEnsemble(parameters['forecast_model'])
                flag = False
            truth = pickle.load(f)
            filter_ens = pickle.load(f)

        y_filter, labels = filter_ens.getObservableHist()
        y_truth = truth['y'][:len(y_filter)]

        # Observable bias
        b_obs = y_truth - np.mean(y_filter, -1)

        # ESN bias
        spline = CubicSpline(filter_ens.bias.hist_t, filter_ens.bias.hist, extrapolate=False)
        b_ESN = spline(filter_ens.hist_t)

        # plt.figure()
        # plt.plot(filter_ens.hist_t, b_obs[:, 0])
        # plt.plot(filter_ens.bias.hist_t, b_ESN[:, 0])
        # plt.plot(filter_ens.hist_t, spline(filter_ens.hist_t)[:,0])
        # plt.show()

        kinterval = int(0.03 / filter_ens.dt)
        bias = []
        bias_esn = []
        esn_err = []
        t = filter_ens.hist_t[kinterval::kinterval]
        for j in range(int(len(b_obs) // kinterval)):
            i = j * kinterval
            mean_bias_obs = np.mean(abs(b_obs[i:i + kinterval]), 0)
            mean_bias_esn = np.mean(abs(b_ESN[i:i + kinterval]), 0)
            mean_unbiased_error = np.mean(abs(b_obs[i:i + kinterval] - b_ESN[i:i + kinterval]), 0)

            bias.append(mean_bias_obs)
            bias_esn.append(mean_bias_esn)
            esn_err.append(mean_unbiased_error)

            # errors.append([k, np.array(error)])
        biases.append(np.array(bias))
        biases_ESN.append(np.array(bias_esn))
        esn_errors.append(np.array(esn_err))
        ks.append(k)

        u1, _ = getEnvelope(filter_ens.hist_t, b_ESN[:, 0])
        env_esn.append(u1(filter_ens.hist_t))
        u1, _ = getEnvelope(filter_ens.hist_t, b_obs[:, 0])
        env_obs.append(u1(filter_ens.hist_t))
        u1, _ = getEnvelope(filter_ens.hist_t, abs(b_ESN[:-1, 0] - b_obs[:-1, 0]))
        env_err.append(u1(filter_ens.hist_t))

    t = filter_ens.hist_t[kinterval::kinterval]

    # plt.figure()
    # plt.plot(filter_ens.hist_t, b_ESN[:, 0] ** 2, '-')
    # u1 = getEnvelope(filter_ens.hist_t, b_ESN[:, 0] ** 2)
    # plt.plot(filter_ens.hist_t, u1(filter_ens.hist_t), 'r--')
    # plt.show()

    i0 = np.argmin(abs(t - truth['t_obs'][0]))
    iend = np.argmin(abs(t - truth['t_obs'][parameters['num_DA']]))

    true_env, prop = getEnvelope(truth['t'][-kinterval * 10:], y_truth[-kinterval * 10:, 0])
    scale = np.mean(prop['peak_heights'])

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Mean errors')
    for mic in [0]:
        norm = mpl.colors.Normalize(vmin=min(ks), vmax=max(ks))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        fig.colorbar(cmap, orientation='vertical')

        for i, metric in enumerate([biases, biases_ESN, esn_errors]):
            # for i, metric in enumerate(zip([biases, biases_ESN, esn_errors], [1, 2, 3])):

            # scale = metric[0][i0, mic]
            errors = [b[:, mic] / scale for b in metric]

            for err, k in zip(errors, ks):
                if k == 0.0:
                    ax[i].plot(t, err, color='r', label='k = 0')
                else:
                    ax[i].plot(t, err, color=cmap.to_rgba(k))

            # for i in [0,1]:
            ax[i].plot((truth['t_obs'][0], truth['t_obs'][0]), (1E-10, 1E1), '--', color='dimgray')
            ax[i].plot((t[i0] - 0.05, t[iend] + 0.05), (1, 1), '--', color='dimgray')
            ax[i].plot((t[iend], t[iend]), (1E-10, 1E1), '--', color='dimgray')

            ax[i].legend()
            # ax.set(xlim=[0.9, 1.6], yscale='log')
            ax[i].set(xlim=[t[i0] - 0.02, t[iend] + 0.05], ylim=[0, 3], xlabel='$t$')

        ax[0].set(ylabel='Observable bias')
        ax[1].set(ylabel='ESN prediction')
        ax[2].set(ylabel='||Observable bias - ESN bias prediction||')

    plt.tight_layout()

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Envelopes errors')
    for mic in [0]:
        norm = mpl.colors.Normalize(vmin=min(ks), vmax=max(ks))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        fig.colorbar(cmap, orientation='vertical')

        for i, metric in enumerate([env_obs, env_esn, env_err]):
            # for i, metric in enumerate(zip([biases, biases_ESN, esn_errors], [1, 2, 3])):

            errors = [b / scale for b in metric]

            for err, k in zip(errors, ks):
                if k == 0.0:
                    ax[i].plot(filter_ens.hist_t, err, color='r', label='k = 0')
                else:
                    ax[i].plot(filter_ens.hist_t, err, color=cmap.to_rgba(k))

            # for i in [0,1]:
            ax[i].plot((truth['t_obs'][0], truth['t_obs'][0]), (1E-10, 1E1), '--', color='dimgray')
            ax[i].plot((t[i0] - 0.05, t[iend] + 0.05), (1, 1), '--', color='dimgray')
            ax[i].plot((t[iend], t[iend]), (1E-10, 1E1), '--', color='dimgray')

            ax[i].legend(loc='lower left')
            # ax.set(xlim=[0.9, 1.6], yscale='log')
            ax[i].set(xlim=[t[i0] - 0.02, t[iend] + 0.05], ylim=[0, 10])

        plt.tight_layout()
        plt.show()
