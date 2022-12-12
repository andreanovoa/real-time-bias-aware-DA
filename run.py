import os
import numpy as np
import pylab as plt
import pickle
import matplotlib as mpl

from Util import createObservations, CR
from DA import dataAssimilation

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

rng = np.random.default_rng(0)


def main(filter_ens, truth, filter_p,
         results_folder="results/", figs_folder='results/figs/', save_=False):
    # Create parent directory
    if not os.path.isdir(figs_folder):
        os.makedirs(figs_folder)
        print('created dir', figs_folder)

    # ===============================  PERFORM DATA ASSIMILATION =============================== #
    filter_ens = dataAssimilation(filter_ens, truth['p_obs'], truth['t_obs'], method=filter_p['filt'])

    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    Nt_extra = 0
    if filter_ens.hist_t[-1] < truth['t'][-1]:
        Nt_extra = int(min((truth['t'][-1] - filter_ens.hist_t[-1]), 0.2) / filter_ens.dt) + 1
        psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
        filter_ens.updateHistory(psi, t)
        if filter_ens.bias is not None:
            y = filter_ens.getObservableHist(Nt_extra)[0]
            b, t_b = filter_ens.bias.timeIntegrate(t=t, y=y)
            filter_ens.bias.updateHistory(b, t_b)
    filter_ens.close()

    # ================================== SAVE DATA  ================================== #
    parameters = dict(biasType=filter_p['biasType'], forecast_model=filter_ens.name,
                      true_model=truth['model'], num_DA=len(truth['t_obs']), Nt_extra=Nt_extra)
    if save_:
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)
            print('created dir', results_folder)

        filename = '{}{}-{}_F-{}_B-{}_k{}'.format(results_folder, filter_p['filt'], truth['name'],
                                                    filter_ens.name, filter_p['biasType'].name, filter_ens.bias.k)
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)
            pickle.dump(truth, f)
            pickle.dump(filter_ens, f)
    return filter_ens, truth, parameters


# ====================================================================================================
def createEnsemble(true_p, forecast_p, filter_p, bias_p, folder="results"):

    if not os.path.isdir(folder + 'figs/'):
        os.makedirs(folder + 'figs/')
        print('created dir', folder + 'figs/')
    # %% =====================================  CREATE OBSERVATIONS ===================================== #
    y_true, t_true, name_truth = createObservations(true_p, t_max=5.)
    print(name_truth)
    if true_p['manual_bias']:
        b_true = np.cos(y_true)
        y_true += b_true
        name_truth += '_+cosy'

    t_start = filter_p['t_start']
    t_stop = filter_p['t_stop']
    dt_true = t_true[1] - t_true[0]

    obs_idx = np.arange(round(t_start / dt_true), round(t_stop / dt_true) + 1, filter_p['kmeas'])
    t_obs = t_true[obs_idx]
    obs = y_true[obs_idx]

    truth = dict(y=y_true, t=t_true, name=name_truth, t_obs=t_obs, p_obs=obs, b_true=b_true,
                 true_params=true_p, model=true_p['model'])
    # %% =====================================  DEFINE BIAS ======================================== #
    forecast_model = forecast_p['model']
    if filter_p['biasType'] is not None:
        if filter_p['biasType'].name == 'ESN':
            b_args = (filter_p, forecast_model, y_true, t_true, t_obs, name_truth, folder)
            filter_p['Bdict'] = createESNbias(*b_args, bias_param=bias_p)
        else:
            raise ValueError('Bias model not defined')
    else:
        b_args = 'None'

    # ===========================================  INITIALISE ENSEMBLE  ========================================== #
    ensemble = forecast_model(forecast_p, filter_p)
    return ensemble, truth, b_args


# ====================================================================================================
def createESNbias(filter_p, model, y_true, t_true, t_obs, name_truth, folder, bias_param=None, L=10):
    bias_p = bias_param.copy()
    train_params = bias_p['train_params'].copy()
    train_params['m'] = L
    # Compute reference bias. Create an ensemble of training data
    ref_ens = model(train_params, train_params)
    name_train = folder + 'Truth_{}_{}'.format(ref_ens.name, ref_ens.law)
    for k, v in ref_ens.getParameters().items():
        name_train += '_{}{}'.format(k, v)
    name_train += '_std{:.2}_m{}_{}'.format(ref_ens.std_a, ref_ens.m, ref_ens.alpha_distr)
    # Load or create refeence ensemble (multi-parameter solution)
    if os.path.isfile(name_train):
        print('Loading Reference solution(s)')
        with open(name_train, 'rb') as f:
            ref_ens = pickle.load(f)
    else:
        print('Creating Reference solution(s)')
        psi, t = ref_ens.timeIntegrate(Nt=len(t_true) - 1)
        ref_ens.updateHistory(psi, t)
        ref_ens.close()
        with open(name_train, 'wb') as f:
            pickle.dump(ref_ens, f)

    y_ref, lbl = ref_ens.getObservableHist()
    biasData = np.expand_dims(y_true, -1) - y_ref  # [Nt x Nmic x Ntrain]
    biasData = np.append(biasData, ref_ens.hist[:, -len(ref_ens.est_p):], axis=1)
    # provide data for washout before first observation
    i1 = int(np.where(t_true == t_obs[0])[0]) - filter_p['kmeas']
    dt_true = t_true[1] - t_true[0]
    i0 = i1 - int(0.1 / dt_true)

    # create bias dictionary
    bias_p['trainData'] = biasData[int(1. / dt_true):]  # remove transient - steady state solution
    bias_p['washout_obs'] = y_true[i0:i1 + 1]
    bias_p['washout_t'] = t_true[i0:i1 + 1]
    bias_p['filename'] = folder + name_truth + '_' + name_train.split('Truth_')[-1] + '_bias'
    bias_p['L'] = L
    # Plot training data -------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    norm = mpl.colors.Normalize(vmin=-5, vmax=y_ref.shape[-1])
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma)
    fig.suptitle('Training data')
    ax[0].plot(t_true, y_true, color='silver', linewidth=6, alpha=.8)
    Nt = int(.2 // dt_true)
    for ii in range(y_ref.shape[-1]):
        C, R = CR(y_true[-Nt:], y_ref[-Nt:, :, ii])
        line = ax[0].plot(t_true, y_ref[:, :, ii], color=cmap.to_rgba(ii))
        ax[1].plot(ii, C, 'o', color=cmap.to_rgba(ii))
        ax[2].plot(ii, R, 'x', color=cmap.to_rgba(ii))
    plt.tight_layout()
    ax[0].legend(['Truth'], bbox_to_anchor=(0., 1.25), loc="upper left")
    ax[0].set(xlabel='$t$', ylabel=lbl, xlim=[t_true[-1] - 0.05, t_true[-1]])
    ax[1].set(xlabel='$l$', ylabel='Correlation')
    ax[2].set(xlabel='$l$', ylabel='RMS error')
    ax[0].plot(t_true, y_true, color='silver', linewidth=6, alpha=.8)
    for ax1, l in zip(ax[:], [.5, 1., 1.]):
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()
        ax1.set_aspect(l * (x1 - x0) / (y1 - y0))
    plt.savefig(folder + 'figs/L{}_training_data.svg'.format(train_params['m']), dpi=350)
    plt.close()

    return bias_p

