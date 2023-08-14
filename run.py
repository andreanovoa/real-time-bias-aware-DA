import matplotlib.pyplot as plt
import numpy as np

from Util import *
from DA import dataAssimilation
from scipy import signal

rng = np.random.default_rng(6)
path_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'


def main(filter_ens, truth, results_dir="results/", save_=False):
    os.makedirs(results_dir, exist_ok=True)

    # =========================  PERFORM DATA ASSIMILATION ========================== #
    filter_ens = dataAssimilation(filter_ens, truth['p_obs'], truth['t_obs'], std_obs=truth['std_obs'])
    filter_ens.is_not_physical(print_=True)

    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    Nt_extra = 0
    if filter_ens.hist_t[-1] < truth['t'][-1]:
        Nt_extra = int(min((truth['t'][-1] - filter_ens.hist_t[-1]), filter_ens.t_CR) / filter_ens.dt) + 1
        psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
        filter_ens.updateHistory(psi, t)
        if filter_ens.bias is not None:
            y = filter_ens.getObservableHist(Nt_extra)
            b, t_b = filter_ens.bias.timeIntegrate(t=t, y=y)
            filter_ens.bias.updateHistory(b, t_b)
    filter_ens.close()

    # =============================== SAVE SIMULATION  =============================== #
    parameters = dict(biasType=filter_ens.biasType, forecast_model=filter_ens.name,
                      true_model=truth['model'], num_DA=len(truth['t_obs']), Nt_extra=Nt_extra)
    # filter_ens = filter_ens.getOutputs()
    if save_:
        filename = '{}{}-{}_F-{}'.format(results_dir, filter_ens.filter, truth['name'], filter_ens.name)
        if filter_ens.bias.name == 'ESN':
            filename += '_k{}'.format(filter_ens.bias.k)
        save_to_pickle_file(filename, parameters, truth, filter_ens)

    return filter_ens, truth, parameters


# ------------------------------------------------------------------------------------------------------------------- #
def create_ensemble(true_p, forecast_p, filter_p, bias_p=None):
    # =============================  CREATE OBSERVATIONS ============================== #
    y_true, t_true, name_truth = createObservations(true_p)

    # Add bias ------------------------------------------------------------
    if 'manual_bias' in true_p.keys() and true_p['manual_bias'] is not None:
        if true_p['manual_bias'] == 'time':
            b_true = .4 * y_true * np.sin((np.expand_dims(t_true, -1) * np.pi * 2) ** 2)
        elif true_p['manual_bias'] == 'periodic':
            b_true = 0.2 * np.max(y_true, 0) * np.cos(2 * y_true / np.max(y_true, 0))
        elif true_p['manual_bias'] == 'linear':
            b_true = .1 * np.max(y_true, 0) + .3 * y_true
        elif true_p['manual_bias'] == 'cosine':
            b_true = np.cos(y_true) + 0.3 * np.mean(abs(y_true), axis=0)
        else:
            raise ValueError("Bias type not recognised choose: 'linear', 'periodic', 'time', 'cosine (VdP)")
        y_true += b_true

    dt_t = t_true[1] - t_true[0]
    obs_idx = np.arange(round(filter_p['t_start'] / dt_t),
                        round(filter_p['t_stop'] / dt_t) + 1,
                        filter_p['dt_obs'])
    t_obs = t_true[obs_idx]
    q = np.shape(y_true)[1]
    if 'std_obs' not in true_p.keys():
        true_p['std_obs'] = 0.01

    # Add noise ------------------------------------------------------------
    Cdd = np.eye(q) * true_p['std_obs'] ** 2

    noise = rng.multivariate_normal(np.zeros(q), Cdd, len(obs_idx))
    obs = y_true[obs_idx] * (1. + noise)
    truth = dict(y=y_true, t=t_true, b=b_true, dt=dt_t,
                 t_obs=t_obs, p_obs=obs, dt_obs=t_obs[1] - t_obs[0],
                 true_params=true_p, name=name_truth,
                 model=true_p['model'], std_obs=true_p['std_obs'])

    # Remove transient to save up space
    i_transient = np.argmin(abs(truth['t'] - forecast_p['model'].attr['t_transient']))
    for key in ['y', 't', 'b']:
        truth[key] = truth[key][i_transient:]

    # ===============================  INITIALISE ENSEMBLE  =============================== #
    ensemble = forecast_p['model'](**forecast_p)
    ensemble.initEnsemble(**filter_p)
    ensemble.initBias(**bias_p)

    if ensemble.bias.N_wash > 0:
        # Add washout ----------------------------------------------------------------
        i1 = truth['t_obs'][0] - truth['dt_obs']
        i1 = np.argmin(abs(truth['t'] - i1))
        i0 = i1 - ensemble.bias.N_wash * ensemble.bias.upsample
        if i0 < 0:
            min_Nt = (ensemble.bias.N_wash * ensemble.bias.upsample + truth['dt_obs']) * ensemble.dt
            raise ValueError('increase t_start to > t_wash + dt_obs = {}'.format(min_Nt))
        ensemble.bias.wash_obs = truth['y'][i0:i1 + 1:ensemble.bias.upsample]
        ensemble.bias.wash_time = truth['t'][i0:i1 + 1:ensemble.bias.upsample]

    return ensemble, truth


# ------------------------------------------------------------------------------------------------------------------- #

