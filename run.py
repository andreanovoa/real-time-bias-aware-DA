import numpy as np
import os as os

from Util import save_to_pickle_file
from DA import dataAssimilation
from create import create_bias_model


rng = np.random.default_rng(6)
path_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'


def main(filter_ens, truth):
    observations = dict()
    for key in ['y_obs', 't_obs', 'std_obs']:
        observations[key] = truth[key]
    if 'wash_obs' in truth.keys():
        for key in ['wash_obs', 'wash_t']:
            observations[key] = truth[key]

    # =========================  PERFORM DATA ASSIMILATION ========================== #

    filter_ens = dataAssimilation(filter_ens, **observations)
    filter_ens.is_not_physical(print_=True)

    # Integrate further without assimilation as ensemble mean (if truth very long, integrate only .2s more)
    if hasattr(filter_ens, 't_max'):
        Nt_extra = int(min((filter_ens.t_max - filter_ens.hist_t[-1]), filter_ens.t_CR) / filter_ens.dt) + 1
    else:
        Nt_extra = int(filter_ens.t_CR / filter_ens.dt) + 1

    psi, t = filter_ens.timeIntegrate(Nt_extra, averaged=True)
    filter_ens.updateHistory(psi, t)
    if filter_ens.bias is not None:
        y = filter_ens.getObservableHist(Nt_extra)
        b, t_b = filter_ens.bias.timeIntegrate(t=t, y=y)
        filter_ens.bias.updateHistory(b, t_b)

    # =============================== CLOSE SIMULATION  =============================== #
    filter_ens.close()

    return filter_ens


# ------------------------------------------------------------------------------------------------------------------- #
#
#

def run_Lk_loop(ensemble, truth, bias_params, Ls=10, ks=1., folder=''):
    truth = truth.copy()
    bias_params = bias_params.copy()
    if type(Ls) is int:
        Ls = [Ls]
    if type(ks) is float:
        ks = [ks]

    for L in Ls:  # LOOP OVER Ls
        blank_ens = ensemble.copy()
        # Reset ESN
        bias_params['L'] = L
        bias_name = 'ESN_L{}'.format(bias_params['L'])
        create_bias_model(blank_ens, truth, bias_params, bias_name,
                          bias_model_folder=folder, plot_train_data=False)
        results_folder = folder + 'L{}/'.format(L)
        for k in ks:
            filter_ens = blank_ens.copy()
            filter_ens.regularization_factor = k  # Reset regularization value
            # ------------------ RUN & SAVE SIMULATION  -------------------
            filter_ens = main(filter_ens, truth)
            save_simulation(filter_ens, truth, results_dir=results_folder)


#

# ------------------------------------------------------------------------------------------------------------------- #

def save_simulation(filter_ens, truth, extra_parameters=None, results_dir="results/"):
    os.makedirs(results_dir, exist_ok=True)

    parameters = dict(biasType=filter_ens.biasType, forecast_model=filter_ens.name,
                      true_model=truth['model'], num_DA=len(truth['t_obs']))

    if extra_parameters is not None:
        for key, val in extra_parameters.items():
            parameters[key] = val
    # =============================== SAVE SIMULATION  =============================== #
    filename = '{}{}-{}_F-{}'.format(results_dir, filter_ens.filter, truth['name'], filter_ens.name)
    if hasattr(filter_ens, 'regularization_factor'):
        filename += '_k{}'.format(filter_ens.regularization_factor)
    # save simulation
    save_to_pickle_file(filename, parameters, truth, filter_ens)
