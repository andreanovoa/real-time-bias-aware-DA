import numpy as np
import os as os

from Util import save_to_pickle_file
from DA import dataAssimilation


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
# ------------------------------------------------------------------------------------------------------------------- #
#
#
# def run_Lk_loop(ensemble,  Ls, ks, folder):
#
#     for L in Ls:
#         results_folder = folder + 'L{}/'.format(L)
#         blank_ens = ensemble.copy()
#
#         # Reset ESN -------------------------------------------------------------
#         bias_paramsL = bias_params.copy()
#         bias_paramsL['L'] = L
#         Bdict = create_ESN_train_dataset(*args, bias_param=bias_paramsL)
#         blank_ens.initBias(Bdict)
#
#         for k in ks:  # Reset gamma value
#             filter_ens = blank_ens.copy()
#             filter_ens.bias.k = k
#             # Run simulation------------------------------------------------------------
#             main(filter_ens, truth, method='rBA_EnKF', results_dir=results_folder, save_=True)
#


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

