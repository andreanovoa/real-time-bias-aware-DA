from src.DA import dataAssimilation
from src.create import *
import numpy as np

rng = np.random.default_rng(6)
path_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'


def main(filter_ens, y_obs, t_obs, std_obs=0.2, wash_obs=None, wash_t=None, max_t=None):

    # =========================  PERFORM DATA ASSIMILATION ========================== #

    filter_ens = dataAssimilation(filter_ens, y_obs, t_obs, std_obs=std_obs,
                                  wash_obs=wash_obs, wash_t=wash_t)

    # =========================  EXTRA FORCAST POST-DA  ========================== #

    filter_ens.is_not_physical(print_=True)
    # If the true signal very long, integrate only one reference time more
    if hasattr(filter_ens, 't_max'):
        Nt_extra = int(min((filter_ens.t_max - filter_ens.hist_t[-1]),
                           filter_ens.t_CR) / filter_ens.dt) + 1
    elif max_t is not None:
        Nt_extra = int((max_t - filter_ens.hist_t[-1]) / filter_ens.dt) + 1
    else:
        Nt_extra = int(filter_ens.t_CR / filter_ens.dt) + 1

    psi, t = filter_ens.time_integrate(Nt_extra)
    filter_ens.update_history(psi, t)
    if filter_ens.bias is not None:
        y = filter_ens.get_observable_hist(Nt_extra)
        b, t_b = filter_ens.bias.time_integrate(t=t, y=y)
        filter_ens.bias.update_history(b, t_b)

    # =============================== CLOSE SIMULATION  =============================== #
    filter_ens.close()

    return filter_ens


# ------------------------------------------------------------------------------------------------------------------- #
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

    parameters = dict(biasType=filter_ens.bias_type, forecast_model=filter_ens.name,
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
