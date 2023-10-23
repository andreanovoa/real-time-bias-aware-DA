from physical_models import Lorenz63
from bias_models import *
from run import main
from create import create_truth, create_ensemble, create_bias_model
from plot_functions.plotResults import *

folder = 'results/Lorenz/'
figs_dir = folder + 'figs/'

os.makedirs(figs_dir, exist_ok=True)


# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #


def model_bias(yy, tt):
    return 0 * yy, 'nobias'
    # return .3 * np.max(yy, axis=0) + .3 * yy, 'linear'   # Linear bias
    # return .3 * np.max(yy, axis=0) * (np.cos((yy / np.max(yy, axis=0))**2) + 1), 'nonlinear+offset'  # Non-linear bias


true_params = {'model': Lorenz63,
               't_max': 100.,
               'std_obs': 0.01,
               'manual_bias': model_bias,
               'rho': 28.,
               'sigma': 10.,
               'beta': 8. / 3.,
               }

forecast_params = {'model': Lorenz63,
                   'rho': 28.,
                   'sigma': 10.,
                   'beta': 8. / 3.,
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
params_IC = dict(rho=(27.5, 32), sigma=(9.5, 12.5))

filter_params = {'filter': 'EnKF',
                 'constrained_filter': False,
                 'regularization_factor': 0.,
                 'm': 10,
                 'est_a': [*params_IC],
                 'std_psi': 0.1,
                 'std_a': params_IC,
                 'alpha_distr': 'uniform',
                 # Define the observation time-window
                 't_start': 5.,
                 't_stop': 30.,
                 'dt_obs': 25,
                 # Inflation
                 'inflation': 1.0,
                 }

bias_params = {'biasType': ESN,
               'N_wash': 30,
               'upsample': 3,
               'N_units': 600,
               'est_a': filter_params['est_a'],
               'ensure_mean': True,
               'connect': 3,
               # 't_train': Lorenz63.t_transient * 2,
               # 't_val': Lorenz63.t_CR * 2,
               'L': 10,
               'rho_range': [0.5, 1.1],
               'sigma_in_range': [np.log10(1e-5), np.log10(.5)],
               'tikh_range': np.array([1e-6, 1e-9, 1e-12]),
               'augment_data': True,
               }

if __name__ == '__main__':
    # ================================================================================== #

    ensemble = create_ensemble(forecast_params, filter_params)
    truth = create_truth(true_params, filter_params)

    bias_name = 'ESN_{}_L{}'.format(truth['name_bias'], bias_params['L'])
    create_bias_model(ensemble, truth, bias_params, bias_name, bias_model_folder=folder, plot_train_data=True)

    # ================================================================================== #

    out = main(ensemble, truth)
    post_process_single(ensemble, truth, reference_p=Lorenz63.defaults, reference_t=Lorenz63.t_lyap)

    plt.show()
