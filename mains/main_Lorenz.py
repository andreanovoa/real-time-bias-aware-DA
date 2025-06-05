from essentials.models_physical import Lorenz63
from essentials.bias_models import *
from essentials.run import main
from essentials.create import create_truth, create_ensemble, create_bias_model
from essentials.plotResults import *

folder = 'results/Lorenz/'

os.makedirs(folder, exist_ok=True)

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #

from default_parameters.lorenz63 import *


def model_bias(yy, tt):
    return 0 * yy, 'nobias'
    # return .3 * np.max(yy, axis=0) + .3 * yy, 'linear'  # Linear bias
    # return .3 * np.max(yy, axis=0) * (np.cos((yy / np.max(yy, axis=0))**2) + 1), 'nonlinear+offset'  # Non-linear bias


true_params = {'model': Lorenz63,
               't_max': 80. * t_lyap,
               'std_obs': 0.1,
               'manual_bias': model_bias,
               'rho': 28.,
               'sigma': 10.,
               'beta': 8. / 3.,
               }

# ==================================== SELECT FILTER PARAMETERS =================================== #
# params_IC = dict(rho=(27.5, 32),
#                  sigma=(9.5, 12.5))


if __name__ == '__main__':
    # ================================================================================== #
    ensemble = create_ensemble(forecast_params, **filter_params)
    truth = create_truth(**true_params, **filter_params)

    bias_name = 'ESN_{}_L{}'.format(truth['name_bias'], bias_params['L'])
    bias, wash_obs, wash_t = create_bias_model(ensemble, truth, bias_params,
                                               bias_filename=bias_name, folder=folder)

    # ================================================================================== #

    filter_ens = ensemble.copy()
    filter_ens.bias = bias.copy()

    # ================================================================================== #

    filter_ens = main(filter_ens, truth['y_obs'], truth['t_obs'],
                      std_obs=0.01, wash_obs=wash_obs, wash_t=wash_t)

    plot_timeseries(filter_ens=filter_ens, truth=truth,
                    reference_t=filter_ens.t_lyap, plot_ensemble_members=True)

    plt.show()