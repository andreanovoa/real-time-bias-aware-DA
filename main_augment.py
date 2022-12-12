import numpy as np
import TAModels
import Bias
from run import main
from plotResults import *

rng = np.random.default_rng(0)

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/VdP_12.06_LargeBeta_avg'
Ls = [1, 10, 50, 100]
stds = [0.01, 0.1, 0.25]
ks = np.linspace(0., 48., 13)

save_ = True

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.VdP,
               'manual_bias': True,
               'law': 'tan',
               'beta': 80.,  # forcing
               'zeta': 60.,  # damping
               'kappa': 3.4,  # nonlinearity
               'omega': 2 * np.pi * 120.
               }

forecast_params = true_params.copy()
forecast_params = {'model': TAModels.VdP,
                   'beta': 70,
                   'zeta': 55,
                   'kappa': 4.2,
                   }
# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 10,  # Dictionary of DA parameters
                 'est_p': ['beta', 'zeta', 'kappa'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 # Define the observation timewindow
                 't_start': 2.0,
                 't_stop': 4.5,
                 'kmeas': 25,
                 # Inflation and optional parameters
                 'inflation': 1.002,
                 'num_DA_blind': 0,  # int(0.0 / t_obs[1] - t_obs[0]),  # num of obs to start accounting for the bias
                 'num_SE_only': 0  # int(0.0 / t_obs[1] - t_obs[0])  # num of obs to start parameter estimation
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    train_params = forecast_params.copy()
    train_params = {'std_a': 0.5,
                    'std_psi': 0.5,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform'  # training reference data created with uniform distributions
                    }
    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'N_units': 200,
                   't_train': 1.0,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None

folder += '_{}PE_{}kmeas/'.format(len(filter_params['est_p']), filter_params['kmeas'])

out = main(true_params, forecast_params, filter_params, bias_params, Ls, stds, ks, folder, save_)[:2]
plotResults(folder, stds, Ls, k_plot=(0, 10, 50))

