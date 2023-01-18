import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/Rijke_mod_Model/'


# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': 'wave',
               't_max': 4.,
               'std_obs': 0.05
               }

forecast_params = {'model': TAModels.Rijke,
                   'beta': 1.,
                   'tau': 2.E-3,
                   'C1': 0.1,
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 10,
                 'est_p': ['beta', 'C1', 'tau'],
                 'biasType': Bias.NoBias,  # Bias.ESN  # None
                 'std_a': 0.1,
                 'std_psi': 0.1,
                 # Define the observation timewindow
                 't_start': 2.0,  # ensure SS
                 't_stop': 3.5,
                 'kmeas': 25,
                 # Inflation
                 'inflation': 1.002
                 }

if filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': TAModels.Rijke,
                    'beta': 1.,
                    'tau': 2.E-3,
                    'C1': 0.1,
                    'std_psi': 0.5,
                    'std_a': 0.5,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform'
                    }
    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'N_units': 100,
                   't_train': 1.0,
                   'L': 10,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================
results_folder = folder #+ 'L{}_m{}/'.format(bias_params['L'], filter_params['m'])
figs_folder = results_folder + 'figs/'

ensemble, truth, b_args = createEnsemble(true_params, forecast_params,
                                         filter_params, bias_params,
                                         folder=results_folder, folderESN=folder)

# ================================================================================== #
# ================================================================================== #

blank_ens = ensemble.copy()
# ks = np.linspace(50, 100, 11)[1:]
# ks = np.linspace(0, 50, 11)
ks = np.linspace(0, 10, 11)
for k in ks:  # Reset gamma value
    filter_ens = blank_ens.copy()
    filter_ens.bias.k = k

    out = main(filter_ens, truth, filter_params, results_folder=results_folder, figs_folder=figs_folder, save_=True)

    # filename = '{}L{}_std{}_k{}'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a, k)
    filename = None
    post_process_single_SE_Zooms(*out[:2], filename=filename + '_time')
    post_process_single(*out, filename=filename)
#
# filename = '{}CR_L{}_std{}_results'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a)
# post_process_multiple(results_folder, filename)
# plt.close('all')
