import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/Rijke_betaC1/'


# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': 'wave',
               't_max': 5.,
               'std_obs': 0.05
               }

forecast_params = {'model': TAModels.Rijke,
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 100,
                 'est_p': ['beta', 'C1'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 'std_a': 0.2,
                 'std_psi': 0.2,
                 # Define the observation time-window
                 't_start': 2.0,  # ensure SS
                 't_stop': 4.5,
                 'kmeas': 25,
                 # Inflation
                 'inflation': 1.002
                 }

if filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': TAModels.Rijke,
                    'std_psi': 0.8,
                    'std_a': 0.8,
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
results_folder = folder + 'L{}_m{}/'.format(bias_params['L'], filter_params['m'])
figs_folder = results_folder + 'figs/'


if __name__ == '__main__':
    ensemble, truth, b_args = createEnsemble(true_params, forecast_params,
                                             filter_params, bias_params,
                                             folder=results_folder, folderESN=folder)

    # ================================================================================== #
    # ================================================================================== #

    blank_ens = ensemble.copy()
    ks = np.linspace(0, 5, 11)
    for k in ks:  # Reset gamma value
        filter_ens = blank_ens.copy()
        filter_ens.bias.k = k
        out = main(filter_ens, truth, filter_params, results_folder=results_folder, figs_folder=figs_folder, save_=True)

        filename = '{}L{}_std{}_k{}'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a, k)
        # filename = None
        # post_process_single_SE_Zooms(*out[:2], filename=filename)
        post_process_single(*out, filename=filename)
    #
    filename = '{}CR_L{}_std{}_results'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a)
    post_process_multiple(results_folder, filename)
    # plt.close('all')

    # plt.show()
