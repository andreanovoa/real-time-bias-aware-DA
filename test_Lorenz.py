import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

folder = 'results/Lorenz_twinESN_0125/'

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.Lorenz63,
               't_max': 500.,
               'std_obs': 0.01,
               'manual_bias': False,
               'rho': 20,
               'sigma': 10,
               'beta': 1.8,
               }

forecast_params = {'model': TAModels.Lorenz63,
                   't_max': 500.,
                   'rho': 20,
                   'sigma': 10,
                   'beta': 1.8,
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKF',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 500,
                 'est_p': [],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 'std_psi': 0.2,
                 # Define the observation time-window
                 't_start': 200.,  # ensure SS
                 't_stop': 240.,
                 'kmeas': 500,
                 # Inflation
                 'inflation': 1.00
                 }

if filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': forecast_params['model'],
                    'est_p': filter_params['est_p'],
                    'std_psi': 0.2,
                    'rho': 20,
                    'sigma': 10,
                    'beta': 1.8,
                    }

    t_lyap = 0.906 ** (-1)
    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'N_units': 200,
                   'noise_level': 0.03,
                   't_train': t_lyap * 5,
                   't_val': t_lyap,
                   'connect': 3,
                   'L': 10,
                   'rho_': [0.7, 1.1],
                   'sigin_': [np.log10(0.05), np.log10(5.)],
                   'tikh_': np.array([1e-6, 1e-9, 1e-12]),
                   'N_fo': 4,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================
results_folder = folder #+ 'L{}_m{}/'.format(bias_params['L'], filter_params['m'])
figs_folder = results_folder + 'figs/'

if __name__ == '__main__':
    ensemble, truth, b_args = createEnsemble(true_params, forecast_params,
                                             filter_params, bias_params,
                                             folder=results_folder, folderESN=folder)

    # ================================================================================== #
    # ================================================================================== #

    blank_ens = ensemble.copy()
    ks = [1.]  # np.linspace(0, 10, 11)
    for k in ks:  # Reset gamma value
        filter_ens = blank_ens.copy()

        filter_ens.bias.k = k
        out = main(filter_ens, truth, filter_params,
                   results_folder=results_folder, figs_folder=figs_folder, save_=True)

        # filename = '{}L{}_std{}_k{}'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a, k)
        filename = None
        # post_process_single_SE_Zooms(*out[:2], filename=filename)
        post_process_single(*out, filename=filename)
        plt.show()
    # filename = '{}CR_L{}_std{}_results'.format(figs_folder, filter_ens.bias.L, filter_ens.std_a)
    # post_process_multiple(results_folder, filename)
    # plt.close('all')
