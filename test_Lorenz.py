import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

folder = 'results/Lorenz_twinESN/'


# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.Lorenz63,
               't_max': 500.,
               'std_obs': 0.05,
               'manual_bias': False
               }

forecast_params = {'model': TAModels.Lorenz63,
                   't_max': 30.
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnSRKF',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 100,
                 'est_p': [],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 'std_a': 0.0,
                 'std_psi': 0.1,
                 # Define the observation time-window
                 't_start': 20.,  # ensure SS
                 't_stop': 120.,
                 'kmeas': 10000,
                 # Inflation
                 'inflation': 1.00
                 }

if filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': forecast_params['model'],
                    'std_psi': 0.1,
                    'std_a': 0.5,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform'
                    }

    t_lyap = 0.906**(-1)
    print(t_lyap)
    bias_params = {'N_wash': 50,
                   'upsample': 2,
                   'N_units': 200,
                    'noise_level': 1e-6,
                   't_train': t_lyap * 50,
                   't_val': t_lyap * 3,
                   'connect': 3,
                   'L': 1,
                   'rho_': [0.1, 1.],
                   'sigin_': [np.log10(0.5), np.log10(5.)],
                   'tikh': [1e-6,1e-9,1e-12],
                   'N_fo': 25,
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
    ks = [0., 1., 10.] #np.linspace(0, 10, 11)
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
