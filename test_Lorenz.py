import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

folder = 'results/Lorenz_0131_good/'
# psi0 = [-7.72733655, -6.74391408, 22.38752501]  # LC
psi0 = [3.41862453, 2.01878628, 25.73177281]  # CH

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.Lorenz63,
               't_max': 100.,
               'std_obs': 0.15,
               'manual_bias': True,
               'psi0': psi0
               }

forecast_params = {'model': TAModels.Lorenz63,
                   't_max': 100.,
                   'psi0': np.array(psi0)
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 100,
                 'est_p': [],
                 'biasType': Bias.ESN,  # Bias.ESN
                 'std_psi': 0.3,
                 # Define the observation time-window
                 't_start': 2.,
                 't_stop': 60.,
                 'kmeas': 30,
                 # Inflation
                 'inflation': 1.0,
                 # Ensemble forecast
                 'start_ensemble_forecast': .20
                 }

if filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': forecast_params['model'],
                    'est_p': filter_params['est_p'],
                    'psi0': forecast_params['psi0'],
                    'std_psi': 0.001,  # 0.1 LC
                    }
    t_lyap = 0.906 ** (-1)
    bias_params = {'N_wash': 10,
                   'upsample': 5,
                   'N_units': 200,
                   't_train': t_lyap * 10,  # 10 LC
                   't_val': t_lyap * 2,  # 2 LC
                   'connect': 3,
                   'L': 10,
                   'noise_level': 1E-1,
                   'rho_': [0.5, 1.1],
                   'sigin_': [np.log10(1e-5), np.log10(.5)],
                   'tikh_': np.array([1e-6, 1e-9, 1e-12]),
                   # 'N_fo': 10,
                   'augment_data': True,
                   'train_params': train_params
                   }
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================
results_folder = folder + '{}_{}/'.format(filter_params['biasType'].name, filter_params['filt'])  #'L{}_m{}/'.format(bias_params['L'], filter_params['m'])
figs_folder = results_folder + 'figs/'

if __name__ == '__main__':
    ensemble, truth, b_args = createEnsemble(true_params, forecast_params,
                                             filter_params, bias_params,
                                             folder=results_folder, folderESN=folder)

    # ================================================================================== #
    # ================================================================================== #

    blank_ens = ensemble.copy()
    ks =[0., 1., 5.] #np.linspace(0, 5, 11)
    for k in ks:  # Reset gamma value
        filter_ens = blank_ens.copy()

        filter_ens.bias.k = k
        out = main(filter_ens, truth, filter_params,
                   results_folder=results_folder, figs_folder=figs_folder, save_=True)

        filename = '{}time_k{}'.format(figs_folder, k)
        # filename = None
        # post_process_single_SE_Zooms(*out[:2], filename=filename)
        post_process_single(*out, filename=filename)
        plt.close()
    # filename = '{}CR__results'.format(figs_folder)
    post_process_multiple(results_folder, filename)
    # plt.show()


