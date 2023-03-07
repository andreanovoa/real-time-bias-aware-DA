import os
import pickle
import TAModels
import Bias
import numpy as np
from run import main, createESNbias, createEnsemble, get_error_metrics
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/Rijke_test_linear/'
figs_folder = folder + 'figs/'
run_loopParams = True

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.Rijke,
               't_max': 2.0,
               'beta': 4.2,
               'tau': 1.4E-3,
               'manual_bias': 'linear'
               }
forecast_params = {'model': TAModels.Rijke,
                   't_max': 2.0
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 100,
                 'est_p': ['beta', 'tau'], #, 'C1', 'C2'],
                 'biasType': Bias.ESN,
                 # Define the observation timewindow
                 't_start': 1.5,  # ensure SS
                 't_stop': 1.7,
                 'kmeas': 10,
                 # Inflation
                 'inflation': 1.01,
                 'start_ensemble_forecast': 1
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'model': TAModels.Rijke,
                    'std_a': 0.2,
                    'std_psi': 0.2,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True
                    }
    bias_params = {'N_wash': 50,
                   'upsample': 2,
                   'L': 10,
                   'N_units': 500,
                   'augment_data': True,
                   't_val': 0.02,
                   't_train': 0.5,
                   'train_params': train_params,
                   'tikh_': np.array([1e-16]),
                   'sigin_': [np.log10(1e-5), np.log10(1e-2)],
                   }
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================

name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

if __name__ == '__main__':
    ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                           filter_params, bias_params,
                                           working_dir=folder, filename=name)
    if run_loopParams:
        Ls = [10, 20,  30,  40, 50, 60, 70, 80, 90, 100]
        ks = np.linspace(0., 5., 21)
        stds = [.10, .25]
        for std in stds:
            blank_ens = ensemble.copy()
            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                     std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std
            std_folder = folder + 'm{}/results_loopParams/std{}/'.format(blank_ens.m, std)
            for L in Ls:
                # Reset ESN
                bias_params['L'] = L
                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
                blank_ens.initBias(filter_params['Bdict'])

                results_folder = std_folder + 'L{}/'.format(L)
                for k in ks:  # Reset gamma value
                    filter_ens = blank_ens.copy()
                    filter_ens.bias.k = k
                    # Run simulation
                    main(filter_ens, truth, filter_params, results_dir=results_folder, save_=True)
            # Compute the error metrics for all the L-k pairs
            get_error_metrics(std_folder)

        # for ff in os.listdir(results_folder):
        #     with open(results_folder+ff, 'rb') as f:
        #         print(f)
        #         params = pickle.load(f)
        #         truth = pickle.load(f)
        #         filter_dict = pickle.load(f)
        #
        #     # post_process_single_SE_Zooms(filter_dict, truth)
        #     post_process_single(filter_dict, truth, params)

