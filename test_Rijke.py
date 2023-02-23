import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/Rijke_test_m100/'
# folder = 'results/Rijke_test_m10/'

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.Rijke,
               't_max': 2.0,
               'beta': 4.2,
               'tau': 1.4E-3,
               # 'C1': 0.06,
               # 'C2': 0.008,
               'manual_bias': 'cosine'
               }

forecast_params = {'model': TAModels.Rijke,
                   't_max': 2.0
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                 'm': 100,
                 'est_p': ['beta', 'tau'],#, 'C1', 'C2'],
                 'biasType': Bias.ESN,
                 # Define the observation timewindow
                 't_start': 1.5,  # ensure SS
                 't_stop': 1.7,
                 'kmeas': 10,
                 # Inflation
                 'inflation': 1.002,
                 'start_ensemble_forecast': 1
                 }

folder_suffix = 'm{}'.format(filter_params['m'])

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
    folder_suffix += '_ESN{}_L{}/'.format(bias_params['N_units'], bias_params['L'])
else:
    bias_params = None
# ================================== CREATE REFERENCE ENSEMBLE =================================

name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                       filter_params, bias_params,
                                       working_dir=folder, filename=name)

# results_folder = folder + folder_suffix
figs_folder = folder + 'figs/'

run_loopParams = True


if __name__ == '__main__':

    if run_loopParams:

        # Ls = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        Ls = [10, 30, 50, 70, 90]
        stds = [.1, .25]
        ks = np.linspace(0., 10., 21)

        for L in Ls:
            blank_ens = ensemble.copy()
            # Reset ESN
            bias_params['L'] = L
            filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
            blank_ens.initBias(filter_params['Bdict'])
        #
            for std in stds:
                # Reset std
                blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                         std, blank_ens.m, method='normal')
                blank_ens.hist[-1] = blank_ens.psi
                blank_ens.std_psi, blank_ens.std_a = std, std

                results_folder = folder + 'results_loopParams/std{}/L{}/'.format(std, L)
                for k in ks:  # Reset gamma value
                    filter_ens = blank_ens.copy()
                    filter_ens.bias.k = k

                    out = main(filter_ens, truth, filter_params,
                               results_dir=results_folder, figs_dir=figs_folder, save_=True)

                    # if int(k) in (0, 2, 10, 30):
                    #     filename = '{}L{}_std{}_k{}_time'.format(figs_folder, L, std, k)
                    #     post_process_single_SE_Zooms(*out[:2], filename=filename)

                # filename = '{}CR_L{}_std{}_results'.format(figs_folder, L, std)
                # post_process_multiple(results_folder, filename)
                # plt.close('all')

        # get_CR_values(results_folder)

        # fig2(folder + 'results_loopParams/', Ls, stds, figs_folder)

