import TAModels
import Bias
from run import main, createESNbias, createEnsemble
from plotResults import *

# %% ========================== SELECT LOOP PARAMETERS ================================= #
folder = 'results/VdP_final_short_val_0.01/'

run_whyAugment, run_loopParams = 1, 0
plot_whyAugment, plot_loopParams = 1, 0

# %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
true_params = {'model': TAModels.VdP,
               'manual_bias': 'cos_state',
               'std_obs': 0.01,
               't_max': 5.0
               }

forecast_params = {'model': TAModels.VdP,
                   'beta': 77.,  # forcing
                   'zeta': 52.,  # damping
                   'kappa': 4.7,  # nonlinearity
                   }

# ==================================== SELECT FILTER PARAMETERS =================================== #
filter_params = {'filt': 'EnKFbias',
                 'm': 10,
                 'est_p': ['beta', 'zeta', 'kappa'],
                 'biasType': Bias.ESN,  # Bias.ESN  # None
                 # Define the observation timewindow
                 't_start': 2.5,
                 't_stop': 4.5,
                 'kmeas': 25,
                 # Inflation
                 'inflation': 1.002,
                 'start_ensemble_forecast': 2
                 }

if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
    # using default TA parameters for ESN training
    train_params = {'std_a': 0.2,
                    'std_psi': 0.2,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True
                    }

    for key, val in train_params.items():
        train_params[key] = val

    bias_params = {'N_wash': 50,
                   'upsample': 5,
                   'L': 1,
                   't_val': 0.04,
                   'augment_data': True,
                   'train_params': train_params,
                   'tikh_': np.array([1e-16]),
                   'sigin_': [np.log10(1e-5), np.log10(1e0)],
                   }

else:
    bias_params = None

name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

# ================================================================================== #
# ================================================================================== #

if __name__ == '__main__':
    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                           filter_params, bias_params,
                                           working_dir=folder, filename=name)

    if run_whyAugment:
        results_folder = folder + 'results_whyAugment_uniform/'
        flag = True

        # Add standard deviation to the state
        blank_ens = ensemble.copy()
        std = 0.1

        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std,
                                                 blank_ens.m, method='uniform')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi, blank_ens.std_a = std, std

        # print(blank_ens.psi)
        order = -1
        for L, augment in [(1, False), (1, True), (10, True), (30, True)]:
            ks = [0., 1., 5., 10.]
            order += 1
            for ii, k in enumerate(ks):
                filter_ens = blank_ens.copy()

                # Reset ESN
                bias_params['augment_data'] = augment
                bias_params['L'] = L

                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)  # reset bias
                filter_ens.initBias(filter_params['Bdict'])

                filter_ens.bias.k = k
                # ======================= RUN DATA ASSIMILATION  =================================
                name = results_folder + '{}_L{}_Augment{}/'.format(order, L, augment)
                main(filter_ens, truth, filter_params, results_dir=name, save_=True)

    if run_loopParams:
        Ls = [1, 10, 50, 100]
        stds = [.1, 0.25]
        ks = np.linspace(0., 50., 26)

        for std in stds:
            blank_ens = ensemble.copy()
            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                     std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std

            std_folder = folder + 'results_loopParams/std{}/'.format(std)
            for L in Ls:
                # Reset ESN
                bias_params['L'] = L
                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
                blank_ens.initBias(filter_params['Bdict'])

                results_folder = std_folder + 'L{}/'.format(L)
                for k in ks:  # Reset gamma value
                    filter_ens = blank_ens.copy()
                    filter_ens.bias.k = k

                    main(filter_ens, truth, filter_params, results_dir=results_folder, save_=True)

    if plot_whyAugment:
        # ------------------------------------------------------------------------------------------------ #
        results_folder = folder + 'results_whyAugment_uniform/'
        if not os.path.isdir(results_folder):
            print('results_whyAugment not run')
        else:
            post_process_WhyAugment(results_folder)
        # ------------------------------------------------------------------------------------------------ #
    if plot_loopParams:
        results_folder = folder + 'results_loopParams/'
        if not os.path.isdir(results_folder):
            print('results_loopParams not run')
        else:
            my_dirs = os.listdir(results_folder)
            post_process_loopParams(results_folder, k_plot=(0, 10, 50))
