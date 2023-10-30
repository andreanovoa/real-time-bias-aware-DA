from physical_models import Rijke
from bias_models import ESN, NoBias
from create import *
from run import *
from plot_functions.plotResults import *
import os as os

if __name__ == '__main__':

    bias_form = 'periodic'  # linear, periodic, time
    run_multiple_ensemble_sizes = False

    run_loopParams, plot_loopParams = 0, 1
    run_optimal, plot_optimal = 0, 0

    # %% ========================== SELECT WORKING PATHS ================================= #
    folder = 'results/new_arch/Rijke_{}/'.format(bias_form)
    path_dir = os.path.realpath(__file__).split('main')[0]
    os.chdir('/mscott/an553/')  # set working directory to mscott

    # %% ======================= SELECT RANGE OF PARAMETERS ============================== #
    ms = [50]
    stds = [.25]
    Ls = np.linspace(10, 100, 10, dtype=int)
    ks = np.linspace(0., 10., 41)
    # Special cases
    if bias_form == 'time':
        ks = np.linspace(0.25, 4.75, 10)
    if run_multiple_ensemble_sizes:
        ms = [10, 50, 80]

    # %% ====================== SELECT TRUE AND FORECAST MODELS ========================== #
    true_params = {'model': Rijke,
                   't_max': 2.5,
                   'beta': 4.2,
                   'tau': 1.4E-3,
                   'manual_bias': bias_form
                   }
    forecast_params = {'model': Rijke,
                       't_max': 2.5
                       }
    # %% ============================ SELECT FILTER PARAMETERS ============================ #
    filter_params = {'filter': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'est_a': ['beta', 'tau'],
                     # Define the observation time window
                     't_start': 1.5,  # ensure SS
                     't_stop': 2.0,
                     'dt_obs': 20,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 1
                     }
    bias_params = {'biasType': ESN,
                   'est_a': filter_params['est_a'],
                   'std_a': 0.3,
                   'N_wash': 50,
                   'upsample': 2,
                   'N_units': 500,
                   'augment_data': True,
                   't_val': 0.02,
                   't_train': 0.5,
                   'rho_range': [0.5, 1.0],
                   'tikh_range': np.array([1e-16]),
                   'sigma_in_range': [np.log10(1e-5), np.log10(1e-2)],
                   }
    if bias_form == 'time':
        bias_params['t_train'] = 1.5
        filter_params['dt_obs'] = 10
    # %% ============================ RUN SIMULATIONS ================================= #

    for m in ms:  #
        loopParams_folder = folder + 'm{}/results_loopParams/'.format(m)
        optimal_folder = folder + 'm{}/results_optimal/'.format(m)
        if run_loopParams:
            filter_params['m'] = m
            truth_og = create_truth(true_params, filter_params)
            for std in stds:  # LOOP OVER STDs
                filter_params['std_psi'] = std
                filter_params['std_a'] = std
                # --------------------- CREATE REFERENCE ENSEMBLE ------------------
                ensemble = create_ensemble(forecast_params, filter_params)
                std_folder = loopParams_folder + 'std{}/'.format(std)
                for L in Ls:  # LOOP OVER Ls
                    blank_ens = ensemble.copy()
                    truth = truth_og.copy()
                    # Reset ESN
                    bias_params['L'] = L
                    bias_name = 'ESN_L{}'.format(bias_params['L'])
                    create_bias_model(blank_ens, truth, bias_params, bias_name,
                                      bias_model_folder=std_folder, plot_train_data=False)
                    results_folder = std_folder + 'L{}/'.format(L)
                    for k in ks:
                        filter_ens = blank_ens.copy()
                        filter_ens.regularization_factor = k  # Reset gamma value
                        # ------------------ RUN & SAVE SIMULATION  -------------------
                        filter_ens = main(filter_ens, truth)
                        save_simulation(filter_ens, truth, results_dir=results_folder)
        if plot_loopParams:
            if not os.path.isdir(loopParams_folder):
                raise ValueError('results_loopParams not run')
            figs_dir = path_dir + loopParams_folder
            post_process_loopParams(loopParams_folder, k_plot=(None,), figs_dir=figs_dir)

        # -------------------------------------------------------------------------------------------------------------
        if run_optimal:
            filter_params['m'] = m
            truth_og = create_truth(true_params, filter_params)
            std = 0.25
            filter_params['std_psi'] = std
            filter_params['std_a'] = std

            # --------------------- CREATE REFERENCE ENSEMBLE ------------------
            ensemble = create_ensemble(forecast_params, filter_params)
            std_folder = loopParams_folder + 'std{}/'.format(std)

            # These are manually defined after running the loops
            if bias_form == 'linear':
                L, k = 100, 1.75
            elif bias_form == 'periodic':
                L, k = 60, 2.75
            elif bias_form == 'time':
                L, k = 10,  1.25
            else:
                raise ValueError("Select 'linear', 'periodic' or 'time' bias form")

            # ------------------ RUN & SAVE SIMULATION  -------------------
            filter_ens = main(filter_ens, truth)
            save_simulation(filter_ens, truth, results_dir=optimal_folder)


            blank_ens = ensemble.copy()
            truth = truth_og.copy()
            # Reset ESN
            bias_params['L'] = L
            bias_name = 'ESN_L{}'.format(bias_params['L'])
            create_bias_model(blank_ens, truth, bias_params, bias_name,
                              bias_model_folder=std_folder, plot_train_data=False)
            results_folder = std_folder + 'L{}/'.format(L)
            for k in ks:
                filter_ens = blank_ens.copy()
                filter_ens.regularization_factor = k  # Reset gamma value
                # ------------------ RUN & SAVE SIMULATION  -------------------
                filter_ens = main(filter_ens, truth)
                save_simulation(filter_ens, truth, results_dir=optimal_folder)

            # Run reference solution with bias-blind EnKF -----------------------------
            filter_ens = blank_ens.copy()

        # -------------------------------------------------------------------------------------------------------------
        if plot_optimal:
            if not os.path.isdir(optimal_folder):
                raise ValueError('results_loopParams not run')
            figs_dir = path_dir + optimal_folder
            plot_Rijke_animation(optimal_folder, figs_dir)
