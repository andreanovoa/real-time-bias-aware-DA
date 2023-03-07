
if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createESNbias, createEnsemble
    from plotResults import *

    # %% ========================== SELECT LOOP PARAMETERS ================================= #
    folder = 'results/VdP_final_check/'
    whyAugment_folder = folder + 'results_whyAugment_picked25/'
    loopParams_folder = folder + 'results_loopParams/'

    run_whyAugment, run_loopParams = 1,  1
    plot_whyAugment, plot_loopParams = 1, 1

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': TAModels.VdP,
                   'manual_bias': 'cosine',
                   'law': 'tan',
                   'beta': 75.,  # forcing
                   'zeta': 55.,  # damping
                   'kappa': 3.4,  # nonlinearity
                   'std_obs': 0.01,
                   }

    forecast_params = {'model': TAModels.VdP
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filt': 'EnKFbias',  # 'EnKFbias' 'EnKF' 'EnSRKF'
                     'm': 10,
                     'est_p': ['beta', 'zeta', 'kappa'],
                     'biasType': Bias.ESN,  # Bias.ESN  # None
                     # Define the observation timewindow
                     't_start': 2.0,
                     't_stop': 4.5,
                     'kmeas': 25,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 1
                     }

    if filter_params['biasType'].name == 'ESN':
        # using default TA parameters for ESN training
        train_params = {'model': TAModels.VdP,
                        'std_a': 0.3,
                        'std_psi': 0.3,
                        'est_p': filter_params['est_p'],
                        'alpha_distr': 'uniform',
                        'ensure_mean': True
                        }

        bias_params = {'N_wash': 50,
                       'upsample': 5,
                       'L': 10,
                       'augment_data': True,
                       'train_params': train_params
                       }
    else:
        bias_params = None

    name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

    # ================================================================================== #
    # ================================================================================== #

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                           filter_params, bias_params,
                                           working_dir=folder, filename=name)

    # ------------------------------------------------------------------------------------------------ #
    if run_whyAugment:
        # Add standard deviation to the state
        blank_ens = ensemble.copy()
        std = 0.25
        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std,
                                                 blank_ens.m, method='uniform')
        blank_ens.hist[-1] = blank_ens.psi
        blank_ens.std_psi, blank_ens.std_a = std, std

        order = -1
        for L, augment in [(1, False), (1, True), (10, True), (50, True)]:
            ks = [0., 9.]
            order += 1
            for ii, k in enumerate(ks):
                filter_ens = blank_ens.copy()

                # Reset ESN
                bias_params['augment_data'] = augment
                bias_params['L'] = L
                bias_params['k'] = k

                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)  # reset bias
                filter_ens.initBias(filter_params['Bdict'])

                filter_ens.bias.k = k
                # ======================= RUN DATA ASSIMILATION  =================================
                name = whyAugment_folder + '{}_L{}_Augment{}/'.format(order, L, augment)
                main(filter_ens, truth, filter_params, results_dir=name, save_=True)

    # ------------------------------------------------------------------------------------------------ #
    if run_loopParams:
        Ls = [1, 10, 50, 100]
        stds = [.1, .25]
        ks = np.linspace(0., 40., 41)

        for std in stds:
            blank_ens = ensemble.copy()
            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                     std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std

            std_folder = loopParams_folder + 'std{}/'.format(std)
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
            get_error_metrics(std_folder)
    # ------------------------------------------------------------------------------------------------ #
    if plot_whyAugment:
        if not os.path.isdir(whyAugment_folder):
            print('results_whyAugment not run')
        else:
            post_process_WhyAugment(whyAugment_folder)
    # ------------------------------------------------------------------------------------------------ #
    if plot_loopParams:
        if not os.path.isdir(loopParams_folder):
            print('results_loopParams not run')
        else:
            my_dirs = os.listdir(loopParams_folder)
            post_process_loopParams(loopParams_folder, k_plot=(0., 10., 40.))
