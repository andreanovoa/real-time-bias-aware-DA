
if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createESNbias, createEnsemble
    from plotResults import *
    import os as os

    path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott
    folder = 'results/VdP_final/'

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    run_whyAugment, run_loopParams = 0, 1
    plot_whyAugment, plot_loopParams = 0, 1

    whyAugment_folder = folder + 'results_whyAugment_final/'
    whyAug_params = [(1, False), (1, True), (10, True), (50, True)]
    whyAug_ks = [0., 6., 10., 20.]

    loopParams_folder = folder + 'results_loopParams_final/'
    loop_Ls = [1, 10, 50, 100]
    loop_stds = [.25]  # [.1, .25]
    loop_ks = np.linspace(0., 20., 21)


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
    filter_params = {'filt': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'm': 10,
                     'est_p': ['beta', 'zeta', 'kappa'],
                     'biasType': Bias.ESN,  # Bias.ESN  # None
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 3.0,
                     'kmeas': 30,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 4
                     }

    if filter_params['biasType'].name == 'ESN':
        # using default TA parameters for ESN training
        train_params = {'model': TAModels.VdP,
                        'std_a': 0.3,
                        'std_psi': 0.3,
                        'est_p': filter_params['est_p'],
                        'alpha_distr': 'uniform',
                        'ensure_mean': True,
                        }

        bias_params = {'N_wash': 30,
                       'upsample': 5,
                       'L': 1,
                       'augment_data': True,
                       'train_params': train_params,
                       'tikh_': np.array([1e-16]),
                       'sigin_': [np.log10(1e-5), np.log10(1e0)],
                       }
    else:
        bias_params = None

    name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                           filter_params, bias_params,
                                           working_dir=folder, filename=name)

    # ------------------------------------------------------------------------------------------------ #
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
        for L, augment in whyAug_params:
            order += 1
            for ii, k in enumerate(whyAug_ks):
                filter_ens = blank_ens.copy()
                for key, val in zip(['augment_data', 'L', 'k'], [augment, L, k]):
                    bias_params[key] = val
                # Reset ESN
                filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)  # reset bias
                filter_ens.initBias(filter_params['Bdict'])
                filter_ens.bias.k = k
                # ======================= RUN DATA ASSIMILATION  =================================
                name = whyAugment_folder + '{}_L{}_Augment{}/'.format(order, L, augment)
                main(filter_ens, truth, 'rBA_EnKF', results_dir=name, save_=True)
    # ------------------------------------------------------------------------------------------------ #
    if plot_whyAugment:
        if not os.path.isdir(whyAugment_folder):
            raise ValueError('results_whyAugment not run')
        else:
            figs_dir = path_dir + whyAugment_folder
            # post_process_WhyAugment(whyAugment_folder, k_plot=[0., 7., 19.], L_plot=(1,), figs_dir=figs_dir)
            post_process_WhyAugment(whyAugment_folder, k_plot=(0., 6., 10., 20.), L_plot=None, figs_dir=figs_dir)

    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #

    if run_loopParams:
        for std in loop_stds:
            blank_ens = ensemble.copy()
            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std

            std_folder = loopParams_folder + 'std{}/'.format(std)
            for L in loop_Ls:
                # Reset ESN
                if bias_params is not None:
                    bias_params['L'] = L
                    filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
                    blank_ens.initBias(filter_params['Bdict'])
                results_folder = std_folder + 'L{}/'.format(L)
                for k in loop_ks:  # Reset gamma value
                    filter_ens = blank_ens.copy()
                    filter_ens.bias.k = k
                    # Run main ---------------------
                    main(filter_ens, truth, 'rBA_EnKF', results_dir=results_folder, save_=True)
            # get_error_metrics(std_folder)
    # ------------------------------------------------------------------------------------------------ #
    if plot_loopParams:
        if not os.path.isdir(loopParams_folder):
            raise ValueError('results_loopParams not run')
        else:
            figs_dir = path_dir + loopParams_folder
            post_process_loopParams(loopParams_folder, k_max=40., k_plot=(0., 10., 20.), figs_dir=figs_dir)
