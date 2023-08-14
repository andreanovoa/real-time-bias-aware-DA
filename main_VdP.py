
if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main, create_ESN_train_dataset, create_ensemble
    from plotResults import *
    import os as os

    path_dir = os.path.realpath(__file__).split('main')[0]
    os.chdir('/mscott/an553/')  # set working directory to mscott
    folder = 'results/VdP_final/'

    whyAugment_folder = folder + 'results_whyAugment/'
    loopParams_folder = folder + 'results_loopParams/'
    figs_dir = path_dir + '/paper_figs/'

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    run_whyAugment, run_loopParams = 0, 0
    plot_whyAugment, plot_loopParams = 0, 1


    if run_whyAugment or run_loopParams:

        whyAug_params = [(1, False), (1, True), (10, True), (50, True)]
        loop_Ls = [1, 10, 50, 100]
        loop_stds = [.25]  # [.1, .25]

        whyAug_ks = [0., 10., 20.]
        loop_ks = np.linspace(0., 20., 21)
        # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
        true_params = {'model': physical_models.VdP,
                       'manual_bias': 'cosine',
                       'law': 'tan',
                       'beta': 75.,  # forcing
                       'zeta': 55.,  # damping
                       'kappa': 3.4,  # nonlinearity
                       'std_obs': 0.01,
                       }

        forecast_params = {'model': physical_models.VdP
                           }

        # ==================================== SELECT FILTER PARAMETERS =================================== #
        filter_params = {'filt': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                         'm': 10,
                         'est_p': ['beta', 'zeta', 'kappa'],
                         'biasType': bias_models.ESN,  # Bias.ESN  # None
                         # Define the observation time window
                         't_start': 2.0,
                         't_stop': 3.0,
                         'dt_obs': 30,
                         # Inflation
                         'inflation': 1.002,
                         'start_ensemble_forecast': 2,
                         'regularization_factor': 0.
                         }

        if filter_params['biasType'].name == 'ESN':
            # using default TA parameters for ESN training
            train_params = {'model': physical_models.VdP,
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

        name = 'reference_Ensemble_m{}_dt_obs{}'.format(filter_params['m'], filter_params['dt_obs'])

        # ======================= CREATE REFERENCE ENSEMBLE =================================
        ensemble, truth, esn_args = create_ensemble(true_params, forecast_params, filter_params, bias_params)

    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #

    if run_whyAugment:
        # Add standard deviation to the state
        blank_ens = ensemble.copy()
        std = 0.25

        blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std, blank_ens.m, method='normal')
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
                filter_params['Bdict'] = create_ESN_train_dataset(*esn_args, bias_param=bias_params)  # reset bias
                filter_ens.initBias(filter_params['Bdict'])
                filter_ens.regularization_factor = k
                # ======================= RUN DATA ASSIMILATION  =================================
                name = whyAugment_folder + '{}_L{}_Augment{}/'.format(order, L, augment)
                main(filter_ens, truth, results_dir=name, save_=True)
    # ------------------------------------------------------------------------------------------------ #
    if plot_whyAugment:
        if not os.path.isdir(whyAugment_folder):
            raise ValueError('results_whyAugment not run')
        else:
            plot_ks = (0., 10., 20.)
            post_process_WhyAugment(whyAugment_folder, k_plot=plot_ks,
                                    J_plot=plot_ks, figs_dir=figs_dir)

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
                    filter_params['Bdict'] = create_ESN_train_dataset(*esn_args, bias_param=bias_params)
                    blank_ens.initBias(filter_params['Bdict'])
                results_folder = std_folder + 'L{}/'.format(L)
                for k in loop_ks:
                    filter_ens = blank_ens.copy()
                    # Reset gamma value
                    filter_ens.regularization_factor = k
                    # Run main ---------------------
                    main(filter_ens, truth, results_dir=results_folder, save_=True)
    # ------------------------------------------------------------------------------------------------ #
    if plot_loopParams:
        if not os.path.isdir(loopParams_folder):
            raise ValueError('results_loopParams not run')
        else:
            figs_dir = path_dir + loopParams_folder
            post_process_loopParams(loopParams_folder, k_max=20.,
                                    k_plot=(0., 10., 20.), figs_dir=figs_dir)
