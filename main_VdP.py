if __name__ == '__main__':
    from physical_models import *
    from bias_models import *
    from run import main, save_simulation
    from create import *
    from plotResults import *
    import os as os

    path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott
    folder = 'results/VdP/'

    whyAugment_folder = folder + 'results_whyAugment/'
    loopParams_folder = folder + 'results_loopParams/'

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    run_whyAugment, run_loopParams = 0, 0
    plot_whyAugment, plot_loopParams = 0, 0
    test = 1

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': VdP,
                   'manual_bias': 'cosine',
                   'law': 'tan',
                   'beta': 75.,  # forcing
                   'zeta': 55.,  # damping
                   'kappa': 3.4,  # nonlinearity
                   'std_obs': 0.2,
                   }

    forecast_params = {'model': VdP
                       }

    params_IC = dict(beta=(60, 80),
                     zeta=(50, 60),
                     kappa=(3, 4))

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filter': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'constrained_filter': False,
                     'regularization_factor': 10.,
                     'm': 10,
                     'std_psi': 0.2,
                     'est_a': [*params_IC],
                     'std_a': params_IC,
                     'alpha_distr': 'uniform',
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 3.0,
                     'dt_obs': 30,
                     # Inflation
                     'inflation': 1.001,
                     }

    # using default TA parameters for ESN training
    bias_params = {'biasType': ESN,  # Bias.ESN  # None
                   'std_a': filter_params['std_a'],
                   'est_a': filter_params['est_a'],
                   'N_wash': 30,
                   'upsample': 3,
                   'noise': 0.01,
                   't_train': VdP.t_transient,
                   't_val': VdP.t_CR,
                   'L': 10,
                   'augment_data': True,
                   'tikh_range': [1e-12, 1e-16],
                   'sigma_in_range': [np.log10(1e-6), np.log10(1e0)],
                   'rho_range': [0.5, 1.0]
                   }

    whyAug_params = [(1, False), (1, True), (10, True), (50, True)]
    loop_Ls = [1, 10, 50, 100]
    loop_stds = [.25]

    whyAug_ks = [0., 1., 5., 10., 20.]
    plot_ks = (0., 1., 5., 10., 20.)
    loop_ks = np.linspace(0., 20., 21)

    # ------------------------------------------------------------------------------------------------ #
    if test:
        ensemble = create_ensemble(forecast_params, filter_params)
        truth = create_truth(true_params, filter_params)

        bias_name = 'ESN_{}_L{}'.format(truth['name_bias'], bias_params['L'])
        create_bias_model(ensemble, truth, bias_params, bias_name,
                          bias_model_folder=folder, plot_train_data=False)

        # ================================================================================== #

        out = main(ensemble, truth)
        post_process_single(ensemble, truth, reference_p=true_params)

        plt.show()
    # ------------------------------------------------------------------------------------------------ #

    if run_whyAugment:
        # ======================= CREATE REFERENCE ENSEMBLE =================================
        ensemble = create_ensemble(forecast_params, filter_params)
        truth_og = create_truth(true_params, filter_params)
        order = -1
        for L, augment in whyAug_params:
            for key, val in zip(['augment_data', 'L'], [augment, L]):
                bias_params[key] = val

            blank_ens = ensemble.copy()
            truth = truth_og.copy()
            bias_filename = 'ESN_L{}_augment{}'.format(bias_params['L'], bias_params['augment_data'])
            create_bias_model(blank_ens, truth, bias_params, bias_filename,
                              bias_model_folder=whyAugment_folder, plot_train_data=True)
            order += 1
            for k in whyAug_ks:
                filter_ens = blank_ens.copy()
                filter_ens.regularization_factor = k

                print(filter_ens.est_a)

                # ======================= RUN DATA ASSIMILATION  =================================
                folder_name = whyAugment_folder + '{}_L{}_Augment{}/'.format(order, L, augment)
                filter_ens = main(filter_ens, truth)
                # ======================= SAVE SIMULATION  =================================
                save_simulation(filter_ens, truth, results_dir=folder_name)

    if run_loopParams:
        # ======================= CREATE REFERENCE ENSEMBLE =================================
        ensemble = create_ensemble(forecast_params, filter_params)
        truth_og = create_truth(true_params, filter_params)
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
    if plot_whyAugment:
        if not os.path.isdir(whyAugment_folder):
            raise ValueError('results_whyAugment not run')
        else:
            post_process_WhyAugment(whyAugment_folder, k_plot=plot_ks,
                                    J_plot=plot_ks, figs_dir=whyAugment_folder + 'figs/')

    # ------------------------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------------------------ #
    if plot_loopParams:
        if not os.path.isdir(loopParams_folder):
            raise ValueError('results_loopParams not run')
        else:
            figs_dir = path_dir + loopParams_folder
            post_process_loopParams(loopParams_folder, k_max=20.,
                                    k_plot=(0., 10., 20.), figs_dir=figs_dir)
