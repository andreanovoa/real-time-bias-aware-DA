if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main, create_ensemble
    from plotResults import *
    from Util import save_to_pickle_file, load_from_pickle_file, create_bias_training_dataset, check_valid_file

    folder = 'results/test_ESN/'
    os.makedirs(folder, exist_ok=True)

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    model = physical_models.VdP
    true_params = {'model': model,
                   'manual_bias': 'cosine',
                   'std_obs': 0.2,
                   'beta': 50,
                   'zeta': 25,
                   'kappa': 9.0
                   }
    forecast_params = {'model': model,
                       'beta': 50,
                       'zeta': 25,
                       'kappa': 9.0
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(beta=(30, 60), zeta=(20, 30), kappa=(8., 10.))

    filter_params = {'filter': 'rBA_EnKF',
                     'constrained_filter': False,
                     'regularization_factor': 20.,
                     'k': 1.0,
                     'm': 10,
                     'est_a': ('beta', 'zeta', 'kappa'),
                     'std_a': parameters_IC,
                     'std_psi': 0.8,
                     'alpha_distr': 'uniform',
                     # # initial parameter and state uncertainty
                     'biasType': bias_models.ESN,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 3.0,
                     't_stop': 4.5,
                     'dt_obs': 30,
                     # Inflation
                     'inflation': 1.001,
                     'reject_inflation': 1.001,
                     'start_ensemble_forecast': 5
                     }
    # ==================================== SELECT ESN PARAMETERS =================================== #

    bias_params = {'L': 100,  # ESN training specs
                   'std_a': parameters_IC,
                   'std_psi': 0.3,
                   'est_a': filter_params['est_a'],
                   'alpha_distr': 'uniform',
                   'ensure_mean': True,
                   'N_wash': 30,
                   'upsample': 2,
                   'augment_data': True,
                   'tikh_range': [1e-16],
                   'sigma_in_range': (np.log10(1e-6), np.log10(1e-3)),
                   'rho_range': (0.5, 1.1),
                   'plot_training': True,
                   'biasType': bias_models.ESN,  # Bias.ESN / Bias.NoBias
                   'perform_test': True,
                   }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    ensemble, truth = create_ensemble(true_params, forecast_params, filter_params, bias_params)
    filter_ens = ensemble.copy()

    # ================================= START BIAS MODEL ======================================== #

    bias_filename = folder + 'ESN_L{}'.format(bias_params['L'])
    rerun = True
    if os.path.isfile(bias_filename):
        bias = load_from_pickle_file(bias_filename)
        rerun = check_valid_file(bias, bias_params)

    if rerun:
        print('Train and save bias case')
        filter_ens.filename = ensemble.filename + '_L{}'.format(bias_params['L'])
        # Create training data on a multi-parameter approach
        train_data_filename = folder + filter_ens.filename + '_train_data'
        train_data = create_bias_training_dataset(truth['y'], filter_ens, bias_params,
                                                  train_data_filename, plot_=False)
        # Run ESN training
        filter_ens.bias.train_bias_model(train_data, folder=folder, plot_training=True)
        # Save
        save_to_pickle_file(bias_filename, filter_ens.bias)
    else:
        filter_ens.bias = bias
        print(filter_ens.bias.name)

    # ================================= START BIAS MODEL ======================================== #
    # std = 0.2
    # mean = np.mean(filter_ens.psi, axis=-1)
    # psi = filter_ens.addUncertainty(mean, std, filter_ens.m, method='normal')
    # filter_ens.hist[-1] = filter_ens.psi
    # mean = np.mean(filter_ens.psi, axis=-1)

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    # post_process_pdf(*out, reference_p=true_params, normalize=False)
    post_process_single(*out, reference_p=true_params)

    plt.show()
