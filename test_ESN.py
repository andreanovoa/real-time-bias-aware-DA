if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main
    from create import create_ensemble, create_bias_model
    from plotResults import *

    folder = 'results/test_ESN/'
    os.makedirs(folder, exist_ok=True)

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    model = physical_models.VdP

    true_params = {'model': model,
                   'manual_bias': 'cosine',
                   'std_obs': 0.01,
                   'beta': 50,
                   'zeta': 25,
                   'kappa': 9.0
                   }

    forecast_params = {'model': model
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(beta=(40., 60.), zeta=(20., 30.), kappa=(8., 10.))

    filter_params = {'filter': 'rBA_EnKF',
                     'constrained_filter': False,
                     'regularization_factor': 2.,
                     'm': 10,
                     # # initial parameter and state uncertainty
                     'est_a': ('beta', 'kappa'),
                     'std_a': parameters_IC,
                     'std_psi': 0.2,
                     'alpha_distr': 'uniform',
                     # Define the observation time window
                     't_start': 1.0,
                     't_stop': 2.5,
                     'dt_obs': 60,
                     # Inflation
                     'inflation': 1.002,
                     'reject_inflation': 1.001
                     }
    # ==================================== SELECT ESN PARAMETERS =================================== #

    bias_params = {'biasType': bias_models.ESN,  # Bias.ESN / Bias.NoBias
                   'augment_data': True,
                   'alpha_distr': 'uniform',
                   'L': 10,  # ESN training specs
                   'ensure_mean': True,
                   'est_a': filter_params['est_a'],
                   'N_wash': 30,
                   'plot_training': True,
                   'perform_test': True,
                   'rho_range': (0.5, 1.1),
                   'std_a': parameters_IC,
                   'std_psi': 2.,
                   'sigma_in_range': (np.log10(1e-5), np.log10(1e0)),
                   'tikh_range': [1e-12],
                   'upsample': 2,
                   }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    ensemble, truth = create_ensemble(true_params,
                                      forecast_params,
                                      filter_params,
                                      bias_params)
    filter_ens = ensemble.copy()

    # ================================= START BIAS MODEL ======================================== #

    bias_filename = 'ESN_L{}'.format(bias_params['L'])
    bias = create_bias_model(filter_ens, truth['y'], bias_params,
                             bias_filename, folder, plot_train_data=True)

    # ================================= START BIAS MODEL ======================================== #

    filtered_ens = main(filter_ens, truth)

    # Plot results -------
    # post_process_pdf(*out, reference_p=true_params, normalize=False)
    post_process_single(filtered_ens, truth, reference_p=true_params)

    plt.show()
