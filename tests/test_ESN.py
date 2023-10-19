if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main
    from create import create_ensemble, create_bias_model, create_washout
    from plotResults import *

    os.chdir('../')  # set working directory to mscott
    folder = 'results/test_ESN/Annular/'
    os.makedirs(folder, exist_ok=True)

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    model = physical_models.Annular


    def model_bias(yy, tt):
        # return .3 * np.max(yy, axis=0) + .3 * yy, 'linear'   # Linear bias
        return .3 * np.max(yy, axis=0) * (np.cos((yy / np.max(yy, axis=0))**2) + 1), 'nonlinear+offset'  # Non-linear bias


    true_params = {'model': model,
                   'manual_bias': model_bias,
                   'noise_type': 'white',
                   'std_obs': 0.1,
                   'nu': 17.,
                   'beta_c2': 17.,
                   'kappa': 1.2e-4
                   }

    forecast_params = true_params.copy()

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(beta_c2=(20., 30.),
                         kappa=(0.2e-5, 2.4e-4)
                         )

    filter_params = {'filter': 'rBA_EnKF',
                     'constrained_filter': False,
                     'regularization_factor': 5.,
                     'm': 10,
                     # # initial parameter and state uncertainty
                     'est_a': [*parameters_IC],
                     'std_a': parameters_IC,
                     'std_psi': 0.2,
                     'alpha_distr': 'uniform',
                     # Define the observation time window
                     't_start': model.t_transient * 1.5,
                     't_stop': model.t_transient * 3.,
                     'dt_obs': 180,
                     # Inflation
                     'inflation': 1.001,
                     'reject_inflation': 1.001
                     }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    ensemble_og, truth_og = create_ensemble(true_params, forecast_params, filter_params)

    # ==================================== SELECT ESN PARAMETERS =================================== #

    ESN_params = {'biasType': bias_models.ESN,  # Bias.ESN / Bias.NoBias
                  'augment_data': True,
                  'alpha_distr': 'uniform',
                  'L': 20,  # ESN training specs
                  'ensure_mean': True,
                  'est_a': filter_params['est_a'],
                  'N_wash': 20,
                  'plot_training': True,
                  'perform_test': True,
                  'rho_range': (0.5, 1.1),
                  'std_a': parameters_IC,
                  't_val': model.t_CR * 2,
                  'std_psi': 2.,
                  'sigma_in_range': (np.log10(1e-5), np.log10(1e0)),
                  'tikh_range': [1e-10, 1e-12, 1e-16],
                  'upsample': 10,
                  }
    NoBias_params = {'biasType': bias_models.NoBias}

    for bias_params in [ESN_params, NoBias_params]:
        ensemble = ensemble_og.copy()

        bias_filename = '{}_{}'.format(bias_params['biasType'].name, truth_og['name_bias'])
        print(bias_filename)

        # START BIAS MODEL -----------------------------------------------------------

        bias = create_bias_model(ensemble, truth_og, bias_params,
                                 bias_filename, folder, plot_train_data=True)
        ensemble.bias = bias

        # Add washout if needed ------------------------------------------------------
        truth = create_washout(bias, truth_og)

        # Run main DA code -----------------------------------------------------------
        filtered_ens = main(ensemble, truth)

        # Plot results --------------------------------------------------------------
        post_process_single(filtered_ens, truth, reference_p=true_params)
        # plot_parameters(filtered_ens, truth, reference_p=None)
    plt.show()
