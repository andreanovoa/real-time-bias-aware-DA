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
        return .1 * np.max(yy, 0) + .3 * yy

    true_params = {'model': model,
                   'manual_bias': model_bias,
                   'std_obs': 0.01,
                   'nu': 17.,
                   'beta_c2': 17.,
                   'kappa': 1.2e-4
                   }

    forecast_params = {'model': model
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(beta_c2=(10., 20.))

    filter_params = {'filter': 'rBA_EnKF',
                     'constrained_filter': False,
                     'regularization_factor': 2.,
                     'm': 10,
                     # # initial parameter and state uncertainty
                     'est_a': [*parameters_IC],
                     'std_a': parameters_IC,
                     'std_psi': 0.2,
                     'alpha_distr': 'uniform',
                     # Define the observation time window
                     't_start': 1.0,
                     't_stop': 2.5,
                     'dt_obs': 60,
                     # Inflation
                     'inflation': 1.001,
                     'reject_inflation': 1.001
                     }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    ensemble_og, truth_og = create_ensemble(true_params,
                                            forecast_params,
                                            filter_params)

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
                   'upsample': 5,
                   }

    for L in [10, 1]:
        ensemble = ensemble_og.copy()

        bias_params['L'] = L
        bias_filename = '{}_L{}'.format(bias_params['biasType'].name, bias_params['L'])

        # START BIAS MODEL -----------------------------------------------------------

        bias = create_bias_model(ensemble, truth_og['y'], bias_params,
                                 bias_filename, folder, plot_train_data=True)
        ensemble.bias = bias

        # Add washout if needed ------------------------------------------------------
        truth = create_washout(bias, truth_og)

        filtered_ens = main(ensemble, truth)

        # Plot results -------
        # post_process_pdf(*out, reference_p=true_params, normalize=False)
        post_process_single(filtered_ens, truth, reference_p=true_params)

