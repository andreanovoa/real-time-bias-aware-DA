if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main, createEnsemble, create_bias_training_dataset
    from plotResults import *
    from Util import save_to_pickle_file

    folder = 'results/test_Gaussians/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': physical_models.VdP,
                   'manual_bias': 'cosine',
                   'std_obs': 0.05,
                   'beta': 50,
                   'zeta': 25,
                   'kappa': 9.0
                   }

    forecast_params = {'model': physical_models.VdP,
                       'beta': 50,
                       'zeta': 25,
                       'kappa': 9.0
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filter': 'EnKF',
                     'constrained_filter': 1,
                     'm': 10,
                     'est_p': ['beta'],
                     # # initial parameter and state uncertainty
                     'biasType': bias_models.ESN,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 3.0,
                     't_stop': 3.5,
                     'dt_obs': 35,
                     # Inflation
                     'inflation': 1.005,
                     'reject_inflation': 1.02,
                     'start_ensemble_forecast': 10
                     }
    # ==================================== SELECT ESN PARAMETERS =================================== #

    train_params = {'model': physical_models.VdP,
                    'std_a': 0.3,
                    'std_psi': 0.3,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True,
                    'L': 10,
                    }

    bias_params = {'N_wash': 30,
                   'upsample': 5,
                   'augment_data': True,
                   'tikh_range': [1e-16],
                   'sigma_in_range': (np.log10(1e-6), np.log10(1e-2)),
                   'test_run': True,
                   'plot_training': True
                   }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    name = 'reference_Ensemble_m{}_dt_obs{}'.format(filter_params['m'], filter_params['dt_obs'])
    ensemble, truth = createEnsemble(true_params, forecast_params, filter_params, bias_params)

    filter_ens = ensemble.copy()
    # ================================= START BIAS MODEL ======================================== #

    y_train = create_bias_training_dataset(ensemble.model, truth, folder, bias_param=None)

    train_data = truth['y'] - y_train  # [Nt x Nmic x L]



    # Store file
    save_to_pickle_file(folder + name, filter_ens, truth)

    std = 0.2
    mean = np.mean(filter_ens.psi, axis=-1)
    filter_ens.psi = filter_ens.addUncertainty(mean, std,
                                               filter_ens.m, method='normal')
    filter_ens.hist[-1] = filter_ens.psi
    mean = np.mean(filter_ens.psi, axis=-1)

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    # post_process_pdf(*out, reference_p=true_params, normalize=False)
    post_process_single(*out, reference_p=true_params)

    plt.show()
