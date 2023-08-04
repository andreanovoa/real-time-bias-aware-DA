if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createEnsemble
    from plotResults import *

    folder = 'results/test_Gaussians/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': TAModels.VdP,
                   'manual_bias': 'cosine',
                   'std_obs': 0.05,
                   'beta': 50,
                   'zeta': 25,
                   'kappa': 9.0
                   }

    forecast_params = {'model': TAModels.VdP,
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
                     'biasType': Bias.ESN,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 3.0,
                     't_stop': 3.5,
                     'kmeas': 35,
                     # Inflation
                     'inflation': 1.005,
                     'reject_inflation': 1.02,
                     'start_ensemble_forecast': 10
                     }
    # ==================================== SELECT ESN PARAMETERS =================================== #

    train_params = {'model': TAModels.VdP,
                    'std_a': 0.3,
                    'std_psi': 0.3,
                    'est_p': filter_params['est_p'],
                    'alpha_distr': 'uniform',
                    'ensure_mean': True,
                    }

    bias_params = {'N_wash': 30,
                   'upsample': 5,
                   'L': 10,
                   'augment_data': True,
                   'train_params': train_params,
                   'tikh_range': [1e-16],
                   'sigma_in_range': (np.log10(1e-6), np.log10(1e-2)),
                   'test_run': True,
                   'plot_training': True
                   }

    # ============================ CREATE REFERENCE ENSEMBLE =================================== #
    name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])
    ensemble, truth, esn_args = createEnsemble(true_params, forecast_params,
                                               filter_params, bias_params,
                                               working_dir=folder, filename=name)

    filter_ens = ensemble.copy()

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
