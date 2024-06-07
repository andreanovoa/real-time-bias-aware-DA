if __name__ == '__main__':
    from functions import bias_models, physical_models
    from run import main, create_ensemble
    from plotResults import *

    folder = 'results/test_Gaussians/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model':       physical_models.VdP,
                   'manual_bias': None,
                   'std_obs':     0.05,
                   'beta':        110,
                   'zeta':        25,
                   'kappa':       9.0
                   }

    forecast_params = {'model': physical_models.VdP,
                       'beta':  50,
                       'zeta':  25,
                       'kappa': 9.0
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filter': 'EnKF',
                     'constrained_filter': 1,
                     'm': 100,
                     'est_p': ['beta'],
                     # # initial parameter and state uncertainty
                     'bias_type': bias_models.NoBias,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.5,
                     'kmeas': 35,
                     # Inflation
                     'inflation': 1.005,
                     'reject_inflation': 1.02,
                     'start_ensemble_forecast': 10
                     }

    ensemble, truth = create_ensemble(true_params, forecast_params, filter_params)

    filter_ens = ensemble.copy()

    std = 0.2
    mean = np.mean(filter_ens.psi, axis=-1)
    filter_ens.psi = filter_ens.add_uncertainty(mean, std,
                                                filter_ens.m, method='normal')
    filter_ens.hist[-1] = filter_ens.psi
    mean = np.mean(filter_ens.psi, axis=-1)

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    post_process_pdf(*out, reference_p=true_params, normalize=False)
    post_process_single(*out, reference_p=true_params)

    plt.show()
