if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createEnsemble
    from plotResults import *

    folder = 'results/test_constrainedKF/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': TAModels.VdP,
                   'manual_bias': None,
                   'std_obs': 0.01,
                   'beta': 115
                   }

    forecast_params = {'model': TAModels.VdP,
                       'beta': 70,
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filter': 'EnKF',
                     'constrained_filter': False,
                     'm': 10,
                     'est_p': ['beta'],
                     # initial parameter and state uncertainty
                     'std_a': 0.3,
                     'std_psi': 0.3,
                     'biasType': Bias.NoBias,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.6,
                     'kmeas': 35,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 10
                     }

    ensemble, truth = createEnsemble(true_params, forecast_params, filter_params, working_dir=folder, save_=False)

    filter_ens = ensemble.copy()

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    post_process_single(*out, reference_p=true_params)