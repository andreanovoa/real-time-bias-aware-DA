if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createEnsemble
    from plotResults import *
    import os as os

    folder = 'results/test_constrainedKF/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': TAModels.VdP,
                   'manual_bias': None,
                   'law': 'tan',
                   'std_obs': 0.1
                   }

    forecast_params = {'model': TAModels.VdP,
                       'law': 'tan',
                       'beta': 100,
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filt': 'EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'm': 10,
                     'est_p': ['beta', 'kappa'],
                     # initial parameter and state uncertainty
                     'std_a': 0.1,
                     'std_psi': 0.1,
                     'biasType': Bias.NoBias,  # Bias.ESN  # Bias.NoBias
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.5,
                     'kmeas': 35,
                     # Inflation
                     'inflation': 1.05,
                     'start_ensemble_forecast': 10
                     }

    ensemble, truth = createEnsemble(true_params, forecast_params, filter_params, working_dir=folder)


    filter_ens = ensemble.copy()
    out = main(filter_ens, truth, filter_params['filt'], save_=False)

    # Plot results -------
    post_process_single(*out, reference_p=TAModels.VdP.attr)
