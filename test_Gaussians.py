if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createEnsemble
    from plotResults import *

    folder = 'results/test_Gaussians/'
    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
    true_params = {'model': TAModels.VdP,
                   'manual_bias': None,
                   'std_obs': 0.01,
                   'beta': 90,
                   'zeta': 60
                   }

    forecast_params = {'model': TAModels.VdP,
                       'beta': 70,
                       'zeta': 60
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filter': 'EnKF',
                     'constrained_filter': True,
                     'm': 500,
                     'est_p': ['beta'],
                     # # initial parameter and state uncertainty
                     # 'std_a': 0.25,
                     # 'std_psi': 0.25,
                     'biasType': Bias.NoBias,  # Bias.ESN / Bias.NoBias
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.1,
                     'kmeas': 35,
                     # Inflation
                     'inflation': 1.00,
                     'start_ensemble_forecast': 10
                     }

    ensemble, truth = createEnsemble(true_params, forecast_params,
                                     filter_params, working_dir=folder, save_=False)

    filter_ens = ensemble.copy()

    std = 0.15
    mean = np.mean(filter_ens.psi, axis=-1)
    print(mean, np.std(filter_ens.psi, axis=-1))
    filter_ens.psi = filter_ens.addUncertainty(mean, std,
                                               filter_ens.m, method='normal')
    filter_ens.hist[-1] = filter_ens.psi
    mean = np.mean(filter_ens.psi, axis=-1)
    print(mean, np.std(filter_ens.psi, axis=-1))

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    post_process_pdf(*out, reference_p=true_params, normalize=False)
    post_process_single(*out, reference_p=true_params)

    plt.show()
