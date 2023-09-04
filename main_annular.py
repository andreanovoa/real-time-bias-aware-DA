if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main, create_ESN_train_dataset, create_ensemble
    from plotResults import *
    import os as os

    # path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott

    folder = '/results/Annular/'
    figs_dir = folder + 'figs/'

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #

    true_params = {'model': physical_models.Annular,
                   'std_obs': 0.010,
                   # slect truth parameters
                   'beta_c2': 17.,
                   'nu': 17.,
                   'kappa': 1.2E-4,
                   'omega': 1090,
                   'epsilon': 0.0023,
                   'theta_b': 0.63,
                   'theta_e': 0.66,
                   }

    forecast_params = {'model': physical_models.Annular,
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(nu=(-15., 25.), beta_c2=(5, 40), kappa=(.5E-4, 2.E-4), epsilon=(0.001, 0.003),
                         omega=(1080, 1100), theta_b=(0.2, 0.7), theta_e=(0.6, 0.7))

    filter_params = {'filt': 'EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'constrained_filter': 0,
                     'm': 50,
                     'est_a': ['nu', 'beta_c2', 'kappa', 'epsilon', 'omega', 'theta_b', 'theta_e'],
                     'std_a': parameters_IC,
                     'alpha_distr': 'uniform',
                     'std_psi': 1.,
                     'biasType': bias_models.NoBias,  # Bias.ESN  # None
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.5,
                     'dt_obs': 320,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 10
                     }

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble, truth = create_ensemble(true_params, forecast_params, filter_params)
    filter_ens = ensemble.copy()

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    post_process_single(*out, reference_p=true_params, plot_params=True)

    plt.show()
