if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, create_ESN_train_dataset, createEnsemble
    from plotResults import *
    import os as os

    # path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott

    folder = '/results/Annular/'
    figs_dir = folder + 'figs/'

    # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #

    true_params = {'model': TAModels.Annular,
                   'std_obs': 0.010,
                   'beta_c2': 17.,
                   'nu': 17.
                   }

    forecast_params = {'model': TAModels.Annular,
                       'beta_c2': 20.,
                       'nu': 10.
                       }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    filter_params = {'filt': 'EnSRKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'constrained_filter': 0,
                     'm': 10,
                     'est_p': ['beta_c2', 'nu'],
                     'biasType': Bias.NoBias,  # Bias.ESN  # None
                     # Define the observation time window
                     't_start': 2.0,
                     't_stop': 2.3,
                     'kmeas': 160,
                     # Inflation
                     'inflation': 1.002,
                     'start_ensemble_forecast': 10
                     }

    name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble, truth = createEnsemble(true_params, forecast_params, filter_params,
                                     working_dir=folder, filename=name, save_=False)
    filter_ens = ensemble.copy()

    std = 0.25
    mean = np.mean(filter_ens.psi, axis=-1)
    filter_ens.psi = filter_ens.addUncertainty(mean, std,
                                               filter_ens.m, method='normal')
    filter_ens.hist[-1] = filter_ens.psi
    mean = np.mean(filter_ens.psi, axis=-1)

    out = main(filter_ens, truth, save_=False)

    # Plot results -------
    post_process_single(*out, reference_p=true_params)

    plt.show()
