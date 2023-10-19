
if __name__ == '__main__':
    import physical_models
    import bias_models
    from run import main
    from create import create_truth, create_ensemble, create_bias_model
    from plotResults import *

    # path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott

    folder = 'results/Annular/'
    figs_dir = folder + 'figs/'

    os.makedirs(figs_dir, exist_ok=True)
    # %% ==================================== SELECT TRUE MODEL ======================================= #

    # def model_bias(yy, tt):
    #     # return .3 * np.max(yy, axis=0) + .3 * yy, 'linear'   # Linear bias
    #     return .3 * np.max(yy, axis=0) * (np.cos((yy / np.max(yy, axis=0))**2) + 1), 'nonlinear+offset'  # Non-linear bias

    ER = np.round(np.linspace(start=0.4875, stop=0.575, num=8, endpoint=True), 6)  # equivalence ratios 0.4875-0.575 (by steps of 0.0125)}

    true_params = {'model': 'annular/ER_{}'.format(ER[-3]),
                   'std_obs': 0.1
                   }

    # ==================================== SELECT FILTER PARAMETERS =================================== #
    parameters_IC = dict(
                         nu=(5., 100.),
                         beta_c2=(5, 100),
                         kappa=(1.E-4, 1.3E-4),
                         epsilon=(0.0001, 0.03),
                         omega=(1100, 1200),
                         theta_b=(0.2, 0.7),
                         theta_e=(0.15, 0.8),
                         )

    filter_params = {'biasType': bias_models.ESN,
                     'filt': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'constrained_filter': False,
                     'regularization_factor': 10.,
                     'm': 100,
                     # Parameter estimation options
                     'est_a': [*parameters_IC],
                     'std_a': parameters_IC,
                     'alpha_distr': 'uniform',
                     'std_psi': .2,
                     # Define the observation time window
                     't_start': 1.0,
                     't_stop': 2.,
                     'dt_obs': 10,
                     # Inflation
                     'inflation': 1.002
                     }

    truth = create_truth(true_params, filter_params)

    # %% ================================= SELECT  FORECAST MODEL ===================================== #
    model = physical_models.Annular
    forecast_params = {'model': model,
                       'dt': truth['dt']
                       }

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

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble = create_ensemble(forecast_params, filter_params)
    filter_ens = ensemble.copy()

    # START BIAS MODEL -----------------------------------------------------------
    bias_name = 'ESN_{}_L{}'.format(truth['name_bias'], ESN_params['L'])
    filter_ens.bias = create_bias_model(filter_ens, truth, ESN_params, bias_name,
                                        bias_model_folder=folder, plot_train_data=True)

    # Add washout if needed ------------------------------------------------------
    truth = create_truth(true_params, filter_params)

    print(len(truth['wash_t']))

    filter_ens = main(filter_ens, truth)

    # Plot results -------

    plot_parameters(filter_ens, truth, reference_p=model.defaults)
    post_process_single(filter_ens, truth)

    plt.show()
