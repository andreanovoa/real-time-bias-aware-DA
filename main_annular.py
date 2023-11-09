from physical_models import Annular
from bias_models import *
from run import main, save_simulation
from create import *
from plot_functions.plotResults import *

path_dir = os.path.realpath(__file__).split('main')[0]

data_folder = '/mscott/an553/'
os.chdir(data_folder)  # set working directory to mscott


folder = 'results/Annular/'
figs_dir = folder + 'figs/'
out_dir = folder+"/out/"

os.makedirs(figs_dir, exist_ok=True)

if __name__ == '__main__':
    # %% ==================================== SELECT TRUE MODEL ======================================= #

    ERs = 0.4875 + np.arange(0, 8) * 0.0125  # equivalence ratios 0.4875-0.575 (by steps of 0.0125)
    ER = ERs[-1]
    true_params = dict(model=data_folder + 'annular/ER_{}'.format(ER),
                       std_obs=0.2,
                       noise_type='white'
                       )

    # ==================================== SELECT FILTER PARAMETERS =================================== #

    parameters_IC = dict(
                        nu=(40., 50.),
                        beta_c2=(40., 50.),
                        kappa=(1.E-4, 1.3E-4),
                        epsilon=(1E-3, 5E-3),
                        omega=(1080 * 2 * np.pi, 1095 * 2 * np.pi),
                        theta_b=(0.5, 0.7),
                        theta_e=(0.4, 0.8),
                        )

    filter_params = dict(filter='rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                         constrained_filter=False,
                         m=10,
                         regularization_factor=2.,
                         # Parameter estimation options
                         est_a=[*parameters_IC],
                         std_a=parameters_IC,
                         alpha_distr='uniform',
                         std_psi=.5,
                         # Define the observation time window
                         t_start=5.0,
                         t_stop=5.05,
                         dt_obs=60,
                         # Inflation parameters
                         inflation=1.00,
                         reject_inflation=1.00
                         )

    # %% ================================= SELECT  FORECAST MODEL ===================================== #

    forecast_params = dict(model=Annular,
                           ER=ER,
                           )

    bias_params = dict(biasType=ESN,   # ESN / NoBias
                       upsample=5,
                       N_units=500,
                       std_a=filter_params['std_a'],
                       est_a=filter_params['est_a'],
                       # Training data generation  options
                       augment_data=True,
                       L=10,
                       # Training, val and wash times
                       N_wash=20,
                       noise=0.001,
                       # Hyperparameter search ranges
                       rho_range=(0.5, 1.1),
                       sigma_in_range=(np.log10(1e-5), np.log10(1e0)),
                       tikh_range=[1e-16]
                       )

    # ======================= CREATE TRUTH AND ENSEMBLE  =================================
    truth = create_truth(true_params, filter_params, post_processed=False)
    forecast_params['dt'] = truth['dt']

    ensemble = create_ensemble(forecast_params, filter_params)

    # START BIAS MODEL -----------------------------------------------------------
    ESN_name = 'ESN_{}_L{}'.format(truth['name_bias'], bias_params['L'])
    filter_ens = ensemble.copy()
    create_bias_model(filter_ens, truth, bias_params, ESN_name,
                      bias_model_folder=folder, plot_train_data=True)

    filter_ens = main(filter_ens, truth)

    #%%  Plot results -------

    # Save simulation
    # save_simulation(filter_ens, truth, extra_parameters=None, results_dir=out_dir)

    nu_1, nu_2 = 633.77, -331.39
    c2b_1, c2b_2 = 258.3, -108.27

    reference_params = dict(theta_b=0.63, theta_e=0.66, omega=1090*2*np.pi, epsilon=2.3E-3,
                            nu=nu_1 * ER + nu_2, beta_c2=c2b_1 * ER + c2b_2, kappa=1.2E-4)  # values in Matlab codes
    if filter_ens.est_a:
        plot_parameters(filter_ens, truth, reference_p=reference_params, twin=True)

    # post_process_single(filter_ens, truth, reference_p=Annular.defaults)
    plot_timeseries(filter_ens, truth, reference_y=1.)
    plt.show()
