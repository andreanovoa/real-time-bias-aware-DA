


from physical_models import Annular
from bias_models import *
from run import main
from create import *
import numpy as np
from plot_functions.plotResults import *

# path_dir = os.path.realpath(__file__).split('main')[0]
# os.chdir('/mscott/an553/')  # set working directory to mscott

folder = 'results/Annular/'
figs_dir = folder + 'figs/'

os.makedirs(figs_dir, exist_ok=True)

if __name__ == '__main__':
    # %% ==================================== SELECT TRUE MODEL ======================================= #

    ERs = 0.4875 + np.arange(0, 8) * 0.0125  # equivalence ratios 0.4875-0.575 (by steps of 0.0125)}

    ER = ERs[-1]

    true_params = {'model': 'annular/ER_{}'.format(ER),
                   'std_obs': 0.1
                   }

    # ==================================== SELECT FILTER PARAMETERS =================================== #

    parameters_IC = dict(
        nu=(40., 50.),
        beta_c2=(40., 50.),
        kappa=(1.E-4, 1.3E-4),
        epsilon=(1E-3, 5E-3),
        omega=(1090 * 2 * np.pi, 1095 * 2 * np.pi),
        theta_b=(0.5, 0.7),
        theta_e=(0.4, 0.8),
    )

    filter_params = {'filter': 'EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     'constrained_filter': False,
                     'regularization_factor': 0.,
                     'm': 10,
                     # Parameter estimation options
                     'est_a': [*parameters_IC],
                     'std_a': parameters_IC,
                     'alpha_distr': 'uniform',
                     'std_psi': .5,
                     # Define the observation time window
                     't_start': 5.0,
                     't_stop': 5.1,
                     'dt_obs': 60,
                     # Inflation
                     'inflation': 1.00,
                     'reject_inflation': 1.00
                     }

    truth_og = create_truth(true_params, filter_params, post_processed=False)
    # for ER in ERs:
    #     true_params['model'] = 'annular/ER_{}'.format(ER)
    #     truth = create_truth(true_params, filter_params, post_processed=False)
    #
    #     print(true_params['model'])
    #     # Plot true data
    #     plot_truth(truth.copy(), plot_time=True, Nq=4, filename=figs_dir + 'truth_ER_{}.pdf'.format(ER))

    # %% ================================= SELECT  FORECAST MODEL ===================================== #

    k = 1.2e-4
    nu_1 = 633.77
    nu_2 = -331.39
    c2b_1 = 258.3
    c2b_2 = -108.27

    omega = 1090 * 2 * np.pi

    # %initial values
    C0 = 10
    X0 = 0
    th0 = 0.63
    ph0 = 0

    # %Conversion of the initial conditions from the quaternion formalism to the AB
    # %formalism
    Ai = C0 * np.sqrt(np.cos(th0) ** 2 * np.cos(X0) ** 2 + np.sin(th0) ** 2 * np.sin(X0) ** 2)
    Bi = C0 * np.sqrt(np.sin(th0) ** 2 * np.cos(X0) ** 2 + np.cos(th0) ** 2 * np.sin(X0) ** 2)
    phai = ph0 + np.arctan2(np.sin(th0) * np.sin(X0), np.cos(th0) * np.cos(X0))
    phbi = ph0 - np.arctan2(np.cos(th0) * np.sin(X0), np.sin(th0) * np.cos(X0))

    # %initial conditions for the fast oscillator equations
    eai = Ai * np.cos(phai)
    dteai = -omega * Ai * np.sin(phai)

    ebi = Bi * np.cos(phbi)
    dtebi = -omega * Bi * np.sin(phbi)
    psi0 = [eai, dteai, ebi, dtebi]

    forecast_params = {'model': Annular,
                       'dt': truth_og['dt'],
                       'nu': nu_1 * ER + nu_2,
                       'beta_c2': c2b_1 * ER + c2b_2,
                       'omega': omega,
                       'psi0': psi0
                       }

    bias_params = {'biasType': NoBias,  # Bias.ESN / Bias.NoBias
                   'std_a': filter_params['std_a'],
                   'est_a': filter_params['est_a'],
                   'augment_data': True,
                   'upsample': 5,
                   'L': 10,
                   'N_wash': 20,
                   'noise': 0.01,
                   'rho_range': (0.5, 1.1),
                   'sigma_in_range': (np.log10(1e-5), np.log10(1e0)),
                   'tikh_range': [1e-16],
                   }

    # ======================= CREATE REFERENCE ENSEMBLE =================================
    ensemble = create_ensemble(forecast_params, filter_params)
    filter_ens = ensemble.copy()

    truth = create_truth(true_params, filter_params, post_processed=True)

    # from plot_functions.plot_annular_model import *
    # plot_annular_model()

    # START BIAS MODEL -----------------------------------------------------------
    ESN_name = 'ESN_{}_L{}'.format(truth['name_bias'], bias_params['L'])
    create_bias_model(filter_ens, truth, bias_params, ESN_name,
                      bias_model_folder=folder, plot_train_data=True)

    # Add washout if needed ------------------------------------------------------

    filter_ens = main(filter_ens, truth)

    #%%  Plot results -------
    if filter_ens.est_a:
        reference_p = dict(omega=2 * np.pi,
                           kappa=1e-4,
                           epsilon=1e-3)

        plot_parameters(filter_ens, truth, reference_p=dict())

    post_process_single(filter_ens, truth, reference_p=Annular.defaults)

    #%%
    plot_timeseries(filter_ens, truth, reference_y=1.)
    plt.show()
