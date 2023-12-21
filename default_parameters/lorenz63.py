from essentials.physical_models import Lorenz63
import numpy as np

rnd = np.random.RandomState(6)

dt_model = 0.015
t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent

forecast_params = dict(dt=dt_model,
                       psi0=rnd.random(3),
                       t_transient=10 * t_lyap,
                       t_lyap=t_lyap)

bias_params = dict(t_train=40 * t_lyap,
                   t_val=2 * t_lyap,
                   t_test=10 * t_lyap,
                   N_wash=2,
                   N_units=80,
                   N_folds=8,
                   N_split=5,
                   connect=6,
                   plot_training=True,
                   rho_range=(.1, 1.),
                   tikh_range=[1E-6, 1E-9, 1E-12],
                   N_func_evals=30,
                   sigma_in_range=(np.log10(0.5), np.log10(50.)),
                   N_grid=4,
                   noise=1e-3,
                   perform_test=True
                   )

parameters_IC = dict(
                     # rho=[25., 30.],
                     # sigma=[9, 11.],
                     # beta=[8. / 3., 4]
                     )

filter_params = dict(filter='EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     constrained_filter=False,
                     m=10,
                     regularization_factor=2.0,
                     # Parameter estimation options
                     est_a=[*parameters_IC],
                     std_a=parameters_IC,
                     alpha_distr='uniform',
                     std_psi=.5,
                     # Define the observation time window
                     t_start=2.0,
                     t_stop=2.3,
                     dt_obs=40,
                     # Inflation parameters
                     inflation=1.00,
                     reject_inflation=1.00
                     )

