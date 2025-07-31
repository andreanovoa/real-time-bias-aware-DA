from src.models_physical import Lorenz63
from src.bias import ESN
import numpy as np

rnd = np.random.RandomState(6)

dt_model = 0.015
t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent

forecast_params = dict(model=Lorenz63,
                       dt=dt_model,
                       psi0=rnd.random(3),
                       t_transient=10 * t_lyap,
                       t_lyap=t_lyap)

bias_params = dict(bias_model=ESN,
                   t_train=40 * t_lyap,
                   t_val=6 * t_lyap,
                   N_wash=20,
                   N_units=50,
                   N_folds=4,
                   N_split=5,
                   connect=3,
                   plot_training=True,
                   rho_range=(.2, .8),
                   tikh_range=[1E-6, 1E-9, 1E-12],
                   N_func_evals=20,
                   sigma_in_range=(np.log10(0.5), np.log10(50.)),
                   N_grid=4,
                   noise=1e-2,
                   perform_test=True,
                   Win_type='sparse',
                   upsample=2,
                   L=20,
                   )

parameters_IC = dict(
                     # rho=[25., 30.],
                     # sigma=[9, 11.],
                     # beta=[8. / 3., 4]
                     )

filter_params = dict(filter='EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     constrained_filter=False,
                     m=20,
                     regularization_factor=2.0,
                     # Parameter estimation options
                     est_a=[*parameters_IC],
                     std_a=parameters_IC,
                     alpha_distr='uniform',
                     std_psi=.5,
                     # Define the observation time window
                     t_start=20,
                     t_stop=40.,
                     dt_obs=25,
                     # Inflation input_parameters
                     inflation=1.00,
                     reject_inflation=1.00
                     )

