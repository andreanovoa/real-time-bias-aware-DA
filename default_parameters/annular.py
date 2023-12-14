from essentials.create import *
from essentials.physical_models import *
from essentials.Util import *

# ============================================================================================== #

path_dir = os.path.realpath(__file__).split('main')[0]
#
if os.path.isdir('/mscott/'):
    data_folder = '/mscott/an553/data/'  # set working directory to mscott
    # os.chdir(data_folder)  # set working directory to mscott
else:
    data_folder = "data/"


folder = 'results/Annular/'
figs_dir = folder + 'figs/'
out_dir = folder+"/out/"

os.makedirs(figs_dir, exist_ok=True)


ERs = 0.4875 + np.arange(0, 8) * 0.0125  # equivalence ratios 0.4875-0.575 (by steps of 0.0125)
ER = ERs[-1]
true_params = dict(model=data_folder + 'annular/ER_{}'.format(ER),
                   std_obs=0.1,
                   noise_type='white'
                   )

# ==================================== SELECT FILTER PARAMETERS =================================== #

parameters_IC = dict(
        nu=(40., 50.),
        beta_c2=(40., 50.),
        kappa=(1.E-4, 1.3E-4),
        epsilon=(1E-3, 5E-3),
        theta_b=(0.5, 0.7),
        omega=(1085 * 2 * np.pi, 1092 * 2 * np.pi),
        theta_e=(0.4, 0.8),
)

filter_params = dict(filter='rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     constrained_filter=False,
                     m=50,
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

# %% ================================= SELECT  FORECAST MODEL ===================================== #

forecast_params = dict(model=Annular,
                       nu=Annular.nu_from_ER(ER),
                       beta_c2=Annular.beta_c2_from_ER(ER),
                       )

bias_params = dict(biasType=ESN,   # ESN / NoBias
                   upsample=5,
                   N_units=100,
                   std_a=filter_params['std_a'],
                   est_a=filter_params['est_a'],
                   # Training data generation  options
                   augment_data=True,
                   L=20,
                   # Training, val and wash times
                   N_wash=5,
                   # Hyperparameter search ranges
                   rho_range=(0.5, 1.1),
                   sigma_in_range=(np.log10(1e-5), np.log10(1e1)),
                   tikh_range=[1e-16]
                   )