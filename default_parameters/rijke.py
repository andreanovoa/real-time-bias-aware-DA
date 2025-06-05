from essentials.bias_models import *
from essentials.create import *
from essentials.models_physical import *
from essentials.plotResults import *
from essentials.run import *
from essentials.Util import *


# %% ====================== SELECT TRUE AND FORECAST MODELS ========================== #
true_params = dict(model=Rijke,
                   std_obs=0.05,
                   t_max=2.5,
                   beta=4.2,
                   tau=1.4E-3,
                   manual_bias='linear'
                   )
forecast_params = dict(model=Rijke,
                       t_max=2.5
                       )
# %% ============================ SELECT FILTER PARAMETERS ============================ #
filter_params = dict(filter='rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                     est_a=['beta', 'tau'],
                     std_a=0.25,
                     std_psi=0.25,
                     t_start=1.5,
                     t_stop=2.0,
                     dt_obs=20,
                     inflation=1.002,
                     start_ensemble_forecast=1)

bias_params = dict(bias_type=ESN,  # ESN / NoBias
                   N_units=500,
                   upsample=2,
                   # Training data generation  options
                   augment_data=True,
                   L=10,
                   est_a=filter_params['est_a'],
                   std_a=0.3,
                   # Training, val and wash times
                   t_val=0.02,
                   t_train=0.5,
                   N_wash=50,
                   # Hyperparameter search ranges
                   rho_range=[0.5, 1.0],
                   tikh_range=np.array([1e-16]),
                   sigma_in_range=[np.log10(1e-5), np.log10(1e-2)]
                   )