
import numpy as np
from essentials.physical_models import Lorenz63
from essentials.create import create_truth, create_ensemble
from essentials.DA import dataAssimilation
from essentials.plotResults import plot_timeseries, plot_parameters, plot_truth


t_lyap = Lorenz63.t_lyap
dt_t = 0.015

rng = np.random.default_rng(0)

true_params = dict(model=Lorenz63,
                   t_start=t_lyap * 10,
                   t_stop=t_lyap * 80,
                   t_max=100 * t_lyap,
                   Nt_obs=(t_lyap * .5) // dt_t,
                   dt=dt_t,
                   rho=28.,
                   sigma=10.,
                   beta=8. / 3.,
                   psi0=rng.random(3)+10,
                   std_obs=0.005,
                   noise_type='gauss,additive'
                   )

truth = create_truth(**true_params)
y_obs, t_obs = [truth[key].copy() for key in ['y_obs', 't_obs']]
plot_truth(**truth)

forecast_params = dict(filter='EnKF',
                       m=50,
                       dt=dt_t,
                       model=Lorenz63,
                       est_a=dict(rho=(25., 35.),
                                  beta=(2, 4),
                                  sigma=(5, 15)),
                       std_psi=0.3,
                       alpha_distr='uniform',
                       inflation=1.01
                       )

ensemble = create_ensemble(**forecast_params)
filter_ens = dataAssimilation(ensemble.copy(), y_obs=y_obs, t_obs=t_obs, std_obs=0.005)

# Visualize attractors
case0 = truth['case'].copy()
case1 = filter_ens.copy()

## Plot timeseries results
ens = filter_ens.copy()

# Forecast the ensemble further without assimilation
psi, t = ens.time_integrate(int(4 * t_lyap / ens.dt), averaged=False)
ens.update_history(psi, t)

plot_timeseries(ens, truth, reference_t=t_lyap, plot_ensemble_members=False)
plot_parameters(ens, truth, reference_p=true_params)

# Forecast both cases
Nt = 40 * int(t_lyap / filter_ens.dt)
psi0, t0 = case0.time_integrate(Nt=Nt)
psi1, t1 = case1.time_integrate(Nt=Nt, averaged=True)

plot_attractor([psi0, psi1], color=['w', 'teal'])

