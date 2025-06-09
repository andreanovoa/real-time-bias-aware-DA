from default_parameters.lorenz63 import bias_params
from src.data_assimilation import *
from src.create import *
from src.bias import ESN
from src.models_physical import Annular
from dev.plotting_annular_results import *

# This code does not run without the Azimuthal data. Contact @andreanovoa for access this data.

name = 'simulation_output_all_new'
ERs = 0.4875 + np.arange(0, 4) * 0.025

suffix = '_short'
rng = np.random.default_rng(0)

alpha0 = dict(nu=(-10., 30.),
              c2beta=(10, 50),
              kappa=(1.E-4, 2.E-4),
              epsilon=(5e-3, 8e-3),
              omega=(1090 * 2 * np.pi, 1095 * 2 * np.pi),
              theta_b=(0.5, 0.7),
              theta_e=(0.4, 0.6)
              )

forecast_params = dict(model=Annular,
                       std_psi=0.3,
                       std_a=alpha0.copy(),
                       inflation=1.00,
                       reject_inflation=1.0
                       )

train_params = dict(bias_model=ESN,
                    upsample=5,
                    N_units=50,
                    N_wash=10,
                    t_train=Annular.t_transient / 3.,
                    t_test=Annular.t_CR * 2,
                    t_val=Annular.t_CR * 2,
                    # Training data generation options
                    augment_data=True,
                    biased_observations=True,
                    seed_W=0,
                    N_folds=4,
                    L=50,
                    std_a=alpha0,
                    # Hyperparameter search ranges
                    rho_range=(0.5, 1.),
                    sigma_in_range=(np.log10(1e-5), np.log10(1e1)),
                    tikh_range=[1e-12, 1e-9],
                    )


data_folder, results_folder, figs_folder = set_working_directories('annular')

if __name__ == '__main__':

    for m in [20]:
    # for m in [10, 20, 40, 60, 80]:

        for ER in ERs[2:]:
            t_start = Annular.t_transient

            if suffix == '_long':
                t_stop = t_start + Annular.t_CR * 80
            else:
                t_stop = t_start + Annular.t_CR * 35

            truth_og = create_truth(model=data_folder + 'ER_{}'.format(ER),
                                    t_start=t_start,
                                    t_stop=t_stop,
                                    Nt_obs=35,
                                    t_max=t_stop + Annular.t_transient,
                                    post_processed=False
                                    )
            truth = truth_og.copy()

            forecast_params['dt'] = truth['dt']
            forecast_params['m'] = m
            train_params['t_init'] = truth['t_obs'][0] - 2 * truth['dt_obs']

            ensemble = create_ensemble(**forecast_params)
            ensemble.t_init = truth['t_obs'][0]

            parent_dir = results_folder + f'ER{ER}/'
            os.makedirs(parent_dir, exist_ok=True)


            ensemble_ba = ensemble.copy()
            ensemble_bb = ensemble.copy()
            bias_og, wash_obs, wash_t = create_bias_model(ensemble_ba,
                                                          bias_params=train_params,
                                                          training_dataset=truth,
                                                          folder=parent_dir,
                                                          bias_filename="ESN_case_annular_raw")
            ensemble_ba.bias = bias_og.copy()

            truth = truth_og.copy()

            out = []


            for rho in [1.0, 1.001]:
                results_dir = parent_dir + f'm{m}/rho{rho}{suffix}/'

                os.makedirs(results_dir, exist_ok=True)

                ens_bb = ensemble_bb.copy()
                ens_ba = ensemble_ba.copy()

                DA_kwargs = dict(y_obs=truth['y_obs'].copy(), t_obs=truth['t_obs'].copy(), std_obs=0.1,
                                 wash_obs=wash_obs.copy(), wash_t=wash_t.copy())

                for kf in ['rBA_EnKF', 'EnKF']:

                    if kf[0] == 'r':
                        ks = np.linspace(0, 5, 12)
                        blank_ens = ens_ba.copy()
                    else:
                        ks = [None]
                        blank_ens = ens_bb.copy()

                    blank_ens.filter = kf
                    blank_ens.inflation = rho
                    blank_ens.reject_inflation = rho

                    for kk in ks:
                        ens = blank_ens.copy()

                        if kf[0] == 'r':
                            ens.regularization_factor = kk

                        filter_ens = dataAssimilation(ens, **DA_kwargs.copy())

                        # Forecast the ensemble further without assimilation
                        Nt_extra = int(filter_ens.t_CR / filter_ens.dt) * 5

                        psi, t = filter_ens.time_integrate(Nt_extra)

                        filter_ens.update_history(psi, t)

                        y = filter_ens.get_observable_hist(Nt_extra)
                        b, t_b = filter_ens.bias.time_integrate(t=t, y=y)
                        filter_ens.bias.update_history(b, t_b)

                        filter_ens.close()

                        out.append(filter_ens.copy())

                save_to_pickle_file(results_dir + name, truth, out, bias_og.copy(), ensemble.copy())

            # run_plotting_annular_results(m=m, rho=rho, suffix=suffix)

