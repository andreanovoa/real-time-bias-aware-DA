from essentials.DA import *
from essentials.create import *
from essentials.physical_models import Annular
from essentials.bias_models import ESN
from essentials.plotResults import plot_truth, print_parameter_results, plot_states_PDF, plot_RMS_pdf

rng = np.random.default_rng(0)

# This code does not run without the Azimuthal data.
# Please contact @andreanovoa for access this data.

data_folder, results_folder, figs_folder = set_working_directories('annular')

if __name__ == '__main__':
    ERs = 0.4875 + np.arange(0, 4) * 0.025

    for ER in [ERs[3]]:
        t_start = Annular.t_transient
        t_stop = t_start + Annular.t_CR * 35

        truth_og = create_truth(model=data_folder + 'ER_{}'.format(ER),
                                t_start=t_start,
                                t_stop=t_stop,
                                Nt_obs=30,
                                t_max=t_stop + Annular.t_transient,
                                post_processed=False
                                )
        truth = truth_og.copy()

        alpha0 = dict(nu=(-10., 30.),
                      c2beta=(10, 50),
                      kappa=(1.E-4, 2.E-4),
                      epsilon=(5e-3, 8e-3),
                      omega=(1090 * 2 * np.pi, 1095 * 2 * np.pi),
                      theta_b=(0.5, 0.7),
                      theta_e=(0.4, 0.6)
                      )

        forecast_params = dict(model=Annular,
                               dt=truth['dt'],
                               m=50,
                               std_psi=0.3,
                               std_a=alpha0.copy(),
                               inflation=1.001,
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
                            t_init=truth['t_obs'][0] - 2 * truth['dt_obs'],
                            )

        # plot_truth(**truth)
        ensemble = create_ensemble(**forecast_params)

        results_dir = results_folder + 'ER{}/m{}_rho{}/'.format(ER, ensemble.m, ensemble.inflation)
        os.makedirs(results_dir, exist_ok=True)

        ensemble_ESN = ensemble.copy()
        bias, wash_obs, wash_t = create_bias_model(ensemble_ESN, truth, train_params,
                                                   folder=results_dir, bias_filename="ESN_case_annular_raw")
        truth = truth_og.copy()

        for regularized_filter, name in zip(['rBA_EnKF', 'rBA_EnKF_CMAME'],
                                            ['simulation_output_all_new', 'simulation_output_all_CMAME']):

            out = []

            ens_bb = ensemble.copy()
            ens_bb.t_init = truth['t_obs'][0]

            ens_ba = ens_bb.copy()
            ens_ba.bias = bias.copy()

            DA_kwargs = dict(y_obs=truth['y_obs'].copy(), t_obs=truth['t_obs'].copy(), std_obs=0.1,
                             wash_obs=wash_obs.copy(), wash_t=wash_t.copy())

            for kf in [regularized_filter, 'EnKF']:

                if kf[0] == 'r':
                    ks = np.linspace(0, 5, 6)
                    blank_ens = ens_ba.copy()
                else:
                    ks = [None]
                    blank_ens = ens_bb.copy()

                blank_ens.filter = kf
                for kk in ks:
                    ens = blank_ens.copy()

                    if kf[0] == 'r':
                        ens.regularization_factor = kk

                    filter_ens = dataAssimilation(ens, **DA_kwargs.copy())

                    # Forecast the ensemble further without assimilation
                    Nt_extra = int(filter_ens.t_CR / filter_ens.dt) * 10

                    psi, t = filter_ens.time_integrate(Nt_extra)
                    filter_ens.update_history(psi, t)

                    y = filter_ens.get_observable_hist(Nt_extra)
                    b, t_b = filter_ens.bias.time_integrate(t=t, y=y)
                    filter_ens.bias.update_history(b, t_b)

                    filter_ens.close()

                    out.append(filter_ens.copy())

            save_to_pickle_file(results_dir + name, truth, out, bias, ensemble)

        truth_params = dict()
        for param in ens_ba.params:
            if param == 'nu':
                truth_params[param] = ens_ba.nu_from_ER(ER)
            elif param == 'c2beta':
                truth_params[param] = ens_ba.c2beta_from_ER(ER)
            else:
                truth_params[param] = ens_ba.alpha0[param]

        print_parameter_results(out, true_values=truth_params)
