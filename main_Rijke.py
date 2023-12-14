
from default_parameters.rijke import *

if __name__ == '__main__':

    bias_form = 'linear'  # linear, periodic, time

    true_params['manual_bias'] = bias_form
    run_multiple_ensemble_sizes = False

    run_loopParams, plot_loopParams = 1, 1
    run_optimal, plot_optimal = 0, 0

    # %% ========================== SELECT WORKING PATHS ================================= #
    folder = 'results/new_arch/Rijke_{}/'.format(bias_form)

    path_dir = os.path.realpath(__file__).split('main')[0]
    os.chdir('/mscott/an553/')  # set working directory to mscott

    # %% ======================= SELECT RANGE OF PARAMETERS ============================== #
    ms = [50]
    Ls = np.linspace(10, 100, 10, dtype=int)
    ks = np.linspace(0., 10., 41)

    # Special cases
    if bias_form == 'time':
        ks = np.linspace(0.25, 4.75, 10)
    if run_multiple_ensemble_sizes:
        ms = [10, 50, 80]

    if bias_form == 'time':
        bias_params['t_train'] = 1.5
        filter_params['dt_obs'] = 10
    # %% ============================ RUN SIMULATIONS ================================= #

    truth_og = create_truth(true_params, filter_params)

    for m in ms:  # LOOP OVER ENSEMBLE SIZES
        loopParams_folder = folder + 'm{}/results_loopParams/'.format(m)
        optimal_folder = folder + 'm{}/results_optimal/'.format(m)

        filter_params['m'] = m

        #  CREATE REFERENCE ENSEMBLE
        ensemble = create_ensemble(forecast_params, filter_params)

        # LOOP OVER Ls AND ks
        run_Lk_loop(ensemble, truth_og, bias_params, Ls=Ls, ks=ks, folder=loopParams_folder)

        if plot_loopParams:
            if not os.path.isdir(loopParams_folder):
                raise ValueError('results_loopParams not run')
            figs_dir = path_dir + loopParams_folder
            post_process_loopParams(loopParams_folder, k_plot=(None,), figs_dir=figs_dir)

        # -------------------------------------------------------------------------------------------------------------
        if run_optimal:
            filter_params['m'] = m
            truth_og = create_truth(true_params, filter_params)

            # --------------------- REFERENCE BIAS-BLIND SOLUTION ------------------
            filter_ens = main(ensemble, truth_og)
            save_simulation(filter_ens, truth_og, results_dir=optimal_folder)

            # ------------------ BIAS-AWARE SIMULATION  -------------------
            # These are manually defined after running the loops
            if bias_form == 'linear':
                L, k = 100, 1.75
            elif bias_form == 'periodic':
                L, k = 60, 2.75
            elif bias_form == 'time':
                L, k = 10, 1.25
            else:
                raise ValueError("Select 'linear', 'periodic' or 'time' bias form")

            blank_ens = ensemble.copy()
            truth = truth_og.copy()
            # Set ESN model
            bias_params['L'] = L
            bias_name = 'ESN_L{}'.format(bias_params['L'])
            create_bias_model(blank_ens, truth, bias_params, bias_name,
                              bias_model_folder=loopParams_folder, plot_train_data=False)

            results_folder = loopParams_folder + 'L{}/'.format(L)
            for k in ks:
                filter_ens = blank_ens.copy()
                filter_ens.regularization_factor = k  # Reset gamma value
                # ------------------ RUN & SAVE SIMULATION  -------------------
                filter_ens = main(filter_ens, truth)
                save_simulation(filter_ens, truth, results_dir=optimal_folder)

            # Run reference solution with bias-blind EnKF -----------------------------
            filter_ens = blank_ens.copy()

        # -------------------------------------------------------------------------------------------------------------
        if plot_optimal:
            if not os.path.isdir(optimal_folder):
                raise ValueError('results_loopParams not run')
            figs_dir = path_dir + optimal_folder
            plot_Rijke_animation(optimal_folder, figs_dir)
