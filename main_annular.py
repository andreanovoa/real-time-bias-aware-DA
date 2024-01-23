


from default_parameters.annular import *

plt.rcParams['text.usetex'] = True # enable LaTeX rendering

if __name__ == '__main__':
    # ========================== CREATE TRUTH AND ENSEMBLE  =================================
    truth = create_truth(true_params, filter_params, post_processed=False)
    forecast_params['dt'] = truth['dt']

    ensemble = create_ensemble(forecast_params, filter_params)

    # %% ========================== SELECT BIAS MODEL ======================================= #
    ESN_name = 'ESN{}_L{}_Nw{}_{}1'.format(bias_params['N_units'], bias_params['L'],
                                           bias_params['N_wash'], truth['name_bias'])

    filter_ens = ensemble.copy()
    create_bias_model(filter_ens, truth, bias_params, ESN_name, bias_model_folder=folder)

    # %% ========================== RUN SIMULATION ====================================== #
    filter_ens = main(filter_ens, truth)

    # Save simulation
    # save_simulation(filter_ens, truth, extra_parameters=None, results_dir=out_dir)

    # %% ========================== PLOT RESULTS ======================================= #

    reference_params = dict(theta_b=0.63,
                            theta_e=0.66,
                            omega=1090*2*np.pi,
                            epsilon=2.3E-3,
                            nu=Annular.nu_from_ER(ER),
                            beta_c2=Annular.beta_c2_from_ER(ER),
                            kappa=1.2E-4)  # values in Matlab codes
    if filter_ens.est_a:
        plot_parameters(filter_ens, truth, reference_p=reference_params, twin=True)

    # post_process_single(filter_ens, truth, reference_p=Annular.defaults)
    plot_timeseries(filter_ens, truth, reference_y=1.,
                    plot_states=True, plot_bias=True)

    # Save results and plot
    save_properties = ['m', 'N_wash', 'L', 'regularization_factor', 'N_units', 'dt_obs']
    filename = ''
    for key in save_properties:
        if len(key.split('_')) > 2:
            filename += ''.join([ks[0] for ks in key.split('_')])
        elif len(key) > 2:
            filename += key[:3]
        else:
            filename += key
        try:
            filename += str(getattr(filter_ens.bias, key))
        except:
            filename += str(getattr(filter_ens, key))

        if key != save_properties[-1]:
            filename += '_'

    pdf = plt_pdf.PdfPages(figs_dir + filename + '.pdf')
    for fig_num in plt.get_fignums():
        current_fig = plt.figure(fig_num)
        pdf.savefig(current_fig)
        # current_fig.savefig(filename + str(fignum))
    pdf.close()
