import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    import TAModels
    import Bias
    from run import main, createESNbias, createEnsemble
    from plotResults import *

    bias_form = 'linear'
    run_loopParams, plot_loopParams = 0, 0
    run_optimal, plot_optimal = 0, 1

    for mm in [50]:
        Ls = np.linspace(10, 100, 10, dtype=int)
        if bias_form == 'time':
            ks = np.linspace(0.25, 4.75, 10)
        else:
            ks = np.linspace(0., 10., 41)
        stds = [.25]

        # %% ========================== SELECT WORKING PATHS ================================= #
        folder = 'results/Rijke_final_{}/'.format(bias_form)
        path_dir = os.path.realpath(__file__).split('main')[0]
        # os.chdir('/mscott/an553/')  # set working directory to mscott

        loopParams_folder = folder + 'm{}/results_loopParams_final/'.format(mm)
        optimal_folder = folder + 'm{}/results_optimal//'.format(mm)
        # %% ============================= SELECT TRUE AND FORECAST MODELS ================================= #
        true_params = {'model': TAModels.Rijke,
                       't_max': 2.5,
                       'beta': 4.2,
                       'tau': 1.4E-3,
                       'manual_bias': bias_form
                       }
        forecast_params = {'model': TAModels.Rijke,
                           't_max': 2.5
                           }
        # ==================================== SELECT FILTER PARAMETERS =================================== #
        filter_params = {'filt': 'rBA_EnKF',  # 'rBA_EnKF' 'EnKF' 'EnSRKF'
                         'm': mm,
                         'est_p': ['beta', 'tau'],
                         'biasType': Bias.ESN,
                         # Define the observation time window
                         't_start': 1.5,  # ensure SS
                         't_stop': 2.0,
                         'kmeas': 20,
                         # Inflation
                         'inflation': 1.002,
                         'start_ensemble_forecast': 1
                         }
        if filter_params['biasType'] is not None and filter_params['biasType'].name == 'ESN':
            train_params = {'model': TAModels.Rijke,
                            'std_a': 0.2,
                            'std_psi': 0.2,
                            'est_p': filter_params['est_p'],
                            'alpha_distr': 'uniform',
                            'ensure_mean': True
                            }
            bias_params = {'N_wash': 50,
                           'upsample': 2,
                           'L': 10,
                           'N_units': 500,
                           'augment_data': True,
                           't_val': 0.02,
                           't_train': 0.5,
                           'train_params': train_params,
                           'tikh_': np.array([1e-16]),
                           'sigin_': [np.log10(1e-5), np.log10(1e-2)],
                           }
            if bias_form == 'time':
                bias_params['t_train'] = 1.5
                filter_params['kmeas'] = 10
        else:
            bias_params = None
        # ================================== CREATE REFERENCE ENSEMBLE =================================
        name = 'reference_Ensemble_m{}_kmeas{}'.format(filter_params['m'], filter_params['kmeas'])
        ensemble, truth, args = createEnsemble(true_params, forecast_params,
                                               filter_params, bias_params,
                                               working_dir=folder, filename=name)

        if run_loopParams:
            # =========================================== RUN LOOP ==========================================
            for std in stds:
                blank_ens = ensemble.copy()
                # Reset std
                blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1),
                                                         std, blank_ens.m, method='normal')
                blank_ens.hist[-1] = blank_ens.psi
                blank_ens.std_psi, blank_ens.std_a = std, std
                std_folder = loopParams_folder + 'std{}/'.format(std)
                for L in Ls:
                    # Reset ESN
                    bias_params['L'] = L
                    filter_params['Bdict'] = createESNbias(*args, bias_param=bias_params)
                    blank_ens.initBias(filter_params['Bdict'])

                    results_folder = std_folder + 'L{}/'.format(L)
                    for k in ks:  # Reset gamma value
                        filter_ens = blank_ens.copy()
                        filter_ens.bias.k = k
                        # Run simulation
                        main(filter_ens, truth, filter_params, results_dir=results_folder, save_=True)
                get_error_metrics(std_folder)
        # -------------------------------------------------------------------------------------------------------------
        if plot_loopParams:
            if not os.path.isdir(loopParams_folder):
                print('results_loopParams not run')
            else:
                figs_dir = path_dir + loopParams_folder
                post_process_loopParams(loopParams_folder, k_plot=(None,), figs_dir=figs_dir)
        # -------------------------------------------------------------------------------------------------------------
        if run_optimal:
            blank_ens = ensemble.copy()
            std = 0.25
            if bias_form == 'linear':
                L, k = 100, 1.75
            elif bias_form == 'periodic':
                L, k = 60, 2.75
            elif bias_form == 'time':
                L, k = 10,  1.25

            # Reset std
            blank_ens.psi = blank_ens.addUncertainty(np.mean(blank_ens.psi, 1), std, blank_ens.m, method='normal')
            blank_ens.hist[-1] = blank_ens.psi
            blank_ens.std_psi, blank_ens.std_a = std, std

            # Run simulation with ESN and bias-aware EnKF -----------------------------
            filter_ens = blank_ens.copy()
            if bias_params is not None:
                bias_params['L'] = L
                Bdict = createESNbias(*args, bias_param=bias_params)
                filter_ens.initBias(Bdict)
            filter_ens.bias.k = k
            main(filter_ens, truth, 'rBA_EnKF', results_dir=optimal_folder, save_=True)

            # run reference solution with bias-blind EnKF -----------------------------
            filter_ens = blank_ens.copy()
            filter_ens.biasType = Bias.NoBias
            filter_ens.initBias()
            main(filter_ens, truth, 'EnKF', results_dir=optimal_folder, save_=True)

        # ========================================================================================================
        if plot_optimal:

            from compress_pickle import load

            files = os.listdir(optimal_folder)
            for file in files:
                if file[0] != 'E':
                    continue
                with open(optimal_folder + file, 'rb') as f:
                    params = pickle.load(f)
                    truth = pickle.load(f)
                    filter_ens = pickle.load(f)
                # filename = '{}results_{}_{}_J'.format(optimal_folder, filter_ens.filt, filter_ens.bias.name)
                # post_process_single(filter_ens, truth, params, filename=filename, mic=5)

                if filter_ens.filt == 'EnKF':
                    filter_ens_BB = filter_ens.copy()
                    print('ok1')
                elif filter_ens.filt == 'rBA_EnKF':
                    filter_ens_BA = filter_ens.copy()
                    print('ok2')

            # load truth
            name_truth = truth['name'].split('_{}'.format(true_params['manual_bias']))[0]
            with open('data/Truth_{}.lzma'.format(name_truth), 'rb') as f:
                truth_ens = load(f)


            # extract history of observables along the tube
            locs = np.linspace(0, filter_ens.L, 10)
            locs_obs = filter_ens.x_mic


            # Bias-aware EnKF sol
            y_BA = filter_ens_BA.getObservableHist(loc=locs)
            y_BA = np.mean(y_BA, -1)
            # Bias-blind EnKF sol
            y_BB = filter_ens_BB.getObservableHist(loc=locs)
            y_BB = np.mean(y_BB, -1)
            # truth
            y_t = truth_ens.getObservableHist(loc=locs).squeeze()
            y_t += .4 * y_t * np.sin((np.expand_dims(truth_ens.hist_t, -1) * np.pi * 2) ** 2)

            y_t = interpolate(truth_ens.hist_t, y_t, filter_ens.hist_t)

            max_v = [np.max(abs(yy)) for yy in [y_t, y_BA, y_BB]]
            max_v = np.max(max_v)

            contours = False
            if contours:
                fig = plt.figure(figsize=[15, 5], layout="constrained")
                sub_figs = fig.subfigures(1, 3)
                norm = colors.Normalize(vmin=-max_v, vmax=max_v)
                cmap = plt.cm.RdBu_r
                plot_times = [t_obs[0]-filter_ens.t_CR, t_obs[-1], t_obs[-1] + filter_ens.t_CR]

                xv, yv = np.meshgrid(locs, [0, 1], indexing='ij')
                for sf, t_sf in zip(sub_figs, plot_times):
                    axs = sf.subplots(3, 1)
                    tj = np.argmin(abs(filter_ens.hist_t - t_sf))
                    for ax, yy in zip(axs, [y_t, y_BA, y_BB]):
                        zz = np.tile(yy[tj], (2, 1))
                        print(xv.shape, yv.shape, zz.shape, yy.shape)
                        im = ax.contourf(xv, yv, zz.T, cmap=cmap, norm=norm)
                        sf.colorbar(im, ax=ax, orientation='horizontal', shrink=1.5)
                        ax.contour(xv, yv, zz.T, cmap=cmap, norm=norm)
                        ax.set(xlabel='$x$ [m]', yticks=[])
                        for xx in locs_obs:
                            ax.plot([xx, xx], [0, 1], 'k--')
                # plt.show()


            # -----------------------

            import matplotlib.animation
            fig = plt.figure(figsize=[10, 5], layout='constrained')
            sub_figs = fig.subfigures(2, 1)
            ax1 = sub_figs[0].subplots()
            ax2 = sub_figs[1].subplots(1, 2)

            t0 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][0] - filter_ens.t_CR/2)))
            t1 = np.argmin(abs(filter_ens.hist_t - (truth['t_obs'][-1] + filter_ens.t_CR)))

            t_gif = filter_ens.hist_t[t0:t1:5]

            # all pressure points
            y_BA = filter_ens_BA.getObservableHist(loc=locs)
            y_BB = filter_ens_BB.getObservableHist(loc=locs)
            y_t, y_BB, y_BA = [interpolate(filter_ens.hist_t, yy, t_gif) for yy in [y_t, y_BB, y_BA]]


            max_v = np.max(abs(y_t))

            # observation points
            y_BB_obs = filter_ens_BB.getObservableHist()
            y_BA_obs = filter_ens_BA.getObservableHist()

            y_BA_obs = y_BA_obs[::filter_ens.bias.upsample] + np.expand_dims(filter_ens_BA.bias.hist, -1)

            y_BA_obs = interpolate(filter_ens_BA.bias.hist_t, y_BA_obs, t_gif)

            y_BB_obs = interpolate(filter_ens.hist_t, y_BB_obs, t_gif)


            def animate(ii):
                ax1.clear()
                ax1.set(ylim=[-max_v, max_v], xlim=[0, 1], title='$t={:.4}$'.format(t_gif[ii]),
                        xlabel='$x$ [m]', ylabel="$p'$ [Pa]")
                ax1.plot(locs, y_t[ii], color='lightgray', linewidth=3)
                for yy, c, l in zip([y_BA[ii], y_BB[ii]], ['lightseagreen', 'orange'], ['--', '-']):
                    y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
                    ax1.plot(locs, y_mean, l, color=c)
                    ax1.fill_between(locs, y_mean+y_std, y_mean-y_std, alpha=0.1, color=c)

                # Plot observables
                if any(abs(t_gif[ii] - truth['t_obs']) < 1E-6):
                    jj = np.argmin(abs(t_gif[ii] - truth['t_obs']))
                    ax1.plot(locs_obs, truth['p_obs'][jj], 'o', color='red', markersize=6,
                             markerfacecolor=None, markeredgewidth=2)
                    for yy, c in zip([y_BA_obs[ii], y_BB_obs[ii]], ['lightseagreen', 'orange']):
                        y_mean, y_std = np.mean(yy, -1), np.std(yy, -1)
                        ax1.plot(locs_obs, y_mean, 'x', color=c, markeredgewidth=2)
                    if jj < 10 or jj > len(truth['t_obs'])-10:
                        plt.pause(.5)
                        # plt.resume()



                # for p, c in zip(filter_ens.est_p, cs):
                #     mean_p = hist_mean[:, ii].squeeze() / reference_p[p]
                #     std = np.std(hist[:, ii] / reference_p[p], axis=1)
                #
                #     max_p, min_p = max(max_p, max(mean_p)), min(min_p, min(mean_p))
                #     if p in ['C1', 'C2']:
                #         params_ax.plot(hist_t, mean_p, color=c, label='$' + p + '/' + p + superscript)
                #     else:
                #         params_ax.plot(hist_t, mean_p, color=c, label='$\\' + p + '/\\' + p + superscript)

            ani = mpl.animation.FuncAnimation(fig, animate, frames=len(t_gif), interval=10, repeat=False)

            plt.show()
            # ani.save('file2.mp4')
#



