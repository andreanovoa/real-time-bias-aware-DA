from src.plotResults import *
from src.Util import load_from_pickle_file
from src.models_physical import Annular
import matplotlib.pyplot as plt

def run_plotting_annular_results(m, rho,
                                 plot_best_stats=1,
                                 plot_all_stats=1,
                                 plot_nu_beta_ERs=1,
                                 plot_rest_of_params_ERs=1,
                                 plot_experimental_data=1,
                                 plot_biases=1,
                                 output_filename='simulation_output_all_new',  # 'simulation_output_all_CMAME',
                                 save_figs=True,
                                 figs_format='.png',
                                 dpi=200,
                                 ERs=0.4875 + np.arange(0, 4) * 0.025,
                                 folder='results/annular/',
                                 suffix=''
                                 ):
    def set_results_dir(_ER):
        if rho is not None:
            return folder + f'ER{_ER}/m{m}/rho{rho}{suffix}/'
        else:
            return folder + f'ER{_ER}/m{m}/'

    def get_reference(_window):
        _jjs = [np.argmin(abs(truth['t'] - tt)) for tt in _window]
        return [truth[key][_jjs[0]:_jjs[1]] for key in ['t', 'y_true', 'y_raw']]

    def select_best_case():
        _best_case = None
        _best_rmse = 1000

        for _ens in results[:-1]:
            _y_est = _ens.get_observable_hist()
            _y_mean = np.mean(_y_est, axis=-1)
            _y_mean = interpolate(_ens.hist_t, _y_mean, t_ref)

            _Rm = np.sqrt(np.sum((y_ref - _y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
            _Rm = np.mean(_Rm)

            _bias = np.mean(_ens.bias.get_bias(_ens.bias.hist), axis=-1)
            _b_mean = interpolate(_ens.bias.hist_t, _bias, t_ref)
            _y_mean += _b_mean

            _Rm_u = np.sqrt(np.sum((y_ref - _y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
            _Rm_u = np.mean(_Rm_u)

            _Rm = (_Rm + _Rm_u) / 2.

            if _Rm < _best_rmse:
                _best_rmse = _Rm.copy()
                _best_case = _ens.copy()
        return _best_case, _best_rmse

    for ER in ERs:

        results_dir = set_results_dir(ER)

        params_scales = {'kappa': 1e-4, 'epsilon': 1e-3, 'theta_b': np.pi / 180.,
                         'theta_e': np.pi / 180., 'omega': np.pi * 2}

        print(f'...plotting simulations in folder {results_dir}')
        if plot_biases:

            truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + output_filename)

            print(bias)
            print(results[0].bias)

            long_window = [truth['t_obs'][-1], results[0].hist_t[-1]]
            short_window = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]

            for window, lbl in zip([long_window, short_window], ['long_window', 'short_window']):
                t_ref, y_ref, y_raw = get_reference(window)
                bias.t_init = truth['t_obs'][0] - 2 * truth['dt_obs']
                i1 = np.argmin(abs(bias.t_init - truth['t']))
                i0 = i1 - bias.N_wash * bias.upsample
                truth['wash_obs'] = truth['y_raw'][i0:i1 + 1:bias.upsample]
                truth['wash_t'] = truth['t'][i0:i1 + 1:bias.upsample]

                best_case, best_rms = select_best_case()

                fig, axs = plt.subplots(figsize=(10, 5), nrows=4, ncols=2, sharey='col',
                                        sharex=True, layout='constrained')

                bias = best_case.bias
                bias_hist = interpolate(bias.hist_t, bias.hist, t_ref)

                model_bias = bias.get_bias(state=bias_hist)
                innovations = bias_hist[:, bias.observed_idx]
                sensor_bias = innovations - model_bias

                y = np.mean(best_case.get_observable_hist(), axis=-1)
                actual_innovation = y_raw - interpolate(best_case.hist_t, y, t_ref)
                actual_model_bias = y_ref - interpolate(best_case.hist_t, y, t_ref)
                actual_sensor_bias = y_raw - y_ref

                for qq, ax in enumerate(axs):
                    ax[0].plot(t_ref, actual_model_bias[:, qq], c='k', lw=1)
                    ax[0].plot(t_ref, model_bias[:, qq], c='orchid', linestyle='--', dashes=(8, 1), lw=1.5)
                    ax[1].plot(t_ref, actual_sensor_bias[:, qq], lw=.5, c='silver')
                    ax[1].axhline(np.mean(actual_sensor_bias[:, qq]), ls='-', c='k', lw=1)
                    ax[1].plot(t_ref, sensor_bias[:, qq], c='orangered', lw=1.5, ls='--', dashes=(8, 5))

                axs[0, 0].set(xlim=[t_ref[0], t_ref[-1]])
                axs[-1, 0].set(xlabel='$t$ [s]')
                axs[-1, 1].set(xlabel='$t$ [s]')
                axs[0, 0].set(title='Model bias')
                axs[0, 0].legend(['$d^\\dagger - \\bar{y}$', '$b^\\mathrm{f}$'])
                axs[0, 1].legend(['$d^\\dagger - d_\\mathrm{raw}$',
                                  '$\\langle d^\\dagger - d_\\mathrm{raw}\\rangle$', '$b_{d}^\\mathrm{f}$'])
                axs[0, 1].set(title='Sensor bias')
                if save_figs:
                    figs_dir = results_dir + 'figs/'
                    os.makedirs(figs_dir, exist_ok=True)
                    plt.savefig(figs_dir + f'biases_{lbl}{figs_format}', dpi=dpi)
                    plt.close()

        if plot_experimental_data:
            truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + output_filename)

            fig, axs_all = plt.subplots(nrows=3, ncols=2, figsize=(10, 9), layout='constrained', width_ratios=[3, 1])

            t_ref, y_ref, y_raw = get_reference([truth['t_obs'][0], truth['t_obs'][-1]])

            R_truth = np.sqrt(np.sum((y_ref - y_raw) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))

            colors = ['dodgerblue', 'tab:green', 'crimson', 'orange']

            for qq in range(results[0].Nq):
                for ax, yy, lw in zip(axs_all, [y_raw, y_ref], [0.7, 1.5]):
                    ax[0].plot(t_ref, yy[:, qq], color=colors[qq], lw=lw)

                    # f, PSD = fun_PSD(truth['dt'], yy[:, qq])
                    # ax[1].semilogy(f, PSD[0], color=colors[qq])

                    ax[1].axhline(np.mean(yy[:, qq]), ls='-', color=colors[qq], lw=0.8)
                    ax[1].hist(yy[:, qq], color=colors[qq], orientation='horizontal', histtype='stepfilled', bins=40,
                               alpha=0.3)
                    ax[1].hist(yy[:, qq], color=colors[qq], orientation='horizontal', histtype='step', bins=40)

            leg = results[0].obs_labels
            axs_all[1, 0].legend(leg, ncols=2)

            axs_all[-1, 0].hist(R_truth, histtype='step', color='k', lw=1.5, bins=20, density=True,
                                orientation='vertical')
            axs_all[-1, 0].axvline(np.mean(R_truth), c='k', lw=1, ls='--')
            axs_all[0, 0].set(ylabel='Raw', xlim=[t_ref[0], t_ref[0] + 0.02])
            axs_all[1, 0].set(ylabel='Truth', xlim=[t_ref[0], t_ref[0] + 0.02])
            axs_all[2, 0].set(ylabel='RMS error', xlim=[0, 2.5])

            ylims = axs_all[0, 0].get_ylim()
            axs_all[1, 0].set(ylim=ylims)
            ylims = axs_all[0, 1].get_ylim()
            axs_all[1, 1].set(ylim=ylims)
            ylims = axs_all[1, 1].get_xlim()
            axs_all[0, 1].set(xlim=ylims)

            if save_figs:
                figs_dir = results_dir + 'figs/'
                os.makedirs(figs_dir, exist_ok=True)
                plt.savefig(figs_dir + f'experimental_data{figs_format}', dpi=dpi)

                plt.close()

        if plot_best_stats or plot_all_stats:
            truth_og, results, bias, blank_ensemble = load_from_pickle_file(results_dir + output_filename)
            truth = truth_og.copy()

            long_window = [truth['t_obs'][-1], results[0].hist_t[-1]]
            short_window = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]

            compare_cases = None
            for window, lbl in zip([long_window, short_window], ['long_window', 'short_window']):

                t_ref, y_ref, y_raw = get_reference(window)
                bias.t_init = truth['t_obs'][0] - 2 * truth['dt_obs']

                i1 = np.argmin(abs(bias.t_init - truth['t']))
                i0 = i1 - bias.N_wash * bias.upsample

                truth['wash_obs'] = truth['y_raw'][i0:i1 + 1:bias.upsample]
                truth['wash_t'] = truth['t'][i0:i1 + 1:bias.upsample]
                best_case, best_rms = select_best_case()
                compare_cases = [best_case, results[-1]]

                for cases, name, plot_flag in zip([compare_cases, results], ['_best', '_all'],
                                                  [plot_best_stats, plot_all_stats]):
                    if plot_flag:
                        plot_states_PDF(ensembles=cases, truth=truth_og, window=window)
                        plt.gcf().set_size_inches(8, 3 * len(cases))
                        if save_figs:
                            figs_dir = results_dir + 'figs/'
                            os.makedirs(figs_dir, exist_ok=True)
                            plt.savefig(figs_dir + f'PDF_{lbl}{name}{figs_format}', dpi=dpi)
                            plt.close()

            for cases, name, plot_flag in zip([compare_cases, results], ['_best', '_all'],
                                              [plot_best_stats, plot_all_stats]):
                if plot_flag:
                    plot_RMS_pdf(ensembles=cases, truth=truth)
                    plt.gcf().set_size_inches(8, 3 * len(cases))
                    if save_figs:
                        figs_dir = results_dir + 'figs/'
                        os.makedirs(figs_dir, exist_ok=True)
                        plt.savefig(figs_dir + f'RMS{name}{figs_format}', dpi=dpi)
                        plt.close()

            if plot_best_stats:
                for case, lbl in zip(compare_cases, ['rEnKF', 'EnKF']):
                    plot_timeseries(filter_ens=case, truth=truth, max_time=case.hist_t[-1])
                    plt.gcf().set_size_inches(8, 8)
                    if save_figs:
                        figs_dir = results_dir + 'figs/'
                        os.makedirs(figs_dir, exist_ok=True)
                        plt.savefig(figs_dir + f'Timeseries_{lbl}{figs_format}', dpi=dpi)
                        plt.close()

                    plot_covariance(case)
                    plt.gcf().set_size_inches(8, 5)
                    if save_figs:
                        figs_dir = results_dir + 'figs/'
                        os.makedirs(figs_dir, exist_ok=True)
                        plt.savefig(figs_dir + f'Covariances_{lbl}{figs_format}', dpi=dpi)
                        plt.close()

                plot_parameters(ensembles=compare_cases, truth=truth, reference_p=params_scales, cmap='tab10')
                plt.gcf().set_size_inches(8, 5)

                if save_figs:
                    figs_dir = results_dir + 'figs/'
                    os.makedirs(figs_dir, exist_ok=True)
                    plt.savefig(figs_dir + f'Parameters{figs_format}', dpi=dpi)
                    plt.close()
        print('\t.... done')

    if plot_nu_beta_ERs or plot_rest_of_params_ERs:

        print(f'...plotting parameters for all ERs in folder {folder} for m={m} and rho={rho}')
        if plot_rest_of_params_ERs:

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True)
            axs = axs.ravel()

            ms = [R_to_size(xx) for xx in [5, 2, .9, .6, .4, .2]]
            for mm in ms:
                axs[0].plot(0, 0, 'o', ms=mm)
            axs[0].legend(['{}'.format(xx) for xx in [5, 2, .9, .6, .4, .2]])

            scs = 6
            cs = np.arange(0, 5) * scs
            cs = cs[::-1]

            for ER in ERs:
                results_dir = set_results_dir(ER)

                xval = ER.copy()

                truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + output_filename)

                norm = mpl.colors.Normalize(vmin=0, vmax=7)
                cmaps = categorical_cmap(5, 6, cmap="tab10")
                t_ref = truth['t']
                tts = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]
                jjs = [np.argmin(abs(t_ref - tt)) for tt in tts]
                t_ref, y_ref = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true']]

                for kk, ens in enumerate(results):

                    y_est = ens.get_observable_hist()
                    y_mean = np.mean(y_est, axis=-1)
                    y_mean = interpolate(ens.hist_t, y_mean, t_ref)

                    Rm = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
                    Rm = np.mean(Rm)

                    bias = np.mean(ens.bias.get_bias(ens.bias.hist), axis=-1)
                    b_mean = interpolate(ens.bias.hist_t, bias, t_ref)
                    y_mean += b_mean
                    Rm_u = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
                    Rm_u = np.mean(Rm_u)
                    Rm = (Rm + Rm_u) / 2.
                    ms = R_to_size(Rm)
                    params = ['kappa', 'epsilon', 'theta_b', 'theta_e', 'omega']
                    mks = ['o', 'o', 'o', '^', 'o']
                    scales = [1e-4, 1e-2, np.pi / 180., np.pi / 180., 2 * np.pi]
                    scale_lbl = ['$10^{-4}$', '$10^{-2}$', '', Annular.alpha_labels['theta_b'], '$2 \\pi$']
                    axs_id = [0, 2, 3, 3, 1]

                    for ax, param, mk, cat, scale, lbl in zip(axs_id, params, mks, cs, scales, scale_lbl):
                        ax = axs[ax]
                        param_idx = ens.est_a.index(param)
                        params = ens.get_current_state[ens.Nphi + param_idx]
                        params /= scale
                        # c = cmap.to_rgba(ens.regularization_factor)
                        c = cmaps[int(ens.regularization_factor + cat)]
                        if ens != results[-1]:
                            if param[-1] == 'e':
                                xval = ER - 0.002 * (kk - 3)
                            else:
                                xval = ER + 0.002 * (kk - 3)
                            ax.errorbar(xval, np.mean(params), np.std(params), alpha=0.7, marker=mk, c=c, ms=ms, capsize=2)
                        else:
                            ax.errorbar(ER, np.mean(params), np.std(params), alpha=0.7, marker=mk, c='k', ms=ms, capsize=2)
                            ax.axhline(getattr(Annular, param) / scale, ls='--', c=c)
                            ax.set_xticks(ERs, ERs)
                            title = f'{getattr(Annular, param)} {lbl}'
                            ax.set(title=title)
                        ax.set(xlim=[ERs[0] - .01, ERs[-1] + .01])
            # plt.show()

            if save_figs:
                figs_dir = folder + f'figs_m{m}_rho{rho}{suffix}/'
                os.makedirs(figs_dir, exist_ok=True)
                plt.savefig(figs_dir + f'plot_rest_of_params_ERs{figs_format}', dpi=dpi)

                plt.close()

        if plot_nu_beta_ERs:
            plt.figure(figsize=(10, 5))
            lbls = []
            for ER in ERs:
                results_dir = set_results_dir(ER)

                truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + output_filename)

                norm = mpl.colors.Normalize(vmin=0, vmax=7)
                cmap_nu = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greens_r)
                cmap_beta = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Oranges_r)

                t_ref = truth['t']

                tts = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]

                jjs = [np.argmin(abs(t_ref - tt)) for tt in tts]

                t_ref, y_ref = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true']]

                plt.plot([ER - 0.01, ER + 0.01], [Annular.c2beta_from_ER(ER), Annular.c2beta_from_ER(ER)], ls='--', c='C1')
                plt.plot([ER - 0.01, ER + 0.01], [Annular.nu_from_ER(ER), Annular.nu_from_ER(ER)], ls='--', c='C2')

                for kk, ens in enumerate(results):

                    y_est = ens.get_observable_hist()
                    y_mean = np.mean(y_est, axis=-1)
                    y_mean = interpolate(ens.hist_t, y_mean, t_ref)
                    Rm = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
                    Rm = np.mean(Rm)

                    bias = np.mean(ens.bias.get_bias(ens.bias.hist), axis=-1)
                    b_mean = interpolate(ens.bias.hist_t, bias, t_ref)
                    y_mean += b_mean

                    Rm_u = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
                    Rm_u = np.mean(Rm_u)

                    Rm = (Rm + Rm_u) / 2.

                    nu_idx, c2beta_idx = [ens.est_a.index(key) for key in ['nu', 'c2beta']]
                    nus = ens.get_current_state[ens.Nphi + nu_idx]
                    c2betas = ens.get_current_state[ens.Nphi + c2beta_idx]

                    if ens != results[-1]:
                        c_b = cmap_beta.to_rgba(ens.regularization_factor)
                        c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                        ms = R_to_size(Rm)

                        xval = ER + 0.002 * (kk - 3)
                        xval_b = ER + 0.002 * (kk - 2)
                        plt.errorbar(xval, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c=c_nu, ms=ms, capsize=2)
                        plt.errorbar(xval_b, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c=c_b, ms=ms,
                                     capsize=2)

                    else:
                        c_b = cmap_beta.to_rgba(ens.regularization_factor)
                        c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                        ms = R_to_size(Rm)

                        plt.errorbar(ER - 0.01, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c='k', ms=ms, capsize=2)
                        plt.errorbar(ER - 0.01, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c='k', ms=ms,
                                     capsize=2)

                    lbls.append('{:.3}'.format(Rm))
                    lbls.append('{:.3}'.format(Rm))

            plt.legend(lbls, loc='upper left', bbox_to_anchor=(1.1, 1.1), ncol=3, fontsize='xx-small')
            plt.xticks(ERs, ERs)
            plt.xlim([ERs[0] - .015, ERs[-1] + .015])
            plt.xlabel('Equivalence ratio $\\Phi$')
            plt.colorbar(cmap_nu, ax=plt.gca(), shrink=.8)
            plt.colorbar(cmap_beta, ax=plt.gca(), shrink=.8, label='$\\gamma c_2\\beta \\nu$')

            if save_figs:
                figs_dir = folder + f'figs_m{m}_rho{rho}{suffix}/'
                os.makedirs(figs_dir, exist_ok=True)
                plt.savefig(figs_dir + f'plot_nu_beta_ERs{figs_format}', dpi=dpi)

                plt.close()

        print('\t.... done')

def R_to_size(R):
    if R > 3:
        return 2
    elif R > 1:
        return 3
    elif R > .7:
        return 5
    elif R > .5:
        return 8
    elif R > 0.3:
        return 10
    else:
        return 13


def categorical_cmap(nc, nsc, cmap="Set2", continuous=False):
    # number of categories(nc) and the number of subcategories(nsc)
    # and returns a colormap with nc * nsc different colors, where for
    # each category there are nsc colors of same hue.

    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    # return matplotlib.colors.ListedColormap(cols)
    return cols


if __name__ == '__main__':
    run_plotting_annular_results(m=10, rho=1.001,
                                 suffix='_long', folder='../results/annular/',
                                 plot_nu_beta_ERs=0,
                                 plot_rest_of_params_ERs=0,
                                 plot_experimental_data=1,
                                 )
