from essentials.plotResults import *

from essentials.Util import load_from_pickle_file
from scipy.io import savemat
from essentials.physical_models import Annular
import matplotlib.pyplot as plt
import numpy as np

ERs = 0.4875 + np.arange(0, 4) * 0.025
ER = ERs[2]

m = 20


folder = 'results/'

plot_nu_beta_ERs = 0
plot_timeseries_windows = 1
plot_rest_of_params_ERs = 0
plot_experimental_data = 0
plot_biases = 0

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


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
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


if plot_biases:
    results_dir = folder + 'ER{}/m{}/'.format(ER, m)

    truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')

    t_ref = truth['t']
    window = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]
    jjs = [np.argmin(abs(t_ref - tt)) for tt in window]
    t_ref, y_ref, y_raw = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true', 'y_raw']]


    bias.t_init = truth['t_obs'][0] - 2 * truth['dt_obs']

    i1 = np.argmin(abs(bias.t_init - truth['t']))
    i0 = i1 - bias.N_wash * bias.upsample

    truth['wash_obs'] = truth['y_raw'][i0:i1 + 1:bias.upsample]
    truth['wash_t'] = truth['t'][i0:i1 + 1:bias.upsample]

    best_case = None
    best_rmse = 1000

    for ens in results[:-1]:
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

        if Rm < best_rmse:
            best_rmse = Rm.copy()
            best_case = ens.copy()

    params_scales = {'kappa': 1e-4, 'epsilon': 1e-2, 'theta_b': np.pi / 180.,
                     'theta_e': np.pi / 180., 'omega': np.pi * 2}



    fig, axs = plt.subplots(figsize=(10, 5), nrows=4, ncols=2, sharey='col', sharex=True, layout='constrained')

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
        ax[0].plot(t_ref, actual_model_bias[:, qq],  c='k', lw=1)
        ax[0].plot(t_ref, model_bias[:, qq], c='orchid', linestyle='--', dashes=(8, 1), lw=1.5)
        ax[1].plot(t_ref, actual_sensor_bias[:, qq], lw=.5, c='silver')
        ax[1].axhline(np.mean(actual_sensor_bias[:, qq]), ls='-', c='k', lw=1)
        ax[1].plot(t_ref, sensor_bias[:, qq], c='orangered', lw=1.5, ls='--', dashes=(8, 5))

    axs[0,0].set(xlim=[t_ref[0], t_ref[-1]])
    axs[-1, 0].set(xlabel='$t$ [s]')
    axs[-1, 1].set(xlabel='$t$ [s]')
    axs[0, 0].set(title='Model bias')
    axs[0, 0].legend(['$d^\\dagger - \\bar{y}$', '$b^\\mathrm{f}$'])
    axs[0, 1].legend(['$d^\\dagger - d_\\mathrm{raw}$',
                      '$\\langle d^\\dagger - d_\\mathrm{raw}\\rangle$',  '$b_{d}^\\mathrm{f}$'])
    axs[0, 1].set(title='Sensor bias')
    plt.show()


if plot_experimental_data:
    results_dir = folder + 'ER{}/m{}/'.format(ER, m)
    truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')


    fig, axs_all = plt.subplots(nrows=3, ncols=2, figsize=(10, 9), layout='constrained',width_ratios=[3,1])

    tts = [truth['t_obs'][0], truth['t_obs'][-1]]

    j0, j1 = [np.argmin(abs(truth['t'] - tt)) for tt in tts]

    t_ref, y_ref, y_raw = [truth[key][j0:j1] for key in ['t', 'y_true', 'y_raw']]


    R_truth = np.sqrt(np.sum((y_ref - y_raw) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))

    colors = ['dodgerblue', 'tab:green', 'crimson', 'orange']


    for qq in range(results[0].Nq):
        for ax, yy,lw in zip(axs_all, [y_raw, y_ref], [0.7, 1.5]):
            ax[0].plot(t_ref, yy[:, qq], color=colors[qq], lw=lw)
        # axs_all[1,0].plot(t_ref, y_ref[:, qq], color=colors[qq], lw=1.5)

            # f, PSD = fun_PSD(truth['dt'], yy[:, qq])
            # ax[1].semilogy(f, PSD[0], color=colors[qq])

            ax[1].axhline(np.mean(yy[:, qq]), ls='-', color=colors[qq], lw=0.8)
            ax[1].hist(yy[:, qq], color=colors[qq], orientation='horizontal', histtype='stepfilled',bins=40, alpha=0.3)
            ax[1].hist(yy[:, qq], color=colors[qq], orientation='horizontal', histtype='step', bins=40)

    leg = results[0].obs_labels
    axs_all[1,0].legend(leg, ncols=2)

    axs_all[-1,0].hist(R_truth, histtype='step', color='k', lw=1.5, bins=20, density=True, orientation='vertical')
    axs_all[-1,0].axvline(np.mean(R_truth), c='k', lw=1, ls='--')
    axs_all[0,0].set(ylabel='Raw', xlim=[t_ref[0], t_ref[0] + 0.02])
    axs_all[1,0].set(ylabel='Truth', xlim=[t_ref[0],  t_ref[0] + 0.02])
    axs_all[2,0].set(ylabel='RMS error', xlim=[0,2.5])

    ylims = axs_all[0,0].get_ylim()
    axs_all[1, 0].set(ylim=ylims)
    ylims = axs_all[0,1].get_ylim()
    axs_all[1,1].set(ylim=ylims)
    ylims = axs_all[1,1].get_xlim()
    axs_all[0,1].set(xlim=ylims)


    plt.show()


if plot_timeseries_windows:
    results_dir = folder + 'ER{}/m{}/'.format(ER, m)

    truth_og, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')
    truth = truth_og.copy()

    t_ref = truth['t']
    window = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]
    jjs = [np.argmin(abs(t_ref - tt)) for tt in window]
    t_ref, y_ref, y_raw = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true', 'y_raw']]


    bias.t_init = truth['t_obs'][0] - 2 * truth['dt_obs']

    i1 = np.argmin(abs(bias.t_init - truth['t']))
    i0 = i1 - bias.N_wash * bias.upsample

    truth['wash_obs'] = truth['y_raw'][i0:i1 + 1:bias.upsample]
    truth['wash_t'] = truth['t'][i0:i1 + 1:bias.upsample]

    best_case = None
    best_rmse = 1000

    for ens in results[:-1]:
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

        if Rm < best_rmse:
            best_rmse = Rm.copy()
            best_case = ens.copy()

    #
    # params_scales = {'kappa': 1e-4, 'epsilon': 1e-3, 'theta_b': np.pi / 180.,
    #                  'theta_e': np.pi / 180., 'omega': np.pi * 2}
    #
    # ref = {'nu': Annular.nu_from_ER(ER),
    #        'c2beta': Annular.c2beta_from_ER(ER)
    #        }
    # for param in best_case.est_a:
    #     if param not in ['nu', 'c2beta']:
    #         ref[param] = Annular.defaults[param]
    # print(ER)
    # pppp = best_case.Nphi
    # psi = best_case.get_current_state
    # psi_bb = results[-1].get_current_state
    #
    # for param in best_case.est_a:
    #     if param in params_scales:
    #         scale = params_scales[param]
    #     else:
    #         scale = 1.
    #     aaa = psi[pppp] / scale
    #     mean, std = np.mean(aaa), np.std(aaa)
    #     print('\t\t {:.2f}'.format(ref[param]/ scale))
    #     print('{} \t {:.2f} pm {:.2f}'.format(param, mean, std))
    #     aaa = psi_bb[pppp] / scale
    #     mean, std = np.mean(aaa), np.std(aaa)
    #     print('\t\t {:.2f} pm {:.2f}'.format(mean, std))
    #     pppp += 1

    # pdf_file = plt_pdf.PdfPages(folder + 'timeseries_results_ER{}.pdf'.format(ER))
    #
    # def save_fig_to_pdf(fig, file):
    #     # file.savefig(fig)
    #     plt.close(fig)
    #
    # # for ens in [best_case, results[-1]]:
    # #     plot_timeseries(ens, truth)
    # #     save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    # #     plot_parameters(ens, truth, reference_p=params_scales)
    # #     save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    #
    #
    #
    #
    # plot_parameters([best_case, results[-1]], truth, reference_p=params_scales)
    #
    # save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    #
    # plot_RMS_pdf(ensembles=[results[-1], best_case], truth=truth)
    # plt.show()
    #
    # save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    plot_states_PDF(ensembles=[results[-1], best_case], truth=truth_og, window=window)
    plt.show()
    #
    #
    #
    # save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    #
    # pdf_file.close()


if plot_rest_of_params_ERs:

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True)
    axs = axs.ravel()
    lbls = []

    ms = [R_to_size(xx) for xx in [5, 2, .9, .6, .4, .2]]
    for mm in ms:
        axs[0].plot(0, 0, 'o', ms=mm)
    axs[0].legend(['{}'.format(xx) for xx in [5, 2, .9, .6, .4, .2]])

    scs = 6
    cs = np.arange(0, 5) * scs
    cs = cs[::-1]

    for ER in ERs:
        results_dir = folder + 'ER{}/m{}/'.format(ER, m)

        xval = ER.copy()

        truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')

        norm = colors.Normalize(vmin=0, vmax=7)
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
            scales = [1e-4, 1e-2, np.pi/180., np.pi/180., 2*np.pi]
            scale_lbl = ['$10^{-4}$', '$10^{-2}$', '', Annular.params_labels['theta_b'], '$2 \\pi$']
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
                    ax.axhline(Annular.defaults[param]/ scale, ls='--', c=c)
                    ax.set_xticks(ERs, ERs)
                    title = Annular.params_labels[param] + '  ' + lbl
                    ax.set(title=title)
                ax.set(xlim =[ERs[0] - .01, ERs[-1] + .01])

    plt.show()


if plot_nu_beta_ERs:

    plt.figure(figsize=(10,5))

    lbls = []

    for ER in ERs:
        results_dir = folder + 'ER{}/m{}/'.format(ER, m)

        truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir+'simulation_output_all')

        norm = colors.Normalize(vmin=0, vmax=7)
        cmap_nu = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greens_r)
        cmap_beta = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Oranges_r)

        t_ref = truth['t']

        tts = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]

        jjs = [np.argmin(abs(t_ref - tt)) for tt in tts]

        t_ref, y_ref = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true']]

        plt.plot([ER- 0.01, ER + 0.01], [Annular.c2beta_from_ER(ER), Annular.c2beta_from_ER(ER)], ls='--', c='C1')
        plt.plot([ER- 0.01, ER + 0.01], [Annular.nu_from_ER(ER), Annular.nu_from_ER(ER)] , ls='--', c='C2')


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

                xval = ER - 0.002 * (kk - 3)
                xval_b = ER + 0.002 * (kk - 2)
                plt.errorbar(xval, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c=c_nu, ms=ms, capsize=2)
                plt.errorbar(xval_b, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c=c_b, ms=ms, capsize=2)

            else:
                c_b = cmap_beta.to_rgba(ens.regularization_factor)
                c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                ms = R_to_size(Rm)

                plt.errorbar(ER-0.01, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c='k', ms=ms, capsize=2)
                plt.errorbar(ER-0.01, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c='k', ms=ms, capsize=2)

            lbls.append('{:.3}'.format(Rm))
            lbls.append('{:.3}'.format(Rm))

    plt.legend(lbls, loc='upper left', bbox_to_anchor=(1.1, 1.1), ncol=3, fontsize='xx-small')
    plt.xticks(ERs, ERs)
    plt.xlim([ERs[0]-.015, ERs[-1]+.015])
    plt.xlabel('Equivalence ratio $\\Phi$')
    plt.colorbar(cmap_nu, ax=plt.gca(), shrink=.8)
    plt.colorbar(cmap_beta, ax=plt.gca(), shrink=.8, label='$\\gamma c_2\\beta \\nu$')
    plt.show()
