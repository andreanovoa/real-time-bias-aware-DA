from essentials.plotResults import *

from essentials.Util import load_from_pickle_file
from scipy.io import savemat
from essentials.physical_models import Annular
import matplotlib.pyplot as plt
import numpy as np

ERs = 0.4875 + np.arange(0, 4) * 0.025
m = 20


# folder = '/Users/andreanovoa/OneDrive - Imperial College London/results/'
# folder = '/home/an553/Downloads/'
folder = '/Users/anovoama/Desktop/results/'


print(os.getcwd())

plot_nu_beta_ERs = 0
plot_timeseries_windows = 1
plot_rest_of_params_ERs = 0




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


if plot_timeseries_windows:
    ER = ERs[-1]
    results_dir = folder + 'ER{}/m{}/'.format(ER, m)

    truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')

    t_ref = truth['t']
    window = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]
    jjs = [np.argmin(abs(t_ref - tt)) for tt in window]
    t_ref, y_ref = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true']]

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

    pdf_file = plt_pdf.PdfPages(folder + 'timeseries_results_ER{}.pdf'.format(ER))

    def save_fig_to_pdf(fig, file):
        file.savefig(fig)
        plt.close(fig)

    for ens in [best_case, results[-1]]:
        plot_timeseries(ens, truth)
        save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
        plot_parameters(ens, truth, reference_p=params_scales)
        save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)

    plot_RMS_pdf(ensembles=[best_case, results[-1]], truth=truth)
    save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)
    plot_states_PDF(ensembles=[best_case, results[-1]], truth=truth, window=window)
    save_fig_to_pdf(fig=plt.gcf(), file=pdf_file)

    pdf_file.close()




if plot_rest_of_params_ERs:

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True)
    axs = axs.ravel()
    lbls = []

    ms = [R_to_size(xx) for xx in [5, 2, .9, .6, .4, .2]]
    for mm in ms:
        axs[0].plot(0, 0, 'o', ms=mm)
    axs[0].legend(['{}'.format(xx) for xx in [5, 2, .9, .6, .4, .2]])

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
                        xval = ER - 0.001 * (kk + 1)
                    else:
                        xval = ER + 0.001 * (kk + 1)

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

        xval = ER.copy()
        xval_b = ER.copy()

        truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir+'simulation_output_all')

        C, N = [], []

        norm = colors.Normalize(vmin=0, vmax=7)
        cmap_nu = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Greens_r)
        cmap_beta = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Oranges_r)

        t_ref = truth['t']

        tts = [truth['t_obs'][-1], truth['t_obs'][-1] + Annular.t_CR * 2]

        jjs = [np.argmin(abs(t_ref - tt)) for tt in tts]

        t_ref, y_ref = [truth[key][jjs[0]:jjs[1]] for key in ['t', 'y_true']]

        for ens in results:

            y_est = ens.get_observable_hist()
            y_mean = np.mean(y_est, axis=-1)
            y_mean = interpolate(ens.hist_t, y_mean, t_ref)
            Rm = np.sqrt(np.sum((y_ref - y_mean) ** 2, axis=1) / np.sum(y_ref ** 2, axis=1))
            Rm = np.mean(Rm)

            nu_idx, c2beta_idx = [ens.est_a.index(key) for key in ['nu', 'c2beta']]
            nus = ens.get_current_state[ens.Nphi + nu_idx]
            c2betas = ens.get_current_state[ens.Nphi + c2beta_idx]

            if ens != results[-1]:
                c_b = cmap_beta.to_rgba(ens.regularization_factor)
                c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                ms = R_to_size(Rm)

                xval += 0.001
                xval_b -= 0.001
                plt.errorbar(xval, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c=c_nu, ms=ms, capsize=2)
                plt.errorbar(xval_b, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c=c_b, ms=ms, capsize=2)

            else:
                c_b = cmap_beta.to_rgba(ens.regularization_factor)
                c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                ms = R_to_size(Rm)

                plt.errorbar(ER, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c='k', ms=ms, capsize=2)
                plt.errorbar(ER, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c='k', ms=ms, capsize=2)


            C.append(c2betas)
            N.append(nus)

            lbls.append('{:.3}'.format(Rm))
            lbls.append('{:.3}'.format(Rm))



    plt.legend(lbls, loc='upper left', bbox_to_anchor=(1.1, 1.1), ncol=3, fontsize='xx-small')




    vals = 0.4875 + np.arange(-1, 5) * 0.025
    plt.plot(vals, [Annular.c2beta_from_ER(er) for er in vals], ls='--', c='C1')
    plt.plot(vals, [Annular.nu_from_ER(er) for er in vals], ls='--', c='C2')

    plt.xticks(ERs, ERs)
    plt.xlim([ERs[0]-.01, ERs[-1]+.01])
    plt.xlabel('Equivalence ratio $\\Phi$')
    plt.colorbar(cmap_nu, ax=plt.gca(), shrink=.8)
    plt.colorbar(cmap_beta, ax=plt.gca(), shrink=.8, label='$\\gamma c_2\\beta \\nu$')
    plt.show()
