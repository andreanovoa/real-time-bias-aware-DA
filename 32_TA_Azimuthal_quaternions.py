from essentials.plotResults import *

from essentials.Util import load_from_pickle_file
from scipy.io import savemat
from essentials.physical_models import Annular
import matplotlib.pyplot as plt
import numpy as np

ERs = 0.4875 + np.arange(0, 4) * 0.025


plot_nu_beta_ERs = False
plot_timeseries_windows = False
plot_rest_of_params_ERs = False

if plot_timeseries_windows:
    ER = 0.5625
    m = 20
    results_dir = '/home/an553/Downloads/ER{}/m{}/'.format(ER, m)

    truth, results, bias, blank_ensemble = load_from_pickle_file(results_dir + 'simulation_output_all')




if plot_rest_of_params_ERs:
    print('TODO')



if plot_nu_beta_ERs:

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

    plt.figure(figsize=(10,5))

    lbls = []

    for ER in ERs:
        results_dir = '/home/an553/Downloads/ER{}/m{}/'.format(ER, m)

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

                plt.errorbar(ER-.005, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c=c_nu, ms=ms, capsize=2)
                plt.errorbar(ER+.005, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c=c_b, ms=ms, capsize=2)

            else:
                c_b = cmap_beta.to_rgba(ens.regularization_factor)
                c_nu = cmap_nu.to_rgba(ens.regularization_factor)

                ms = R_to_size(Rm)

                plt.errorbar(ER-.005, np.mean(nus), np.std(nus), marker='^', alpha=0.7, c='k', ms=ms, capsize=2)
                plt.errorbar(ER+.005, np.mean(c2betas), np.std(c2betas), alpha=0.7, marker='o', c='k', ms=ms, capsize=2)


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
