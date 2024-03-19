from essentials.create import create_ensemble, create_truth, create_bias_model
from essentials.DA import dataAssimilation
from essentials.physical_models import Annular
from essentials.bias_models import ESN
from essentials.plotResults import *

rng = np.random.default_rng(0)


if os.path.isdir('/mscott/'):
    data_folder = '/mscott/an553/data/'  # set working directory to
else:
    data_folder = "../data/"

folder = 'results/AnnularEperiments/'

if __name__ == '__main__':

    ER = 0.5625  # 0.4875 + np.arange(0, 4) * 0.025

    t_start = Annular.t_transient
    t_stop = t_start + Annular.t_CR * 5

    truth = create_truth(model=data_folder + 'annular/ER_{}'.format(ER),
                         t_start=t_start,
                         t_stop=t_stop,
                         Nt_obs=35,
                         t_max=t_stop + Annular.t_transient,
                         post_processed=False
                         )

    alpha0 = dict(nu=(20., 40.),
                  c2beta=(10, 50),
                  kappa=(1.E-4, 2.E-4),
                  epsilon=(5e-3, 8e-3),
                  omega=(1090 * 2 * np.pi, 1095 * 2 * np.pi),
                  theta_b=(0.5, 0.7),
                  theta_e=(0.4, 0.6)
                  )

    ensemble = create_ensemble(model=Annular, dt=truth['dt'],
                               m=20, std_psi=0.3, std_a=alpha0)

    train_params = dict(bias_model=ESN,
                        upsample=5,
                        N_units=100,
                        N_wash=10,
                        t_train=ensemble.t_CR * 20,
                        t_test=ensemble.t_CR * 4,
                        t_val=ensemble.t_CR * 4,
                        # Training data generation options
                        augment_data=True,
                        bayesian_update=False,
                        biased_observations=True,
                        seed_W=0,
                        L=10,
                        # Hyperparameter search ranges
                        rho_range=(0.5, 1.),
                        sigma_in_range=(np.log10(1e-5), np.log10(1e1)),
                        tikh_range=[1e-12, 1e-9]
                        )

    bias, wash_obs, wash_t = create_bias_model(ensemble, truth, bias_params=train_params)

    DA_kwargs = dict(y_obs=truth['y_obs'],
                     t_obs=truth['t_obs'], std_obs=0.1, wash_obs=wash_obs, wash_t=wash_t)


    out = []
    for kf in ['rBA_EnKF', 'EnSRKF']:

        ens = ensemble.copy()
        if kf[0] == 'r':
            ens.bias = bias

        ens.filter = kf
        ens.regularization_factor = 2.

        ens.t_init = truth['t_obs'][0]
        ens.inflation = 1.001

        filter_ens = dataAssimilation(ens, **DA_kwargs.copy())

        # Forecast the ensemble further without assimilation
        Nt_extra = int(filter_ens.t_CR / filter_ens.dt) * 5

        psi, t = filter_ens.time_integrate(Nt_extra)
        filter_ens.update_history(psi, t)

        y = filter_ens.get_observable_hist(Nt_extra)
        b, t_b = filter_ens.bias.time_integrate(t=t, y=y)
        filter_ens.bias.update_history(b, t_b)

        out.append(filter_ens)

    truth_params = dict()
    for param in Annular.params:
        if param == 'nu':
            truth_params[param] = Annular.nu_from_ER(ER)
        elif param == 'c2beta':
            truth_params[param] = Annular.c2beta_from_ER(ER)
        else:
            truth_params[param] = Annular.defaults[param]

    print_parameter_results(out, true_values=truth_params)

    truth['wash_t'] = wash_t
    truth['wash_obs'] = wash_obs

    window = (truth['t_obs'][-1], truth['t_obs'][-1] + out[0].t_CR * 2)
    plot_states_PDF(out, truth, nbins=20, window=window)
    plot_RMS_pdf(out, truth, nbins=20)

    for filter_ens in out:
        plot_timeseries(filter_ens, truth)
        plot_parameters(filter_ens, truth)

    for filter_ens in [out[0]]:
        bias = filter_ens.bias
        ESN_prediction = bias.hist
        bias_hist = bias.get_bias(ESN_prediction)
        inn_hist = ESN_prediction[:, bias.observed_idx]

        xls = [[DA_kwargs['wash_t'][0], bias.hist_t[-1]],
               [DA_kwargs['t_obs'][-5], bias.hist_t[-1]]]
        fig, axs = plt.subplots(bias.N_dim_in, 2, sharex='col', sharey=True, figsize=(10, 10))
        for kk, axs_ in enumerate(axs):
            for ax in axs_:
                ax.plot(bias.hist_t, bias_hist[:, kk])
                ax.plot(bias.hist_t, inn_hist[:, kk])
                ax.plot(bias.hist_t, bias_hist[:, kk] - inn_hist[:, kk])
                ax.axvline(x=DA_kwargs['t_obs'][0], ls='--', color='k', linewidth=.8)
                ax.axvline(x=DA_kwargs['t_obs'][-1], ls='--', color='k', linewidth=.8)
        axs[-1, 0].set_xlim(xls[0])
        axs[-1, 1].set_xlim(xls[-1])

    for filter_ens in [out[0]]:
        bias = filter_ens.bias

        y_raw, y_true = [np.expand_dims(interpolate(truth['t'], yy, bias.hist_t), axis=-1) for yy in
                         [truth['y_raw'], truth['y_true']]]
        y_filter = filter_ens.get_observable_hist()
        y_filter = interpolate(filter_ens.hist_t, y_filter, bias.hist_t)

        inn_ESN = bias.hist[:, bias.observed_idx]
        b_ESN = bias.get_bias(state=bias.hist)
        bd_ESN = b_ESN - inn_ESN

        inn_obs = y_raw - y_filter
        b_obs = y_true - y_filter
        bd_obs = y_true - y_raw

        inn_RMS = np.sqrt(np.sum((inn_obs - inn_ESN) ** 2, axis=1) / np.sum(y_true ** 2, axis=1))
        b_RMS = np.sqrt(np.sum((b_obs - b_ESN) ** 2, axis=1) / np.sum(y_true ** 2, axis=1))
        bd_RMS = np.sqrt(np.sum((bd_obs - bd_ESN) ** 2, axis=1) / np.sum(y_true ** 2, axis=1))

        window = int(filter_ens.t_CR / 5 // filter_ens.dt)
        average_RMS_i, average_RMS_b, average_RMS_bd = [], [], []
        average_t = []
        for ind in range(len(inn_RMS)):
            average_RMS_i.append(np.mean(inn_RMS[ind:ind + window]))
            average_RMS_b.append(np.mean(b_RMS[ind:ind + window]))
            average_RMS_bd.append(np.mean(bd_RMS[ind:ind + window]))
            average_t.append(bias.hist_t[ind])

        xls = [[DA_kwargs['wash_t'][0] - .2, DA_kwargs['wash_t'][-1] + .5],
               [DA_kwargs['t_obs'][-5], DA_kwargs['t_obs'][-1] + filter_ens.t_CR * 2]]
        xls = [[DA_kwargs['wash_t'][0] - .2, bias.hist_t[-1]],
               [DA_kwargs['t_obs'][-5], bias.hist_t[-1]]]
        fig, axs = plt.subplots(1, 2, sharex='col', sharey=True, figsize=(10, 4))
        for kk, ax in enumerate(axs):
            # ax.plot(bias.hist_t, inn_RMS)
            ax.plot(average_t, average_RMS_i)
            ax.plot(average_t, average_RMS_b)
            ax.plot(average_t, average_RMS_bd)
            # ax.axhline(y=np.mean(inn_obs[:, kk], axis=0), ls='--', color='g')
            ax.axvline(x=DA_kwargs['t_obs'][0], ls='--', color='k', linewidth=.8)
            ax.axvline(x=DA_kwargs['t_obs'][-1], ls='--', color='k', linewidth=.8)
        axs[0].set_xlim(xls[0])
        axs[1].set_xlim(xls[-1])

    for filter_ens in [out[0]]:
        bias = filter_ens.bias

        y_raw, y_true = [np.expand_dims(interpolate(truth['t'], yy, bias.hist_t), axis=-1) for yy in
                         [truth['y_raw'], truth['y_true']]]
        y_filter = filter_ens.get_observable_hist()
        y_filter = interpolate(filter_ens.hist_t, y_filter, bias.hist_t)

        b_ESN = bias.get_bias(state=bias.hist)
        bd_ESN = b_ESN - inn_ESN

        y_RMS = np.sqrt(np.sum((y_true - (y_filter + b_ESN)) ** 2, axis=1) / np.max(y_true ** 2))
        Mpsi_RMS = np.sqrt(np.sum((y_true - y_filter) ** 2, axis=1) / np.max(y_true ** 2))

        window = int(filter_ens.t_CR // filter_ens.dt)
        average_RMS_y, average_RMS_Mpsi = [], []
        average_t = []
        for ind in range(len(inn_RMS)):
            average_RMS_y.append(np.mean(y_RMS[ind:ind + window], axis=0))
            average_RMS_Mpsi.append(np.mean(Mpsi_RMS[ind:ind + window], axis=0))
            average_t.append(bias.hist_t[ind])

        if len(out) > 1:
            for filter_ens_bb in [out[1]]:
                y_filter = filter_ens_bb.get_observable_hist()
                y_filter = interpolate(filter_ens_bb.hist_t, y_filter, bias.hist_t)

                Mpsi_RMS = np.sqrt(np.sum((y_true - y_filter) ** 2, axis=1) / np.max(y_true ** 2))

                average_RMS_Mpsi_BB = []
                for ind in range(len(inn_RMS)):
                    average_RMS_Mpsi_BB.append(np.mean(Mpsi_RMS[ind:ind + window], axis=0))

    xls = [[DA_kwargs['wash_t'][0] - .5, bias.hist_t[-1]],
           [DA_kwargs['t_obs'][-5], bias.hist_t[-1]]]
    fig, axs = plt.subplots(1, 2, sharex='col', sharey=True, figsize=(10, 4))
    for kk, ax in enumerate(axs):
        # ax.plot(bias.hist_t, inn_RMS)
        ax.plot(average_t, average_RMS_y, c='navy')
        ax.plot(average_t, average_RMS_Mpsi, ':', c='lightseagreen')

        if len(out) > 1:
            ax.plot(average_t, average_RMS_Mpsi_BB, ':', c='orange')
        # ax.axhline(y=np.mean(inn_obs[:, kk], axis=0), ls='--', color='g')
        ax.axvline(x=DA_kwargs['t_obs'][0], ls='--', color='k', linewidth=.4)
        ax.axvline(x=DA_kwargs['t_obs'][-1], ls='--', color='k', linewidth=.4)
    axs[0].set_xlim(xls[0])
    axs[1].set_xlim(xls[-1])


    plt.show()