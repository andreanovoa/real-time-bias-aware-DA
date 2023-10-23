
if __name__ == '__main__':
    from create import create_truth, create_ensemble
    from plotResults import *
    from physical_models import Annular

    # path_dir = os.path.realpath(__file__).split('main')[0]
    # os.chdir('/mscott/an553/')  # set working directory to mscott

    folder = 'results/Annular/'
    figs_dir = folder + 'figs/'

    os.makedirs(figs_dir, exist_ok=True)
    # %% ==================================== SELECT TRUE MODEL ======================================= #

    ER = np.round(np.linspace(start=0.4875, stop=0.575, num=8, endpoint=True), 6)  # equivalence ratios 0.4875-0.575 (by steps of 0.0125)}

    true_params = {'model': 'annular/ER_{}'.format(ER[-3]),
                   'std_obs': 0.1,
                   'psi0': [.1, .1, .1, .1]  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}
                   }

    parameters_IC = dict(
                         nu=(40., 50.),
                         beta_c2=(5, 20),
                         kappa=(1.E-4, 1.3E-4),
                         epsilon=(0.0001, 0.03),
                         omega=(1090*2*np.pi, 1100*2*np.pi),
                         theta_b=(0.2, 0.7),
                         theta_e=(0.15, 0.8),
                         )

    filter_params = {'m': 10,
                     'est_a': [*parameters_IC],
                     'std_a': parameters_IC,
                     'alpha_distr': 'uniform',
                     # Define the observation time window
                     't_start': 1.0,
                     't_stop': 2.,
                     'dt_obs': 10,
                     # Inflation
                     'inflation': 1.002
                     }

    truth = create_truth(true_params, filter_params)

    forecast_params = {'model': Annular}

    # Forecast model and update
    ensemble = create_ensemble(forecast_params, filter_params)

    print(ensemble.hist.shape, ensemble.psi.shape)

    # Forecast ensemble
    Nt = ensemble.t_transient // ensemble.dt
    state, t_ = ensemble.timeIntegrate(Nt)

    # %% Plot ensemble

    print(ensemble.hist.shape, ensemble.psi.shape, state.shape)


    ensemble.updateHistory(psi=state, t=t_)
    y_model = ensemble.getObservableHist()


    fig1 = plt.figure()

    axs = fig1.subplots(3, 1, sharex='all', sharey='all')

    for ax, yy, c in zip(axs, [truth['y_noise'], truth['y']], ['firebrick', 'royalblue']):
        ax.plot(truth['t'], yy[:, 0], color=c)

    axs[-1].plot(ensemble.hist_t, y_model[:, 0])
    axs[-1].set(xlim=[ensemble.hist_t[-1]-ensemble.t_CR, ensemble.hist_t[-1]])

    plt.show()