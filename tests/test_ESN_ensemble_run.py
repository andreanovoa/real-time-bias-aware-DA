"""
Test if I can run an ensemble of instances within the ESN.
Compare the results with the predictions from a for loop.
"""


from default_parameters.annular import *


os.chdir('../')
folder = 'results/tests/Annular/'
figs_dir = folder + 'figs/'
out_dir = folder+"/out/"

if __name__ == '__main__':
    filter_params['m'] = 10
    bias_params['L'] = 10
    bias_params['N_units'] = 50
    bias_params['N_wash'] = 5

    # ======================= CREATE TRUTH AND ENSEMBLE  =================================
    truth = create_truth(true_params, filter_params, post_processed=False)
    forecast_params['dt'] = truth['dt']

    ensemble = create_ensemble(forecast_params, filter_params)

    # START BIAS MODEL -----------------------------------------------------------
    ESN_name = 'ESN{}_L{}_Nw{}_{}_extraL'.format(bias_params['N_units'], bias_params['L'],
                                                 bias_params['N_wash'], truth['name_bias'])

    filter_ens = ensemble.copy()
    create_bias_model(filter_ens, truth, bias_params, ESN_name,
                      bias_model_folder=folder, plot_train_data=True)

    # %%
    # FORECAST UNTIL FIRST OBS ##
    Nt = int(np.round((truth['t_obs'][0] - filter_ens.t) / filter_ens.dt))

    # Forecast ensemble and update the history
    psi, t = filter_ens.timeIntegrate(Nt=Nt, averaged=False, alpha=None)
    filter_ens.updateHistory(psi, t)
    filter_ens.close()

    # %%
    import time

    def get_cmap(n, name='viridis'):
        cs = plt.get_cmap(name, n)
        return [cs(j) for j in range(cs.N)]

    t1 = time.time()
    plt.figure(3)

    plt.title('ensemble forecast')
    ESN = deepcopy(filter_ens.bias)

    bias_idx = ESN.observed_idx

    y = filter_ens.getObservableHist(Nt)
    ESN.N_ens = y.shape[-1]

    colors = get_cmap(y.shape[-1])

    b2, t_b = ESN.timeIntegrate(t=t, y=y, wash_t=truth['wash_t'], wash_obs=truth['wash_obs'])

    wash_model = interpolate(t, y, truth['wash_t'])
    washout = np.expand_dims(truth['wash_obs'], -1) - wash_model
    for ii in range(ESN.N_ens):
        plt.plot(t_b[-100:], b2[-100:, bias_idx[0], ii], '--', lw=.8, c=colors[ii])
        plt.plot(truth['wash_t'], washout[:, 0, ii], 'x-', lw=.8, c=colors[ii])

    plt.ylim([np.min(washout) * 2, np.max(washout) * 2])
    plt.xlim([truth['wash_t'][0], t_b[-1]])

    print('\n\nEnsemble time: ', time.time()-t1)
    plt.plot([t_b[-100], t_b[-1]], [0, 0], color='r')
    xlims = plt.gca().get_xlim()
    ylims = plt.gca().get_ylim()

    ## %% Forecast ensemble bias and update its history
    t1 = time.time()

    plt.figure(2)

    ESN = deepcopy(filter_ens.bias)

    bias_idx = [a for a in np.arange(ESN.N_dim) if a in ESN.observed_idx]

    y = filter_ens.getObservableHist(Nt)
    for ii in range(y.shape[-1]):
        yy = y[:, :, ii]
        ESN = deepcopy(filter_ens.bias)
        b2, t_b = ESN.timeIntegrate(t=t, y=yy, wash_t=truth['wash_t'], wash_obs=truth['wash_obs'])
        line = plt.plot(t_b[-100:], b2[-100:, bias_idx[0]], '--', lw=.8, color=colors[ii])
        wash_model = interpolate(t, yy, truth['wash_t'])
        washout = truth['wash_obs'] - wash_model
        plt.plot(truth['wash_t'], washout[:, 0], 'x-', lw=.8, color=colors[ii])

    plt.ylim(ylims)
    plt.xlim(xlims)

    print('Loop time: ', time.time()-t1)

    plt.plot([t_b[-100], t_b[-1]], [0, 0], color='r')

    plt.show()


