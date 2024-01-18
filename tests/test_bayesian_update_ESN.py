"""
Test using an ensemble approach to estimating the full states from partial observations with an ESN

1. Select working model e.g., Lorenz63
2. Create a true state from the model
    2.1 The true states are (i) the model predictions and (ii) noisy/biased model predictions
    2.2 The inputs are only of (ii), either full or partial noisy state
3. Train the ESN
4. Data assimilation
    4.1 Initialise an ensemble of states --- may need an ensemble of washouts?
    4.2 Forecast the ESN in closed-loop
    4.3 Feed observations on the noisy/biased data and use the EnKF/EnSRKF to estimate the full state
    4.4 Reinitialise with the analysis state and go back to 4.2
"""

# %% 1. EMPLOY LORENZ63 MODEL

from default_parameters.lorenz63 import *
from ML_models.EchoStateNetwork import *
from essentials.create import create_ensemble
from essentials.DA import EnSRKF, EnKF, inflateEnsemble
from copy import deepcopy
cs = ['lightblue', 'tab:blue', 'navy']

observe_idx = np.array([0, 2])  # Number of dimensions the ESN prediction
update_reservoir = 1
plot_training_data = 0
plot_timeseries_flag = 1


ESN_params = bias_params.copy()

ESN_params['N_wash'] = 5
ESN_params['N_units'] = 80
ESN_params['N_folds'] = 8
ESN_params['connect'] = 3

ESN_params['noise'] = 0.01
ESN_params['upsample'] = 3
ESN_params['Win_type'] = 'sparse'

ESN_params['rho_range'] = (0.2, 0.9)

ESN_params['t_val'] = 2 * t_lyap
ESN_params['t_test'] = 5 * t_lyap


dt_ESN = dt_model * ESN_params['upsample']

dt_DA = 1. * t_lyap
total_time = 100. * t_lyap
num_DA_steps = int(total_time / dt_DA)


N_train = int(ESN_params['t_train'] / dt_ESN)
N_val = int(ESN_params['t_val'] / dt_ESN)
N_test = int(ESN_params['t_test'] / dt_ESN) * 20

t_ref = t_lyap


def plot_data(axs, type_plot='train'):
    if type_plot == 'train':
        ci = -1
        for yyy, ttt in out:
            ci += 1
            for axj, axx in enumerate(axs):
                axx.plot(ttt / t_ref, yyy[:, axj], '-', c=cs[ci])
                axx.set(ylabel=truth.obsLabels[axj])
            if ci == 2:
                axs[0].set(xlim=[0, ttt[-1] / t_ref])
                axs[0].legend(['Transient', 'Train+val', 'Tests'], ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    else:
        if jj == 0:
            for ii, idx in enumerate(ESN_case.observed_idx):
                axs[idx].plot(t_wash_model / t_ref, wash_model[:, ii], '.-', c=cs[-1])
                axs[idx].plot(t_wash / t_ref, wash_data[:, ii], 'x-', c='tab:green')
        lbl_pred = ['ESN_prediction'] + [None] * (Aa.shape[-1]-1)
        lbl_a = ['Analysis'] + [None] * (Aa.shape[-1]-1)
        lbl_f = ['Forecast'] + [None] * (Aa.shape[-1]-1)

        for axi, ax in enumerate(axs):
            ax.plot(tt / t_ref, yy[:, axi], '-', c='k', lw=2, alpha=.8, label='Truth')
            if axi < ESN_case.N_dim:
                ax.plot(tt_up / t_ref, u_closed[:, axi], '--', c='#20b2aae5', ms=4, label=lbl_pred)
                ax.plot(tt_up[-1] / t_ref, Af[[axi]], '^', c='#20b2aae5', ms=4, label=lbl_f)
                ax.plot(tt_up[-1] / t_ref, Aa[[axi]], '*', c='#0054D8', ms=7, label=lbl_a)
                if axi in observe_idx:
                    for yy_obs in yy[-1, axi]:
                        ax.plot(tt[-1] / t_ref, yy_obs, 'o', c='r', ms=8, alpha=.8, label='Data')
                if jj == 0 and axi == 0:
                    axs[0].legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))

        if jj == num_tests:
            margin = dt_ESN * ESN_case.N_wash
            axs[-1].set(xlabel='$t/T$', xlim=[t_wash[0] / t_ref - margin, tt_up[-1] / t_ref + margin])


def plot_RMS(axs):
    yy_up = interpolate(tt, yy, tt_up)
    # yy_est = np.mean(u_closed, axis=-1)
    std = np.std(u_closed, axis=-1)
    rms = np.sqrt(np.sum((yy_up - u_closed) ** 2, axis=1))  #  / np.sum(yy_up ** 2, axis=1))
    axs[0].plot(tt_up / t_ref, rms, 'r.', ms=.5)
    for ss, cc in zip(std.T, ['b', 'c', 'g']):
        axs[1].plot(tt_up / t_ref, ss * 2, c=cc)
    axs[1].set(xlabel='$t/T$', ylabel='2 std')
    axs[0].set(ylabel='RMS')


if __name__ == '__main__':
    # %% 2. CREATE TRUTH

    forecast_params['model'] = Lorenz63
    forecast_params['dt'] = dt_model

    truth = Lorenz63(**forecast_params)

    # %% 3. TRAIN THE ESN

    ESN_name = 'my_esn_partial'
    try:
        ESN_case, ESN_train_data = load_from_pickle_file(ESN_name)
    except FileNotFoundError:
        filter_params['m'] = 20
        train_model = create_ensemble(forecast_params, filter_params)
        N_wtv_model = (N_train + N_val + ESN_params['N_wash'] * 2) * ESN_params['upsample']
        N_test_model = N_test * ESN_params['upsample']
        N_transient_model = int(train_model.t_transient / dt_model)
        out = []
        for Nt in [N_transient_model, N_wtv_model, N_test_model]:
            state, t1 = train_model.timeIntegrate(Nt)
            train_model.updateHistory(state, t1, reset=False)
            yy = train_model.getObservableHist(Nt)
            out.append((yy, t1))
        train_model.close()

        rng = np.random.default_rng(0)

        if plot_training_data:
            fig1 = plt.figure(figsize=(6, 3), layout="tight")
            axs_train = fig1.subplots(nrows=truth.Nphi, ncols=1, sharex='col', sharey='row')
            plot_data(axs_train, type_plot='train')

        # Build training data dictionary
        Y = train_model.getObservableHist(N_wtv_model + N_test_model)
        Y = Y.transpose(2, 0, 1)

        # ESN class
        ESN_case = EchoStateNetwork(Y[0, 0, observe_idx], dt=dt_model, **ESN_params)

        # Train
        ESN_train_data = dict(inputs=Y[:, :, observe_idx],
                              labels=Y,
                              observed_idx=observe_idx)
        ESN_case.train(ESN_train_data, validation_strategy=EchoStateNetwork.RVC_Noise)
        save_to_pickle_file(ESN_name, deepcopy(ESN_case), ESN_train_data)

    # %% INITIALIZE ENSEMBLE OF ESN
    num_tests = num_DA_steps
    Nt_tests = int(dt_DA / dt_ESN)
    Nt_tests_model = Nt_tests * ESN_case.upsample

    N_wash_model = ESN_case.N_wash * ESN_case.upsample
    wash_model = ESN_train_data['inputs'][:, -N_wash_model:].transpose(1, 2, 0)

    t_wash_model = np.arange(0, N_wash_model) * dt_model

    wash_data, t_wash = wash_model[::ESN_case.upsample], t_wash_model[::ESN_case.upsample]
    u_wash, r_wash = ESN_case.openLoop(wash_data)

    ESN_case.reset_state(u=u_wash[-1], r=r_wash[-1])

    for Nt in [int(truth.t_transient/dt_model), N_wash_model]:
        truth.t = 0.
        psi, tt = truth.timeIntegrate(Nt)
        truth.updateHistory(psi, tt, reset=True)

    #  Observation operator

    if update_reservoir:
        M = np.zeros([ESN_case.N_dim_in, ESN_case.N_dim + ESN_case.N_units])
    else:
        M = np.zeros([ESN_case.N_dim_in, ESN_case.N_dim])

    for dim_i, obs_i in enumerate(observe_idx):
        M[dim_i, obs_i] = 1.

    fig1 = plt.figure(figsize=(6, 3), layout="tight")
    axs_test = fig1.subplots(nrows=ESN_case.N_dim, ncols=1, sharex='col', sharey='row')

    fig2 = plt.figure(figsize=(6, 3), layout="tight")
    axs_RMS = fig2.subplots(nrows=2, ncols=1, sharex='col')

    for jj in range(num_tests + 1):
        # Forecast model and ESN
        psi, tt = truth.timeIntegrate(Nt_tests_model)
        truth.updateHistory(psi, tt, reset=False)

        u_closed, r_closed = ESN_case.closedLoop(Nt_tests)
        u_closed, r_closed = u_closed[1:], r_closed[1:]

        tt_up = tt[::ESN_case.upsample].copy()

        if not Nt_tests_model % Nt_tests:
            u_closed[-1] = interpolate(tt_up, u_closed, tt[-1], method='linear')
            r_closed[-1] = interpolate(tt_up, r_closed, tt[-1], method='linear')
            tt_up[-1] = tt[-1].copy()

        # Take measurement of the truth
        yy = truth.getObservableHist(Nt_tests_model)
        d = yy[-1, observe_idx, 0]

        if update_reservoir:
            Af = np.concatenate([u_closed[-1], r_closed[-1]], axis=0)
        else:
            Af = u_closed[-1].copy()

        # Compute ensemble statistics
        Cdd = (0.05 * 20) ** 2 * np.eye(len(d))

        # Apply ensemble square-root Kalman filter
        Aa = EnSRKF(Af, d, Cdd, M)[0]

        # Update ESN with analysis state
        if update_reservoir:
            ESN_case.reset_state(u=Aa[:ESN_case.N_dim], r=Aa[ESN_case.N_dim:])
        else:
            ESN_case.reset_state(u=Aa[:ESN_case.N_dim])
        if plot_timeseries_flag:
            plot_data(axs_test, type_plot='test')
        plot_RMS(axs_RMS)

    plt.show()



