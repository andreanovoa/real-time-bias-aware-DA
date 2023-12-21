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


observe_idx = np.array([0])  # Number of dimensions the ESN predicts

ESN_params = bias_params.copy()

ESN_params['N_wash'] = 5
ESN_params['upsample'] = 5

filter_params['m'] = 2
observe_idx = np.array([0, 1, 2])  # Select observable states


t_transient = Lorenz63.defaults['t_transient']
upsample = ESN_params['upsample']
dt_ESN = dt_model * upsample


ESN_params['t_val'] = 2 * t_lyap
ESN_params['t_test'] = 20 * t_lyap

N_train = int(ESN_params['t_train'] / dt_ESN)
N_val = int(ESN_params['t_val'] / dt_ESN)
N_test = int(ESN_params['t_test'] / dt_ESN)

N_wash = ESN_params['N_wash']

N_wtv_model = (N_train + N_val + N_wash) * upsample
N_test_model = N_test * upsample

N_transient_model = int(t_transient / dt_model)

# %% 2. CREATE TRUTH


if __name__ == '__main__':

    forecast_params['model'] = Lorenz63
    forecast_params['dt'] = dt_model
    model = create_ensemble(forecast_params, filter_params)

    t_ref = t_lyap

    N_wtv_model += ESN_params['N_wash'] * upsample

    out = []
    for Nt in [N_transient_model, N_wtv_model, N_test_model]:
        state, t1 = model.timeIntegrate(Nt)
        model.updateHistory(state, t1, reset=False)
        yy = model.getObservableHist(Nt)
        out.append((t1, yy))

    # %% 3. TRAIN THE ESN

    # Build training data dictionary
    Y = model.getObservableHist(N_wtv_model + N_test_model)
    Y = Y.transpose(2, 0, 1)

    # ESN class
    ESN_case = EchoStateNetwork(Y[0, 0, observe_idx], dt=dt_model, **ESN_params)

    # Train
    ESN_train_data = dict(inputs=Y[:, :, observe_idx],
                          labels=Y,
                          observed_idx=observe_idx)
    ESN_case.train(ESN_train_data, validation_strategy=EchoStateNetwork.RVC_Noise)


    # %% 4.1 INITIALISE ESN WITH ENSEMBLE

    rng = np.random.default_rng(0)


    wash_model = model.getObservableHist(ESN_case.N_wash * upsample)[:, observe_idx]
    t_wash_model = model.hist_t[-ESN_case.N_wash * upsample:]

    wash_data, t_wash = wash_model[::upsample], t_wash_model[::upsample]


    # %%
    # run closed loops and initialize with observations every loop
    num_tests = 4
    Nt_tests = int(num_tests * t_lyap / dt_ESN)
    Nt_tests_model = Nt_tests * upsample

    u_wash, r_wash = ESN_case.openLoop(wash_data)

    ESN_case.reset_state(u=u_wash[-1], r=r_wash[-1])


    fig1 = plt.figure(figsize=(13, 6), layout="tight")
    axs = fig1.subplots(nrows=model.Nq, ncols=2, sharex='col', sharey='row')

    cs = ['lightblue', 'tab:blue', 'navy']
    ci = -1
    for t1, yy in out:
        ci += 1
        for ii, ax in enumerate(axs[:, 0]):
            ax.plot(t1 / t_ref, yy[:, ii], '-', c=cs[ci])
            ax.set(ylabel=model.obsLabels[ii])
    axs[0, 0].set(xlim=[0, model.hist_t[-1] / t_ref])
    axs[0, 0].legend(['Transient', 'Train+val', 'Train tests'],
                     ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))

    for ii, idx in enumerate(ESN_case.observed_idx):
        axs[idx, 1].plot(t_wash_model / t_ref, wash_model[:, ii], '.-', c=cs[-1])
        axs[idx, 1].plot(t_wash / t_ref, wash_data[:, ii], 'x-', c='tab:green')

    for jj in range(num_tests + 1):
        # Forecast model and ESN and compare
        out = model.timeIntegrate(Nt_tests_model)
        model.updateHistory(*out, reset=False)

        yy = model.getObservableHist(Nt_tests_model)
        tt = model.hist_t[-Nt_tests_model:]
        tt_up = out[-1][::ESN_case.upsample]

        u_closed = ESN_case.closedLoop(len(tt_up))[0]
        ESN_case.reset_state(u=yy[-1])

        for axs_col in [axs[:, 1]]:
            for ii, ax in enumerate(axs_col):
                ax.plot(tt / t_ref, yy[:, ii], '-', c='k', lw=2, alpha=.8, label='Truth')
                if ii < ESN_case.N_dim:
                    ax.plot(tt_up / t_ref, u_closed[1:, ii], 'x--', c='r', ms=4, label='ESN prediction')
                    if ii in observe_idx:
                        for yy_obs in yy[-1, ii]:
                            ax.plot(out[-1][-2] / t_ref, yy_obs, 'o', c='r', ms=8, alpha=.8, label='Data')
        if jj == 0:
            axs[0, 1].legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
        if jj == num_tests:
            margin = dt_ESN * ESN_case.N_wash
            for ax, xl in zip(axs[-1, :], [[0, t_wash[0] / t_ref],
                                           [t_wash[0] / t_ref - margin, tt_up[-1] / t_ref + margin]]):
                ax.set(xlabel='$t/T$', xlim=xl)
    plt.show()
    # %% 4. UPDATE STATES USING DATA ASSIMILATION

    observables = dict()

