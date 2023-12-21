"""
Test the EchoStateNetwork class.
--------------------------------
1)  The newest version of the class allows for partial observability. The default is full observability.
    To remove an index from the observability list you must re-define the variable observe_idx
        Example: 3D system
            - Full observability [default]: observe_idx = np.arange(3)
            - Only the third dimension is measured: observe_idx = np.array([2])
            
2) Test the Jacobian of the network using the function Jacobian_numerical_test in Util
----------------------------------

"""

# from default_parameters.lorenz63 import *
from default_parameters.annular import *
from ML_models.EchoStateNetwork import *

model_params = forecast_params.copy()
defaults = Annular.defaults.copy()


def Jacobian_numerical_test(J_analytical, step_function, y_init, epsilons=np.arange(-5, 15, 0.1)):
    """
    Function that computes numerically the Jacobian of a given step function and
    computes the error with respect to the corresponding analytical form.
        Inputs:
            - y_init: initial condition to the step function
            - epsilons: range of perturbations to consider (its log10)
            - y_out_idx: if the step function returns more than one item, provide index of y_out
    Note: J_analytical is correct if there is a linear decay in the error with decreasing epsilon
    """
    from scipy.sparse import issparse
    # Numerical Jacobian
    errors = []
    y_out = step_function(y_init)[0]

    for epsilon in epsilons:
        eps = 10. ** epsilon
        J_numerical = np.zeros(J_analytical.shape)
        for qi in range(J_analytical.shape[1]):
            y_tilde = y_init.copy()
            y_tilde[qi] = y_tilde[qi] + eps
            y_out_tilde = step_function(y_tilde)[0]

            J_numerical[:, qi] = (y_out_tilde - y_out).squeeze() / eps

        # Compute and store relative norm difference
        errors.append(np.linalg.norm(J_analytical - J_numerical) / np.linalg.norm(J_analytical))
    return epsilons, errors


# Update some of the default parameters as required
bias_params['N_wash'] = 2
bias_params['N_func_evals'] = 35
bias_params['upsample'] = 3
bias_params['L'] = 1
defaults['t_transient'] = 0.05

bias_params['t_train'] = 0.05
bias_params['t_val'] = 0.005
bias_params['t_test'] = 0.03

observe_idx = np.array([0, 1, 3], dtype=int)  # Select observable states
num_tests = 4
test_Jacobian = True
run_tests = True

t_transient = defaults['t_transient']
dt_model = defaults['dt']
upsample = bias_params['upsample']
dt_ESN = dt_model * upsample

t_wtv = bias_params['t_train'] + bias_params['t_val']

N_wtv = int(t_wtv / dt_ESN) + bias_params['N_wash']
N_test = int(bias_params['t_test'] / dt_ESN)

N_wtv_model = N_wtv * upsample
N_test_model = N_test * upsample

N_transient_model = int(t_transient / dt_model)

t_ref = 1.  # t_lyap

if __name__ == '__main__':


    # %% 1. CREATE TRUTH
    # model = Lorenz63(**model_params)
    model = Annular(**model_params)

    out = []
    for Nt in [N_transient_model, N_wtv_model, N_test_model]:
        state, t1 = model.timeIntegrate(Nt)
        model.updateHistory(state, t1, reset=False)
        yy = model.getObservableHist(Nt)
        out.append((t1, yy))
    model.updateHistory(state, t1, reset=False)

    # %% 2. TRAIN THE ESN
    # Build training data dictionary
    Y = model.getObservableHist(N_wtv_model + N_test_model).transpose(2, 0, 1)

    ESN_train_data = dict(inputs=Y[:, :, observe_idx],
                          labels=Y,
                          observed_idx=observe_idx)
    # ESN class
    ESN_case = EchoStateNetwork(Y[0, observe_idx], dt=dt_model, **bias_params)

    # Train
    ESN_case.train(ESN_train_data,
                   validation_strategy=EchoStateNetwork.RVC_Noise)

    # %% 3. INITIALISE THE ESN
    #  First initialise the network - washout phase
    wash_model = model.getObservableHist(ESN_case.N_wash * upsample)
    t_wash_model = model.hist_t[-ESN_case.N_wash * upsample:]

    wash_model = wash_model[:, observe_idx, 0]
    wash_data, t_wash = wash_model[::upsample], t_wash_model[::upsample]

    # Second run closed loops tests and initialize with observations every loop
    u_wash, r_wash = ESN_case.openLoop(wash_data)
    ESN_case.reset_state(u=u_wash[-1], r=r_wash[-1])

    # %% 4. TEST THE ESN
    if run_tests:

        Nt_tests = int(0.5 * bias_params['t_val'] / dt_ESN)
        Nt_tests_model = Nt_tests * upsample

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
                            ax.plot(out[-1][-2] / t_ref, yy[-1, ii], 'o', c='r', ms=8, alpha=.8, label='Data')
            if jj == 0:
                axs[0, 1].legend(ncols=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
            if jj == num_tests:
                margin = dt_ESN * ESN_case.N_wash
                for ax, xl in zip(axs[-1, :], [[0, t_wash[0] / t_ref],
                                               [t_wash[0] / t_ref - margin, tt_up[-1] / t_ref + margin]]):
                    ax.set(xlabel='$t/T$', xlim=xl)



    # %% 5. TEST IF THE JACOBIAN IS PROPERLY DEFINED
    if test_Jacobian:

        u_init, r_init = ESN_case.getReservoirState()

        J_ESN = ESN_case.Jacobian(open_loop_J=True, state=(u_init, r_init))

        fun = partial(ESN_case.step, r=r_init)
        eps, errs = Jacobian_numerical_test(J_ESN, step_function=fun, y_init=u_init)

        plt.figure(figsize=(6, 6), layout="tight")
        plt.semilogy(-eps, errs, 'rx')
        plt.xlabel('$-\\log_{10}(\\epsilon)$')


    plt.show()