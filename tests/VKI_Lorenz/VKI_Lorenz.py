# %% 1. EMPLOY LORENZ63 MODEL
import matplotlib.pyplot as plt

from ML_models.EchoStateNetwork import *
from essentials.physical_models import Lorenz63

dt_model = 0.015
t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent


rnd = np.random.RandomState(6)

t_ref = t_lyap

forecast_params = dict(dt=dt_model,
                       psi0=rnd.random(3),
                       t_transient=10 * t_lyap,
                       t_lyap=t_lyap)

ESN_params = dict(t_train=40 * t_lyap,
                  t_val=6 * t_lyap,
                  t_test=10 * t_lyap,
                  upsample=3,
                  N_wash=10,
                  N_units=60,
                  N_folds=8,
                  N_split=5,
                  connect=3,
                  plot_training=True,
                  rho_range=(.1, 1.),
                  tikh_range=[1E-6, 1E-9, 1E-12],
                  N_func_evals=32,
                  sigma_in_range=(-.5, 2.),
                  N_grid=5,
                  noise=1e-3,
                  perform_test=True
                  )


dt_ESN = dt_model * ESN_params['upsample']

dt_DA = .25 * t_lyap
total_time = 20. * t_lyap

N_train = int(ESN_params['t_train'] / dt_ESN)
N_val = int(ESN_params['t_val'] / dt_ESN)
N_test = int(ESN_params['t_test'] / dt_ESN) * 20

if __name__ == '__main__':
    # %% 2. CREATE TRUTH

    forecast_params['model'] = Lorenz63
    forecast_params['dt'] = dt_model

    truth = Lorenz63(**forecast_params)

    # %% 3. TRAIN THE ESN

    N_wtv_model = (N_train + N_val + ESN_params['N_wash'] * 2) * ESN_params['upsample']
    N_test_model = N_test * ESN_params['upsample']
    N_transient_model = int(truth.t_transient / dt_model)
    out = []
    for Nt in [N_transient_model, N_wtv_model, N_test_model]:
        state, t1 = truth.time_integrate(Nt)
        truth.update_history(state, t1, reset=False)
        yy = truth.get_observable_hist(Nt)
        out.append((yy, t1))

    # Build training data dictionary
    Y = truth.get_observable_hist(N_wtv_model + N_test_model)
    Y = Y.transpose(2, 0, 1)

    # %%
    Wouts = []
    for seed in np.arange(6):
        ESN_params['seed_W'] = seed

        # ESN class
        ESN_case = EchoStateNetwork(Y[0, 0, :], dt=dt_model, **ESN_params)
        ESN_case.filename = 'ESN_seed{}_'.format(seed)
        # Train
        ESN_train_data = dict(data=Y,
                              observed_idx=np.arange(3)
                              )
        ESN_case.train(ESN_train_data, validation_strategy=EchoStateNetwork.RVC_Noise)


        Wouts.append(ESN_case.Wout)

    # %%

    plt.rc('text', usetex=True)
    plt.rc('font', family='times', size=14, serif='Times New Roman')
    plt.rc('mathtext', rm='times', bf='times:bold')
    plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


    wr = np.ones(len(Wouts)).tolist()
    wr += [1]

    fig, ax = plt.subplots(nrows=len(Wouts)+1, ncols=1, height_ratios=wr, layout='tight')


    maxxx = 0
    for Wout in Wouts:
        maxxx = np.mean([np.mean(abs(Wout)), maxxx])

    maxxx = 100

    print(maxxx)
    for ii, Wout in enumerate(Wouts):
        im = ax[ii].matshow(Wout.T/maxxx, cmap="PRGn", aspect=4., vmin=-2, vmax=2)

        ax[ii].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

        # ax[ii].set(='seed {}'.format(ii))
        if ii < len(Wouts)-1:
            ax[ii].set(xticks=[])

    im = ax[-1].matshow(Wout *0., cmap="PRGn", aspect=4., vmin=-2, vmax=2)
    cbar=plt.colorbar(im, orientation='horizontal')

    ax[-1].set(yticks=[], xticks=[])

    plt.show()
