import os

from itertools import product
os.environ["OMP_NUM_THREADS"] = '1'  # imposes cores
import numpy as np
from functions import physical_models

import matplotlib.pyplot as plt
from skopt.space import Real
import matplotlib as mpl
import time
from skopt.plots import plot_convergence

plt.style.use('dark_background')
# Latex
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
# Validation strategies

exec(open('ESN_AR.py').read())

dim = 3

upsample = 5  # to increase the dt of the ESN wrt the numerical integrator
dt = 0.005 * upsample  # time step
t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent)
N_lyap = int(t_lyap / dt)  # number of time steps in one Lyapunov time

N_transient = int(200 / dt)
u0 = np.random.random((dim))  # initial random condition

# number of time steps for washout, train, validation, test
N_washout = 80
N_train = 50 * N_lyap
N_val = 3 * N_lyap
N_test = 1000 * N_lyap

# # generate data
N_forecast = (N_transient+N_washout+N_train+N_val+N_test) * upsample


# U, T = solve_ode(N_forecast, dt / upsample, u0)
#
# fig1 = plt.figure(figsize=[12, 9], layout="constrained")
# axs = fig1.subplots(3, 1)
#
# for ax, u in zip(axs, U.T):
#     ax.plot(T, u)


model = physical_models.Lorenz63({'dt':   dt / upsample,
                                  'psi0': u0})

state1, t1 = model.time_integrate(N_forecast)
model.update_history(state1, t1)
U, T = model.get_observable_hist().squeeze(), model.hist_t

#
# for ax, u in zip(axs, U.T):
#     ax.plot(model.hist_t[N_transient * upsample:], u, '--')
#
# print(U.shape)
# plt.show()


U, T = U[N_transient*upsample:], T[N_transient*upsample:]
U = U[::upsample, :].copy()

# compute normalization factor (range component-wise)
U_data = U.copy()#[:N_washout + N_train].copy()
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M - m
u_mean = U_data.mean(axis=0)

Y_tv = U[N_washout + 1:N_washout + N_train].copy()  # data to match at next timestep

# adding noise to training set inputs with sigma_n the noise of the data
# improves performance and regularizes the error as a function of the hyperparameters

seed = 0
rnd1 = np.random.RandomState(seed)
noisy = True
if noisy:
    data_std = np.std(U, axis=0)
    sigma_n = 1e-6  # change this to increase/decrease noise in training inputs (up to 1e-1)
    for i in range(dim):
        U[:, i] += rnd1.normal(0, sigma_n * data_std[i], len(U))

# washout
U_washout = U[:N_washout].copy()

# data to be used for training + validation
U_tv = U[N_washout:N_washout + N_train - 1].copy()  # inputs

# network parameters
bias_in = np.array([np.mean(np.abs((U_data - u_mean) / norm))])  # input bias (average absolute value of the inputs)
bias_out = np.array([1.])  # output bias

N_units = 200  # neurons
connectivity = 3
sparseness = 1 - connectivity / (N_units - 1)

tikh = np.array([1e-6, 1e-9, 1e-12, 1e-16])  # Tikhonov factor (optimize among the values in this list)

n_grid = 6
n_tot = 40

ranges = dict(rho_range=(.1, 1.),
              sigma_range=(np.log10(0.05), np.log10(5.)))
param_grid, search_space = [[None] * 2 for _ in range(2)]
for hpi, key in enumerate(ranges.keys()):
    param_grid[hpi] = np.linspace(*ranges[key], n_grid)
    search_space[hpi] = Real(*ranges[key], name=key)

# The first n_grid^2 points are from grid search
x1 = product(*param_grid, repeat=1)  # list of tuples
x1 = [list(sg) for sg in x1]



# Number of Networks in the ensemble
ensemble = 3
# Which validation strategy (implemented in Val_Functions.ipynb)
val = RVC_Noise
N_fo = 25  # number of validation intervals
N_in = N_washout  # timesteps before the first validation interval (can't be 0 due to implementation)
N_fw = (N_train - N_val) // (
        N_fo - 1)  # how many steps forward the validation interval is shifted (in this way they are evenly spaced)
N_splits = 4  # reduce memory requirement by increasing N_splits

# Quantities to be saved
par = np.zeros((ensemble, 4))  # GP parameters
x_iters = np.zeros((ensemble, n_tot, 2))  # coordinates in hp space where f has been evaluated
f_iters = np.zeros((ensemble, n_tot))  # values of f at those coordinates
minimum = np.zeros((ensemble, 4))  # minima found per each member of the ensemble

# to store optimal hyperparameters and matrices
tikh_opt = np.zeros(n_tot)
Woutt = np.zeros(((ensemble, N_units + 1, dim)))
Win_ensemble = []  # save as list to keep single elements sparse
W_ensemble = []

# save the final gp reconstruction for each network
gps = [None] * ensemble

# to print performance of every set of hyperparameters
print_flag = False

# optimize ensemble networks (to account for the random initialization of the input and state matrices)
for i in range(ensemble):

    print('Realization    :', i + 1)
    k = 0

    # Win and W generation
    seed = i + 1
    W, Win = generate_W_Win(seed)

    # Bayesian Optimization
    tt = time.time()
    res = g(val)
    print('Total time for the network:', time.time() - tt)

    # Saving Quantities for post_processing
    gps[i] = res.models[-1]
    gp = gps[i]
    x_iters[i] = np.array(res.x_iters)
    f_iters[i] = np.array(res.func_vals)
    minimum[i] = np.append(res.x, [tikh_opt[np.argmin(f_iters[i])], res.fun])

    # saving matrices
    Woutt[i] = train_save_n(U_washout, U_tv, Y_tv,
                            minimum[i, 2], 10 ** minimum[i, 1], minimum[i, 0], minimum[i, 3])
    Win_ensemble += [Win]
    W_ensemble += [W]

    # Plotting Optimization Convergence for each network
    print('Best Results: x', minimum[i, 0], 10 ** minimum[i, 1], minimum[i, 2], 'f', -minimum[i, -1])
    plt.rcParams["figure.figsize"] = (15, 2)
    plt.figure()
    plot_convergence(res)

# ================================ TSTS =====================================#
N_test = 50  # number of intervals in the test set
N_tstart = N_washout + N_train  # where the first test interval starts
N_intt = 15 * N_lyap  # length of each test set interval

# #prediction horizon normalization factor and threshold
sigma_ph = np.sqrt(np.mean(np.var(U, axis=1)))
threshold_ph = 0.5

ensemble_test = 3

for j in range(ensemble_test):

    print('Realization    :', j + 1)

    # load matrices and hyperparameters
    Wout = Woutt[j].copy()
    Win = Win_ensemble[j]  # csr_matrix(Win_ensemble[j])
    W = W_ensemble[j]  # csr_matrix(W_ensemble[j])
    rho = minimum[j, 0].copy()
    sigma_in = 10 ** minimum[j, 1].copy()
    print('Hyperparameters:', rho, sigma_in)

    # to store prediction horizon in the test set
    PH = np.zeros(N_test)

    # to plot results
    plot = True
    if plot:
        n_plot = 2
        plt.rcParams["figure.figsize"] = (15, 3 * n_plot)
        plt.figure()
        plt.tight_layout()

    # run different test intervals
    for i in range(N_test):

        # data for washout and target in each interval
        U_wash = U[N_tstart - N_washout + i * N_intt: N_tstart + i * N_intt].copy()
        Y_t = U[N_tstart + i * N_intt: N_tstart + i * N_intt + N_intt].copy()

        # washout for each interval
        Xa1 = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)
        Uh_wash = np.dot(Xa1, Wout)

        # Prediction Horizon
        Yh_t = closed_loop(N_intt - 1, Xa1[-1], Wout, sigma_in, rho)[0]
        Y_err = np.sqrt(np.mean((Y_t - Yh_t) ** 2, axis=1)) / sigma_ph
        PH[i] = np.argmax(Y_err > threshold_ph) / N_lyap
        if PH[i] == 0 and Y_err[0] < threshold_ph:
            PH[i] = N_intt / N_lyap  # (in case PH is larger than interval)

        if plot:
            # left column has the washout (open-loop) and right column the prediction (closed-loop)
            # only first n_plot test set intervals are plotted
            if i < n_plot:
                plt.subplot(n_plot, 2, 1 + i * 2)
                xx = np.arange(U_wash[:, -2].shape[0]) / N_lyap
                plt.plot(xx, U_wash[:, -2], 'w', label='True')
                plt.plot(xx, Uh_wash[:-1, -2], '--r', label='ESN')
                plt.plot(xx, U_wash, 'w')
                plt.plot(xx, Uh_wash[:-1], '--r')
                plt.ylim(Y_t.min() - .1, Y_t.max() + .1)
                plt.xlabel('Time[Lyapunov Times]')
                if i == 0:
                    plt.legend(ncol=2)

                plt.subplot(n_plot, 2, 2 + i * 2)
                xx = np.arange(Y_t[:, -2].shape[0]) / N_lyap
                plt.plot(xx, Y_t, 'w')
                plt.plot(xx, Yh_t, '--r')
                plt.ylim(Y_t.min() - .1, Y_t.max() + .1)
                plt.xlabel('Time [Lyapunov Times]')

    # Percentiles of the prediction horizon
    print('PH quantiles [Lyapunov Times]:',
          np.quantile(PH, .75), np.median(PH), np.quantile(PH, .25))
    print('')

plt.show()
