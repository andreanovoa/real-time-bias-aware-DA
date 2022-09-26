# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:06:13 2022

@author: an553
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
# import os as os
# os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
import time

from datetime import date

# % LOAD REQUIRED FUNCTIONS AND ENVIRONMENTS 
exec(open("fns_training.py").read())

##  SET TRAINING PARAMETERS (from input dict) __________________________________________________
try:
    data = trainData
    plt.figure()
    plt.plot(data[:,0])
    plt.show()
except:
    filename = 'data/Rijke_2022-08-31_short_bias'
    # data = np.load(filename + '.npz')
    # file = data.files[0]
    # data = data[file]
    # raise Exception('Bias not defined')
    # plt.figure()
    # plt.plot(data[:,0])
    # plt.show()
try:
    dt = dt
except:
    dt = 1E-4
    print('Set default value for dt =', dt)
try:
    dt_ESN = dt_ESN
    upsample = int(dt_ESN / dt)
    # print('upsample =', upsample)
    # print('dt =', dt)
    # print('dt =', dt_ESN)
except:
    try:
        upsample = upsample
    except:
        upsample = 5
        print('Set default value for upsample =', upsample)

    dt_ESN = dt * upsample  # ESN time step

try:
    t_wash = t_wash
except:
    t_wash = 2.5e-2
    print('Set default value for t_wash =', t_wash)
try:
    t_train = t_train
except:
    t_train = 1.
    print('Set default value for t_train =', t_train)
try:
    t_val = t_val
except:
    t_val = 0.2
    print('Set default value for t_val =', t_val)
try:
    test_run = test_run
except:
    test_run = True
    print('Set default value for test_run =', test_run)

print('\n -------------------- TRAINING PARMETERS -------------------- \n',
      'Data filename: ', filename, '\n', 'Training time: ', t_train,
      's \n Validation time: ', t_val, 's', '\n', 'Washout time: ', t_wash,
      's \n Data length: ', len(data) * dt,
      's \n Upsample: ', upsample, '\n', 'Run test?: ', test_run)

# %%
U = data[::upsample]
if len(np.shape(U)) == 1:
    U = np.expand_dims(U, 1)
N_lat = U.shape[1]  # number of dimensions

#  SEPARATE INTO WASH/TRAIN/VAL SETS ________________________________________
N_wash = int(t_wash / dt_ESN)
N_train = int(t_train / dt_ESN)
N_val = int(t_val / dt_ESN)  # length of the data used is train+val
N_tv = N_train + N_val
N_wtv = N_wash + N_tv  # useful for compact code later

U_data = U[:N_wtv]
m = U_data.min(axis=0)
M = U_data.max(axis=0)
norm = M - m  # compute norm (normalize inputs by component range)

U_wash = U[:N_wash]
U_tv = U[N_wash:N_wtv - 1]
Y_tv = U[N_wash + 1:N_wtv].reshape(1, N_tv - 1, N_lat)

# ADD NOISE _________________________________________________________________
# Add noise to inputs and targets during training. Larger noise_level
# promote stability in long term, but hinders time accuracy
noisy = True
noise_level = 0.03 # try increasing if blowing up
noises = np.array([noise_level])  # target noise
U_std = np.std(U, axis=0)

seed = 0
rnd = np.random.RandomState(seed)

if noisy:  # input noise (it is not optimized)
    for i in range(N_lat):
        U_tv[:, i] = U_tv[:, i].copy() + \
                     rnd.normal(0, noise_level * U_std[i], N_tv - 1)

    Y_tv = np.zeros((len(noises), N_tv - 1, N_lat))
    for jj in range(noises.size):
        for i in range(N_lat):
            Y_tv[jj, :, i] = U[N_wash + 1:N_wtv, i].copy() + \
                             rnd.normal(0, noises[jj] * U_std[i], N_tv - 1)

# INITIALISE ESN HYPERPARAMETRES ____________________________________________
bias_in = .1  # input bias
bias_out = 1.0  # output bias
N_units = 100  # neurones in the reservoir

dim = U.shape[1]  # dimension of inputs (and outputs)
tikh = np.array([1e-4, 1e-8, 1e-12, 1e-16])  # Tikhonov

connect = 5  # average neuron connections
sparse = 1 - connect / (N_units - 1)

# GRID SEARCH AND BAYESIAN OPTIMISATION PARAMS ______________________________
n_tot = 20  # Total Number of Function Evaluatuions
n_in = 0  # Number of Initial random points
# Range for hyperparametera (spectral radius and input scaling)
rho_ = [.5, 1.2]
sigin_ = [np.log10(1e-3), .3]

# The first n_grid^2 points are from grid search 
# if n_grid**2 < n_tot, perform Bayesian Optimization
n_grid = 4
if n_grid > 0:
    x1 = [[None] * 2 for i in range(n_grid ** 2)]
    k = 0
    for i in range(n_grid):
        for j in range(n_grid):
            x1[k] = [rho_[0] + (rho_[1] - rho_[0]) / (n_grid - 1) * i,
                     sigin_[0] + (sigin_[1] - sigin_[0]) / (n_grid - 1) * j]
            k += 1

# range for hyperparameters
search_space = [Real(rho_[0], rho_[1], name='rho_W'),
                Real(sigin_[0], sigin_[1], name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) * \
          Matern(length_scale=[0.2, 0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1))

print('\n -------------------- HYPERPARAMETER SEARCH ---------------------\n',
      str(n_grid) + 'x' + str(n_grid) +
      ' grid points and ' + str(n_tot - n_grid ** 2) +
      ' points with Bayesian Optimization')


## HYPERPARAMS OPTIMISATION FUN ______________________________________________     nested function?
def g(val):
    # Gaussian Process reconstruction
    b_e = GPR(kernel=kernell,
              normalize_y=True,  # mean = avg of objective fun
              n_restarts_optimizer=3,  # num of random starts
              noise=1e-10,  # for numerical stability
              random_state=10)  # seed
    # Bayesian Optimization
    res = skopt.gp_minimize(val,  # function to minimize
                            search_space,  # bounds
                            base_estimator=b_e,  # GP kernel
                            acq_func="gp_hedge",  # acquisition function
                            n_calls=n_tot,  # num of evaluations
                            x0=x1,  # Initial grid points
                            n_random_starts=n_in,  # num of random inits
                            n_restarts_optimizer=3,  # tries per acquisition
                            random_state=10)  # seed
    return res


## TRAIN & VALIDATE NETWORK __________________________________________________
ti = time.time()  # check time
ens = 1  # Number of Networks in the ensemble
val = RVC_Noise  # Which validation strategy
N_fo = 15  # number of folds
N_in = 0  # interval before the first fold
N_fw = N_wash // 2  # NUM Steps forward the val interval is shifted
# (N_fw*N_fo has to be smaller than N_train)
# Quantities to be saved
par = np.zeros((ens, 4))  # GP parameters
x_iters = np.zeros((ens, n_tot, 2))  # Coord in hp space to evaluate f
f_iters = np.zeros((ens, n_tot))  # values of f at those coordinates
mins = np.zeros((ens, 5))  # minima found per each ensemble member
tikh_opt = np.zeros(n_tot)  # optimal tikhonov
noise_opt = np.zeros(n_tot)  # optimal noise
Woutt = np.zeros(((ens, N_units + 1, dim)))  # Output matrix
Winn = np.zeros((ens, dim + 1, N_units))  # Input matrix
Ws = np.zeros((ens, N_units, N_units))  # State matrix
gps = [None] * ens  # save the gp reconstruction for each network
for i in range(ens):
    k = 0
    seed = i + 1
    rnd = np.random.RandomState(seed)
    ## ======================== Win & W generation ======================== ##
    Win = np.zeros((dim + 1, N_units))
    for j in range(N_units):  # only one element different from zero per row
        Win[rnd.randint(0, dim + 1), j] = rnd.uniform(-1, 1)
    W = rnd.uniform(-1, 1, (N_units, N_units)) \
        * (rnd.rand(N_units, N_units) < (1 - sparse))  # set sparseness
    rho_W = np.max(np.abs(np.linalg.eigvals(W)))
    W /= rho_W  # scaled to have unitary spec radius
    ## ======================= Bayesian Optimization ====================== ##
    res = g(val)
    gps[i] = res.models[-1]
    gp = gps[i]
    x_iters[i] = np.array(res.x_iters)
    f_iters[i] = np.array(res.func_vals)
    mins[i] = np.append(res.x, [tikh_opt[np.argmin(f_iters[i])],
                                noise_opt[np.argmin(f_iters[i])], res.fun])
    params = gp.kernel_.get_params()
    key = sorted(params)
    par[i] = np.array([params[key[2]], params[key[5]][0], params[key[5]][1], gp.noise_])
    # ============================ Train Wout ============================ ##
    Woutt[i] = train_save_n(U_wash, U_tv, U[N_wash + 1:N_wtv],
                            mins[i, 2], 10 ** mins[i, 1],
                            mins[i, 0], mins[i, 3])
    # ========== Save Optimization Convergence for each network ========== ##
    Winn[i] = Win.copy()
    Ws[i] = W.copy()
    print('Realization:', i + 1,
          '\n Time per hyperparameter eval.:', (time.time() - ti) / n_tot,
          '\n Best Results: x', mins[i, 0], 10 ** mins[i, 1], mins[i, 2],
          mins[i, 3], ', f', -mins[i, -1])

# RUN TEST IF REQUESTED _____________________________________________________
if test_run:

    # Select number of tests
    N_t0 = N_wtv + 10  # start test after washout, training and validation data
    max_test_time = len(U[N_t0:]) * dt_ESN
    N_test = 50
    if N_test * t_val > max_test_time:
        N_test = int(np.floor(max_test_time / t_val))
        # print('N_test = ', N_test)

    # Break if not enough data for testing
    if N_test < 1:
        print('Test not performed. Not enough data')
    else:
        subplots = min(10, N_test)  # number of plotted intervals
        plt.rcParams["figure.figsize"] = (10, subplots * 2)
        plt.subplots(subplots, 1)

        sub_loop = 5  # number of closed loop re-initialisations with true data
        N_reinit = N_val // sub_loop  # time between closed loop re-initialisations
        for k in range(ens):
            # load matrices and hyperparameters
            Win = Winn[k].copy()
            W = Ws[k].copy()
            Wout = Woutt[k].copy()
            rho = mins[k, 0].copy()
            sigma_in = 10 ** mins[k, 1].copy()
            errors = np.zeros(N_test)
            # Different intervals in the test set
            i = -1
            ii = -1
            # for i in range(N_test):
            while True:
                i += 1
                ii += 1
                if i >= N_test:
                    break
                # data for washout and target in each interval
                U_wash = U[N_t0 - N_wash + i * N_val: N_t0 + i * N_val]
                Y_t = U[N_t0 + i * N_val: N_t0 + i * N_val + N_reinit * sub_loop]
                # washout for each interval
                xa1 = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]
                Yh_t, xa1 = closed_loop_test(N_reinit - 1, xa1, Y_t[0], Wout, sigma_in, rho)
                try:
                    # if True:
                    # Do the multiple sub_loop inside each test interval
                    if sub_loop > 1:
                        for j in range(sub_loop - 1):
                            Y_start = Y_t[(j + 1) * N_reinit - 1].copy()  #
                            # Y_start    = Yh_t[-1].copy()# #uncomment this to not update input
                            Y1, xa1 = closed_loop_test(N_reinit, xa1, Y_start, Wout, sigma_in, rho)
                            Yh_t = np.concatenate((Yh_t, Y1[1:]))
                    errors[i] = np.log10(np.mean((Yh_t - Y_t) ** 2) / np.mean(norm ** 2))
                    if i < subplots:
                        plt.subplot(subplots, 1, i + 1)
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Y_t[:, ii], 'k',
                                 label='truth dim ' + str(ii))
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Yh_t[:, ii], '--r',
                                 label='ESN dim ' + str(ii))
                        # plt.ylabel('Test ' + str(i))
                        plt.legend(title='Test \\#' + str(i), loc='upper left', bbox_to_anchor=(1.01, 1.01))
                    if ii == dim - 1:
                        ii = -1
                except:
                    raise Exception('error at i = ', i, ', N_test = ', N_test)

        print('Median and max error in test:', np.median(errors), errors.max())
        plt.tight_layout()
        plt.savefig(filename[:-len('bias')] + 'Test_run.pdf')
        plt.close()

# %%
# OUTPUTS: PLOT SEARCH AND SAVE DATA ________________________________________
# Plot Gaussian Process reconstruction for each network in the ensemble after
# n_tot evaluations. The GP reconstruction is based on the n_tot function 
# evaluations decided in the search
n_len = 100  # points to evaluate the GP at
xx, yy = np.meshgrid(np.linspace(rho_[0], rho_[1], n_len),
                     np.linspace(sigin_[0], sigin_[1], n_len))
x_x = np.column_stack((xx.flatten(), yy.flatten()))
x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format
y_pred = np.zeros((ens, n_len, n_len))

fig = plt.figure(figsize=[10, 5], tight_layout=True)
for i in range(ens):
    # retrieve the gp reconstruction
    plt.subplot(ens, 1, 1 + i)
    gp = gps[i]
    amin = np.amin([10, f_iters.max()])
    # Final GP reconstruction for each realization at the evaluation points
    y_pred[i] = np.clip(-gp.predict(x_gp), a_min=-amin,
                        a_max=-f_iters.min()).reshape(n_len, n_len)
    # Plot GP Mean
    plt.xlabel('Spectral Radius')
    plt.ylabel('Input Scaling (log-scale)')
    CS = plt.contourf(xx, yy, y_pred[i], levels=20, cmap='Blues')
    cbar = plt.colorbar();
    cbar.set_label('-$\log_{10}$(MSE)', labelpad=15)
    CSa = plt.contour(xx, yy, y_pred[i], levels=20, colors='black',
                      linewidths=1, linestyles='solid', alpha=0.3)
    #   Plot the n_tot search points
    plt.plot(x_iters[i, :n_grid ** 2, 0], x_iters[i, :n_grid ** 2, 1], 'v', c='w',
             alpha=0.8, markeredgecolor='k', markersize=10)
    plt.plot(x_iters[i, n_grid ** 2:, 0], x_iters[i, n_grid ** 2:, 1], 's', c='w',
             alpha=0.8, markeredgecolor='k', markersize=8)

plt.savefig(filename[:-len('bias')] + 'Hyperparameter_search.pdf')
plt.close()

# print('sigma_in', 10 ** mins[0, 1])
# print('rho', mins[0, 0])
# print('bias_in', bias_in)
# print('bias_out', bias_out)
# print('norm', norm)
# %% ====================================================================== %%#
np.savez(filename[:-len('bias')] + 'ESN',
         t_train=t_train,
         t_wash=t_wash,
         t_val=t_val,
         norm=norm,
         Win=Winn,
         Wout=Woutt,
         W=Ws,
         dt_ESN=dt_ESN,
         N_wash=int(N_wash),
         N_unit=int(N_units),
         N_dim=int(dim),
         bias_in=bias_in,
         bias_out=bias_out,
         rho=mins[0, 0],
         sigma_in=10 ** mins[0, 1],
         upsample=int(upsample),
         hyperparameters=[mins[0, 0], 10 ** mins[0, 1], bias_in],
         training_time=(N_train + N_val) * dt_ESN
         )
# print('save location: ' + filename)

print('\n --------------------------------------------------------------- \n')

# %% ====================================================================== %%#
