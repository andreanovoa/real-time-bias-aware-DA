# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:06:13 2022

@author: an553
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf
import h5py
import os as os

os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
from skopt.plots import plot_convergence
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from datetime import date
from scipy.io import loadmat, savemat
import time

# % LOAD REQUIRED FUNCTIONS AND ENVIRONMENTS 
exec(open("fns_training.py").read())


##  SET TRAINING PARAMETERS (from input dict) __________________________________________________
try:
    data = trainData
    np.save('data/test_results_bias', data, allow_pickle=True)
    # plt.figure()
    # plt.plot(data[:,0])
    # plt.show()
except:
    filename = 'data/test_results_bias.npy'
    data = np.load(filename)
    # data = data['data']
    # plt.figure()
    # plt.plot(data[:, 0])
    # plt.show()

try:
    dt = dt  # original signal dt
except:
    dt = 1E-4
    print('Set default value for dt =', dt)
try:
    upsample = upsample
except:
    upsample = 5
    print('Set default value for upsample =', upsample)

try:
    N_wash = N_wash
except:
    N_wash = 100
    print('Set default value for N_wash =', N_wash)
try:
    t_train = t_train
except:
    t_train = 1.
    print('Set default value for t_train =', t_train)
try:
    t_val = t_val
except:
    t_val = 0.1
    print('Set default value for t_val =', t_val)
try:
    test_run = test_run
except:
    test_run = True
    print('Set default value for test_run =', test_run)

try:
    N_units = N_units  # neurones in the reservoir
except:
    N_units = 100  # neurones in the reservoir
    print('Set default value for N_units =', N_units)

# Force data to be (Nalpha, Nt, Nmic)
if len(np.shape(data)) == 1:  # (Nt,)
    data = np.expand_dims(data, 1)
    data = np.expand_dims(data, 0)
elif len(np.shape(data)) == 2:  # (Nt, Nmic)
    data = np.expand_dims(data, 0)
else:
    if np.argmax(np.shape(data)) == 0:
        data = data.transpose((2, 0, 1))

is_param = data.min(axis=1)[0]-data.max(axis=1)[0] == 0
if any(is_param):
    alpha = data[:, 0, is_param]
    data = data[:, :, ~is_param]
    norm_alpha = np.mean(0.1 / alpha, axis=0)
else:
    alpha = []


#  ____________________________ APPLY UPSAMPLE - CONVERT INTO ESN dt ______________________________________
dt_ESN = dt * upsample  # ESN time step
data = data[:, ::upsample]

#  _____________________________________ DATA AUGMENTATION _____________________________________________
augment_data = True
if augment_data:
    norm_alpha = None
    U = np.vstack([data * 1., data * 1e-2, data * 1e-4])

#  _______________________________ SEPARATE INTO WASH/TRAIN/VAL SETS ________________________________________
N_dim = U.shape[-1]  # dimension of inputs (and outputs)
N_alpha = U.shape[0] # number of training timeseries
N_train = int(t_train / dt_ESN)
N_val = int(t_val / dt_ESN)  # length of the data used is train+val
N_tv = N_train + N_val
N_wtv = N_wash + N_tv  # useful for compact code later

U_data = U[:, :N_wtv]
# m = U_data.min(axis=1).min(axis=0)
# M = U_data.max(axis=1).max(axis=0)
m = np.mean(U_data.min(axis=1), axis=0)
M = np.mean(U_data.max(axis=1), axis=0)

norm = M - m  # compute norm (normalize inputs by component range)

U_wash = U[:, :N_wash]
U_tv = U[:, N_wash:N_wtv - 1]
Y_tv = U[:, N_wash + 1:N_wtv]

# ___________________________________________ ADD NOISE ______________________________________________________
# Add noise to inputs and targets during training. Larger noise_level promotes stability in long term,
# but hinders time accuracy
noisy = True
noise_level = 0.03  # try increasing if blowing up
U_std = np.std(U, axis=1)

seed = 0
rnd = np.random.RandomState(seed)

if noisy:  # input noise (it is not optimized)
    for j in range(N_dim):
        for i in range(N_alpha):
            U_tv[i, :, j] += rnd.normal(0, noise_level * U_std[i, j], N_tv - 1)

# ________________________________________ INITIALISE ESN HYPERPARAMETRES _________________________________________
bias_in = np.array([.1])  # input bias
bias_out = np.array([1.0])  # output bias
connect = 5  # average neuron connections
sparse = 1 - connect / (N_units - 1)

# Range for hyperparametera (spectral radius and input scaling)
rho_ = [.7, 1.1]
sigin_ = [np.log10(1e-4), np.log10(1e-1)]
tikh = np.array([1e-10, 1e-12, 1e-16])  # Tikhonov

# _________________________________ GRID SEARCH AND BAYESIAN OPTIMISATION PARAMS ____________________________________
n_tot = 20  # Total Number of Function Evaluatuions
n_in = 0  # Number of Initial random points

# The first n_grid^2 points are from grid search. If n_grid**2 < n_tot, perform Bayesian Optimization
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
search_space = [Real(rho_[0], rho_[1], name='spectral_radius'),
                Real(sigin_[0], sigin_[1], name='input_scaling')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) * \
          Matern(length_scale=[0.2, 0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1))

print('\n -------------------- HYPERPARAMETER SEARCH ---------------------\n', str(n_grid) + 'x' + str(n_grid) +
      ' grid points and ' + str(n_tot - n_grid ** 2) + ' points with Bayesian Optimization')


# _____________________________________ HYPERPARAMS OPTIMISATION FUN _____________________________________
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


# ________________________________ TRAIN & VALIDATE NETWORK __________________________________
ti = time.time()  # check time

val = RVC_Noise  # Which validation strategy
N_fo = 2  # number of folds
N_in = 0  # interval before the first fold
N_fw = (N_train-N_val)//(N_fo-1)  # num steps forward the validation interval is shifted (evenly spaced)

# Quantities to be saved
tikh_opt = np.zeros(n_tot)  # optimal tikhonov

gps = [None]  # save the gp reconstruction for each network

k = 0
seed = 1
rnd = np.random.RandomState(seed)

# ==================================== Win & W generation ==================================== ##
N_aug = 1
if norm_alpha is not None:
    N_aug += sum(is_param)
Win = lil_matrix((N_units, N_dim + N_aug))
for j in range(N_units):
    Win[j, rnd.randint(0, N_dim + 1)] = rnd.uniform(-1, 1)
Win = Win.tocsr()

W = csr_matrix( rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1 - sparse)))
spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
W = (1 / spectral_radius) * W  # scaled to have unitary spec radius

# ====================================  Bayesian Optimization ==================================== ##
res = g(val)
gp = res.models[-1]
x_iters = np.array(res.x_iters)
f_iters = np.array(res.func_vals)
minimum = np.append(res.x, [tikh_opt[np.argmin(f_iters)], res.fun])
params = gp.kernel_.get_params()
key = sorted(params)
params_gp = np.array([params[key[2]], params[key[5]][0], params[key[5]][1], gp.noise_])

# =========================================  Train Wout ========================================== ##
Wout = train_save_n(U_wash, U_tv, U[:, N_wash + 1:N_wtv], minimum[2], 10**minimum[1], minimum[0])
if len(Wout.shape) == 1:
    Wout = np.expand_dims(Wout, axis=1)
    print(Wout.shape)


print('\n Time per hyperparameter eval.:', (time.time() - ti) / n_tot,
      '\n Best Results: x', minimum[0], 10 ** minimum[1], minimum[2], ', f', -minimum[-1])


pdf = plt_pdf.PdfPages(filename[:-len('bias')] + 'ESN'+str(N_units) +'_Training.pdf')

fig = plt.figure()
plot_convergence(res)
pdf.savefig(fig)
plt.close(fig)

# %%
# ________________________________ OUTPUTS: PLOT SEARCH AND SAVE DATA ________________________________
# Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations.
# The GP reconstruction is based on the n_tot function evaluations decided in the search
n_len = 100  # points to evaluate the GP at
xx, yy = np.meshgrid(np.linspace(rho_[0], rho_[1], n_len),
                     np.linspace(sigin_[0], sigin_[1], n_len))
x_x = np.column_stack((xx.flatten(), yy.flatten()))
x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

fig = plt.figure(figsize=[10, 5], tight_layout=True)
# retrieve the gp reconstruction
amin = np.amin([10, f_iters.max()])
# Final GP reconstruction for each realization at the evaluation points
y_pred = np.clip(-gp.predict(x_gp), a_min=-amin,
                 a_max=-f_iters.min()).reshape(n_len, n_len)
# Plot GP Mean
plt.xlabel('Spectral Radius')
plt.ylabel('Input Scaling (log-scale)')
CS = plt.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
cbar = plt.colorbar()
cbar.set_label('-$\log_{10}$(MSE)', labelpad=15)
CSa = plt.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid', alpha=0.3)
#   Plot the n_tot search points
plt.plot(x_iters[:n_grid ** 2, 0], x_iters[:n_grid ** 2, 1], 'v', c='w',
         alpha=0.8, markeredgecolor='k', markersize=10)
plt.plot(x_iters[n_grid ** 2:, 0], x_iters[n_grid ** 2:, 1], 's', c='w',
         alpha=0.8, markeredgecolor='k', markersize=8)

pdf.savefig(fig)
plt.close(fig)

# RUN TEST IF REQUESTED _____________________________________________________
if test_run:
    # Select number of tests
    N_t0 = N_wtv + 10  # start test after washout, training and validation data
    max_test_time = np.shape(U[:, N_t0:])[1] * dt_ESN
    N_test = 50
    if N_test * t_val > max_test_time:
        N_test = int(np.floor(max_test_time / t_val))
    # Break if not enough data for testing
    if N_test < 1:
        print('Test not performed. Not enough data')
    else:
        medians_alpha = []
        maxs_alpha = []
        alph = None
        sub_loop = 5  # number of closed loop re-initialisations with true data
        N_reinit = N_val // sub_loop  # time between closed loop re-initialisations

        # load matrices and hyperparameters
        rho = minimum[0].copy()
        sigma_in = 10 ** minimum[1].copy()
        for kk in range(N_alpha):
            if norm_alpha is not None:
                alph = alpha[kk]
            subplots = min(10, N_test)  # number of plotted intervals
            plt.rcParams["figure.figsize"] = (10, subplots * 2)
            plt.subplots(subplots, 1)

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
                U_wash = U[kk, N_t0 - N_wash + i * N_val: N_t0 + i * N_val]
                Y_t = U[kk, N_t0 + i * N_val: N_t0 + i * N_val + N_reinit * sub_loop]
                # washout for each interval
                xa1 = open_loop(U_wash, np.zeros(N_units), sigma_in, rho, alph)[-1]
                Yh_t, xa1 = closed_loop(N_reinit - 1, xa1, Wout, sigma_in, rho, alph)
                try:
                    # Do the multiple sub_loop inside each test interval
                    if sub_loop > 1:
                        for j in range(sub_loop - 1):
                            Y_start = Y_t[(j + 1) * N_reinit - 1].copy()  #
                            Y1, xa1 = closed_loop(N_reinit, xa1, Wout, sigma_in, rho, alph)
                            Yh_t = np.concatenate((Yh_t, Y1[1:]))
                    errors[i] = np.log10(np.mean((Yh_t - Y_t) ** 2) / np.mean(norm ** 2))
                    if i < subplots:
                        plt.subplot(subplots, 1, i + 1)
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Y_t[:, ii], 'k',
                                 label='truth N_dim ' + str(ii))
                        plt.plot(np.arange(N_reinit * sub_loop) * dt_ESN, Yh_t[:, ii], '--r',
                                 label='ESN N_dim ' + str(ii))
                        plt.legend(title='Test \#' + str(i), loc='upper left', bbox_to_anchor=(1.01, 1.01))
                    if ii == N_dim - 1:
                        ii = -1
                except:
                    raise Exception('error at i = ', i, ', N_test = ', N_test)
            maxs_alpha.append(errors.max())
            medians_alpha.append(np.median(errors))
            plt.tight_layout()
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)
        print('Median and max error in', N_alpha, ' test:', np.median(medians_alpha), max(maxs_alpha))

# ===================================== Save output and images ========================================== %

save_dict = dict(t_train=t_train,
                 t_val=t_val,
                 norm=norm,
                 Win=Win.T,
                 Wout=Wout,
                 W=W,
                 dt_ESN=dt_ESN,
                 N_augment=int(N_alpha),
                 N_wash=int(N_wash),
                 N_units=int(N_units),
                 N_dim=int(N_dim),
                 bias_in=bias_in,
                 bias_out=bias_out,
                 rho=minimum[0],
                 sigma_in=10 ** minimum[1],
                 upsample=int(upsample),
                 hyperparameters=[minimum[0], 10 ** minimum[1], bias_in],
                 training_time=(N_train + N_val) * dt_ESN,
                 filename=filename
                 )
if norm_alpha is not None:
    save_dict['norm_alpha'] = norm_alpha

savemat(filename[:-len('bias')] + 'ESN'+str(N_units)+'.mat', save_dict, oned_as='column')


pdf.close()


print('\n --------------------------------------------------------------- \n')

# %% ====================================================================== %%#

