import numpy as np
import matplotlib.pyplot as plt
import h5py
import os as os

os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core
import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
import time
from datetime import date


def RVC_Noise(x):
    # chaotic Recycle Validation
    global rho, sigma_in, tikh_opt, k, ti, noise_opt
    rho = x[0]
    sigma_in = 10 ** x[1]

    lenn = tikh.size
    len1 = noises.size
    Mean = np.zeros((lenn, len1))

    # Train using tv: training+val, Wout is passed with all the combinations of tikh and target noise
    Xa_train, Wout, LHS0, RHS0 = train_n(U_wash, U_tv, Y_tv, tikh, sigma_in, rho)

    if k == 0:
        print('\t\t rho \t sigma_in \t tikhonov  \t noise \t MSE val ')

    # Different validation folds
    for i in range(N_fo):

        p = N_in + i * N_fw
        Y_val = U[N_wash + p: N_wash + p + N_val].copy()  # data to compare the cloop prediction with

        for jj in range(len1):
            for j in range(lenn):

                Yh_val = closed_loop(N_val - 1, Xa_train[p], Wout[j, jj], sigma_in, rho)[
                    0]  # cloop for each tikh-noise combinatio
                Mean[j, jj] += np.log10(np.mean((Y_val - Yh_val) ** 2) / np.mean(norm ** 2))

                # prevent from diverging to infinity: put MSE equal to 10^10 (useful for hybrid and similar
                # architectures)
                if np.isnan(Mean[j, jj]) or np.isinf(Mean[j, jj]):
                    Mean[j, jj] = 10 * N_fo

    # select and save the optimal tikhonov and noise level in the targets
    a = np.unravel_index(Mean.argmin(), Mean.shape)
    tikh_opt[k] = tikh[a[0]]
    noise_opt[k] = noises[a[1]]
    k += 1
    # if k % 2 == 0:
    print(k, 'Par: {0:.3f} \t {1:.2e} \t {2:.1e}  \t {3:.2f} \t {4:.4f} '.format(rho, sigma_in, tikh[a[0]],
                                                                                     noises[a[1]], Mean[a] / N_fo))

    return Mean[a] / N_fo


## ESN with bias architecture

def step(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
            sigma_in:
            rho: spectral radius
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack((u / norm, np.array([bias_in])))
    # hyperparameters are explicit here
    x_post = np.tanh(np.dot(u_augmented * sigma_in, Win) + rho * np.dot(x_pre, W))
    # output bias added
    x_augmented = np.concatenate((x_post, np.array([bias_out])))
    return x_augmented


def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N = U.shape[0]
    Xa = np.empty((N + 1, N_units + 1))
    Xa[0] = np.concatenate((x0, np.array([bias_out])))  # , U[0]/norm))
    for i in np.arange(1, N + 1):
        Xa[i] = step(Xa[i - 1, :N_units], U[i - 1], sigma_in, rho)

    return Xa


def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N + 1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1, N + 1):
        xa = step(xa[:N_units], Yh[i - 1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa


def train_n(U_wash, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_wash: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf_washout = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1, :N_units]

    ## open-loop train phase
    Xa = open_loop(U_train, xf_washout, sigma_in, rho)

    ## Ridge Regression
    LHS = np.dot(Xa[1:].T, Xa[1:])

    Wout = np.zeros((len(tikh), len(noises), N_units + 1, dim))
    RHS = np.zeros((len(noises), N_units + 1, dim))
    for jj in range(len(noises)):

        RHS[jj] = np.dot(Xa[1:].T, Y_train[jj])

        for j in range(len(tikh)):
            Wout[j, jj] = np.linalg.solve(LHS + tikh[j] * np.eye(N_units + 1), RHS[jj])

    return Xa, Wout, LHS, RHS


def train_save_n(U_wash, U_train, Y_train, tikh, sigma_in, rho, noise):
    """ Trains ESN.
        Args:
            U_wash: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf_washout = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1, :N_units]

    ## open-loop train phase
    Xa = open_loop(U_train, xf_washout, sigma_in, rho)

    ## Ridge Regression
    LHS = np.dot(Xa[1:].T, Xa[1:])
    sh_0 = Y_train.shape[0]

    for i in range(N_lat):
        Y_train[:, i] = Y_train[:, i] + rnd.normal(0, noise * U_std[i], sh_0)
    RHS = np.dot(Xa[1:].T, Y_train)

    Wout = np.linalg.solve(LHS + tikh * np.eye(N_units + 1), RHS)

    return Wout


def closed_loop_test(N, x0, Y0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N + 1, dim))
    Yh[0] = Y0  # np.dot(xa, Wout)
    for i in np.arange(1, N + 1):
        xa = step(xa[:N_units], Yh[i - 1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa
