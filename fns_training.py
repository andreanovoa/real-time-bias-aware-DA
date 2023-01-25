import os as os
os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core
import numpy as np
import multiprocessing as mp
from functools import partial
import time

def RVC_Noise(x):
    # chaotic Recycle Validation
    global rho, sigma_in, tikh_opt, k, ti
    rho = x[0]
    sigma_in = 10 ** x[1]

    len_tikn = tikh_.size
    Mean = np.zeros(len_tikn)

    # Train using tv: training+val, Wout is passed with all the combinations of tikh_ and target noise
    Xa_train, Wout, LHS0, RHS0 = train_n(U_wash, U_tv, Y_tv, tikh_, sigma_in, rho)

    if k == 0:
        print('\t\t rho \t sigma_in \t tikhonov  \t MSE val ')
    alph = None
    for kk in range(N_alpha):
        if norm_alpha is not None:
            alph = alpha[kk]
        # Different validation folds
        for i in range(N_fo):
            p = N_in + i * N_fw
            Y_val = U[kk, N_wash + p: N_wash + p + N_val].copy()  # data to compare the cloop prediction with

            for j in range(len_tikn):
                Yh_val = closed_loop(N_val - 1, Xa_train[kk, p], Wout[j], sigma_in, rho, alph)[0]  # cloop for each tikh_-noise combinatio
                Mean[j] += np.log10(np.mean((Y_val - Yh_val) ** 2) / np.mean(norm ** 2))

                # prevent from diverging to infinity: put MSE equal to 10^10 (useful for hybrid and similar
                # architectures)
                if np.isnan(Mean[j]) or np.isinf(Mean[j]):
                    Mean[j] = 10 * N_fo

    # select and save the optimal tikhonov and noise level in the targets
    a = Mean.argmin()
    tikh_opt[k] = tikh_[a]
    k += 1
    # if k % 2 == 0:
    print(k, 'Par: {0:.3f} \t {1:.2e} \t {2:.1e}  \t {3:.4f} '.format(rho, sigma_in, tikh_[a], Mean[a] / N_fo / N_alpha))

    return Mean[a] / N_fo / N_alpha


## ESN with bias architecture

def step(x_pre, u, sigma_in, rho, alpha):
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
    u_augmented = np.hstack((u / norm, bias_in))

    if norm_alpha is not None:
        u_augmented = np.hstack((u_augmented, alpha/norm_alpha))


    # hyperparameters are explicit here
    x_post = np.tanh(Win.dot(u_augmented*sigma_in) + W.dot(rho*x_pre))

    # output bias added
    x_augmented = np.concatenate((x_post, bias_out))
    return x_augmented


def open_loop(U_o, x0, sigma_in, rho, alpha):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N = U_o.shape[0]
    Xa = np.empty((N + 1, N_units + 1))
    Xa[0] = np.concatenate((x0, bias_out))
    for i in np.arange(1, N + 1):
        Xa[i] = step(Xa[i - 1, :N_units], U_o[i - 1], sigma_in, rho, alpha)

    return Xa


def closed_loop(N, x0, Wout, sigma_in, rho, alpha):
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
    Yh = np.empty((N + 1, N_dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1, N + 1):
        xa = step(xa[:N_units], Yh[i - 1], sigma_in, rho, alpha)
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

    LHS, RHS = 0., 0.
    Xa = []
    alph = None
    # time1 = time.time()
    for kk in range(N_alpha):
        if norm_alpha is not None:
            alph = alpha[kk]
        # Washout phase
        xf_washout = open_loop(U_wash[kk], np.zeros(N_units), sigma_in, rho, alph)[-1, :N_units]

        # Open-loop train phase
        Xa.append(open_loop(U_train[kk], xf_washout, sigma_in, rho, alph))

        # Compute matrices for linear regression system
        LHS += np.dot(Xa[kk][1:].T, Xa[kk][1:])
        RHS += np.dot(Xa[kk][1:].T, Y_train[kk])

    # print('for loop: ', time.time()-time1)
    #
    # time1 = time.time()
    # with mp.Pool() as pool:
    #     sol = [pool.apply_async(open_loop, (U_wash[kk], np.zeros(N_units), sigma_in, rho, alph))
    #            for kk in range(N_alpha)]
    #     xf_washout = [s.get() for s in sol]
    #
    #     # Open-loop train phase
    #     sol = [pool.apply_async(open_loop, (U_train[kk], xf_washout[kk], sigma_in, rho, alph))
    #           for kk in range(N_alpha)]
    #
    #     Xa = [s.get() for s in sol]
    #
    # LHS2, RHS2 = 0., 0.
    # for kk in range(len(Xa)):
    #     LHS2 += np.dot(Xa[kk][1:].T, Xa[kk][1:])
    #     RHS2 += np.dot(Xa[kk][1:].T, Y_train[kk])
    #
    # print('parallel loop: ', time.time()-time1)
    #
    # print(sum(LHS2-LHS), sum(RHS2-RHS))

    Wout = np.empty((len(tikh), N_units + 1, N_dim))
    for j in range(len(tikh)):
        if j == 0: #add tikhonov to the diagonal (fast way that requires less memory)
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j]
        else:
            LHS.ravel()[::LHS.shape[1]+1] += tikh[j] - tikh[j-1]

        Wout[j] = np.linalg.solve(LHS, RHS)
    Xa = np.array(Xa)
    return Xa, Wout, LHS, RHS


def train_save_n(U_wash, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_wash: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """


    LHS = 0.
    RHS = 0.
    alph = None
    for kk in range(N_alpha):
        if norm_alpha is not None:
            alph = alpha[kk]
        # Washout phase
        xf_washout = open_loop(U_wash[kk], np.zeros(N_units), sigma_in, rho, alph)[-1, :N_units]

        # Open-loop train phase
        Xa = open_loop(U_train[kk], xf_washout, sigma_in, rho, alph)

        # Compute matrices for linear regression system
        LHS += np.dot(Xa[1:].T, Xa[1:])
        RHS += np.dot(Xa[1:].T, Y_train[kk])

    # Add tikhonov regularisation to the diagonal entries
    LHS.ravel()[::LHS.shape[1]+1] += tikh
    # Solve linear regression problem
    Wout = np.linalg.solve(LHS, RHS)

    return Wout
