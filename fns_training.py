import os as os
os.environ["OMP_NUM_THREADS"] = '1'  # imposes only one core
import numpy as np

def RVC_Noise(x):
    # chaotic Recycle Validation
    global rho, sigma_in, tikh_opt, k, ti
    rho = x[0]
    sigma_in = 10 ** x[1]

    len_tikn = tikh.size
    Mean = np.zeros(len_tikn)

    # Train using tv: training+val, Wout is passed with all the combinations of tikh and target noise
    Xa_train, Wout, LHS0, RHS0 = train_n(U_wash, U_tv, Y_tv, tikh, sigma_in, rho)

    if k == 0:
        print('\t\t rho \t sigma_in \t tikhonov  \t MSE val ')

    # Different validation folds
    for i in range(N_fo):

        p = N_in + i * N_fw
        Y_val = U[N_wash + p: N_wash + p + N_val].copy()  # data to compare the cloop prediction with

        for j in range(len_tikn):

            Yh_val = closed_loop(N_val - 1, Xa_train[p], Wout[j], sigma_in, rho)[
                0]  # cloop for each tikh-noise combinatio
            Mean[j] += np.log10(np.mean((Y_val - Yh_val) ** 2) / np.mean(norm ** 2))

            # prevent from diverging to infinity: put MSE equal to 10^10 (useful for hybrid and similar
            # architectures)
            if np.isnan(Mean[j]) or np.isinf(Mean[j]):
                Mean[j] = 10 * N_fo

    # select and save the optimal tikhonov and noise level in the targets
    a = Mean.argmin()
    tikh_opt[k] = tikh[a]
    k += 1
    # if k % 2 == 0:
    print(k, 'Par: {0:.3f} \t {1:.2e} \t {2:.1e}  \t {3:.4f} '.format(rho, sigma_in, tikh[a], Mean[a] / N_fo))

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
    u_augmented = np.hstack((u / norm, bias_in))
    # hyperparameters are explicit here
    # x_post = np.tanh(np.dot(u_augmented * sigma_in, Win) + rho * np.dot(x_pre, W))
    x_post = np.tanh(Win.dot(u_augmented*sigma_in) + W.dot(rho*x_pre))
    # output bias added
    x_augmented = np.concatenate((x_post, bias_out))
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
    Xa[0] = np.concatenate((x0, bias_out)) 
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
    Yh = np.empty((N + 1, N_dim))
    # Yh[0] = np.dot(xa, Wout)
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

    Wout = np.zeros((len(tikh), N_units + 1, N_dim))

    RHS = np.dot(Xa[1:].T, Y_train)

    # print(np.shape(RHS))
    # print(np.shape(Y_train))
    # print(np.shape(Xa))

    for j in range(len(tikh)):
        Wout[j] = np.linalg.solve(LHS + tikh[j] * np.eye(N_units + 1), RHS)

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

    if len(np.shape(U_wash)) < 3:
        num_train_sets = 1
    else:
        num_train_sets = np.shape(U_wash)[-1]

    LHS = 0
    RHS = 0

    for k in range(num_train_sets):
        # washout phase
        xf_washout = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1, :N_units]

        # open-loop train phase
        Xa = open_loop(U_train, xf_washout, sigma_in, rho)

        # Ridge Regression
        LHS += np.dot(Xa[1:].T, Xa[1:])

        for i in range(N_dim):
            Y_train[:, i] = Y_train[:, i] + rnd.normal(0, noise * U_std[i], Y_train.shape[0])

        RHS += np.dot(Xa[1:].T, Y_train)

    # Solve linear regression problem
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
    Yh = np.empty((N + 1, N_dim))
    Yh[0] = Y0  # np.dot(xa, Wout)
    for i in np.arange(1, N + 1):
        xa = step(xa[:N_units], Yh[i - 1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa