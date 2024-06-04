import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
import matplotlib as mpl
from scipy.io import loadmat, savemat
import time
from skopt.plots import plot_convergence

def forward_euler(ddt, u0, T, *args):
    u = np.empty((T.size, u0.size))
    u[0] = u0
    for i in range(1, T.size):
        u[i] = u[i-1] + (T[i] - T[i-1]) * ddt(u[i-1], T[i-1], *args)
    return u

def ddt(u, t, params):
    beta, rho, sigma = params
    x, y, z = u
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def solve_ode(N, dt, u0, params=[8/3, 28, 10]):
    """
        Solves the ODEs for N time steps starting from u0.
        Returned values are normalized.

        Args:
            N: number of time steps
            u0: initial condition
            norm: normalisation factor of u0 (None if not normalised)
            params: parameters for ODE
        Returns:
            normalized time series of shape (N+1, u0.size)
    """

    T = np.arange(N+1) * dt
    U = forward_euler(ddt, u0, T, params)

    return U, T

## ESN with bias architecture

def step(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack(((u - u_mean) / norm, bias_in))
    # reservoir update
    x_post = np.tanh(Win.dot(u_augmented * sigma_in) + W.dot(rho * x_pre))
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
    Yh = np.empty((N + 1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1, N + 1):
        xa = step(xa[:N_units], Yh[i - 1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)  # np.linalg.multi_dot([xa, Wout])

    return Yh, xa


def train_n(U_washout, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## initial washout phase
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1, :N_units]

    ## splitting training in N_splits to save memory
    LHS = 0
    RHS = 0

    U_train_split = np.array_split(U_train, N_splits, axis=0)
    Y_train_split = np.array_split(Y_train, N_splits, axis=0)

    for U_t, Y_t in zip(U_train_split, Y_train_split):
        print(sigma_in)
        ## open-loop train phase
        Xa1 = open_loop(U_t, xf, sigma_in, rho)[1:]
        xf = Xa1[-1, :N_units].copy()

        ##computing the matrices for the linear system
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_t)


    Wout = np.empty((len(tikh), N_units + 1, dim))

    # solve linear system for different Tikhonov
    for j in range(len(tikh)):
        if j == 0:  # add tikhonov to the diagonal (fast way that requires less memory)
            LHS.ravel()[::LHS.shape[1] + 1] += tikh[j]
        else:
            LHS.ravel()[::LHS.shape[1] + 1] += tikh[j] - tikh[j - 1]

        # solve linear system
        Wout[j] = np.linalg.solve(LHS, RHS)

    return Wout, LHS, RHS


def train_save_n(U_washout, U_train, Y_train, tikh, sigma_in, rho, noise):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1, :N_units]

    LHS, RHS = 0, 0

    U_train_split = np.array_split(U_train, N_splits, axis=0)
    Y_train_split = np.array_split(Y_train, N_splits, axis=0)

    for U_t, Y_t in zip(U_train_split, Y_train_split):
        ## open-loop train phase
        Xa1 = open_loop(U_t, xf, sigma_in, rho)[1:]
        xf = Xa1[-1, :N_units].copy()

        ##computing the matrices for the linear system
        LHS += np.dot(Xa1.T, Xa1)
        RHS += np.dot(Xa1.T, Y_t)

    LHS.ravel()[::LHS.shape[1] + 1] += tikh

    Wout = np.linalg.solve(LHS, RHS)

    return Wout


def RVC_Noise(x):
    # Recycle Validation

    global tikh_opt, k, ti

    # setting and initializing
    rho = x[0]
    sigma_in = round(10 ** x[1], 2)
    ti = time.time()
    lenn = tikh.size
    Mean = np.zeros(lenn)

    # Train using tv: training+val
    Wout = train_n(U_washout, U_tv, Y_tv, tikh, sigma_in, rho)[0]

    # Different Folds in the validation set
    t1 = time.time()
    for i in range(N_fo):

        # select washout and validation
        p = N_in + i * N_fw
        Y_val = U[N_washout + p: N_washout + p + N_val].copy()
        U_wash = U[p: N_washout + p].copy()

        # washout before closed loop
        xf = open_loop(U_wash, np.zeros(N_units), sigma_in, rho)[-1]

        for j in range(lenn):
            # Validate
            Yh_val = closed_loop(N_val - 1, xf, Wout[j], sigma_in, rho)[0]
            Mean[j] += np.log10(np.mean((Y_val - Yh_val) ** 2))

    if k == 0: print('closed-loop time:', time.time() - t1)

    # select optimal tikh
    a = np.argmin(Mean)
    tikh_opt[k] = tikh[a]
    k += 1

    # print for every set of hyperparameters
    if print_flag:
        print(k, ': Spectral radius, Input Scaling, Tikhonov, MSE:',
              rho, sigma_in, tikh_opt[k - 1], Mean[a] / N_fo)

    return Mean[a] / N_fo


def generate_W_Win(seed):

    rnd = np.random.RandomState(seed)

    # sparse syntax for the input and state matrices
    Win = lil_matrix((N_units, dim + 1))
    for j in range(N_units):
        Win[j, rnd.randint(0, dim + 1)] = rnd.uniform(-1, 1)  # only one element different from zero
    Win = Win.tocsr()

    W = csr_matrix(  # on average only connectivity elements different from zero
        rnd.uniform(-1, 1, (N_units, N_units)) * (rnd.rand(N_units, N_units) < (1 - sparseness)))

    spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
    W = (1 / spectral_radius) * W  # scaled to have unitary spec radius

    return W, Win

# Hyperparameter Optimization using Grid Search plus Bayesian Optimization
def g(val):
    # ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
    kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) * \
              Matern(length_scale=[0.2, 0.2], nu=2.5, length_scale_bounds=(5e-2, 1e1))

    # Gaussian Process reconstruction
    b_e = GPR(kernel=kernell,
              normalize_y=True,
              # if true mean assumed to be equal to the average of the obj function data, otherwise =0
              n_restarts_optimizer=3,  # number of random starts to find the gaussian process hyperparameters
              noise=1e-10,  # only for numerical stability
              random_state=10)  # seed

    # Bayesian Optimization
    res = skopt.gp_minimize(val,  # the function to minimize
                            search_space,  # the bounds on each dimension of x
                            base_estimator=b_e,  # GP kernel
                            acq_func="EI",  # the acquisition function
                            n_calls=n_tot,  # total number of evaluations of f
                            x0=x1,  # Initial grid search points to be evaluated at
                            n_random_starts=0,  # the number of additional random initialization points
                            n_restarts_optimizer=3,  # number of tries for each acquisition
                            random_state=10,  # seed
                            )
    return res