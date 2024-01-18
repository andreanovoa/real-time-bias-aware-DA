from typing import Any

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf
from numpy import ndarray, dtype

from essentials.Util import *

# Validation methods
from functools import partial
from itertools import product
import skopt
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real
from skopt.plots import plot_convergence

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

XDG_RUNTIME_DIR = 'tmp/'


class EchoStateNetwork:
    augment_data = False
    bias_in = None
    bias_out = np.array([1.0])
    connect = 5
    initialised = False
    L = 1
    N_folds = 4
    N_func_evals = 20
    N_grid = 4
    N_initial_rand = 0
    N_random_points = 0
    N_split = 4
    N_units = 100
    N_wash = 50
    perform_test = True
    seed_W = 1
    seed_noise = 0
    t_val = 0.1
    t_train = 1.0
    t_test = 0.5
    upsample = 5
    wash_obs = None
    wash_time = None
    Win_type = 'sparse'
    # Default hyperparameters and optimization ranges
    noise = 1e-10
    noise_type = 'gauss'
    optimize_hyperparams = ['rho', 'sigma_in', 'tikh']
    rho = 0.9
    rho_range = (.8, 1.05)
    sigma_in = 10 ** -3
    sigma_in_range = (-5, -1)
    tikh = 1e-12
    tikh_range = [1e-10, 1e-12, 1e-16]
    # noise_range=(0.05, 0.5),  # TODO add noise to optimization

    def __init__(self, y, dt, **kwargs):
        """
        :param y: model current state
        :param dt: model dt such that dt_ESN = dt * upsample
        :param kwargs: any argument to re-define the dfault values of the class
        """
        self.u = np.zeros(y.shape)
        self.dt = dt
        self.dt_ESN = self.dt * self.upsample

        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # initially, assume full observability. This may change when providing input data
        self.N_dim = len(y)
        self.observed_idx = np.arange(self.N_dim)
        self.filename = 'ESN'

        # -------------------------- Define time windows ----------------------------- #
        self.N_train = int(round(self.t_train / self.dt_ESN))
        self.N_val = int(round(self.t_val / self.dt_ESN))
        self.N_test = int(round(self.t_test / self.dt_ESN))

        # -----------  Initialise reservoir state and its history to zeros ------------ #
        self.r = np.zeros((self.N_units, 1))
        self.norm = None
        self._WCout = None
        self.trained = False

    @property
    def len_train_data(self):
        val = self.N_train + self.N_val + self.N_wash
        if self.perform_test:
            val += self.N_test * 5
        return val

    def outputs_to_inputs(self, full_state):
        return full_state[self.observed_idx]

    @property
    def N_dim_in(self):
        return len(self.observed_idx)

    @property
    def WCout(self):
        if self._WCout is None:
            self._WCout = np.linalg.lstsq(self.Wout[:-1], self.W.toarray(), rcond=None)[0]
        return self._WCout

    @property
    def sparsity(self):
        return 1 - self.connect / (self.N_units - 1)

    def reset_hyperparams(self, params, names, tikhonov=None):
        for hp, name in zip(params, names):
            if name == 'sigma_in':
                setattr(self, name, 10 ** hp)
            else:
                setattr(self, name, hp)
        if tikhonov is not None:
            setattr(self, 'tikh', tikhonov)

    def reset_state(self, u=None, r=None):
        if u is not None:
            if u.shape[0] != self.u.shape[0]:
                self.u = u[self.observed_idx]
            else:
                self.u = u
        if r is not None:
            self.r = r

    # _______________________________________________________________________________________________________ JACOBIAN

    def Jacobian(self, open_loop_J=True, state=None):
        # Get current state
        if state is None:
            u_in, r_in = self.getReservoirState()
        else:
            u_in, r_in = state

        Win_1 = self.Win[:, :self.N_dim_in]
        Wout_1 = self.Wout[:self.N_units, :].T

        # # Option(i) rin function of bin:
        rout = self.step(u_in, r_in)[1].squeeze()

        tt = 1. - rout ** 2
        g = 1. / self.norm

        dr_di = self.sigma_in * Win_1.multiply(g)

        if not open_loop_J:
            # u_aug = np.concatenate((u_in / self.norm, self.bias_in))
            # rout = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * np.dot(self.WCout.T, u_in))
            # dr_di = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.T
            #  Win_G += dr_di ......
            raise NotImplementedError('Numerical test of closed-loop Jacobian did not pass')

        RHS = dr_di.T.multiply(tt)

        # Compute Jacobian
        return RHS.dot(Wout_1.T).T

    # ___________________________________________________________________________________ FUNCTIONS TO FORECAST THE ESN
    def getReservoirState(self):
        return self.u, self.r

    def step(self, u, r):
        """ Advances one ESN time step.
            Returns:
                new reservoir state (without bias_out) and output state
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)
        if u.ndim == 1:
            u = np.expand_dims(u, axis=-1)
        if r.ndim == 1:
            r = np.expand_dims(r, axis=-1)

        g = np.tile(1. / self.norm, reps=(u.shape[-1], 1)).T
        bias_in = np.tile(self.bias_in, reps=(1, u.shape[-1]))
        bias_out = np.tile(self.bias_out, reps=(1, u.shape[-1]))

        u_aug = np.concatenate((np.multiply(u, g), bias_in))

        # Forecast the reservoir state
        r_out = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * self.W.dot(r))

        # output bias added
        r_aug = np.concatenate((r_out, bias_out))
        # compute output from ESN if not during training
        u_out = np.dot(r_aug.T, self.Wout).T
        return u_out, r_out

    def openLoop(self, u_wash, extra_closed=0):
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - r: time series of reservoir states
        """
        Nt = u_wash.shape[0] - 1

        if extra_closed:
            Nt += extra_closed

        if u_wash.ndim == 3:
            N_ens = u_wash.shape[-1]
        else:
            N_ens = 1

        r = np.empty((Nt + 1, self.N_units, N_ens))
        u = np.empty((Nt + 1, self.N_dim, N_ens))

        r[0] = self.getReservoirState()[-1]

        for i in range(Nt):
            u[i + 1], r[i + 1] = self.step(u_wash[i], r[i])
        return u, r

    def closedLoop(self, Nt):
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """

        u0, r0 = self.getReservoirState()

        r = np.empty((Nt + 1, self.N_units, u0.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, u0.shape[-1]))

        u[0, self.observed_idx], r[0] = u0, r0

        for i in range(Nt):
            u_input = self.outputs_to_inputs(full_state=u[i])
            u[i + 1], r[i + 1] = self.step(u_input, r[i])
        return u, r

    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self, train_data, validation_strategy, plot_training=True, folder='./'):

        # Format training data and divide into wash-train-validate, and test sets
        U_wtv, Y_tv, U_test, Y_test = self.format_training_data(train_data)

        self.Wout = np.zeros([self.N_units + 1, self.N_dim])

        # ======================  Generate matrices W and Win ======================= ##
        self.generate_W_Win(seed=self.seed_W)

        # ==========================  Bayesian Optimization ========================= ##
        ti = time.time()  # check time

        # define grid search and bayesian optimisation hp_names
        search_grid, search_space, hp_names = self.hyperparameter_search()

        tikh_opt = np.zeros(self.N_func_evals)  # optimal tikhonov in each evaluation

        self.val_k = 0  # Validation iteration counter

        # Validation function
        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_tv=Y_tv.copy(),
                           tikh_opt=tikh_opt,
                           hp_names=hp_names
                           )

        # Perform optimization
        res = EchoStateNetwork.hyperparam_optimization(val_func, search_space,
                                                       x0=search_grid, n_calls=self.N_func_evals)
        f_iters = np.array(res.func_vals)

        # Save optimized parameters
        self.reset_hyperparams(res.x, hp_names, tikhonov=tikh_opt[np.argmin(f_iters)])

        for hpi, hp in enumerate(self.optimize_hyperparams):
            self.filename += '_{}_{:.3e}'.format(hp, getattr(self, hp))

        # ============================  Train Wout ================================== ##
        self.reset_state(u=self.u * 0, r=self.r * 0)
        self.Wout = self.solveRidgeRegression(U_wtv, Y_tv)
        print('\n Time per hyperparameter eval.:', (time.time() - ti) / self.N_func_evals,
              '\n Best Results: x ')
        for hp in self.optimize_hyperparams:
            print('\t {:.2e}'.format(getattr(self, hp)), end="")
        print(' f {:.4f}'.format(-res.fun))

        self.trained = True

        # =============================  Plot result ================================ ##
        if plot_training or self.perform_test:
            os.makedirs(folder + 'figs_ESN', exist_ok=True)
            pdf = plt_pdf.PdfPages(folder + 'figs_ESN/' + self.filename + '_Training.pdf')
            if plot_training:
                fig = plt.figure()
                # Plot Bayesian optimization convergence
                plot_convergence(res)
                add_pdf_page(pdf, fig)

                # Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations.
                # The GP reconstruction is based on the n_tot function evaluations decided in the search
                if len(hp_names) >= 2:  # plot GP reconstruction
                    gp = res.models[-1]
                    res_x = np.array(res.x_iters)

                    for hpi in range(len(hp_names) - 1):
                        range_1 = getattr(self, hp_names[hpi] + '_range')
                        range_2 = getattr(self, hp_names[hpi + 1] + '_range')

                        n_len = 100  # points to evaluate the GP at
                        xx, yy = np.meshgrid(np.linspace(*range_1, n_len), np.linspace(*range_2, n_len))

                        x_x = np.column_stack((xx.flatten(), yy.flatten()))
                        x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

                        # Plot GP Mean
                        fig = plt.figure(figsize=[10, 5], tight_layout=True)
                        plt.xlabel(hp_names[hpi])
                        plt.ylabel(hp_names[hpi + 1])

                        # retrieve the gp reconstruction
                        amin = np.amin([10, np.max(f_iters)])

                        # Final GP reconstruction for each realization at the evaluation points
                        y_pred = np.clip(-gp.predict(x_gp), a_min=-amin, a_max=-np.min(f_iters)).reshape(n_len, n_len)

                        plt.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
                        cbar = plt.colorbar()
                        cbar.set_label('-$\\log_{10}$(MSE)', labelpad=15)
                        plt.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid',
                                    alpha=0.3)
                        #   Plot the n_tot search points
                        for rx, mk in zip([res_x[:self.N_grid ** 2], res_x[self.N_grid ** 2:]], ['v', 's']):
                            plt.plot(rx[:, 0], rx[:, 1], mk, c='w', alpha=.8, mec='k', ms=8)
                        # Plot best point
                        best_idx = np.argmin(f_iters)
                        plt.plot(res_x[best_idx, 0], res_x[best_idx, 1], '*r', alpha=.8, mec='r', ms=8)
                        pdf.savefig(fig)
                        plt.close(fig)
            if self.perform_test:
                self.run_test(U_test, Y_test, pdf_file=pdf)  # Run test
            pdf.close()  # Close training results pdf

    def generate_W_Win(self, seed=1):
        rnd0 = np.random.RandomState(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero
        Win = lil_matrix((self.N_units, self.N_dim_in + 1))  # +1 accounts for input bias
        if self.Win_type == 'sparse':
            for j in range(self.N_units):
                Win[j, rnd0.randint(low=0, high=self.N_dim_in + 1)] = rnd0.uniform(low=-1, high=1)
        elif self.Win_type == 'dense':
            for j in range(self.N_units):
                Win[j, :] = rnd0.uniform(low=-1, high=1, size=self.N_dim_in + 1)
        else:
            raise ValueError("Win type {} not implemented. Choose 'sparse' or 'dense'".format(self.Win_type))
        self.Win = Win.tocsr()

        # Reservoir state matrix: Erdos-Renyi network
        W = csr_matrix(rnd0.uniform(-1, 1, (self.N_units, self.N_units)) *
                       (rnd0.rand(self.N_units, self.N_units) < (1 - self.sparsity)))
        # scale W by the spectral radius to have unitary spectral radius
        spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
        self.W = (1. / spectral_radius) * W

    def computeRRterms(self, U_wtv, Y_tv):
        LHS, RHS = 0., 0.
        R_RR = [np.empty([0, self.N_units])] * self.L
        U_RR = [np.empty([0, self.N_dim])] * self.L
        self.r = 0
        for ll in range(self.L):

            # Washout phase. Store the last r value only
            self.r = self.openLoop(U_wtv[ll][:self.N_wash], extra_closed=True)[1][-1]

            # Split training data for faster computations
            U_train = np.array_split(U_wtv[ll][self.N_wash:], self.N_split, axis=0)
            Y_target = np.array_split(Y_tv[ll], self.N_split, axis=0)

            for U_t, Y_t in zip(U_train, Y_target):
                # Open-loop train phase
                u_open, r_open = self.openLoop(U_t, extra_closed=True)

                self.reset_state(u=u_open[-1], r=r_open[-1])

                u_open, r_open = u_open[1:].squeeze(), r_open[1:].squeeze()

                R_RR[ll] = np.append(R_RR[ll], r_open, axis=0)
                U_RR[ll] = np.append(U_RR[ll], u_open, axis=0)

                # Compute matrices for linear regression system
                bias_out = np.ones([r_open.shape[0], 1]) * self.bias_out
                r_aug = np.hstack((r_open, bias_out))
                LHS += np.dot(r_aug.T, r_aug)
                RHS += np.dot(r_aug.T, Y_t)

        return LHS, RHS, U_RR, R_RR

    def solveRidgeRegression(self, U_wtv, Y_tv):
        LHS, RHS = self.computeRRterms(U_wtv, Y_tv)[:2]
        LHS.ravel()[::LHS.shape[1] + 1] += self.tikh  # Add tikhonov to the diagonal
        return np.linalg.solve(LHS, RHS)  # Solve linear regression problem

    def format_training_data(self, train_data):
        """
        :param train_data:  dimensions L x Nt x Ndim .or. dict containing inputs and labels
        """

        #   APPLY UPSAMPLE ________________________
        if type(train_data) is dict:
            U = train_data['inputs'][:, ::self.upsample].copy()
            Y = train_data['labels'][:, ::self.upsample].copy()

            self.N_dim = Y.shape[-1]
            self.observed_idx = train_data['observed_idx']
            # self.reset_state(u=np.zeros(Y.shape[-1]))

            assert Y.shape[-1] >= U.shape[-1]
            assert U.shape[-1] == len(self.observed_idx)
            add_noise = False
        else:
            U = train_data[:, ::self.upsample].copy()
            Y = U.copy()
            add_noise = True

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS _______________
        N_tv = self.N_train + self.N_val
        N_wtv = self.N_wash + N_tv  # useful for compact code later

        if U.shape[1] < N_wtv:
            print(U.shape, N_wtv)
            raise ValueError('Increase the length of the training data signal')

        U_wtv = U[:, :N_wtv - 1].copy()
        Y_tv = Y[:, self.N_wash + 1:N_wtv].copy()
        U_test = U[:, N_wtv:].copy()
        Y_test = Y[:, N_wtv:].copy()

        # compute norm (normalize inputs by component range)
        m = np.mean(U_wtv.min(axis=1), axis=0)
        M = np.mean(U_wtv.max(axis=1), axis=0)
        self.norm = M - m

        if self.bias_in is None:
            setattr(self, 'bias_in', np.array([0.1]))

        if add_noise:
            #  ==================== ADD NOISE TO TRAINING INPUT ====================== ##
            # Add noise to the inputs if distinction inputs/labels is not given.
            # Larger noise promotes stability in long term, but hinders time accuracy
            U_std = np.std(U, axis=1)
            rnd = np.random.RandomState(self.seed_noise)
            for ll in range(self.L):
                for dd in range(self.N_dim):
                    U_wtv[ll, :, dd] += rnd.normal(loc=0, scale=self.noise * U_std[ll, dd], size=U_wtv.shape[1])
                    U_test[ll, :, dd] += rnd.normal(loc=0, scale=self.noise * U_std[ll, dd], size=U_test.shape[1])

        return U_wtv, Y_tv, U_test, Y_test

    # ___________________________________________________________________________________________ BAYESIAN OPTIMIZATION
    def hyperparameter_search(self):
        parameters = []
        for hp in self.optimize_hyperparams:
            if hp != 'tikh':
                parameters.append(hp)
        if 'tikh' not in self.optimize_hyperparams:
            setattr(self, 'tikh_range', [self.tikh])
        param_grid = [None] * len(parameters)
        search_space = [None] * len(parameters)
        for hpi, hyper_param in enumerate(parameters):
            range_ = getattr(self, hyper_param + '_range')
            param_grid[hpi] = np.linspace(*range_, self.N_grid)
            search_space[hpi] = Real(*range_, name=hyper_param)

        # The first n_grid^2 points are from grid search
        search_grid = product(*param_grid, repeat=1)  # list of tuples
        search_grid = [list(sg) for sg in search_grid]

        # Print optimization header
        print('\n ----------------- HYPERPARAMETER SEARCH ------------------\n {0}x{0} grid'.format(self.N_grid) +
              ' and {} points with Bayesian Optimization\n\t'.format(self.N_func_evals - self.N_grid ** 2), end="")
        for kk in self.optimize_hyperparams:
            print('\t {}'.format(kk), end="")
        print('\t MSE val ')

        return search_grid, search_space, parameters


    @staticmethod
    def hyperparam_optimization(val, search_space, x0, n_calls, n_rand=0):

        # ARD 5/2 Matern Kernel with sigma_f in front of the Gaussian Process
        kernel_ = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) * \
                  Matern(length_scale=[0.2, 0.2], nu=2.5, length_scale_bounds=(1e-2, 1e1))

        # Gaussian Process reconstruction
        b_e = GPR(kernel=kernel_,
                  normalize_y=True,  # if true => mean = avg of objective fun
                  n_restarts_optimizer=3,  # num of random starts to find GP hyperparams
                  noise=1e-10,  # for numerical stability
                  random_state=10  # seed
                  )
        # Bayesian Optimization
        result = skopt.gp_minimize(val,  # function to minimize
                                   search_space,  # bounds
                                   base_estimator=b_e,  # GP kernel
                                   acq_func="gp_hedge",  # acquisition function
                                   n_calls=n_calls,  # num of evaluations
                                   x0=x0,  # Initial grid points
                                   n_random_starts=n_rand,  # num of random initializations
                                   n_restarts_optimizer=3,  # tries per acquisition
                                   random_state=10  # seed
                                   )
        return result

    # ______________________________________________________________________________________________ VALIDATION METHODS

    @staticmethod
    def RVC_Noise(x, case, U_wtv, Y_tv, tikh_opt, hp_names):
        """
        Chaotic Recycle Validation
        """
        # Re-set hyperparams as the optimization goes on
        if hp_names:
            case.reset_hyperparams(x, hp_names)

        N_tikh = len(case.tikh_range)
        n_MSE = np.zeros(N_tikh)
        N_fw = (case.N_train - case.N_val - case.N_wash) // (
                    case.N_folds - 1)  # num steps forward the validation interval is shifted

        # Train using tv: Wout_tik is passed with all the combinations of tikh_ and target noise
        # This must result in L-Xa timeseries
        LHS, RHS, U_train, R_train = case.computeRRterms(U_wtv, Y_tv)
        Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))
        for tik_j in range(N_tikh):
            LHS_ = LHS.copy()
            LHS_.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
            Wout_tik[tik_j] = np.linalg.solve(LHS_, RHS)

        # Perform Validation in different folds
        for U_l, Y_l in zip(U_wtv, Y_tv):  # Each set of training data
            for fold in range(case.N_folds):
                case.reset_state(u=case.u * 0, r=case.r * 0)
                p = case.N_wash + fold * N_fw

                # Select washout and validation data
                U_wash = U_l[p:p + case.N_wash]
                Y_val = Y_l[p:p + case.N_val]

                # Perform washout (open-loop without extra forecast step)
                u_open, r_open = case.openLoop(U_wash, extra_closed=False)

                for tik_j in range(N_tikh):  # cloop for each tikh_-noise combination
                    case.reset_state(u=u_open[-1], r=r_open[-1])

                    case.Wout = Wout_tik[tik_j]
                    U_close = case.closedLoop(case.N_val)[0][1:].squeeze()

                    # Compute normalized MSE
                    n_MSE[tik_j] += np.log10(np.mean((Y_val - U_close) ** 2) / np.mean(case.norm ** 2))

                    # prevent from diverging to infinity: MSE=1E10 (useful for hybrid and similar architectures)
                    if np.isnan(n_MSE[tik_j]) or np.isinf(n_MSE[tik_j]):
                        n_MSE[tik_j] = 10 * case.N_folds

        # select and save the optimal tikhonov and noise level in the targets
        a = n_MSE.argmin()
        tikh_opt[case.val_k] = case.tikh_range[a]
        case.tikh = case.tikh_range[a]

        case.val_k += 1
        print(case.val_k, end="")
        for hp in case.optimize_hyperparams:
            print('\t {:.3e}'.format(getattr(case, hp)), end="")
        print('\t {:.4f}'.format(n_MSE[a] / case.N_folds / case.L))

        return n_MSE[a] / case.N_folds / case.L

    # ________________________________________________________________________________________________ TESTING FUNCTION
    def run_test(self, U_test, Y_test, pdf_file=None, folder='./'):

        if pdf_file is None:
            os.makedirs(folder + 'figs_ESN', exist_ok=True)
            pdf_file = plt_pdf.PdfPages(folder + 'figs_ESN/' + self.filename + '_Test.pdf')

        # Number of tests (with a maximum of 10)
        if self.N_test > 3 * self.N_val:
            N_test = self.N_val  # Testing window
        else:
            N_test = self.N_test  # Testing window

        max_test_time = np.shape(U_test)[1] - self.N_wash

        total_tests = min(20, int(np.floor(max_test_time / N_test)))

        # Break if not enough train_data for testing
        if total_tests < 1:
            print('Test not performed. Not enough train_data')
        else:
            medians_alpha, max_alpha = [], -np.inf
            for U_test_l, Y_test_l in zip(U_test, Y_test):
                subplots = min(10, total_tests)  # number of plotted intervals
                fig, _ = plt.subplots(subplots, 1, figsize=[10, 2 * subplots], sharex='all', layout='tight')
                errors = np.zeros(total_tests)
                # Different intervals in the test set
                ti, dim_i = -N_test, -1
                self.reset_state()
                while True:
                    ti += N_test
                    dim_i += 1
                    test_i = ti // N_test
                    if test_i >= total_tests:
                        break

                    # washout for each interval
                    u_open, r_open = self.openLoop(U_test_l[ti: ti + self.N_wash])
                    self.reset_state(u=u_open[-1], r=r_open[-1])

                    # Data to compare with, i.e., labels
                    Y_t = Y_test_l[ti + self.N_wash: ti + self.N_wash + N_test].squeeze()

                    # Closed-loop prediction
                    Yh_t = self.closedLoop(N_test)[0][1:].squeeze()

                    # Do the multiple sub_loop inside each test interval

                    errors[test_i] = np.log10(np.mean((Yh_t - Y_t) ** 2) / np.mean(np.expand_dims(self.norm, -1) ** 2))
                    if test_i < subplots:
                        t_ = np.arange(len(Yh_t)) * self.dt_ESN
                        plt.subplot(subplots, 1, test_i + 1)
                        plt.plot(t_, Y_t[:, dim_i], 'k', label='truth dim ' + str(dim_i))
                        plt.plot(t_, Yh_t[:, dim_i], '--r', label='ESN dim ' + str(dim_i))
                        plt.legend(title='Test ' + str(test_i), loc='upper left', bbox_to_anchor=(1.01, 1.01))
                    if dim_i == self.N_dim - 1:
                        dim_i = -1

                max_alpha = max(max_alpha, max(errors))
                medians_alpha.append(np.median(errors))
                # Save to pdf
                add_pdf_page(pdf_file, fig)

            print('Median and max error in', self.L, ' test:', np.median(medians_alpha), max_alpha)
