import numpy as np
import EchoStateNetwork
import scipy.linalg as la
import time
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf

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


class EchoStateNetwork:
    defaults = dict(N_units=100,
                    upsample=5,
                    L=1,
                    wash_obs=None,
                    wash_time=None,
                    perform_test=True,
                    augment_data=False,
                    connect=5,
                    initialised=False,
                    bias_in=None,  # np.array([0.1]),
                    bias_out=np.array([1.0]),
                    # Validation & training parameters
                    N_ensemble=1,
                    t_val=0.1,
                    t_train=1.0,
                    t_test=0.5,
                    N_wash=50,
                    seed_W=1,
                    N_func_evals=20,
                    N_random_points=0,
                    N_grid=4,
                    N_folds=4,
                    N_split=4,
                    N_initial_rand=0,
                    # Default hyperparameters
                    rho=0.9,
                    sigma_in=-4,
                    noise=0.2,
                    tikh=1e-12,
                    # Hyperparameter optimization
                    optimize_hyperparams=['rho', 'sigma_in', 'tikh'],
                    tikh_range=[1e-10, 1e-12, 1e-16],
                    sigma_in_range=(-5, -1),
                    rho_range=(.8, 1.05),
                    noise_range=(0.05, 0.5)  # TODO: add noise to optimization. Same as Tikhonov
                    )

    # Use slots rather than dict to save memory
    __slots__ = list(defaults.keys()) + ["Win", "Wout", "W", "__dict__"]

    def __init__(self, y, dt, **kwargs):  # TODO: add noise to optimization. Same as Tikhonov):
        self.u = np.zeros(y.shape)
        self.dt = dt
        #
        for key, val in self.defaults.items():
            if key in kwargs.keys():
                val = kwargs[key]
            setattr(self, key, val)

        self.filename = 'ESN_L_{}'.format(self.L)

        # Define time windows
        self.N_train = int(round(self.t_train / self.dt_ESN))
        self.N_val = int(round(self.t_val / self.dt_ESN))
        self.N_test = int(round(self.t_test / self.dt_ESN))


        # -----------  Initialise reservoir state and its history to zeros ------------ #
        self.r = np.zeros(self.N_units)
        self.norm = None
        self._WCout = None
        self.Wout = np.zeros([self.N_units + 1, self.N_dim])

    @property
    def N_dim(self):
        return len(self.u)

    @property
    def len_train_data(self):
        val = self.N_train + self.N_val + self.N_wash
        if self.perform_test:
            val += self.N_test * 10
        return val

    @property
    def dt_ESN(self):
        return self.dt * self.upsample


    @property
    def WCout(self):
        if self._WCout is None:
            self._WCout = la.lstsq(self.Wout[:-1], self.W)[0]
        return self._WCout

    @property
    def sparse(self):
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
            self.u = u
        if r is not None:
            self.r = r

    # _______________________________________________________________________________________________________ JACOBIAN
    def Jacobian(self, open_loop_J=True):
        # Get current state
        u_in, r_in = self.getReservoirState()

        Win_1 = self.Win[:, self.N_dim]
        Wout_1 = self.Wout[:self.N_units, :].T

        # # Option(i) rin function of bin:
        if open_loop_J:
            rout = self.step(u_in, r_in)[1]
            drout_dbin = self.sigma_in * Win_1 / self.norm
        else:
            b_aug = np.concatenate((u_in / self.norm, np.array([self.bias_in])))
            rout = np.tanh(np.dot(b_aug * self.sigma_in, self.Win) + self.rho * np.dot(u_in, self.WCout))
            drout_dbin = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.transpose()

        # Compute Jacobian
        T = 1 - rout ** 2
        return np.dot(Wout_1, np.array(drout_dbin) * np.expand_dims(T, 1))

    # ___________________________________________________________________________________ FUNCTIONS TO FORECAST THE ESN
    def getReservoirState(self):
        return self.u, self.r

    def step(self, u, r):
        """ Advances one ESN time step.
            Returns:
                new reservoir state (without bias_out) and output state
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)
        u_aug = np.concatenate((u / self.norm, self.bias_in))
        # Forecast the reservoir state
        r_out = np.tanh(self.Win.dot(u_aug * self.sigma_in) + self.W.dot(self.rho * r))
        # output bias added
        r_aug = np.concatenate((r_out, self.bias_out))
        # compute output from ESN if not during training
        u_out = np.dot(r_aug, self.Wout)
        return u_out, r_out

    def openLoop(self, u_wash, extra_closed=False):
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - r: time series of reservoir states
        """
        Nt = u_wash.shape[0] - 1
        if extra_closed:
            Nt += 1

        r = np.empty((Nt + 1, self.N_units))
        u = np.empty((Nt + 1, self.N_dim))

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
        r = np.empty((Nt + 1, self.N_units))
        u = np.empty((Nt + 1, self.N_dim))

        u[0], r[0] = self.getReservoirState()

        for i in range(Nt):
            u[i + 1], r[i + 1] = self.step(u[i], r[i])
        return u, r

    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self, train_data, validation_strategy,
              plot_training=True, folder='./'):

        # Format training data and divide into wash-train-validate, and test sets
        U_wtv, Y_tv, U_test = self.format_training_data(train_data)
        print(U_wtv.shape, train_data.shape)

        #  ==================== ADD NOISE TO TRAINING INPUT ====================== ## TODO: add this to the optimization
        # Add noise to inputs during training. Larger noise level
        # promotes stability in long term, but hinders time accuracy
        U_std = np.std(U_wtv, axis=1)
        rnd = np.random.RandomState(0)
        for ll in range(self.L):
            for dd in range(self.N_dim):
                U_wtv[ll, :, dd] += rnd.normal(0, self.noise * U_std[ll, dd], U_wtv.shape[1])

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
        self.reset_state(u=self.u*0, r=self.r*0)
        self.Wout = self.solveRidgeRegression(U_wtv, Y_tv)
        print('\n Time per hyperparameter eval.:', (time.time() - ti) / self.N_func_evals,
              '\n Best Results: x ')
        for hp in self.optimize_hyperparams:
            print('\t {:.2e}'.format(getattr(self, hp)), end="")
        print(' f {:.4f}'.format(-res.fun))
        # =============================  Plot result ================================ ##
        if plot_training or self.perform_test:
            os.makedirs(folder+'figs_ESN', exist_ok=True)
            pdf = plt_pdf.PdfPages(folder + 'figs_ESN/' + self.filename + '_Training.pdf')
            if plot_training:

                # Plot Bayesian optimization convergence
                fig = plt.figure()
                plot_convergence(res)
                pdf.savefig(fig)
                plt.close(fig)

                # Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations.
                # The GP reconstruction is based on the n_tot function evaluations decided in the search

                if len(hp_names) >= 2:  # plot GP reconstruction
                    gp = res.models[-1]
                    x_iters = np.array(res.x_iters)

                    for hpi in range(len(hp_names) - 1):
                        range_1 = getattr(self, hp_names[hpi] + '_range')
                        range_2 = getattr(self, hp_names[hpi + 1] + '_range')

                        n_len = 100  # points to evaluate the GP at
                        xx, yy = np.meshgrid(np.linspace(*range_1, n_len),
                                             np.linspace(*range_2, n_len))

                        x_x = np.column_stack((xx.flatten(), yy.flatten()))
                        x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

                        # Plot GP Mean
                        fig = plt.figure(figsize=[10, 5], tight_layout=True)
                        plt.xlabel(hp_names[hpi])
                        plt.ylabel(hp_names[hpi + 1])

                        # retrieve the gp reconstruction
                        amin = np.amin([10, f_iters.max()])

                        # Final GP reconstruction for each realization at the evaluation points
                        y_pred = np.clip(-gp.predict(x_gp), a_min=-amin, a_max=-f_iters.min()).reshape(n_len, n_len)

                        plt.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
                        cbar = plt.colorbar()
                        cbar.set_label('-$\log_{10}$(MSE)', labelpad=15)
                        plt.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid', alpha=0.3)
                        #   Plot the n_tot search points
                        plt.plot(x_iters[:self.N_grid ** 2, 0], x_iters[:self.N_grid ** 2, 1], 'v', c='w',
                                 alpha=0.8, markeredgecolor='k', markersize=10)
                        plt.plot(x_iters[self.N_grid ** 2:, 0], x_iters[self.N_grid ** 2:, 1], 's', c='w',
                                 alpha=0.8, markeredgecolor='k', markersize=8)
                        pdf.savefig(fig)
                        plt.close(fig)
            if self.perform_test:
                # Run test
                self.run_test(U_test, pdf_file=pdf)

            pdf.close()  # Close training results pdf

    def generate_W_Win(self, seed=1):
        rnd0 = np.random.RandomState(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero
        Win = lil_matrix((self.N_units, self.N_dim + 1))  # +1 accounts for input bias
        for j in range(self.N_units):
            Win[j, rnd0.randint(0, self.N_dim + 1)] = rnd0.uniform(-1, 1)
        self.Win = Win.tocsr()

        # Reservoir state matrix: Erdos-Renyi network
        W = csr_matrix(rnd0.uniform(-1, 1, (self.N_units, self.N_units)) *
                       (rnd0.rand(self.N_units, self.N_units) < (1 - self.sparse))
                       )
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
                u_open, r_open = u_open[1:], r_open[1:]

                self.reset_state(u=u_open[-1], r=r_open[-1])

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
        #   APPLY UPSAMPLE ________________________
        train_data = train_data[:, ::self.upsample].copy()

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS _______________
        N_tv = self.N_train + self.N_val
        N_wtv = self.N_wash + N_tv  # useful for compact code later

        if train_data.shape[1] < N_wtv:
            print(train_data.shape[1], N_wtv)
            raise ValueError('Increase the length of the training data signal')

        U_wtv = train_data[:, :N_wtv - 1].copy()
        Y_tv = train_data[:, self.N_wash + 1:N_wtv].copy()
        U_test = train_data[:, N_wtv + 1:].copy()

        # compute norm (normalize inputs by component range)
        m = np.mean(U_wtv.min(axis=1), axis=0)
        M = np.mean(U_wtv.max(axis=1), axis=0)
        self.norm = M - m

        if self.bias_in is None:
            u_mean = np.mean(np.mean(U_wtv, axis=1), axis=0)
            setattr(self, 'bias_in', np.array([np.mean(np.abs((U_wtv - u_mean) / self.norm))]))

        return U_wtv, Y_tv, U_test

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
        N_fw = (case.N_train - case.N_val) // (case.N_folds - 1)  # num steps forward the validation interval is shifted

        # Train using tv: Wout_tik is passed with all the combinations of tikh_ and target noise
        # This must result in L-Xa timeseries
        LHS, RHS, U_train, R_train = case.computeRRterms(U_wtv, Y_tv)
        Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))
        for tik_j in range(N_tikh):
            LHS_ = LHS.copy()
            LHS_.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
            Wout_tik[tik_j] = np.linalg.solve(LHS_, RHS)

        # Perform Validation in different folds
        for U_l in U_wtv:  # Each set of training data
            for fold in range(case.N_folds):
                case.reset_state(u=case.u*0, r=case.r*0)
                p = case.N_wash + fold * N_fw

                # Select washout and validation data
                U_wash = U_l[p:p + case.N_wash]
                Y_val = U_l[p + case.N_wash:p + case.N_wash + case.N_val]

                # Perform washout
                u_open, r_open = case.openLoop(U_wash, extra_closed=False)

                for tik_j in range(N_tikh):  # cloop for each tikh_-noise combination
                    case.reset_state(u=u_open[-1], r=r_open[-1])
                    case.Wout = Wout_tik[tik_j]
                    U_close = case.closedLoop(case.N_val)[0][1:]

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
    def run_test(self, U_test, pdf_file=None, folder='./'):

        if pdf_file is None:
            os.makedirs(folder+'figs_ESN', exist_ok=True)
            pdf_file = plt_pdf.PdfPages(folder+'figs_ESN/' + self.filename + '_Test.pdf')

        # Testing window
        N_test = self.N_test

        # Number of tests (with a maximum of 10)
        max_test_time = np.shape(U_test)[1]
        total_tests = min(10, int(np.floor(max_test_time / N_test)))

        # Break if not enough train_data for testing
        if total_tests < 1:
            print('Test not performed. Not enough train_data')
        else:
            medians_alpha, max_alpha = [], -np.inf

            for U_test_l in U_test:
                subplots = min(10, total_tests)  # number of plotted intervals
                plt.subplots(subplots, 1, figsize=[10, 2 * subplots])
                errors = np.zeros(total_tests)
                # Different intervals in the test set
                ti, dim_i = -N_test, -1
                self.u, self.r = np.zeros(self.N_dim), np.zeros(self.N_units)
                while True:
                    ti += N_test
                    dim_i += 1
                    test_i = ti // N_test
                    if test_i >= total_tests:
                        break

                    # washout for each interval
                    u_open, r_open = self.openLoop(U_test_l[ti: ti + self.N_wash])
                    self.u, self.r = u_open[-1], r_open[-1]

                    # Data to compare with
                    Y_t = U_test_l[ti + self.N_wash: ti + self.N_wash + N_test]

                    # Closed-loop prediction
                    Yh_t = self.closedLoop(N_test)[0][1:]
                    # Do the multiple sub_loop inside each test interval
                    errors[test_i] = np.log10(np.mean((Yh_t - Y_t) ** 2) / np.mean(self.norm ** 2))
                    if test_i < subplots:
                        t_ = np.arange(len(Yh_t)) * self.dt_ESN
                        plt.subplot(subplots, 1, test_i + 1)
                        plt.plot(t_, Y_t[:, dim_i], 'k', label='truth dim ' + str(dim_i))
                        plt.plot(t_, Yh_t[:, dim_i], '--r', label='ESN dim ' + str(dim_i))
                        plt.legend(title='Test ' + str(test_i), loc='upper left', bbox_to_anchor=(1.01, 1.01))
                    if dim_i == self.N_dim - 1:
                        dim_i = -1

                max_alpha = max(max_alpha, errors.max())
                medians_alpha.append(np.median(errors))
                # Save to pdf
                plt.tight_layout()
                fig = plt.gcf()
                pdf_file.savefig(fig)
                plt.close(fig)
            print('Median and max error in', self.L, ' test:', np.median(medians_alpha), max_alpha)


if __name__ == '__main__':

    plot_ = True

    dim = 3
    upsample = 2  # to increase the dt of the ESN wrt the numerical integrator
    dt_ESN = 0.015 * upsample  # time step
    t_lyap = 0.906 ** (-1)  # Lyapunov Time (inverse of largest Lyapunov exponent)
    N_lyap = int(t_lyap / dt_ESN)  # number of time steps in one Lyapunov time

    # number of time steps for washout, train, validation, test
    N_transient = 20 * N_lyap
    N_washout = 50
    N_train = 100 * N_lyap
    N_val = 5 * N_lyap
    N_test = 10 * N_lyap

    # Create signal to predict ---------------------------------
    # Note: Does not need to be a class. Any signal can be used as U
    import physical_models

    rnd = np.random.RandomState(0)

    model = physical_models.Lorenz63({'dt': dt_ESN / upsample,
                                      'psi0': rnd.random(dim)  # initial random condition
                                      })

    if plot_:
        fig1 = plt.figure(figsize=[12, 9], layout="constrained")
        axs = fig1.subplots(3, 1)
        axs[0].set_title(model.name)

    for Nt in [N_transient, N_washout + N_train + N_val + N_test * 5]:
        state1, t1 = model.timeIntegrate(Nt * upsample)
        model.updateHistory(state1, t1)
        y = model.getObservableHist(len(t1))
        if plot_:
            for ii, ax in enumerate(axs):
                ax.plot(t1, y[:, ii], '.-')

    # number of time steps for washout, train, validation, test
    params_ESN = {'upsample': 2,
                  't_train': N_train * dt_ESN,
                  't_val': N_val * dt_ESN,
                  't_test': N_test * dt_ESN,
                  'N_wash': 50,
                  'N_units': 200,
                  'N_folds': 25,
                  'N_split': 4,
                  'connect': 3,
                  'plot_training': True,
                  'rho_range': (.1, 1.),
                  'sigma_in_range': (np.log10(0.05), np.log10(10.)),
                  'tikh_range': [1E-6, 1E-9, 1E-12, 1E-16],
                  'N_func_evals': 40,
                  'N_grid': 6,
                  'noise': 1e-5,
                  }

    U = y[:, :2]

    ESN_case = EchoStateNetwork(U[0], dt=dt_ESN / upsample, **params_ESN)
    ESN_case.train(U, validation_strategy=EchoStateNetwork.RVC_Noise)

    # Initialise ESN
    t_wash = t1[-ESN_case.N_wash * ESN_case.upsample + 1::ESN_case.upsample]
    u_wash = U[-ESN_case.N_wash * ESN_case.upsample + 1::ESN_case.upsample, :, 0]

    u_wash, r_wash = ESN_case.openLoop(u_wash)
    ESN_case.u = u_wash[-1]
    ESN_case.r = r_wash[-1]

    if plot_:
        for ii, ax in enumerate(axs):
            if ii < ESN_case.N_dim:
                ax.plot(t_wash, u_wash[:, ii], 'o-')
                ax.plot(t_wash[-1], ESN_case.u[ii], '*-', ms=12)
                ax.plot(t_wash, u_wash[:, ii], '*-')

    # Forecast model and ESN and compare
    t_test = 20
    state2, t2 = model.timeIntegrate(int(t_test / model.dt))
    model.updateHistory(state2, t2)
    y = model.getObservableHist(len(t2))

    if plot_:
        for ii, ax in enumerate(axs):
            ax.plot(t2, y[:, ii], '.-')
            ax.set(ylim=[min(y[:, ii]), max(y[:, ii])])

    t2_up = t2[::ESN_case.upsample]
    u_closed = ESN_case.closedLoop(len(t2_up))[0]

    if plot_:
        for ii, ax in enumerate(axs):
            if ii < ESN_case.N_dim:
                ax.plot(t2_up, u_closed[1:, ii], '.-')

        axs[-1].set(xlabel='$t$', ylabel=model.obsLabels[0])
        plt.show()
