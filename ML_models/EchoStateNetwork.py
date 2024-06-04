
import time
import matplotlib.backends.backend_pdf as plt_pdf
from essentials.Util import *
from essentials.DA import EnKF

# Validation methods
from functools import partial
from itertools import product
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real
from skopt.plots import plot_convergence
from multiprocessing import Pool
import random
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

XDG_RUNTIME_DIR = 'tmp/'


class EchoStateNetwork:
    augment_data = False  # Data augmentation during training?
    bayesian_update = False
    bias_in = np.array([0.1])  #
    bias_out = np.array([1.0])  # For symmetry breaking
    connect = 3  # Connectivity between neurons
    L = 1  # Number of augmented datasets
    N_folds = 4  # Folds over the training set
    N_func_evals = 20  # Total evals of Bayesian hyperparameter optimization (BHO)
    N_grid = 4  # BHO grid N_grid x N_grid \geq N_func_evals
    N_initial_rand = 0  # Initial random evaluations at BYO
    N_split = 4  # Splits of training data for faster computation
    N_units = 100  # Number of neurones
    N_wash = 50  # Number of washout steps
    perform_test = True  # Run tests during training?
    seed_W = 0  # Random seed for Win and W definition
    seed_noise = 0  # Random seed for input training data
    t_val = 0.1  # Validation time
    t_train = 1.0  # Training time
    t_test = 0.5  # Testing time
    upsample = 5  # Upsample x dt_model = dt_ESN
    Win_type = 'sparse'  # Type of Wim definition [sparse/dense]
    # Default hyperparameters and optimization ranges -----------------------
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

        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        elif y.ndim > 2:
            raise AssertionError('The input y must have 2 or less dimension')

        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

        # -----------  Initialise state and reservoir state to zeros ------------ #
        self.N_dim = y.shape[0]

        self.u = np.zeros((self.N_dim, y.shape[1]))
        self.r = np.zeros((self.N_units, y.shape[1]))

        self.observed_idx = np.arange(self.N_dim)  # initially, assume full observability.

        # ---------------- Define time steps and time windows -------------------- #
        self.dt = dt
        self.dt_ESN = self.dt * self.upsample
        #
        # self.N_train = int(round(self.t_train / self.dt_ESN))
        # self.N_val = int(round(self.t_val / self.dt_ESN))
        # self.N_test = int(round(self.t_test / self.dt_ESN))

        # ------------------------ Empty arrays and flags -------------------------- #
        self.norm = None
        self._WCout = None
        self.wash_obs = None
        self.wash_time = None
        self.trained = False  # Flag for training
        self.initialised = False  # Flag for washout
        self.filename = 'my_ESN'  # Default ESN file name

    @property
    def N_train(self):
        return int(round(self.t_train / self.dt_ESN))

    @property
    def N_val(self):
        return int(round(self.t_val / self.dt_ESN))

    @property
    def N_test(self):
        return int(round(self.t_test / self.dt_ESN))

    @property
    def len_train_data(self):
        val = self.N_train + self.N_val + self.N_wash
        if self.perform_test:
            val += self.N_test * 5
        return val

    @property
    def N_dim_in(self):
        if self.bayesian_update:
            return self.N_dim
        else:
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
            if u.ndim == 1:
                u = np.expand_dims(u, axis=-1)
            self.u = u

        if r is not None:
            if r.ndim == 1:
                r = np.expand_dims(r, axis=-1)
            self.r = r

        if self.r.shape[-1] != self.u.shape[-1]:
            raise AssertionError(['reset_state', self.r.shape, self.u.shape])

    def get_reservoir_state(self):
        return self.u, self.r

    def reconstruct_state(self, observed_data, filter_=EnKF, update_reservoir=True, Cdd=None, inflation=1.01):

        Nq = len(self.observed_idx)

        # if not hasattr(self, 'M') or if self.M.shape[0] != observed_data.shape[0]:
        if update_reservoir:
            M = np.zeros([Nq, self.N_dim + self.N_units])
        else:
            M = np.zeros([Nq, self.N_dim])

        for dim_i, obs_i in enumerate(self.observed_idx):
            M[dim_i, obs_i] = 1.

        # Define forecast state
        u, r = self.get_reservoir_state()
        if update_reservoir:
            x = np.concatenate([u, r], axis=0)
        else:
            x = u.copy()

        # Define observation error matrix
        if Cdd is None:
            Cdd = (0.1 * np.max(abs(observed_data))) ** 2 * np.eye(Nq)

        # Apply filter
        x_hat = filter_(Af=x, d=observed_data.squeeze(), Cdd=Cdd, M=M)

        if inflation > 1.:
            x_hat_mean = np.mean(x_hat, -1, keepdims=True)
            x_hat = x_hat_mean + (x_hat - x_hat_mean) * inflation

        # Return updates to u and r
        if update_reservoir:
            return x_hat[:self.N_dim], x_hat[self.N_dim:]
        else:
            return x_hat[:self.N_dim], None

    def outputs_to_inputs(self, full_state):
        if self.bayesian_update:
            return full_state
        else:
            return full_state[self.observed_idx]
    # _______________________________________________________________________________________________________ JACOBIAN

    def Jacobian(self, open_loop_J=True, state=None):
        if state is None:
            u_in, r_in = self.get_reservoir_state()
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

    def openLoop(self, u_wash, extra_closed=0, force_reconstruct=True, update_reservoir=True, inflation=1.01):
        """ Modified class of the parent class to account for bayesian update to the network
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - r: time series of reservoir states
        """
        Nt = u_wash.shape[0] - 1
        if extra_closed:
            Nt += extra_closed

        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))

        if self.bayesian_update and self.trained and force_reconstruct:
            u0, r0 = self.reconstruct_state(observed_data=u_wash[0],
                                            update_reservoir=update_reservoir, inflation=inflation)
            self.reset_state(u=u0, r=r0)
        else:
            self.reset_state(u=u_wash[0])

        u[0, self.observed_idx], r[0] = self.get_reservoir_state()

        for ii in range(Nt):
            if self.bayesian_update and self.trained and force_reconstruct:
                u_in, r_in = self.reconstruct_state(observed_data=u_wash[ii],
                                                    update_reservoir=update_reservoir, inflation=inflation)
            else:
                u_in, r_in = u_wash[ii], r[ii]

            u[ii + 1], r[ii + 1] = self.step(u_in, r_in)
        return u, r

    def closedLoop(self, Nt):
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """

        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))
        u[0, self.observed_idx], r[0] = self.get_reservoir_state()

        for i in range(Nt):
            u_input = self.outputs_to_inputs(full_state=u[i])
            u[i + 1], r[i + 1] = self.step(u_input, r[i])
        return u, r

    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self, train_data, add_noise=True, plot_training=True, folder='./', **kwargs):
        """

        train_data: training data shape [Nt x Ndim]
        add_noise : flag to add noise to the input training data

        """

        # The objective is to initialize the weights in Wout
        self.Wout = np.zeros([self.N_units + 1, self.N_dim])

        # =========  Format training data into wash-train-val and test sets ========= ##
        U_wtv, Y_tv, U_test, Y_test = self.format_training_data(data=train_data, add_noise=add_noise)

        # ======================  Generate matrices W and Win ======================= ##
        self.generate_W_Win(seed=self.seed_W)

        # ===================  Bayesian Hyperparameter Optimization ================= ##
        ti = time.time()  # check time

        # define grid search and bayesian optimisation hp_names
        search_grid, search_space, hp_names = self.hyperparameter_search()
        tikh_opt = np.zeros(self.N_func_evals)  # optimal tikhonov in each evaluation
        self.val_k = 0  # Validation iteration counter

        # Validation function
        if 'validation_strategy' not in kwargs.keys():
            validation_strategy = EchoStateNetwork.RVC_Noise
        else:
            validation_strategy = kwargs['validation_strategy']

        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_tv=Y_tv.copy(),
                           tikh_opt=tikh_opt,
                           hp_names=hp_names)

        # Perform optimization
        res = self.hyperparam_optimization(val_func, search_space, x0=search_grid, n_calls=self.N_func_evals)
        f_iters = np.array(res.func_vals)

        # Save optimized parameters
        self.reset_hyperparams(res.x, hp_names, tikhonov=tikh_opt[np.argmin(f_iters)])
        for hpi, hp in enumerate(self.optimize_hyperparams):
            self.filename += '_{}_{:.3e}'.format(hp, getattr(self, hp))

        # ============================  Train Wout ================================== ##
        self.Wout = self.solve_ridge_regression(U_wtv, Y_tv)
        print('\n Time per hyperparameter eval.:', (time.time() - ti) / self.N_func_evals,
              '\n Best Results: x ')
        for hp in self.optimize_hyperparams:
            print('\t {:.2e}'.format(getattr(self, hp)), end="")
        print(' f {:.4f}'.format(-res.fun))

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

        # ====================  Set flags and initialise state ====================== ##
        # Flag case as trained
        self.trained = True

    def initialise_state(self, data, N_ens=1, seed=0):
        if hasattr(self, 'seed'):
            seed = self.seed
        rng0 = np.random.default_rng(seed)
        # initialise state with a random sample from test data
        u_init, r_init = np.empty((self.N_dim, N_ens)), np.empty((self.N_units, N_ens))
        # Random time windows and dimension
        if data.shape[0] == 1:
            dim_ids = [0] * N_ens
        else:
            if N_ens > data.shape[0]:
                replace = False
            else:
                replace = True
            dim_ids = rng0.choice(data.shape[0], size=N_ens, replace=replace)

        t_ids = rng0.choice(data.shape[1] - self.N_wash, size=N_ens, replace=False)

        # Open loop for each ensemble member
        for ii, ti, dim_i in zip(range(N_ens), t_ids, dim_ids):
            self.reset_state(u=np.zeros((self.N_dim, 1)), r=np.zeros((self.N_units, 1)))
            u_open, r_open = self.openLoop(data[dim_i, ti: ti + self.N_wash], force_reconstruct=False)
            u_init[:, ii], r_init[:, ii] = u_open[-1], r_open[-1]
        # Set physical and reservoir states as ensembles
        self.reset_state(u=u_init, r=r_init)

    def generate_W_Win(self, seed=1):
        rnd0 = np.random.default_rng(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero
        Win = lil_matrix((self.N_units, self.N_dim_in + 1))  # +1 accounts for input bias
        if self.Win_type == 'sparse':
            for j in range(self.N_units):
                Win[j, rnd0.choice(self.N_dim_in + 1)] = rnd0.uniform(low=-1, high=1)
        elif self.Win_type == 'dense':
            for j in range(self.N_units):
                Win[j, :] = rnd0.uniform(low=-1, high=1, size=self.N_dim_in + 1)
        else:
            raise ValueError("Win type {} not implemented ['sparse', 'dense']".format(self.Win_type))
        # Make csr matrix
        self.Win = Win.tocsr()

        # Reservoir state matrix: Erdos-Renyi network
        W = csr_matrix(rnd0.uniform(low=-1, high=1, size=(self.N_units, self.N_units)) *
                       (rnd0.random(size=(self.N_units, self.N_units)) < (1 - self.sparsity)))
        # scale W by the spectral radius to have unitary spectral radius
        spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]

        self.W = (1. / spectral_radius) * W

    def compute_RR_terms(self, U_wtv, Y_tv):
        LHS, RHS = 0., 0.
        R_RR = [np.empty([0, self.N_units])] * U_wtv.shape[0]
        U_RR = [np.empty([0, self.N_dim])] * U_wtv.shape[0]
        self.reset_state(u=self.u * 0, r=self.r * 0)

        # todo: pallallelize summations
        # part = partial(EchoStateNetwork.for_func, R_RR=R_RR, U_RR=U_RR)
        # sol = [self.pool.apply_async(part, ) for mi in range(self.m)]
        # psi = [s.get() for s in sol]

        for ll in range(U_wtv.shape[0]):

            # Washout phase. Store the last r value only
            self.r = self.openLoop(U_wtv[ll][:self.N_wash], extra_closed=True)[1][-1]

            # Split training data for faster computations
            U_train = np.array_split(U_wtv[ll][self.N_wash:], self.N_split, axis=0)
            Y_target = np.array_split(Y_tv[ll], self.N_split, axis=0)

            for U_t, Y_t in zip(U_train, Y_target):
                # Open-loop train phase
                u_open, r_open = self.openLoop(U_t, extra_closed=True)

                self.reset_state(u=u_open[-1], r=r_open[-1])

                u_open, r_open = u_open[1:], r_open[1:]
                if u_open.ndim > 2:
                    u_open, r_open = u_open.squeeze(axis=-1), r_open.squeeze(axis=-1)

                R_RR[ll] = np.append(R_RR[ll], r_open, axis=0)
                U_RR[ll] = np.append(U_RR[ll], u_open, axis=0)

                # Compute matrices for linear regression system
                bias_out = np.ones([r_open.shape[0], 1]) * self.bias_out
                r_aug = np.hstack((r_open, bias_out))
                LHS += np.dot(r_aug.T, r_aug)
                RHS += np.dot(r_aug.T, Y_t)

        return LHS, RHS, U_RR, R_RR


    def solve_ridge_regression(self, U_wtv, Y_tv):
        LHS, RHS = self.compute_RR_terms(U_wtv, Y_tv)[:2]
        LHS.ravel()[::LHS.shape[1] + 1] += self.tikh  # Add tikhonov to the diagonal
        return np.linalg.solve(LHS, RHS)  # Solve linear regression problem


    def format_training_data(self, data=None, add_noise=True):
        """
        :param data: training data labels with dimensions L x Nt x Ndim
        :param add_noise:  boolean. Should we add noise to the input data?
        """

        #   APPLY UPSAMPLE AND OBSERVED INDICES ________________________

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Set labels always as the full state
        Y = data[:, ::self.upsample].copy()

        # Case I: Full observability .OR. Case II: Partial observability
        if not self.bayesian_update:
            U = Y[:, :, self.observed_idx]
        # Case III: Full observability with a DA-reconstructed state.
        else:
            U = Y.copy()

        assert Y.shape[-1] >= U.shape[-1]
        assert U.shape[-1] == self.N_dim_in

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

        if add_noise:
            #  ==================== ADD NOISE TO TRAINING INPUT ====================== ##
            # Add noise to the inputs if distinction inputs/labels is not given.
            # Larger noise promotes stability in long term, but hinders time accuracy
            U_std = np.std(U, axis=1)
            rng_noise = np.random.default_rng(self.seed_noise)
            for ll in range(Y.shape[0]):
                for dd in range(self.N_dim_in):
                    U_wtv[ll, :, dd] += rng_noise.normal(loc=0, scale=self.noise * U_std[ll, dd], size=U_wtv.shape[1])
                    U_test[ll, :, dd] += rng_noise.normal(loc=0, scale=self.noise * U_std[ll, dd], size=U_test.shape[1])

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
        result = gp_minimize(val,  # function to minimize
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

        # num steps forward the validation interval is shifted
        N_fw = (case.N_train - case.N_val - case.N_wash) // (case.N_folds - 1)

        # Train using tv: Wout_tik is passed with all the combinations of tikh_ and target noise
        # This must result in L-Xa timeseries
        LHS, RHS, U_train, R_train = case.compute_RR_terms(U_wtv, Y_tv)
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
                    case.reset_state(u=case.outputs_to_inputs(u_open[-1]), r=r_open[-1])

                    case.Wout = Wout_tik[tik_j]
                    U_close = case.closedLoop(case.N_val)[0][1:].squeeze(axis=-1)

                    # Compute normalized MSE
                    n_MSE[tik_j] += np.log10(np.mean((Y_val - U_close) ** 2) / np.mean(case.norm ** 2))

                    # prevent from diverging to infinity: MSE=1E10 (useful for hybrid and similar architectures)
                    if np.isnan(n_MSE[tik_j]) or np.isinf(n_MSE[tik_j]):
                        n_MSE[tik_j] = 10 * case.N_folds

        # select and save the optimal tikhonov and noise level in the targets
        a = n_MSE.argmin()
        tikh_opt[case.val_k] = case.tikh_range[a]
        case.tikh = case.tikh_range[a]
        normalized_best_MSE = n_MSE[a] / case.N_folds / case.L

        case.val_k += 1
        print(case.val_k, end="")
        for hp in case.optimize_hyperparams:
            print('\t {:.3e}'.format(getattr(case, hp)), end="")
        print('\t {:.4f}'.format(normalized_best_MSE))

        return normalized_best_MSE

    # ________________________________________________________________________________________________ TESTING FUNCTION
    def run_test(self, U_test, Y_test, pdf_file=None, folder='./', max_tests=10, seed=0):
        if hasattr(self, 'seed'):
            seed = self.seed

        rng0 = np.random.default_rng(seed)
        if pdf_file is None:
            os.makedirs(folder + 'figs_ESN', exist_ok=True)
            pdf_file = plt_pdf.PdfPages(folder + 'figs_ESN/' + self.filename + '_Test.pdf')

        L, max_test_time, Nq = U_test.shape
        max_test_time -= self.N_wash

        # Select test cases (with a maximum of max_tests)
        total_tests = min(max_tests, L)
        if max_tests != L:
            # L_indices = np.random.random_integers(low=0, high=L, size=total_tests)
            if total_tests <= L:
                L_indices = rng0.choice(L, total_tests, replace=False)
            else:
                L_indices = rng0.choice(L, total_tests, replace=True)
        else:
            L_indices = np.arange(L)

        # Time window of each test
        N_test = self.N_val

        # Random initial time
        if max_test_time > self.N_val:
            # initial_times = np.random.random_integers(low=0, high=max_test_time - N_test, size=total_tests)
            initial_times = rng0.choice(max_test_time - N_test, size=total_tests, replace=False)
        else:
            initial_times = [0] * total_tests

        # Break if not enough train_data for testing
        if total_tests < 1:
            print('Test not performed. Not enough train_data')
        else:
            errors = []
            for test_i, Li, i0 in zip(np.arange(total_tests), L_indices, initial_times):

                # Reset state
                self.reset_state(u=self.u * 0., r=self.r * 0.)

                # Select dataset
                U_test_l, Y_test_l = U_test[Li], Y_test[Li]

                # washout for each interval
                u_open, r_open = self.openLoop(U_test_l[i0: i0 + self.N_wash])

                self.reset_state(u=self.outputs_to_inputs(full_state=u_open[-1]), r=r_open[-1])

                # Data to compare with, i.e., labels
                Y_labels = Y_test_l[i0 + self.N_wash: i0 + self.N_wash + N_test]

                # Closed-loop prediction
                Y_closed = self.closedLoop(N_test)[0][1:].squeeze(axis=-1)

                # compute error
                err = np.log10(np.mean((Y_closed - Y_labels) ** 2) / np.mean(np.expand_dims(self.norm, -1) ** 2))
                errors.append(err)

                # plot test
                fig, axs = plt.subplots(nrows=Nq, ncols=1, figsize=[10, 2 * Nq], sharex='all', layout='tight')
                t_ = np.arange(len(Y_closed)) * self.dt_ESN

                if Nq == 1:
                    axs = [axs]

                for dim_i, ax in enumerate(axs):
                    ax.plot(t_, Y_labels[:, dim_i], 'k', label='truth dim ' + str(dim_i))
                    ax.plot(t_, Y_closed[:, dim_i], '--r', label='ESN dim ' + str(dim_i))
                    ax.legend(title='Test {}: Li = {}'.format(test_i, Li), loc='upper left', bbox_to_anchor=(1, 1))

                # Save to pdf
                if test_i > 0:
                    add_pdf_page(pdf_file, fig, close_figs=True)
                else:
                    add_pdf_page(pdf_file, fig, close_figs=False)

            print('Median and max error in', total_tests, 'tests:', np.median(errors), np.max(errors))

