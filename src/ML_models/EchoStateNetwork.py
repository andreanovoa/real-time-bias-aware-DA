import time
import matplotlib.backends.backend_pdf as plt_pdf
import sys

from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append('../')

from src.data_assimilation import EnKF, EnSRKF
from src.utils import *

# Validation methods
from functools import partial
from itertools import product
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real
from skopt.plots import plot_convergence

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

XDG_RUNTIME_DIR = 'tmp/'


class EchoStateNetwork:
    """
    The EchoStateNetwork class implements a reservoir computing model for time series prediction. It is based on
    the Echo State Network (ESN) approach, which uses a randomly connected reservoir of neurons to map inputs
    into high-dimensional space. This allows the model to capture complex dynamics with efficient training.

    Attributes:
        - Reservoir and network hyperparameters (e.g., N_units, rho, sigma_in, tikh, etc.)
        - Training, validation, and test configuration (e.g., t_train, t_val, t_test, N_wash, etc.)
        - Optimization settings for Bayesian hyperparameter search (e.g., hyperparameters_to_optimize, rho_range, etc.)
        - Input and output weight matrices (Win, Wout) and reservoir state matrix (W)
    """

    augment_data = False  # Data augmentation during training?
    bayesian_update = False
    bias_in = np.array([0.1])  #
    bias_out = np.array([1.0])  # For symmetry breaking
    connect = 3  # Connectivity between neurons
    figs_folder = './figs_ESN/'
    filename = 'my_ESN'  # Default ESN file name
    L = 1  # Number of augmented datasets

    input_parameters = None

    N_folds = 4  # Folds over the training set
    N_func_evals = 20  # Total evals of Bayesian hyperparameter optimization (BHO)
    N_grid = 4  # BHO grid N_grid x N_grid \geq N_func_evals
    N_initial_rand = 0  # Initial random evaluations at BYO
    N_split = 4  # Splits of training data for faster computation
    N_units = 100  # Number of neurones
    N_wash = 50  # Number of washout steps

    max_L_tests = 10
    perform_test = True  # Run tests during training?
    random_initialization = False
    seed_W = 0  # Random seed for Win and W definition
    seed_noise = 0  # Random seed for input training data
    t_val = 0.1  # Validation time
    t_train = 1.0  # Training time
    t_test = 0.5  # Testing time
    upsample = 5  # Upsample x dt_model = dt_ESN
    Win_type = 'sparse'  # Type of Wim definition [sparse/dense]
    norm_method = 'range' # Normalization method for input data

    # Default hyperparameters and optimization ranges -----------------------
    noise = 1e-10
    noise_type = 'gauss'
    hyperparameters_to_optimize = ['rho', 'sigma_in', 'tikh']
    rho = 0.9
    rho_range = (.8, 1.05)
    sigma_in = 10 ** -3
    sigma_in_range = (-5, -1)
    tikh = 1e-12
    tikh_range = [1e-8, 1e-10, 1e-12, 1e-16]

    def __init__(self, y, dt=1, **kwargs):
        """
        Initializes the EchoStateNetwork class with input data, time step, and optional hyperparameters.
        Validates the input dimensions and initializes reservoir states, time steps, and flags.

        Args:
            y (np.ndarray): Initial state of the physical system (dimensions: N_dim x N_samples).
            dt (float): time step of the input data, such that dt_ESN = dt * upsample.
            **kwargs: Optional keyword arguments to override default class attributes.

        Raises:
            AssertionError: If y has more than two dimensions or invalid values in kwargs.

        """

        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        elif y.ndim > 2:
            raise AssertionError(f'y.shape={y.shape}. The input y must have 2 or less dimension')

        [setattr(self, key, val) for key, val in kwargs.items() if hasattr(EchoStateNetwork, key)]

        # -----------  Initialise state and reservoir state to zeros ------------ #
        self.N_dim = y.shape[0]

        self.u = np.zeros((self.N_dim, y.shape[1]))
        self.r = np.zeros((self.N_units, y.shape[1]))

        self.observed_idx = np.arange(self.N_dim)  # initially, assume full observability.

        # ---------------- Define time steps and time windows -------------------- #
        # self.dt = dt
        self.dt_ESN = dt * self.upsample

        # ------------------------ Initialize ESN matrices -------------------------- #
        # self.Win = kwargs.get('Win', None)  # Input matrix (self.N_units, self.N_dim+1)  
        # self.Wout = kwargs.get('Wout', None)  # Output matrix (self.N_units+1, self.N_dim)
        # self.W = kwargs.get('W', None)  # Reservoir state matrix (self.N_units, self.N_units)
        
        self.trained = all([getattr(self, key) is not None for key in ['Wout', 'Win', 'W', 'norm']])
        self.val_k = kwargs.get('val_k', 0)  # Validation counter
        self.initialised = False  # Flag for washout

    @property
    def W(self):
        """
        Returns the reservoir state matrix (W) in CSR format.
        """
        if not hasattr(self, '_W'):
            return None
        return self._W
    
    @W.setter
    def W(self, value):
        """
        Setter for the reservoir state matrix (W). Converts the input to CSR format.
        """
        if not isinstance(value, csr_matrix):
            value = csr_matrix(value)

        # Ensure the matrix is square and has the correct dimensions
        assert value.shape == (self.N_units, self.N_units), \
            f'W must be a square matrix of shape ({self.N_units}, {self.N_units}), but got {value.shape}'
        
        # Set the reservoir state matrix
        self._W = value
    
    @property
    def Win(self):
        """
        Returns the input matrix (Win).
        """
        if not hasattr(self, '_Win'):
            return None
        return self._Win
    
    @Win.setter
    def Win(self, value):
        """
        Setter for the input matrix (Win). Converts the input to CSR format if sparse.
        """
        if self.Win_type == 'sparse':
            value = csr_matrix(value)
        elif self.Win_type == 'dense':
            value = value.toarray() 
        else:
            raise ValueError(f"Win type {self.Win_type} not implemented ['sparse', 'dense']")

        # Ensure the matrix has the correct dimensions
        assert value.shape ==  (self.N_units, self.N_dim_in+1), \
            f'Win must be a square matrix of shape ({self.N_units}, {self.N_dim_in + 1}), but got {value.shape}'
        
        # Set the input matrix
        self._Win = value


    @property
    def Wout(self):
        """
        Returns the output matrix (Wout).
        """
        if not hasattr(self, '_Wout'):
            return None
        return self._Wout

    @Wout.setter
    def Wout(self, value):
        """
        Setter for the reservoir state matrix (W). 
        """
        # Ensure the matrix has the correct dimensions
        assert value.shape == (self.N_units + 1, self.N_dim), \
            f'Wout must be a matrix of shape ({self.N_units + 1}, {self.N_dim}), but got {value.shape}'
        # Set the output matrix
        self._Wout = value

    @property
    def val_k(self):
        """
        Returns the current validation counter.
        """
        if not hasattr(self, '_val_k'):
            return 0
        return self._val_k
    
    @val_k.setter
    def val_k(self, value):
        """
        Setter for the validation counter.
        """
        if not isinstance(value, int):
            raise TypeError('val_k must be an integer')
        self._val_k = value

    @property
    def dt_physical(self):
        """
        Computes the physical time step based on the ESN time step and upsample factor.
        """
        return self.dt_ESN / self.upsample

    @property
    def N_train(self):
        """
        Computes the number of training steps based on training time (t_train) and ESN time step (dt_ESN).
        """
        return int(round(self.t_train / self.dt_ESN))

    @property
    def N_val(self):
        """
        Computes the number of validation steps based on validation time (t_val) and ESN time step (dt_ESN).
        """
        return int(round(self.t_val / self.dt_ESN))

    @property
    def N_test(self):
        """
        Computes the number of testing steps based on testing time (t_val) and ESN time step (dt_ESN).
        """
        return int(round(self.t_test / self.dt_ESN))

    @property
    def WCout(self):
        """
        Lazily computes the closed-loop reservoir weight matrix (W Cout) if it has not been precomputed.
        This matrix is only computed if the Jacobian in closed loop is needed.
        """
        if not hasattr(self, '_WCout'):
            return None
        return self._WCout
    
    @WCout.setter
    def WCout(self, value=None):
        """
        Setter for the closed-loop reservoir weight matrix (W Cout).
        """
        if self._WCout is None:
            self._WCout = np.linalg.lstsq(self.Wout[:-1], self.W.toarray(), rcond=None)[0]
        else:
            self._WCout = value

    @property
    def norm(self):
        """
        Returns the normalization factor for the input data.
        """
        if not hasattr(self, '_norm'):
            return None
        return self._norm

    @norm.setter
    def norm(self, value):
        """
        Setter for the normalization factor. Ensures it is N_dim.
        """
        assert value.shape[0] == self.N_dim_in, \
            f'Normalization factor must be dimension Ndim={self.N_dim}, got {value.shape}'
        self._norm = value

    @property
    def sparsity(self):
        """
        Computes the sparsity level of the reservoir connectivity matrix (W). This is, the
        fraction of connections between neurons in the reservoir that are set to zero.
            sparsity = 1 - #Active connections / Total possible connections
        """
        return 1 - self.connect / (self.N_units - 1)

    def outputs_to_inputs(self, full_state, add_parameters=False):
        """
        Maps the full state (predicted or reconstructed) to input states for the ESN.

        Args:
            full_state (np.ndarray): Full physical state vector.

        Returns:
            np.ndarray: Input state vector mapped from the full state.
        """
        observed_state = full_state[self.observed_idx]
        if not add_parameters:
            return observed_state
        else:
            return np.concatenate([observed_state, self.input_parameters], axis=0)

    @property
    def N_dim_in(self):
        """
        Computes the number of input dimensions.
        """
        if self.input_parameters is None:
            return len(self.observed_idx)
        else:
            return len(self.observed_idx) + self.input_parameters.shape[0]

    # --------------------------------------------------------------------------------------------------------
    def reset_hyperparams(self, params, names, tikhonov=None):
        """
        Updates specific hyperparameters with new values.

        Args:
            params (list): List of hyperparameter values to set.
            names (list): Names of the hyperparameters to update.
            tikhonov (float, optional): Value to set for the Tikhonov regularization parameter.

        Outputs:
            None. Updates internal hyperparameter values.
        """
        for hp, name in zip(params, names):
            if name == 'sigma_in':
                setattr(self, name, 10 ** hp)
            else:
                setattr(self, name, hp)
        if tikhonov is not None:
            setattr(self, 'tikh', tikhonov)

    def reset_state(self, u=None, r=None):
        """
        Resets the physical (u) and reservoir (r) states.

        Args:
            u (np.ndarray, optional): New physical state
                - default: None to retain current state.
            r (np.ndarray, optional): New reservoir state
                 - default: None to retain current state.
        Raises:
            AssertionError: If dimensions of u and r are incompatible.

        """
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
        """
        Retrieves the current physical and reservoir states.

        Returns:
            tuple: (u, r), where u is the physical state and r is the reservoir state.
        """
        return self.u, self.r

    # _______________________________________________________________________________________________________ JACOBIAN

    def Jacobian(self, open_loop_J=True, state=None):
        """
        Computes the Jacobian matrix for the reservoir, either in open-loop or closed-loop mode.

        Args:
            open_loop_J (bool): If True (default), compute the open-loop Jacobian.
            state (tuple, optional): Optional input state (u_in, r_in) to compute Jacobian.

        Returns:
            np.ndarray: Jacobian matrix of the reservoir dynamics.
        """
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
        """
        Advances the reservoir by one time step and updates its internal state.

        Args:
            u (np.ndarray): Input physical state at the current time step.
            r (np.ndarray): Reservoir state at the current time step.

        Returns:
            tuple: (u_out, r_out) where u_out is the output state and r_out is the updated reservoir state.
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
        u_out = self.reservoir_to_physical(r_aug)
        return u_out, r_out

    def reservoir_to_physical(self, r_aug):

        """ Converts the reservoir state to the physical state using the output weight matrix (Wout).
        Args:
            r_aug (np.ndarray): Augmented reservoir state including output bias.
        """
        # print(f'Wout shape: {self.Wout.shape}, r_aug shape: {r_aug.shape}')
        return np.dot(self.Wout.T, r_aug)

    def openLoop(self, u_wash, extra_closed=0):
        """
        Executes the ESN in open-loop mode to wash out initial dynamics and prepare for training/validation.

        Args:
            u_wash (np.ndarray): Input sequence for the washout phase.
            extra_closed (int): Number of additional closed-loop steps to forecast after the washout.
            force_reconstruct (bool): If True, forces Bayesian state reconstruction during washout.
            update_reservoir (bool): If True, updates reservoir during reconstruction.
            inflation (float): Inflation factor for Bayesian reconstruction (default: 1.01).

        Returns:
            tuple: (u, r) where u is the physical state sequence and r is the reservoir state sequence.
        """
        Nt = u_wash.shape[0] - 1
        if extra_closed:
            Nt += extra_closed

        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))

        self.reset_state(u=u_wash[0])

        u[0, self.observed_idx], r[0] = self.get_reservoir_state()

        for ii in range(Nt):
            u_in, r_in = u_wash[ii], r[ii]

            u1, r1 = self.step(u_in, r_in)

            u[ii + 1], r[ii + 1]  = u1, r1
        return u, r

    def closedLoop(self, Nt):
        """
        Advances the ESN in closed-loop mode for prediction.

        Args:
            Nt (int): Number of prediction time steps.

        Returns:
            tuple: (u, r) where u is the forecasted physical state sequence and r is the reservoir state sequence.
        """

        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))
        u[0, self.observed_idx], r[0] = self.get_reservoir_state()

        for i in range(Nt):
            u_input = self.outputs_to_inputs(full_state=u[i])
            u[i + 1], r[i + 1] = self.step(u_input, r[i])
        return u, r

    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self, train_data,
              add_noise=True,
              plot_training=True,
              save_ESN_training=False,
              folder=None,
              validation_strategy=None):
        """
        Trains the ESN using ridge regression and Bayesian hyperparameter optimization.

        Args:
            train_data (np.ndarray): Training data with dimensions [L x Nt x N_dim].
            add_noise (bool): If True, adds noise to the input during training.
            plot_training (bool): If True, visualizes the training process.
            save_ESN_training (bool): If True, saves training plots to a file.
            folder (str): Directory to save training plots (if save_ESN_training=True).
            validation_strategy (function): Custom validation function for hyperparameter tuning.
        """
        if self.trained:
            print("ESN is already trained. Skipping training.")
            return

        # ========================== STEP 1: DATA FORMATTING ==========================
        # Format data into washout, train/validation, and test sets
        U_wtv, Y_tv, U_test, Y_test = self.format_training_data(train_data, add_noise=add_noise)

        # print([xx.shape for xx in [U_wtv, Y_tv, U_test, Y_test]])

        # Ensure W and Win matrices are initialized
        if self.W is None or self.Win is None:
            self.generate_W_Win(seed=self.seed_W)

        self.Wout = np.zeros((self.N_units + 1, self.N_dim))  # Initialize Wout with zeros

        # =================== STEP 2: BAYESIAN HYPERPARAMETER OPTIMIZATION ==============
        self.val_k = 0  # Reset validation counter at the start of training
        # Perform hyperparameter optimization if required
        if self.hyperparameters_to_optimize:
            bo_results = self.optimize_hyperparameters(U_wtv, Y_tv, validation_strategy,
                                                       print_convergence=plot_training)
        else:
            bo_results = None
        # ====================== STEP 3: RIDGE REGRESSION TRAINING =====================
        # Compute the output weight matrix Wout
        self.Wout = self._solve_ridge_regression(U_wtv, Y_tv)

        # ========================== STEP 4: NETWORK INITIALIZATION ======================
        if self.random_initialization:
            self.initialise_state(data=U_wtv)

        # ========================== STEP 5: RESULTS AND PLOTTING ======================
        if plot_training:
            self._plot_training_results(U_test, Y_test, bo_results, save_ESN_training, folder)

        # Mark the model as trained
        self.trained = True
        # print("Training completed successfully.")



    def generate_W_Win(self, seed=1):
        """
        Generates the input weight matrix (Win) and reservoir weight matrix (W) with sparsity constraints.

        Args:
            seed (int): Random seed for reproducibility.

        Raises:
            ValueError: If the specified self.Win_type is unsupported. Allowed values: 'sparse' or 'dense'.

        Outputs:
            None. Updates internal matrices Win and W with appropriate values.
        """

        rnd0 = np.random.default_rng(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero
        if self.Win is None:
            Win = lil_matrix((self.N_units,
                              self.N_dim_in + 1))  # +1 accounts for input bias
            if self.Win_type == 'sparse':
                for j in range(self.N_units):
                    Win[j, rnd0.choice(self.N_dim_in + 1)] = rnd0.uniform(low=-1, high=1)
            elif self.Win_type == 'dense':
                for j in range(self.N_units):
                    Win[j, :] = rnd0.uniform(low=-1, high=1, size=self.N_dim_in + 1)
            else:
                raise ValueError("Win type {} not implemented ['sparse', 'dense']".format(self.Win_type))
            # Store
            self.Win = Win
        else:
            print('Skipping Win generation, using provided Win matrix.')

        # Reservoir state matrix: Erdos-Renyi network
        if self.W is None:
            W = csr_matrix(rnd0.uniform(low=-1, high=1, size=(self.N_units, self.N_units)) *
                        (rnd0.random(size=(self.N_units, self.N_units)) < (1 - self.sparsity)))
            # scale W by the spectral radius to have unitary spectral radius
            spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
            self.W = (1. / spectral_radius) * W
        else:
            print('Skipping W generation, using provided W matrix.')

    def _compute_RR_terms(self, U_wtv, Y_tv):
        """
        Computes the Ridge Regression (RR) terms, including left-hand side (LHS) and right-hand side (RHS)
        matrices, for training the output weights.

        Args:
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_tv (np.ndarray): Corresponding output labels for the train-validation data.

        Returns:
            tuple:
                - LHS (np.ndarray): Left-hand side matrix for ridge regression.
                - RHS (np.ndarray): Right-hand side matrix for ridge regression.
                - U_RR (list): List of input states split by L-segments.
                - R_RR (list): List of reservoir states split by L-segments.
        """

        LHS, RHS = 0., 0.
        R_RR = [np.empty([0, self.N_units])] * U_wtv.shape[0]
        U_RR = [np.empty([0, self.N_dim])] * U_wtv.shape[0]

        self.reset_state(u=np.zeros((self.N_dim, 1)), r=np.zeros((self.N_units, 1)))

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

    def _solve_ridge_regression(self, U_wtv, Y_tv):
        """
        Solves the ridge regression problem to compute the output weight matrix (Wout).

        Args:
            U_wtv (np.ndarray): Input data for ridge regression (train/valiladion).
            Y_tv (np.ndarray): Target labels for ridge regression.

        Returns:
            np.ndarray: Computed output weight matrix (Wout).
        """
        LHS, RHS = self._compute_RR_terms(U_wtv, Y_tv)[:2]
        LHS.ravel()[::LHS.shape[1] + 1] += self.tikh  # Add tikhonov to the diagonal
        return np.linalg.solve(LHS, RHS)  # Solve linear regression problem

    def format_training_data(self, data=None, add_noise=True, observed_idx=None):
        """
        Formats the input data into washout, train/val, and test sets. Optionally adds noise to the input.

        Args:
            - data (np.ndarray): Input time series data with dimensions [L x Nt x N_dim].
            - add_noise (bool): Whether to add noise to the training input data (default: True).
            - observed_idx (list, optional): indices which are observed
        Returns:
            - U_wtv (np.ndarray): Wash-train-validation input data.
            - Y_tv (np.ndarray): Corresponding labels for train/validation data.
            - U_test (np.ndarray): Test input data.
            - Y_test (np.ndarray): Test labels.
        Raises:
            ValueError: If the input data length is insufficient for training.
        """

        #   APPLY UPSAMPLE AND OBSERVED INDICES ________________________
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        if observed_idx is not None:
            self.observed_idx = observed_idx

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

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS ______________________
        N_wtv = self.N_train + self.N_val

        if U.shape[1] < N_wtv:
            print(U.shape, N_wtv)
            raise ValueError('Increase the length of the training data signal')

        U_wtv = U[:, :N_wtv - 1].copy()
        Y_tv = Y[:, self.N_wash + 1:N_wtv].copy()
        U_test = U[:, N_wtv:].copy()
        Y_test = Y[:, N_wtv:].copy()

        # compute norm (normalize inputs by component range)
        self.norm = EchoStateNetwork.__set_norm(U_wtv, method=self.norm_method)

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
    
    @staticmethod
    def __set_norm(train_data, method='range'):
        """
        Computes the normalization factor for the input data.
        Args:
            U_wtv (np.ndarray): Wash-train-validation training input data. (Nens x Nt x Ndim).
        Returns:
            float: Normalization factor based on the range of the input data. 
        """
        assert train_data.ndim == 3, f'U_wtv must be a 3D array, got {train_data.ndim}D: ({train_data.shape})'

        if method == 'std':
            return np.mean(np.std(train_data, axis=1), axis=0)
        elif method == 'max':
            return np.mean(np.max(train_data, axis=1), axis=0)
        elif method == 'mean':
            return np.mean(np.mean(train_data, axis=1), axis=0)
        elif method == 'range':
            m = np.mean(train_data.min(axis=1), axis=0)
            M = np.mean(train_data.max(axis=1), axis=0)
            if np.any(M == m):
                raise ValueError("Normalization range cannot be zero. Check the input data.")
            return M - m
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        

    def initialise_state(self, data, N_ens=1, seed=0):
        if hasattr(self, 'seed'):
            seed = self.seed

        if hasattr(self, 'm'):
            N_ens = getattr(self, 'm')

        rng0 = np.random.default_rng(seed)
        # initialise state with a random sample from test data
        u_init, r_init = (np.empty((self.N_dim, N_ens)),
                          np.empty((self.N_units, N_ens)))
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
            u_open, r_open = self.openLoop(data[dim_i, ti: ti + self.N_wash])
            u_init[:, ii], r_init[:, ii] = u_open[-1, ..., 0], r_open[-1, ..., 0]
        # Set physical and reservoir states as ensembles
        self.reset_state(u=u_init, r=r_init)

    # ___________________________________________________________________________________________ BAYESIAN OPTIMIZATION

    def optimize_hyperparameters(self, U_wtv, Y_tv, validation_strategy=None, print_convergence=True):
        """
        Performs Bayesian hyperparameter optimization to minimize the validation loss.

        Args:
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_tv (np.ndarray): Corresponding labels for train-validation data.
            validation_strategy (function, optional): Validation function for hyperparameter tuning.
                Defaults to `__RVC_Noise`.

        Returns:
            OptimizeResult: Results of the Bayesian optimization process.
        """
        # print("Starting Bayesian hyperparameter optimization...")

        # Prepare search grid, space, and hyperparameter names
        search_grid, search_space, hp_names = self.__hyperparameter_search(print_convergence=print_convergence)
        tikh_opt = np.zeros(self.N_func_evals)  # Track optimal Tikhonov regularization

        # Use default or provided validation strategy
        if validation_strategy is None:
            validation_strategy = self.__RVC_Noise

        # Prepare the validation function
        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_tv=Y_tv.copy(),
                           tikh_opt=tikh_opt,
                           hp_names=hp_names,
                           print_convergence=print_convergence
                           )

        # Configure ARD 5/2 Matern Kernel for Gaussian Process
        kernel_ = (ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0)) *
                   Matern(length_scale=[0.2] * len(search_space), nu=2.5, length_scale_bounds=(1e-2, 1e1)))

        # Gaussian Process reconstruction
        gp_estimator = GPR(kernel=kernel_,
                           normalize_y=True,
                           n_restarts_optimizer=3,
                           noise=1e-10,
                           random_state=10)

        # Perform Bayesian Optimization
        result = gp_minimize(val_func,  # function to minimize
                             search_space,  # bounds
                             base_estimator=gp_estimator,  # GP kernel
                             acq_func="gp_hedge",  # acquisition function
                             n_calls=self.N_func_evals,  # number of evaluations
                             x0=search_grid,  # Initial grid points
                             n_random_starts=self.N_initial_rand,  # random initial points
                             n_restarts_optimizer=3,  # tries per acquisition
                             random_state=10)

        # Process results
        f_iters = np.array(result.func_vals)
        best_idx = np.argmin(f_iters)

        # Update hyperparameters with the best result
        self.reset_hyperparams(result.x, hp_names, tikhonov=tikh_opt[best_idx])

        print(f"seed {self.seed_W} \t Optimal hyperparameters: {result.x}, {self.tikh}, MSE: {result.fun}")

        return dict(res=result,
                    hp_names=hp_names)

    def __hyperparameter_search(self, print_convergence=True):
        """
        Prepares the search grid and search space for Bayesian hyperparameter optimization.
        TODO: add noise to the optional input_parameters to optimize.

        Returns:
            tuple:
                - search_grid (list): List of initial grid points for optimization.
                - search_space (list): Search space objects for each hyperparameter.
                - input_parameters (list): Names of the hyperparameters being optimized.
        """
        parameters = [hp for hp in self.hyperparameters_to_optimize if hp != 'tikh']

        if 'tikh' not in self.hyperparameters_to_optimize:
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
        if print_convergence:
            print('\n ----------------- HYPERPARAMETER SEARCH ------------------\n {0}x{0} grid'.format(self.N_grid) +
                  ' and {} points with Bayesian Optimization\n\t'.format(self.N_func_evals - self.N_grid ** 2), end="")
            for kk in self.hyperparameters_to_optimize:
                print('\t {}'.format(kk), end="")
            print('\t MSE val ')

        return search_grid, search_space, parameters

    @staticmethod
    def __RVC_Noise(x, case, U_wtv, Y_tv, tikh_opt, hp_names, print_convergence=True):
        """
        Implements Chaotic Recycle Validation for hyperparameter optimization.

        Args:
            x (list): Hyperparameter values to evaluate.
            case (EchoStateNetwork): Instance of the ESN being validated.
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_tv (np.ndarray): Corresponding labels for train/validation data.
            tikh_opt (np.ndarray): Array to store optimal Tikhonov regularization values.
            hp_names (list): Names of the hyperparameters being optimized.

        Returns:
            float: Normalized mean squared error (MSE) for the validation set.
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
        LHS, RHS, U_train, R_train = case._compute_RR_terms(U_wtv, Y_tv)
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
        normalized_best_MSE = n_MSE[a] / case.N_folds / case.L

        case.val_k += 1
        if print_convergence:
            print(case.val_k, end="")
            for hp in case.hyperparameters_to_optimize:
                print('\t {:.3e}'.format(getattr(case, hp)), end="")
            print('\t {:.4f}'.format(normalized_best_MSE))

        return normalized_best_MSE

    # _______________________________________________________________________________________ TEST & PLOTTING FUNCTIONS
    def run_test(self, U_test, Y_test, pdf_file=None, Nt_test=None,
                 max_L_tests=10, seed=0, nbins=20, plot_pdf=False, margin=None):
        """
        Evaluates the trained ESN on test data.

        Args:
            U_test (np.ndarray): Test input data [L x Nt x N_dim].
            Y_test (np.ndarray): Ground truth labels for test data.
            pdf_file (PdfPages, optional): File to save test plots
                - default: None.
            max_L_tests (int): Maximum number of L test cases to evaluate
                - default: 10.
            seed (int): Random seed for reproducibility.
            plot_pdf: choose to plot or not the pdf of the prediction
            nbins:
            Nt_test: length of the individual tests
        Returns:
            None. Prints error metrics and optionally saves plots.
        """
        if hasattr(self, 'seed'):
            seed = self.seed
        if max_L_tests is None and hasattr(self, 'max_L_tests'):
            max_L_tests = self.max_L_tests
        if Nt_test is None:
            Nt_test = self.N_val

        if U_test.ndim == 1:
            U_test = U_test[np.newaxis, :, np.newaxis]
        elif U_test.ndim == 2:
            U_test = U_test[np.newaxis, :]
        if Y_test.ndim == 1:
            Y_test = Y_test[np.newaxis, :, np.newaxis]
        elif Y_test.ndim == 2:
            Y_test = Y_test[np.newaxis, :]


        rng0 = np.random.default_rng(seed)

        L, max_test_time, Nq = U_test.shape
        max_test_time -= self.N_wash

        if Nq > 10:
            nrows, dims = 10, rng0.choice(Nq, 10, replace=False)
        else:
            nrows, dims = self.N_dim, np.arange(Nq)
            if Nq == 1:
                dims = [dims]

        observed_idx_np = np.array(self.observed_idx)

        # Select test cases (with a maximum of max_tests)
        if L > 1:
            if max_L_tests != L:
                L_indices = rng0.choice(L, max_L_tests, replace=max_L_tests > L)
                L_indices = sorted(L_indices)
            else:
                L_indices = np.arange(L)
        else:
            L_indices = [0]

        test_counter, errors = 0, []
        for test_i, Li in zip(np.arange(max_L_tests), L_indices):
            i0, err, predictions, clean, noisy = 0, [], [], [], []

            # Select dataset
            U_test_l, Y_test_l = U_test[Li], Y_test[Li]
            t_l = (np.arange(U_test_l.shape[0])) * self.dt_ESN

            # plot tests statistics if the test dataset is long
            if max_test_time // Nt_test > 1 or plot_pdf:
                fig, grid = plt.subplots(nrows=self.N_dim, ncols=2, figsize=[10, 2.5 * self.N_dim],
                                         sharex='col', sharey='row', layout='tight', width_ratios=[5, 1])
                axs, axs_pdf = grid.T
                if Nq == 1:
                    axs = [axs]
                    axs_pdf = [axs_pdf]

                for dim_i, ax in zip(range(self.N_dim), axs):
                    if dim_i in self.observed_idx:
                        _i = np.argmin(abs(observed_idx_np-dim_i))
                        ax.plot(t_l, U_test_l[:, _i], '-k', lw=4, alpha=0.5, label=f'Input')
                    ax.plot(t_l, Y_test_l[:, dim_i], '-k', lw=.85, label=f'Target (truth)')
            else:
                fig = None

            if margin is None:
                margin = np.max(U_test) * 0.1

            figures = []
            while i0 < max_test_time:
                test_counter += 1
                i1 = i0 + Nt_test

                current_input = U_test_l[i0:i1-1].copy()
                current_target = Y_test_l[i0+1:i1].copy()
                current_time = t_l[i0:i1]

                clean.append(current_input[self.N_wash:])
                noisy.append(current_target[self.N_wash:])

                # Reset state
                self.reset_state(u=self.u * 0., r=self.r * 0.)

                # washout
                u_open, r_open = self.openLoop(current_input[:self.N_wash], extra_closed=False)

                self.reset_state(u=self.outputs_to_inputs(full_state=u_open[-1]),
                                 r=r_open[-1])

                # Data to compare with, i.e., labels
                Y_labels = current_target[self.N_wash-1:].squeeze()

                # Closed-loop prediction
                Y_closed = self.closedLoop(Y_labels.shape[0])[0][1:].squeeze()

                if Nq == 1:
                    Y_closed = np.array([Y_closed]).T

                predictions.append(Y_closed)

                # compute error
                current_error = np.log10(np.mean((Y_closed - Y_labels) ** 2) / np.mean(np.atleast_2d(self.norm ** 2)))
                err.append(current_error)

                if fig:
                    for dim_i, ax in zip(range(self.N_dim), axs):

                        if dim_i in self.observed_idx:
                            _i = np.argmin(abs(observed_idx_np-dim_i))
                            ax.plot(current_time[:self.N_wash], current_input[:self.N_wash, _i],
                                    'x', c='C4', ms=5, label=f'Washout input')
                        ax.plot(current_time[:self.N_wash], u_open[:, dim_i].squeeze(), '-c',
                                label=f'ESN open-loop prediction')
                        ax.plot(current_time[self.N_wash:], Y_closed[:, dim_i], '--r', dashes=[2, .5],
                                label=f'ESN closed-loop prediction')
                        ax.set(ylabel=f'$u_{dim_i}$')
                    if i0 == 0:
                        axs[0].legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
                        axs[-1].set(xlabel='$t/T$')

                if max_L_tests == 1 and plot_pdf:
                    pass
                elif test_counter <= max_L_tests:
                    fig2, axs2 = plt.subplots(nrows=nrows, ncols=1, figsize=[8, 1.5 * nrows], sharex='all',
                                              layout='tight')
                    if Nq == 1:
                        axs2 = [axs2]

                    for dim_i, ax in zip(range(self.N_dim), axs2):
                        ax.plot(current_time[1:], current_target[:, dim_i].squeeze(), 'k', label=f'truth dim {dim_i}')
                        if dim_i in self.observed_idx:
                            _i = np.argmin(abs(observed_idx_np-dim_i))
                            ax.plot(current_time[:self.N_wash], current_input[:self.N_wash, _i].squeeze(),
                                    'x', c='C4', ms=5, label=f'Washout')
                        ax.plot(current_time[:self.N_wash], u_open[:, dim_i].squeeze(), '-c', label=f'ESN open loop')
                        ax.plot(current_time[self.N_wash:], Y_closed[:, dim_i], '--r', dashes=[2, .5],
                                label=f'ESN closed-loop prediction \n error = {current_error:.4}')
                        ax.set(ylabel=f'$u_{dim_i}$')
                        if dim_i == 0:
                            ax.legend(title=f'Test {test_counter}: Li = {Li}', loc='upper left',
                                      bbox_to_anchor=(1, 1), fontsize='x-small')
                        ax.set(ylim=[np.min(current_target[:, dim_i])-margin, np.max(current_target[:, dim_i])+margin])
                    figures.append(fig2)
                i0 += Nt_test

            if fig:
                # Plot histogram
                args = dict(bins=nbins, density=True, orientation='horizontal', stacked=False)
                predictions = np.concatenate(predictions)
                clean = np.concatenate(clean)
                noisy = np.concatenate(noisy)
                for dim_i, ax1 in enumerate(axs_pdf):
                    if dim_i in self.observed_idx:
                        _i = np.argmin(abs(observed_idx_np-dim_i))
                        ax1.hist(clean[:, _i].T, color='k', lw=2, alpha=0.5, histtype='step', **args)
                    ax1.hist(noisy[:, dim_i].T, color='k', lw=.85, histtype='step', **args)
                    ax1.hist(predictions[:, dim_i], color='r', ls='--', histtype='stepfilled', alpha=0.5, **args)
                    ax1.hist(predictions[:, dim_i], color='r', ls='--', histtype='step', **args)

                    ax1.set(ylim=[np.min(noisy[:, dim_i]) - margin,
                                  np.max(noisy[:, dim_i]) + margin])

            # Save to pdf
            if pdf_file is not None:
                [add_pdf_page(pdf_file, f, close_figs=test_i > 0) for f in [fig, *figures] if f is not None]
            else:
                plt.show()
            if i0 // Nt_test > 1:
                print(f'Li = {Li}: \t Test min, max and mean MSE in {i0 // Nt_test} tests ='
                      f' {min(err):.4}, {max(err):.4}, {np.mean(err):.4}.')

            errors.append(err)
        # Compute errors over all Lis
        errors = np.array(errors)
        print(f'Overall tests min, max and mean MSE in {test_counter} tests ='
              f' {np.min(errors):.4}, {np.max(errors):.4}, {np.mean(errors):.4}.')

    def _plot_training_results(self, U_test, Y_test, results, save_ESN_training, folder):
        """
        Plots training results, including Bayesian optimization convergence and test results.
        """

        if save_ESN_training:
            if folder is None:
                folder = self.figs_folder
            os.makedirs(folder, exist_ok=True)
            pdf = plt_pdf.PdfPages(f'{folder}{self.filename}_Training.pdf')
        else:
            pdf = None

        # Plot Bayesian optimization convergence
        fig = plt.figure()
        plot_convergence(results['res'])
        if pdf:
            add_pdf_page(pdf, fig)

        # Plot Gaussian Process reconstruction
        self._plot_BO(results, pdf=pdf)

        # Plot test results if applicable
        if self.perform_test and U_test.shape[1] >= (self.N_wash+self.N_val):
            print(U_test.shape)
            self.run_test(U_test, Y_test, pdf_file=pdf)

        if pdf:
            pdf.close()
        else:
            plt.show()

    def _plot_BO(self, results_bayesian_optimization, pdf=None):
        """
        # Plot Gaussian Process reconstruction for each network in the ensemble after n_tot evaluations.
        # The GP reconstruction is based on the n_tot function evaluations decided in the search
        Args:
            results_bayesian_optimization: dictionary containing
                - hp_names: label of the optimized hyperparameters
                - res: result of the GP reconstruction
            pdf: file to save the figures

        Returns:

        """

        hp_names = results_bayesian_optimization['hp_names']
        res = results_bayesian_optimization['res']

        f_iters = np.array(res.func_vals)

        if len(hp_names) >= 2:  # plot GP reconstruction
            gp = res.models[-1]
            res_x = np.array(res.x_iters)

            for hpi in range(len(hp_names) - 1):
                range_1 = getattr(self, f'{hp_names[hpi]}_range')
                range_2 = getattr(self, f'{hp_names[hpi + 1]}_range')

                n_len = 100  # points to evaluate the GP at
                xx, yy = np.meshgrid(np.linspace(*range_1, n_len), np.linspace(*range_2, n_len))

                x_x = np.column_stack((xx.flatten(), yy.flatten()))
                x_gp = res.space.transform(x_x.tolist())  # gp prediction needs norm. format

                # Plot GP Mean
                fig = plt.figure(figsize=(10, 5), tight_layout=True)
                plt.xlabel(hp_names[hpi])
                plt.ylabel(hp_names[hpi + 1])

                # retrieve the gp reconstruction
                amin = np.amin([10, np.max(f_iters)])

                # Final GP reconstruction for each realization at the evaluation points
                y_pred = np.clip(-gp.predict(x_gp), a_min=-amin, a_max=-np.min(f_iters)).reshape(n_len, n_len)

                plt.contourf(xx, yy, y_pred, levels=20, cmap='Blues')
                cbar = plt.colorbar()
                cbar.set_label(label='-$\\log_{10}$(MSE)', labelpad=15)
                plt.contour(xx, yy, y_pred, levels=20, colors='black', linewidths=1, linestyles='solid',
                            alpha=0.3)
                #   Plot the n_tot search points
                for rx, mk in zip([res_x[:self.N_grid ** 2], res_x[self.N_grid ** 2:]], ['v', 's']):
                    plt.plot(rx[:, 0], rx[:, 1], mk, c='w', alpha=.8, mec='k', ms=8)
                # Plot best point
                best_idx = np.argmin(f_iters)
                plt.plot(res_x[best_idx, 0], res_x[best_idx, 1], '*r', alpha=.8, mec='r', ms=8)

                if pdf is not None:
                    add_pdf_page(pdf, fig, close_figs=True)

    def copy(self):
        return deepcopy(self)

    def plot_Wout(self):
        # Visualize the output matrix
        fig, ax = plt.subplots()
        im = ax.matshow(self.Wout.T, cmap="PRGn", aspect=4., vmin=-np.max(self.Wout), vmax=np.max(self.Wout))
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.colorbar(im, orientation='horizontal', extend='both')
        ax.set(ylabel='$N_u$', xlabel='$N_r$', title='$\\mathbf{W}_\\mathrm{out}$')
