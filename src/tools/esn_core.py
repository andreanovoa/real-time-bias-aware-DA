import matplotlib.backends.backend_pdf as plt_pdf
import numpy as np

import os
from utils import add_pdf_page

from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Any, Optional, Union
from functools import cached_property

# Validation methods
from functools import partial
from itertools import product
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.space import Real
from skopt.plots import plot_convergence

from scipy.sparse import csr_matrix, lil_matrix, issparse
from scipy.sparse.linalg import eigs as sparse_eigs

# XDG_RUNTIME_DIR = 'tmp/'


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

    bias_in = np.array([0.1])  #
    bias_out = np.array([1.0])  # For symmetry breaking
    connect = 3  # Connectivity between neurons
    figs_folder = './figs_ESN/'
    filename = 'my_ESN'  # Default ESN file name

    input_parameters: Optional[np.ndarray] = None #TODO: add input parameters functionality to enable parametric ESNs
 
    N_folds = 4  # Folds over the training set
    N_func_evals = 20  # Total evals of Bayesian hyperparameter optimization (BHO)
    N_grid = 4  # BHO grid N_grid x N_grid \geq N_func_evals
    N_initial_rand = 0  # Initial random evaluations at BYO
    N_split = 4  # Splits of training data for faster computation
    N_units = 100  # Number of neurones
    N_wash = 50  # Number of washout steps

    max_L_tests = 10
    perform_test = True  # Run tests during training?
    
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
            y = y[:, np.newaxis]
        elif y.ndim > 2:
            raise AssertionError(f'y.shape={y.shape}. The input y must have 2 dimension')


        #   Initialise state dimensions and reservoir state to zeros ------------ #
        self.N_dim = y.shape[0] # Dimension of the physical system i.e., the output dimension 
        self.observed_idx = kwargs.pop('observed_idx', np.arange(self.N_dim)) # Default to full observability

        # Set provided input parameters ------------------------- #
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(EchoStateNetwork, key)]

        #  Define time steps and time windows -------------------- #
        self.dt_ESN = dt * self.upsample

        #  Initialize ESN matrices -------------------------- #
        self.val_k = kwargs.get('val_k', 0)  # Validation counter
        self.initialised = False  # Flag for washout

    @property
    def trained(self):
        """Flag to check if the model has been trained"""
        return hasattr(self, '_Win') and hasattr(self, '_Wout') and hasattr(self, '_W')

    @property
    def W(self) -> csr_matrix:
        """
        Returns the reservoir state matrix (W) in CSR format.
        """
        return self._W

    
    @property
    def rng(self):
        if not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(self.seed)
        return self._rng
    
    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = 0
        return self._seed
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value
        if hasattr(self, '_rng'):
            del self._rng

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
    def Win(self) -> Union[np.ndarray, csr_matrix]:
        """
        Returns the input matrix (Win).
        """
        return self._Win
    
    @Win.setter
    def Win(self, value):
        """
        Setter for the input matrix (Win). Converts the input to CSR format if sparse.
        """

        assert self.Win_type in ['sparse', 'dense'], \
                f"Win type {self.Win_type} not implemented ['sparse', 'dense']"

        if self.Win_type == 'sparse' and not isinstance(value, csr_matrix):
            value = csr_matrix(value)
        elif self.Win_type == 'dense' and hasattr(value, 'toarray'):
            value = value.toarray() 
        

        # Ensure the matrix has the correct dimensions
        assert value.shape ==  (self.N_units, self.N_dim_in+1), \
            f'Win must be a square matrix of shape ({self.N_units}, {self.N_dim_in + 1}), but got {value.shape}'
        
        # Set the input matrix
        self._Win = value
        self._invalidate_jacobian_cache()


    def _invalidate_jacobian_cache(self):
        self.__dict__.pop('dr_di', None)


    @property
    def Wout(self) -> np.ndarray:
        """
        Returns the output matrix (Wout).
        """
        return self._Wout

    @Wout.setter
    def Wout(self, value: np.ndarray):
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
        # if not hasattr(self, '_WCout'):
        #     return None
        return self._WCout
    
    @WCout.setter
    def WCout(self, value=None):
        """
        Setter for the closed-loop reservoir weight matrix (W Cout).
        """
        assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'
        if value is None:
            self._WCout = np.linalg.lstsq(self.Wout[:-1], self.W.toarray(), rcond=None)[0]
        else:
            self._WCout = value


    @property
    def sparsity(self):
        """
        Computes the sparsity level of the reservoir connectivity matrix (W). This is, the
        fraction of connections between neurons in the reservoir that are set to zero.
            sparsity = 1 - #Active connections / Total possible connections
        """
        return 1. - self.connect / (self.N_units - 1)


    @property
    def N_dim_in(self):
        """
        Computes the number of input dimensions.
        """
        if self.input_parameters is None:
            return len(self.observed_idx)
        else:
            return len(self.observed_idx) + self.input_parameters.shape[0]
        

    @property
    def norm(self):
        """
        Returns the normalization factor for the input data.
        """
        if not hasattr(self, '_norm'):
            return np.ones((self.N_dim_in,))
        return self._norm
    
    @norm.setter
    def norm(self, value):
        """
        Setter for the normalization factor. Ensures it is N_dim_in.
        """
        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Normalization factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._norm = value.flatten()


    @cached_property
    def dr_di(self) -> Union[csr_matrix, np.ndarray]:
        """Open-loop reservoir Jacobian term d(r)/d(i), shape (N_units, N_dim_in)."""
        norm = self.norm.copy()

        Win_1 = self.Win[:, :self.N_dim_in]  # type: Union[csr_matrix, np.ndarray]
        g = self.sigma_in * 1.0 / norm
        
        if issparse(Win_1):
            # .multiply returns a COO matrix: convert back to CSR for efficient products
            return csr_matrix(Win_1.multiply(g[np.newaxis, :]))
        else:
            return Win_1 * g[np.newaxis, :]


    @property
    def shift(self):
        """
        Returns the shift factor for the input data.
        """
        if not hasattr(self, '_shift'):
            return np.zeros((self.N_dim_in,))
        return self._shift
    

    @shift.setter
    def shift(self, value):
        """
        Setter for the shift factor. Ensures it is N_dim_in (nb. after initialization).
        """

        if hasattr(self, 'N_dim_in'):
            assert value.size == self.N_dim_in, \
                f'Shift factor must be dimension Ndim={self.N_dim_in}, got {value.shape}'
        self._shift = value.flatten()   

    # _______________________________________________________________________________________________________ STEP & JACOBIAN
    def step(self, u, r):
        """
        Advances the reservoir by one time step and updates its internal state.

        Args:
            u (np.ndarray): Input physical state at the current time step. Shape = (N_dim x N_ens)
            r (np.ndarray): Reservoir state at the current time step. Shape = (N_units x N_ens)

        Returns:
            tuple: (u_out, r_out) where u_out is the output state and r_out is the updated reservoir state.
        """
        # Normalise input data and augment with input bias (ESN symmetry parameter)

        # assert self.trained, 'ESN must be trained with washout before calling step method. Call ESN.train() first.'

        if u.ndim == 1:
            u = np.expand_dims(u, axis=-1)
        elif u.ndim == 3:
            assert u.shape[0] == 1, f'Input u has shape {u.shape}, only 1 sample at a time is allowed'
            u = u[0]
        if r.ndim == 1:
            r = np.expand_dims(r, axis=-1)
        elif r.ndim == 3:
            assert r.shape[0] == 1, f'Input r has shape {r.shape}, only 1 sample at a time is allowed'
            r = r[0]

        # Normalize input
        u_norm = self.normalize_input(u)

        # Augment input with bias
        bias_in = self.bias_in * np.ones((1, u.shape[-1]))
        u_aug = np.concatenate((u_norm, bias_in))

        # Forecast the reservoir state
        r_out = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * self.W.dot(r))

        # compute output from ESN if not during training
        u_out = self.reservoir_to_physical(r_out)
        return u_out, r_out
    


    def reservoir_to_physical(self, r):
        """ Converts the reservoir state to the physical state using the output weight matrix (Wout).
        Note: I change this in ESN_model
        Args:
            r_aug (np.ndarray): Augmented reservoir state including output bias.
        """
        
        # output bias added
        bias_out = self.bias_out * np.ones((1, r.shape[-1]))
        r_aug = np.concatenate((r, bias_out))

        return np.dot(self.Wout.T, r_aug)

    def normalize_input(self, data):
        """
        Normalizes the input data based on the specified normalization method.

        Args:
            data (np.ndarray): Input data to be normalized.

        Returns:
            np.ndarray: Normalized input data.
        """
        return (data - self.shift[:, np.newaxis]) / self.norm[:, np.newaxis]


    def outputs_to_inputs(self, full_state):
        """
        Maps the full state (predicted or reconstructed) to input states for the ESN.

        Args:
            full_state (np.ndarray): Full physical state vector.

        Returns:
            np.ndarray: Input state vector mapped from the full state.
        """
        assert full_state.shape[0] == self.N_dim, f'full_state has shape {full_state.shape}, expected first dim to be {self.N_dim}'

        observed_state = full_state[self.observed_idx]

        assert observed_state.shape[0] == self.N_dim_in, f'observed_state has shape {observed_state.shape}, expected first dim to be {self.N_dim_in}'

        if self.input_parameters is None:
            return observed_state
        else:
            return np.concatenate([observed_state, self.input_parameters], axis=0)
    

    def Jacobian(self, u_in, r_in, open_loop_J=True):
        """
        Computes the Jacobian matrix for the reservoir, either in open-loop or closed-loop mode.

        Args:
            open_loop_J (bool): If True (default), compute the open-loop Jacobian.
            u_in (np.ndarray): Input state. shape = (N_dim_in x N_ens)
            r_in (np.ndarray): Reservoir state. shape = (N_units x N_ens)

        Returns:
            np.ndarray: Jacobian matrix d(u_out)/d(u_in).
                - If N_ens == 1: shape (N_dim, N_dim_in)
                - If N_ens > 1: shape (N_dim, N_dim_in, N_ens)
        """
        assert self.trained, 'ESN must be trained before computing the Jacobian. Call ESN.train() first.'


        Wout_1 = self.Wout[:self.N_units, :].T

        # # Option(i) rin function of bin:
        rout = self.step(u_in, r_in)[1]

        tt = 1. - rout ** 2
        dr_di = self.dr_di

        if not open_loop_J:
            # u_aug = np.concatenate((u_in / self.norm, self.bias_in))
            # rout = np.tanh(self.sigma_in * self.Win.dot(u_aug) + self.rho * np.dot(self.WCout.T, u_in))
            # dr_di = self.sigma_in * Win_1 / self.norm + self.rho * self.WCout.T
            #  Win_G += dr_di ......
            raise NotImplementedError('Numerical test of closed-loop Jacobian did not pass')

        N_ens = tt.shape[-1]
        if N_ens == 1:
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, 0][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, 0][np.newaxis, :]
            return RHS.dot(Wout_1.T).T

        J = np.zeros((self.N_dim, self.N_dim_in, N_ens))
        for ens_i in range(N_ens):
            if issparse(dr_di):
                RHS = dr_di.T.multiply(tt[:, ens_i][np.newaxis, :])
            else:
                RHS = dr_di.T * tt[:, ens_i][np.newaxis, :]
            J[:, :, ens_i] = RHS.dot(Wout_1.T).T

        return J


    

    # _______________________________________________________________________________________ TRAIN & VALIDATE THE ESN
    def train(self, 
              train_data,
              add_noise=True,
              plot_training=True,
              save_ESN_training=False,
              folder=None,
              validation_strategy=None,
              seed=None,
              **kwargs
              ):
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
            pass #  skip training

        for key, val in kwargs.items():
            if hasattr(self, key):
                print(f'Modifying {key} = {getattr(self, key)} -> {val} at training.')
                setattr(self, key, val)

        # ========================== STEP 1: DATA FORMATTING ==========================
        # Format data into washout, train/validation, and test sets
        U_wtv, Y_wtv, U_test, Y_test = self._split_and_format_data(train_data, add_noise=add_noise)

        # print([xx.shape for xx in [U_wtv, Y_wtv, U_test, Y_test]])

        # Ensure W and Win matrices are initialized
        if not hasattr(self, '_W') or not hasattr(self, '_Win'):
            self._generate_W_Win(seed=seed)

        self.Wout = np.zeros((self.N_units + 1, self.N_dim))  # Initialize Wout with zeros

        # =================== STEP 2: BAYESIAN HYPERPARAMETER OPTIMIZATION ==============
        self.val_k = 0  # Reset validation counter at the start of training
        # Perform hyperparameter optimization if required
        if self.hyperparameters_to_optimize:
            bo_results = self._optimize_hyperparameters(U_wtv, Y_wtv, 
                                                       validation_strategy,
                                                       print_convergence=plot_training)
        else:
            bo_results = None
        # ====================== STEP 3: RIDGE REGRESSION TRAINING =====================
        # Compute the output weight matrix Wout
        self.Wout = self._solve_ridge_regression(U_wtv, Y_wtv)


        # ========================== STEP 4: TEST AND PLOTTING ======================
        if plot_training:
            self._plot_training_results(U_test, Y_test, bo_results, save_ESN_training, folder)


    def copy(self):
        return deepcopy(self)
    
    # _______________________________________________________________________________________ HELPER METHODS FOR ESN INITIALIZATION & TRAINING
        


    def _generate_W_Win(self, seed=None):
        """
        Generates the input weight matrix (Win) and reservoir weight matrix (W) with sparsity constraints.

        Args:
            seed (int): Random seed for reproducibility.

        Raises:
            ValueError: If the specified self.Win_type is unsupported. Allowed values: 'sparse' or 'dense'.

        Outputs:
            None. Updates internal matrices Win and W with appropriate values.
        """
        if seed is None:
            rng0 = self.rng
        else:
            rng0 = np.random.default_rng(seed)

        # Input matrix: Sparse random matrix where only one element per row is different from zero
        if not hasattr(self, '_Win'):
            Win = lil_matrix((self.N_units,
                              self.N_dim_in + 1))  # +1 accounts for input bias
            if self.Win_type == 'sparse':
                for j in range(self.N_units):
                    Win[j, rng0.choice(self.N_dim_in + 1)] = rng0.uniform(low=-1, high=1)
            elif self.Win_type == 'dense':
                for j in range(self.N_units):
                    Win[j, :] = rng0.uniform(low=-1, high=1, size=self.N_dim_in + 1)
            else:
                raise ValueError("Win type {} not implemented ['sparse', 'dense']".format(self.Win_type))
            # Store
            self.Win = Win

        # Reservoir state matrix: Erdos-Renyi network
        if not hasattr(self, '_W'):
            W = csr_matrix(rng0.uniform(low=-1, high=1, size=(self.N_units, self.N_units)) *
                        (rng0.random(size=(self.N_units, self.N_units)) < (1 - self.sparsity)))
            # scale W by the spectral radius to have unitary spectral radius
            spectral_radius = np.abs(sparse_eigs(W, k=1, which='LM', return_eigenvectors=False))[0]
            self.W = (1. / spectral_radius) * W


    def _compute_RR_terms(self, U_wtv, Y_wtv):
        """
        Computes the Ridge Regression (RR) terms, including left-hand side (LHS) and right-hand side (RHS)
        matrices, for training the output weights.

        Args:
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_wtv (np.ndarray): Corresponding output labels for input data.

        Returns:
            tuple:
                - LHS (np.ndarray): Left-hand side matrix for ridge regression.
                - RHS (np.ndarray): Right-hand side matrix for ridge regression.
                - U_RR (list): List of input states split by L-segments.
                - R_RR (list): List of reservoir states split by L-segments.
        """
 
        LHS = np.zeros((self.N_units + 1, self.N_units + 1))
        RHS = np.zeros((self.N_units + 1, self.N_dim))
        R_RR = [np.empty([0, self.N_units])] * U_wtv.shape[0]
        U_RR = [np.empty([0, self.N_dim])] * U_wtv.shape[0]


        for ll in range(U_wtv.shape[0]):

            U_wash_l = U_wtv[ll][:self.N_wash]
            # Y_wash_l = Y_wtv[ll][:self.N_wash]
            Uin_l = U_wtv[ll][self.N_wash:]
            Yout_l = Y_wtv[ll][self.N_wash:]

            assert Uin_l.shape[0] == Yout_l.shape[0], \
                f'Inconsistent shapes for training data at segment {ll}: {Uin_l.shape} vs {Yout_l.shape}'

            assert Uin_l.shape[0] > 0, \
                f'Not enough data for training at segment {ll}: {Uin_l.shape}'
            
            # Washout phase to initialize reservoir state
            N_ens = U_wash_l.shape[-1] if U_wash_l.ndim == 3 else 1
            r = np.zeros((self.N_units, N_ens))
            for u_in in U_wash_l:
                _, r = self.step(u_in, r)

            # Split training data for faster computations
            U_train = np.array_split(Uin_l, self.N_split, axis=0)
            Y_target = np.array_split(Yout_l, self.N_split, axis=0)

            for U_t, Y_t in zip(U_train, Y_target):
                if Y_t.ndim == 3:
                    assert Y_t.shape[-1] == 1, f'Y_t has shape {Y_t.shape}, only 1 sample at a time is allowed'
                    Y_t = Y_t[..., 0]

                # Open-loop train phase
                r_out = r.copy()
                r_open = np.zeros((U_t.shape[0], self.N_units, N_ens))
                y_open = np.zeros((U_t.shape[0], self.N_dim, N_ens))
                for ii, u_in in enumerate(U_t):
                    u_out, r_out = self.step(u_in, r_out)
                    y_open[ii], r_open[ii] = u_out, r_out

                if y_open.ndim > 2:
                    y_open, r_open = y_open.squeeze(axis=-1), r_open.squeeze(axis=-1)

                R_RR[ll] = np.append(R_RR[ll], r_open, axis=0)
                U_RR[ll] = np.append(U_RR[ll], y_open, axis=0)

                # Compute matrices for linear regression system
                bias_out = np.ones([r_open.shape[0], 1]) * self.bias_out
                r_aug = np.hstack((r_open, bias_out)) 
                
                LHS += np.dot(r_aug.T, r_aug) 
                RHS += np.dot(r_aug.T, Y_t)

        return LHS, RHS, U_RR, R_RR


    def _solve_ridge_regression(self, U_wtv, Y_wtv):
        """
        Solves the ridge regression problem to compute the output weight matrix (Wout).

        Args:
            U_wtv (np.ndarray): Input data for ridge regression (train/valiladion).
            Y_wtv (np.ndarray): Target labels for ridge regression.

        Returns:
            np.ndarray: Computed output weight matrix (Wout).
        """
        LHS, RHS = self._compute_RR_terms(U_wtv, Y_wtv)[:2]
        LHS.ravel()[::LHS.shape[1] + 1] += self.tikh  # Add tikhonov to the diagonal
        return np.linalg.solve(LHS, RHS)  # Solve linear regression problem
    

    def _UY_from_raw_data(self, data, add_noise=True, seed=None):
        """
        Extracts input (U) and output (Y) matrices from raw data.

        Args:
            data (np.ndarray): Raw time series data with dimensions [(L) x Nt x N_dim].

        Returns:
            tuple: (U, Y) where U is the input matrix and Y is the output. Shapes: L x Nt x N_dim
        """

        #   APPLY UPSAMPLE AND OBSERVED INDICES ________________________
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        # Set labels always as the full state
        Y = data[:, ::self.upsample].copy()
        # Inputs are the observed components of the state, which can be a subset of the full state
        U = Y[:, :, self.observed_idx].copy()

        assert Y.shape[-1] >= U.shape[-1]
        assert U.shape[-1] == self.N_dim_in

        if add_noise:
            #  ==================== ADD NOISE TO TRAINING INPUT ====================== ##
            # Add noise to the inputs if distinction inputs/labels is not given.
            # Larger noise promotes stability in long term, but hinders time accuracy
            U_std = np.std(U, axis=1, keepdims=True)
            if seed is None:
                rng0 = self.rng
            else:
                rng0 = np.random.default_rng(seed)
            U += rng0.normal(loc=0, scale=self.noise * U_std, size=U.shape)

        return U, Y

    def _split_and_format_data(self, data=None, add_noise=True):
        """
        Formats the input data into washout, train/val, and test sets. Optionally adds noise to the input.

        Args:
            - data (np.ndarray): Input time series data with dimensions [(L) x Nt x N_dim].
            - add_noise (bool): Whether to add noise to the training input data (default: True).
            - observed_idx (list, optional): indices which are observed
        Returns:
            - U_wtv (np.ndarray): Wash-train-validation input data.
            - Y_wtv (np.ndarray): Corresponding labels for train/validation data.
            - U_test (np.ndarray): Test input data.
            - Y_test (np.ndarray): Test labels.
        Raises:
            ValueError: If the input data length is insufficient for training.
        """
        if data is None:
            raise ValueError('No training data provided to format_training_data method.')
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        U, Y = self._UY_from_raw_data(data, add_noise=add_noise) # dimensions: L x Nt x N_dim_in/N_dim

        #   SEPARATE INTO WASH/TRAIN/VAL/TEST SETS ______________________
        N_wtv = self.N_train + self.N_val

        if U.shape[1] < N_wtv:
            raise ValueError(f'Increase the length of the training data signal. {U.shape} < {N_wtv}')

        U_wtv = U[:, :N_wtv - 1].copy()
        Y_wtv = Y[:, 1:N_wtv].copy()

        U_test = U[:, N_wtv:-1].copy()
        Y_test = Y[:, N_wtv+1:].copy()

        assert U_wtv.shape[1] == Y_wtv.shape[1], \
            f'Inconsistent shapes for train/validation data: {U_wtv.shape} vs {Y_wtv.shape}'
        assert U_test.shape[1] == Y_test.shape[1], \
            f'Inconsistent shapes for test data: {U_test.shape} vs {Y_test.shape}'

        if Y_wtv.ndim not in [2, 3]:
            raise ValueError(f'Inconsistent ensemble size for train/validation data: {Y_wtv.shape}')

        # compute norm (normalize inputs by component range)
        self.norm, self.shift = EchoStateNetwork._set_norm(U_wtv, method=self.norm_method)

        return U_wtv, Y_wtv, U_test, Y_test
    

    # ___________________________________________________________________________________________ BAYESIAN OPTIMIZATION
    def _reset_hyperparams(self, params, names, tikhonov=None):
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


    def _optimize_hyperparameters(self, U_wtv, Y_wtv, validation_strategy=None, print_convergence=True):
        """
        Performs Bayesian hyperparameter optimization to minimize the validation loss.

        Args:
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_wtv (np.ndarray): Corresponding labels for train-validation data.
            validation_strategy (function, optional): Validation function for hyperparameter tuning.
                Defaults to `_RVC_Noise`.

        Returns:
            OptimizeResult: Results of the Bayesian optimization process.
        """
        # print("Starting Bayesian hyperparameter optimization...")

        # Prepare search grid, space, and hyperparameter names
        search_grid, search_space, hp_names = self._hyperparameter_search(print_convergence=print_convergence)
        tikh_opt = np.zeros(self.N_func_evals)  # Track optimal Tikhonov regularization

        # Use default or provided validation strategy
        if validation_strategy is None:
            validation_strategy = self._RVC_Noise

        # Prepare the validation function
        val_func = partial(validation_strategy,
                           case=self,
                           U_wtv=U_wtv.copy(),
                           Y_wtv=Y_wtv.copy(),
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
        assert result is not None, 'gp_minimize retuned a None instance'
        # Process results
        f_iters = np.array(result.func_vals)
        best_idx = np.argmin(f_iters)

        # Update hyperparameters with the best result
        self._reset_hyperparams(result.x, hp_names, tikhonov=tikh_opt[best_idx])

        print(f"seed {self.seed} \t Optimal hyperparameters: {result.x}, {self.tikh}, MSE: {result.fun}")  # type: ignore

        return dict(res=result,
                    hp_names=hp_names)

    def _hyperparameter_search(self, print_convergence=True):
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

        param_grid, search_space = [], [] 
        for hyper_param in parameters:
            range_ = getattr(self, hyper_param + '_range')  # type: tuple[float,float]
            param_grid.append(np.linspace(*range_, self.N_grid)) 
            search_space.append(Real(*range_, name=hyper_param))

        # The first n_grid^2 points are from grid search
        search_grid = product(*param_grid, repeat=1) 
        search_grid = [list(sg) for sg in search_grid]

        # Print optimization header
        if print_convergence:
            print('\n ----------------- HYPERPARAMETER SEARCH ------------------\n {0}x{0} grid'.format(self.N_grid) +
                  ' and {} points with Bayesian Optimization\n\t'.format(self.N_func_evals - self.N_grid ** 2), end="")
            for kk in self.hyperparameters_to_optimize:
                print('\t {}'.format(kk), end="")
            print('\t MSE val ')

        return search_grid, search_space, parameters

    # ___________________________________________________________________________________________ NORMALIZATION METHODS

    @staticmethod
    def _set_norm(train_data, method=None):
        """
        Computes the normalization factor for the input data.
        Args:
            train_data (np.ndarray): Wash-train-validation training input data. (Nens x Nt x Ndim).
        Returns:
            float: Normalization factor based on the range of the input data. 
        """
        # assert train_data.ndim in [3, 4], f'U_wtv must be a 3D array, got {train_data.ndim}D: ({train_data.shape})'
        
        if train_data.ndim == 3:
            L, _, Ndim = train_data.shape
            Nens = 1
        elif train_data.ndim == 4:
            L, _, Ndim, Nens = train_data.shape
        elif train_data.ndim == 2:
            L = 1
            Nens = 1
            Ndim = train_data.shape[1]
        else:
            raise ValueError(f'U_wtv must be a 2D, 3D or 4D array, got {train_data.ndim}D: ({train_data.shape})')

        if method is None:
            return np.ones(Ndim), np.zeros(Ndim)

        shift = np.mean(train_data, axis=1) 

        shifted_data  = train_data - shift[:, np.newaxis, :]

        if method == 'std':
            shift = np.mean(train_data, axis=1) 
            norm = np.std(shifted_data, axis=1)
        elif method == 'max':
            norm = np.max(shifted_data, axis=1)
        elif method == 'mean':
            norm = np.mean(abs(shifted_data), axis=1)
        elif method == 'range':
            m = np.min(shifted_data, axis=1)
            M = np.max(shifted_data, axis=1)
            norm = M - m
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if L > 1:
            norm = np.mean(norm, axis=0)
            shift = np.mean(shift, axis=0)
        if Nens > 1:
            norm = np.mean(norm, axis=-1)
            shift = np.mean(shift, axis=-1)

        if np.any(abs(norm) < 1e-12):
            norm[abs(norm) < 1e-12] = 1.0  # Prevent division by zero
            
        return norm, shift
    
    # ___________________________________________________________________________________________ VALIDATION STRATEGIES


    @staticmethod
    def _RVC_Noise(x, case, U_wtv, Y_wtv, tikh_opt, hp_names, print_convergence=True):
        """
        Implements Chaotic Recycle Validation for hyperparameter optimization.

        Args:
            x (list): Hyperparameter values to evaluate.
            case (EchoStateNetwork): Instance of the ESN being validated.
            U_wtv (np.ndarray): Wash-train-validation input data.
            Y_wtv (np.ndarray): Corresponding labels for train/validation data.
            tikh_opt (np.ndarray): Array to store optimal Tikhonov regularization values.
            hp_names (list): Names of the hyperparameters being optimized.

        Returns:
            float: Normalized mean squared error (MSE) for the validation set.
        """
        # Re-set hyperparams as the optimization goes on
        if hp_names:
            case._reset_hyperparams(x, hp_names)

        N_tikh = len(case.tikh_range)
        nRMSE = np.zeros(N_tikh)

        # num steps forward the validation interval is shifted
        N_fw = (case.N_train - case.N_val - case.N_wash) // (case.N_folds - 1)

        # Train using tv: Wout_tik is passed with all the combinations of tikh_ and target noise
        # This must result in L-Xa timeseries
        LHS, RHS, _, _ = case._compute_RR_terms(U_wtv, Y_wtv)
        Wout_tik = np.empty((N_tikh, case.N_units + 1, case.N_dim))

        # print(f'Computing Wout for tikhonov values: {case.tikh_range}')
        # print(f'LHS shape: {LHS.shape}, RHS shape: {RHS.shape}')
        
        for tik_j in range(N_tikh):
            LHS_reg = LHS.copy()
            LHS_reg.ravel()[::LHS.shape[1] + 1] += case.tikh_range[tik_j]
            Wout_tik[tik_j] = np.linalg.solve(LHS_reg, RHS)

        # print(U_wtv.shape, Y_wtv.shape, 'U_wtv, Y_wtv shapes in RVC noise')
        # Perform Validation in different folds
        n_looop = -1 # to count the number of tests performed
        for U_l, Y_l in zip(U_wtv, Y_wtv):  # Each set of training data        
            norm_l = np.max(Y_l, axis=0) - np.min(Y_l, axis=0)

            for fold in range(case.N_folds):
                n_looop += 1
                p = case.N_wash + fold * N_fw

                # Select washout and validation data
                U_wash = U_l[p:p + case.N_wash]
                Y_val = Y_l[p + case.N_wash:p + case.N_wash + case.N_val]

                # Perform washout (open-loop without extra forecast step)
                r_out = np.zeros((case.N_units, 1))
                u_out = np.zeros((case.N_dim, 1))

                for u_in in U_wash:
                    u_out, r_out = case.step(u_in, r_out)

                for tik_j in range(N_tikh):  # cloop for each tikh_-noise combination

                    case.Wout = Wout_tik[tik_j]

                    # Y_close = case.closedLoop(case.N_val)[0][1:].squeeze()
                    Y_closed = np.zeros_like(Y_val)

                    for i in range(Y_closed.shape[0]):
                        u_input = case.outputs_to_inputs(full_state=u_out)
                        u_out, r_out = case.step(u_input, r_out)
                        Y_closed[i] = u_out[:, 0].copy() 

                    # Compute normalized MSE
                    nRMSE[tik_j] += np.log10(case.compute_nRMSE(Y_val, Y_closed, norm=norm_l))
    
                    # prevent from diverging to infinity: MSE=1E10 (useful for hybrid and similar architectures)
                    if np.isnan(nRMSE[tik_j]) or np.isinf(nRMSE[tik_j]):
                        nRMSE[tik_j] = 10 * case.N_folds
                        
        # select and save the optimal tikhonov and noise level in the targets
        a = nRMSE.argmin()
        tikh_opt[case.val_k] = case.tikh_range[a]
        case.tikh = case.tikh_range[a]
        normalized_best_RMSE = nRMSE[a] / n_looop

        case.val_k += 1
        if print_convergence:
            print(case.val_k, end="")
            for hp in case.hyperparameters_to_optimize:
                print('\t {:.3e}'.format(getattr(case, hp)), end="")
            print('\t {:.4f}'.format(normalized_best_RMSE))

        return normalized_best_RMSE

    
    def compute_nRMSE(self, Y_true, Y_pred, norm=1.0):
        """
        Computes the normalized Root Mean Square Error (nRMSE) between true and predicted values.

        Args:
            Y_true (np.ndarray): Ground truth values.
            Y_pred (np.ndarray): Predicted values.
        Returns:
            float: nMSE value.
        """
        return np.mean(np.sqrt((Y_true - Y_pred) ** 2)) / np.mean(np.sqrt(norm**2))

    # _______________________________________________________________________________________ TEST & PLOTTING FUNCTIONS


    def run_test(self, 
                 U_test, 
                 Y_test, 
                 pdf_file=None, 
                 Nt_test=None,
                 max_L_tests=5, 
                 nbins=20, 
                max_short_tests=10,
                long_term=True,
                short_term=True,
                 ):
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

        if max_L_tests is None and hasattr(self, 'max_L_tests'):
            max_L_tests = self.max_L_tests
        if Nt_test is None:
            Nt_test = self.N_val

        if U_test.ndim == 1:
            U_test = U_test[np.newaxis, :, np.newaxis]
        elif U_test.ndim == 2:
            U_test = U_test[np.newaxis, :, :]

        if Y_test.ndim == 1:
            Y_test = Y_test[np.newaxis, :, np.newaxis]
        elif Y_test.ndim == 2:
            Y_test = Y_test[np.newaxis, :, :]


        rng0 = self.rng

        L, max_test_time, Nq = U_test.shape[:3]
        # max_test_time -= self.N_wash

        if Nq > 10:
            nrows, dims = 10, rng0.choice(Nq, 10, replace=False)
        else:
            nrows, dims = self.N_dim, np.arange(Nq)
            if Nq == 1:
                dims = [dims]

        observed_idx_np = np.array(self.observed_idx)

        # Select test cases (with a maximum of max_L_tests)
        if L > 1:
            if max_L_tests != L:
                L_indices = np.sort(rng0.choice(L, max_L_tests, replace=max_L_tests > L))
            else:
                L_indices = np.arange(L)
        else:
            L_indices = [0]
        
        N_ens = U_test.shape[-1] if U_test.ndim == 4 else 1
        # Prediction function
        def predict_Y(_input, _target):

            # Perform washout (open-loop without extra forecast step)
            r_out = np.zeros((self.N_units, N_ens))
            u_out = np.zeros((self.N_dim, N_ens))
            u_open = np.zeros_like(_target[:self.N_wash]) 


            for ii, u_in in enumerate(_input[:self.N_wash]):
                u_out, r_out = self.step(u_in, r_out)
                try:
                    u_open[ii] = u_out.squeeze()
                except:
                    u_open[ii] = u_out.copy()


            Y_closed = np.zeros_like(_target)
            
            for i in range(Y_closed.shape[0]):
                u_input = self.outputs_to_inputs(full_state=u_out)
                u_out, r_out = self.step(u_input, r_out)
                try:
                    Y_closed[i] = u_out.squeeze()
                except:
                    Y_closed[i] = u_out.copy()

            return Y_closed, u_open

        # Plotting function
        def plot_time(_axs, _time, _pred_closed, _pred_open, _inputs, _target, _err=None):
            if not isinstance(_axs, (list, np.ndarray)):
                _axs = [_axs]

            t_wash_in = _time[:self.N_wash] - self.dt_ESN
            t_wash_out = _time[:self.N_wash]
            t_out = _time[self.N_wash:]

            for dim_i, _ax in zip(range(self.N_dim), _axs):
                _ax.plot(t_out, _target[:, dim_i], 'k', label=f'truth dim {dim_i}')
                # Plot the input if observed
                if dim_i in self.observed_idx:
                    _i = np.argmin(abs(observed_idx_np-dim_i))
                    _ax.plot(t_wash_in, _inputs[:self.N_wash, _i], 'x', c='C4', ms=5, label=f'Washout')
                    
                _ax.plot(t_wash_out, _pred_open[:, dim_i], '-co', mfc='none', label=f'ESN open loop')
                _ax.plot(t_out, _pred_closed[:, dim_i], '--r', dashes=[2, .5],
                         label=[f'ESN closed-loop prediction \n error = {_err:.4}' if _err is not None else f'ESN closed-loop prediction'])
                _ax.set(ylabel=f'$u_{dim_i}$')
                _ax.set(ylim=ylims[dim_i])

        test_counter, errors_all = 0, []
        hist_args = dict(bins=nbins, density=True, orientation='horizontal', stacked=False)


        print(f'Running test for L=', end=' ')
        for Li in L_indices:
            print(f'{Li}', end=' ')

            # Select dataset
            U_test_l, Y_test_l = U_test[Li], Y_test[Li]


            norm_l = np.max(Y_test_l, axis=0) - np.min(Y_test_l, axis=0)

            t_l = (np.arange(U_test_l.shape[0])) * self.dt_ESN
            # set ylims for plotting
            ylims = [[np.min(Y_test_l[:, dim_i])*1.05, np.max(Y_test_l[:, dim_i])*1.05] for dim_i in range(self.N_dim)]
            
            # plot tests statistics if the test dataset is long or requested
            if long_term:

                # predict over the entire test set
                Y_closed, U_open = predict_Y(U_test_l[:-1], Y_test_l[self.N_wash:])

                err_long = np.log10(self.compute_nRMSE(Y_closed, Y_test_l[self.N_wash:], norm=norm_l))
                
                fig_long, grid = plt.subplots(nrows=self.N_dim, ncols=2, figsize=[10, 2.5 * self.N_dim],
                                         sharex='col', sharey='row', layout='tight', width_ratios=[5, 1])
                
                if self.N_dim == 1:
                    axs, axs_pdf = [grid[0]], [grid[1]]
                else:
                    axs, axs_pdf = grid[:, 0], grid[:, 1]

                plot_time(_axs=axs, 
                          _time=t_l, 
                          _pred_closed=Y_closed,
                           _pred_open=U_open, 
                          _inputs=U_test_l, 
                          _target=Y_test_l[self.N_wash:],)

                # Plot histograms]
                for dim_i, ax_2 in enumerate(axs_pdf):
                    if dim_i in self.observed_idx:
                        _i = np.argmin(abs(observed_idx_np - dim_i))
                        ax_2.hist(U_test_l[:, _i], color='k', lw=2, alpha=0.6, histtype='step', **hist_args)

                    ax_2.hist(Y_test_l[:, dim_i], color='k', lw=.85, histtype='step', **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='stepfilled', alpha=0.5, **hist_args)
                    ax_2.hist(Y_closed[:, dim_i], color='r', ls='--', histtype='step', **hist_args)

                # axs[0].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
                plt.suptitle(f'Li = {Li}, observed idx = {self.observed_idx}, error = {err_long:.4}')
                axs[-1].set(xlabel='$t/T$')
            else:
                fig_long = None


            if short_term:
                i0 = 0 # reset time index for each Li
                figures_short = []
                short_term_error = 0.

                while i0 + Nt_test < max_test_time:
                    if len(figures_short) >= max_short_tests:
                        break
                    test_counter += 1

                    i1 = i0 + Nt_test + self.N_wash 
                    
                    current_input = U_test_l[i0:i1-1].copy()
                    current_target = Y_test_l[i0+self.N_wash:i1].copy()
                    current_time = t_l[i0:i1]

                    # predict
                    Y_closed, U_open = predict_Y(current_input, current_target)

                    current_error = np.log10(self.compute_nRMSE(current_target, Y_closed, norm=norm_l))

                    short_term_error += current_error
                

                    if test_counter <= max_L_tests:
                        fig_short, axs_short = plt.subplots(nrows=nrows, ncols=1, figsize=[8, 1.5 * nrows], sharex='all', layout='tight')
                        if nrows == 1:
                            axs_short = [axs_short] 

                        plot_time(_axs=axs_short, _time=current_time, _pred_closed=Y_closed, _pred_open=U_open, 
                                  _inputs=current_input, _target=current_target, _err=current_error)
                        

                        axs_short[0].legend(title=f'Test {test_counter}: Li = {Li}', loc='upper left',
                                            bbox_to_anchor=(1, 1), fontsize='x-small')
                        axs_short[-1].set(xlabel='$t/T$')

                        figures_short.append(fig_short)
                    i0 += Nt_test

                errors_all.append(short_term_error / max(1, (i0 // Nt_test)))
            else:
                figures_short = [None]

        else: 
            fig_long = None
            figures_short = [None]

        # Compute errors over all Lis
        if test_counter > 0:
            errors_all = np.array(errors_all) 
            print(f'Overall tests min, max and mean MSE in {test_counter} tests = {np.min(errors_all):.4}, {np.max(errors_all):.4}, {np.mean(errors_all):.4}.')
        
        return [fig_long] + figures_short


    def _plot_training_results(self, U_test, Y_test, results, save_ESN_training, folder):
        """
        Plots training results, including Bayesian optimization convergence and test results.
        """

        all_figs = []

        # Plot Bayesian optimization convergence
        fig1 = plt.figure()
        plot_convergence(results['res'])
        all_figs.append(fig1)
        # Plot Gaussian Process reconstruction
        all_figs.extend(self._plot_BO(results))
        # Plot Wout matrix
        all_figs.append(self.plot_Wout())
        # Plot test results if applicable
        if self.perform_test and U_test.shape[1] >= self.N_val:
            test_figs = self.run_test(U_test, Y_test, 
                                       long_term=True, short_term=True, max_short_tests=5)
            all_figs = all_figs + test_figs

        if save_ESN_training:
            if folder is None:
                folder = self.figs_folder
            os.makedirs(folder, exist_ok=True)
            save_pdf = plt_pdf.PdfPages(f'{folder}{self.filename}_Training.pdf')
            [add_pdf_page(save_pdf, fig) for fig in all_figs]
            save_pdf.close()



    def _plot_BO(self, results_bayesian_optimization):
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
        
        figs_bo = []
        if len(hp_names) >= 2:  # plot GP reconstruction
            gp = res.models[-1]
            res_x = np.array(res.x_iters)

            for hpi in range(len(hp_names) - 1):
                range_1 = getattr(self, f'{hp_names[hpi]}_range')  # type: tuple[float, float]
                range_2 = getattr(self, f'{hp_names[hpi + 1]}_range')  # type: tuple[float, float]

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
                figs_bo.append(fig)

        return figs_bo


    def plot_Wout(self):
        # Visualize the output matrix
        fig, ax = plt.subplots()
        im = ax.matshow(self.Wout.T, cmap="PRGn", aspect=4., vmin=-np.max(self.Wout), vmax=np.max(self.Wout))
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.colorbar(im, orientation='horizontal', extend='both')
        ax.set(ylabel='$N_u$', xlabel='$N_r$', title='$\\mathbf{W}_\\mathrm{out}$')
        return fig




if __name__ == "__main__":

    print(vars(EchoStateNetwork))

    pass