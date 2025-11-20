import os
import matplotlib.pyplot as plt
from integrator import IVPIntegrator
from observations import Observations
from tools_ML import EchoStateNetwork
from typing import Dict, Type, List, Tuple, Union
from utils import mean_vector_to_ensemble, save_to_pickle_file, load_from_pickle_file, check_valid_file
from model import HistoryTracker, Model
from utils import correlation, interpolate
import numpy as np
from copy import deepcopy


class Bias:
    upsample = 1
    L = 1
    augment_data = False

    # Default to not perform bayesian update to state
    bayesian_update = False
    biased_observations = False
    filter = None
    inflation = None
    t_init = None

    keys_to_print = ['bayesian_update', 'upsample', 'N_ens']
    extra_keys_to_print = []

    def __init__(self, b, t, dt, integrator_class=IVPIntegrator, **kwargs):

        self.precision_t = int(-np.log10(dt)) + 2
        self.dt = dt
        self.integrator = integrator_class(self)

        # ===================== ASSIGN PROVIDED KWARGS ======================= ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        # ========================== CREATE HISTORY ========================== ##
        b = self.build_state(b)

        self.history = HistoryTracker()
        if 'initial_capacity' in kwargs.keys():
            self.history._initial_capacity = kwargs.pop('initial_capacity')
        else:
            self.history._initial_capacity = max(1000, b.shape[0]*10)
        self.update_history(b, t=t, reset=True)
                                        
        # Add keys to print out
        if self.bayesian_update:
            self.keys_to_print += ['filter', 'inflation']
        
        self.keys_to_print += self.extra_keys_to_print

    
    def __format_state(self, b):
        """
        Ensure b has shape (nt, nb, nens)
        """
        if b.ndim == 3:
            return b # already (nt, nb, nens)
        if b.ndim == 1: 
            return b.reshape((1, b.size, 1)) # (nb,) -> (1, nb, 1)
        if b.ndim == 2:
            return b.reshape((1, *b.shape))  # (nb, nens) -> (1, nb, nens)
        
        raise AssertionError('b must have 1, 2 or 3 dimensions, got {}'.format(b.ndim))


    def build_state(self, innovation, model_bias=None):
        """
        Build the full bias state from innovations and model bias (if applicable)
        """
        innovation = self.__format_state(innovation)

        if self.biased_observations:
            if model_bias is None and not hasattr(self, 'hist'):
                model_bias = innovation.copy()
            else:
                model_bias = self.__format_state(model_bias)
            
            state = np.concatenate([model_bias, innovation], axis=1)
        else:
            state = innovation

        return state

    @property
    def dt(self):   
        return self._dt

    @dt.setter
    def dt(self, value):
        """Setter for the time step."""
        if value <= 0:
            raise ValueError("Time step must be positive.")
        self._dt = np.round(value, self.precision_t)
    
    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self.history.hist

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self.history.hist_t
    
    @property
    def current_state(self):
        """Returns the current state (last entry in history)."""
        return self.history.current_state

    @property
    def current_time(self):
        """Returns the current time (last entry in time history)."""
        return self.history.current_time


    @property
    def config(self):
        _config = dict()
        for key in self.keys_to_print:
             if hasattr(self, key):
                _config[key] = getattr(self, key)

        return _config


    @property
    def N_ens(self):
        return self.current_state.shape[-1]

    @property
    def current_bias(self):
        """Returns the current bias computed from the current state."""
        return self.get_bias(state=self.current_state)

    @property
    def current_innovations(self):
        """Returns the current innovations computed from the current state."""
        return self.get_innovations(state=self.current_state)

    def get_bias(self, state, **kwargs):

        if self.biased_observations:
            nb = state.shape[1] // 2
            return state[:nb, :, :]
        else:
            return state

    def get_innovations(self, state, **kwargs):
        if self.biased_observations:
            nb = state.shape[1] // 2
            return state[nb:, :, :]
        else:
            return state

    def get_ML_state(self, **kwargs):
        return None
    
    def get_bias_hist(self, mean=False):
        return self.get_bias(state=self.hist, mean=mean)


    def print_bias_parameters(self):
        print('\n ---------------- {} bias model input_parameters --------------- '.format(self.name))
        for key in sorted(set(self.keys_to_print)):
            if hasattr(self, key):
                val = getattr(self, key)
                if type(val) is float:
                    print('\t {} = {:.6}'.format(key, val))
                else:
                    print('\t {} = {}'.format(key, val))


    

    def update_history(self, b, t=None, reset=False, update_last_state=False, **kwargs):
        b = self.__format_state(b)

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(b.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == b.shape[0], f"Length of t ({t.size}) must match number of time steps in b ({b.shape[0]})."
        
        self.history.update_history(b, t=t, reset=reset, update_last_state=update_last_state)



    def copy(self):
        return deepcopy(self)


# =================================================================================================================== #


class NoBias(Bias):
    name = 'NoBias'

    def __init__(self, y, t, dt, **kwargs):
        super().__init__(b=np.zeros(y.shape), t=t, dt=dt, **kwargs)
        self.N_dim = self.hist.shape[1]
        self.observed_idx = np.arange(self.N_dim)

    def state_derivative(self):
        return np.zeros([self.N_dim, self.N_dim])

    def time_integrate(self, t, **kwargs):
        return np.zeros([len(t), self.N_dim, self.N_ens]), t


# =================================================================================================================== #


class ESN_bias(Bias, EchoStateNetwork):
    name = 'ESN_bias'

    biased_observations = True
    update_reservoir = False
    correlation_based_training = True   
    wash_obs , wash_time = None, None

    extra_keys_to_print = ['t_train', 
                           't_val', 
                           'N_wash', 
                           'rho', 
                           'sigma_in',
                           'N_units', 
                           'perform_test', 
                           'L', 
                           'connect', 
                           'tikh',
                           'update_reservoir', 
                           'observed_idx']

    _augment_data_length = 2  # number of augmented data sets (only used if augment_data=True)
    
    # def __init__(self, y, t, dt, **kwargs):
        
    def __init__(self, 
                 forecast_model: Type[Model] = None,
                 reference_data: Union[Type[Observations], List[Type[Observations]]] = None,
                 filename: str = None,
                 **kwargs):
        
        # ----------------------  Initialise parent Bias  and EchoStateNetwork  ------------------------- #
        kwargs_keys = list(kwargs.keys())
        for kwy in kwargs_keys:
            if hasattr(self, kwy) or kwy in self.extra_keys_to_print:
                setattr(self, kwy, kwargs.pop(kwy))


        Bias.__init__(self, **kwargs)

        # assign default EchoStateNetwork parameters if not provided

        kwargs['t_train'] = getattr(self, 't_train', forecast_model.t_transient)
        kwargs['t_val'] = getattr(self, 't_val', forecast_model.t_CR)
        kwargs['t_test'] = getattr(self, 't_test', None)

        EchoStateNetwork.__init__(self, y=self.current_state, **kwargs)

        # --------------------------  Load or Create ESN Bias Model  ------------------------- #
        


        loaded_bias = self.load_bias_model(filename)
        if loaded_bias is None:
            
            if forecast_model is None or reference_data is None:
                raise ValueError('forecast_model and reference_data must be provided to create a new ESN_bias model')
            
            # Create a copy of the forecast model to use for training data generation --------------------------
            print('Creating new ESN bias model......')
            forecast_model = forecast_model.copy()

            t_min = self.t_train + self.t_val
            if self.perform_test:
                if self.t_test is not None:
                    t_min += self.t_test
                else:
                    t_min += self.t_val * 5

            Nt_min = int(np.ceil(t_min / self.dt_ESN)) + self.N_wash
            
            # Create training data------------------------------------------------------------------------------
            training_data = self._load_bias_training_dataset(filename, Nt_min)
            if training_data is None:
                std_alpha = kwargs.pop('std_alpha', None)
                std_phi = kwargs.pop('std_phi', None)
                train_data_dict = self._create_bias_training_dataset(forecast_model, reference_data, Nt_min, 
                                                                     std_phi=std_phi, std_alpha=std_alpha)

                if filename is not None:
                    save_to_pickle_file(filename, train_data_dict)
            else:
                print('Loaded multi-parameter training data')


            if self.plot_training:
                self._plot_training_dataset(plot_data=training_data['data'])

    @property
    def augment_data_length(self):
        if self.augment_data:
            return self._augment_data_length
        else:
            return 0

    def __correlate_data(self, y_L_model, _y_raw, Nt_min):
        """
        Create training data based on correlation between model outputs and observations.
        Inputs:
            y_L_model: Model output history (Nt x Nq x L)
            _y_raw: Raw observation history (Nt x Nq x 1)
            Nt_min: Minimum number of time steps for training data
        """

        # The correlation function is used only locally in this method.
        # If you need to reuse it elsewhere, consider moving it to module level.
        # Otherwise, it remains here for encapsulation and clarity.

        len_augment_set = self.augment_data_length

        Nt, Nq, L = y_L_model.shape
        N_corr = Nt - Nt_min
        train_data_model = np.zeros([Nt_min, Nq, L * len_augment_set]) # shape (Nt_min, Nq, L * len_augment_set)

        lags = np.linspace(start=0, stop=N_corr, num=N_corr, dtype=int)

        y_raw_c = _y_raw[:N_corr, ..., 0] - np.mean(_y_raw[:N_corr, ..., 0], axis=0, keepdims=True) # shape (N_corr, Nq)
        y_model_c = y_L_model[:2*N_corr, :, :] - np.mean(y_L_model[:2*N_corr, :, :], axis=0, keepdims=True)  # shape (2*N_corr, Nq, L)
        # normalize
        epsilon = 1e-8
        y_raw_c /= (np.max(np.abs(y_raw_c), axis=0, keepdims=True) + epsilon)
        y_model_c /= (np.max(np.abs(y_model_c), axis=0, keepdims=True) + epsilon)

        
        shifted_y_model_list = [y_L_model[lag : N_corr + lag] for lag in range(N_corr)] # list of length N_corr, each element shape (N_corr, Nq, L)
        correlations = np.array([correlation(y_raw_c, yy) for yy in shifted_y_model_list]) # shape (N_corr, L)

        # for each lag, find the best, worst, and mid correlation indices and store the corresponding data
        for ii in range(L):
            # correlations[:, ii] gives the correlation values across lags for ensemble member ii
            _corrs = correlations[:, ii]
            best_lag = lags[np.argmax(_corrs)]  # fully correlated (highest correlation value)

            # Store train data: slice the appropriate ensemble member ii from y_L_model
            base_col = len_augment_set * ii
            train_data_model[:, :, base_col] = y_L_model[best_lag:best_lag + Nt_min, :, ii]

            # Fill the "mid" column if at least 2-augment set and the "worst" column if 3-augment set
            if len_augment_set >= 2:
                worst_lag = lags[np.argmin(_corrs)]  # fully uncorrelated (lowest correlation value)
                mid_lag = int(np.mean([best_lag, worst_lag]))  # mid-correlated
                train_data_model[:, :, base_col + 1] = y_L_model[mid_lag:mid_lag + Nt_min, :, ii]
                if len_augment_set >= 3:
                    train_data_model[:, :, base_col + 2] = y_L_model[worst_lag:worst_lag + Nt_min, :, ii]      

        return train_data_model
    

    def load_bias_model(self, bias_filename: str):

        bias = None

        if bias_filename is not None:
            if os.path.isfile(bias_filename):
                load_bias = load_from_pickle_file(bias_filename)
                if check_valid_file(load_bias, self.config) is True:
                    bias = load_bias
            else:
                print('Create bias model: ', bias_filename)
        return bias 


    def create_bias_model(self, 
                            wash_t=None,
                            wash_obs=None,
                            bias_filename=None,
                            folder=None,
                            **bias_params):
        """
        This function creates the bias model for the input ensemble, truth and input_parameters.
        The bias model is added to the ensemble object as ensemble.bias.
        If the bias model requires washout (e.g. ESN) the washout observations are added to
        the truth dictionary.
        """
        if isinstance(training_dataset, dict):
            training_dataset = [training_dataset]
            
        elif not isinstance(training_dataset, list):
            raise ValueError('Training dataset must be a list of dicts or a dict')

        y_raw = [_data['y_raw'].copy() for _data in training_dataset]
        y_true = [_data['y_true'].copy() for _data in training_dataset]
        truth = training_dataset[-1]
        bias_params['noise_type'] = truth['noise_type']

        train_ens = ensemble.copy()
        edited_file = False


        if bias is None:
            edited_file = True
            train_ens.init_bias(**bias_params)
            bias = train_ens.bias.copy()

            # Create training data on a multi-parameter approach
            train_data = create_bias_training_dataset(y_raw=y_raw, 
                                                      y_pp=y_true, 
                                                      ensemble=train_ens,
                                                      **bias_params)

            # Run bias model training
            for key, val in bias_params.items():
                if not hasattr(bias, key):
                    train_data[key] = val

            bias.train_bias_model(**train_data)

            # Create washout if needed
            if bias.t_init is None:
                bias.t_init = truth['t_obs'][0]

            if hasattr(bias, 'N_wash') and wash_t is None:
                wash_t, wash_obs = create_washout(bias, truth['t'], truth['y_raw'])

        # Create washout if needed
        if hasattr(bias, 'N_wash') and wash_t[0] > truth['t_obs'][0]:
            if bias.t_init is None or bias.t_init > truth['t_obs'][0]:
                bias.t_init = truth['t_obs'][0]

            wash_t, wash_obs = create_washout(bias, truth['t'], truth['y_raw'])
            edited_file = True

        if edited_file and bias_filename is not None:
            save_to_pickle_file(bias_filename, bias.copy(), wash_obs, wash_t)


        return bias, wash_obs, wash_t

    # ================================================================================================================== #
    #                      BIAS MODEL  METHODS                       #
    # ================================================================================================================== #
    


    def reset_history(self, b, t):
        self.hist_t = t
        self.hist = b
        r = np.zeros((self.N_units, self.N_ens))
        self.reset_state(u=b, r=r)

    def update_current_state(self, b, **kwargs):
        if 'u' not in kwargs.keys():
            kwargs['u'] = b
        self.reset_state(**kwargs)

    def state_derivative(self):
        u, r = [np.mean(xx, axis=-1, keepdims=True) for xx in self.get_reservoir_state()]
        J = self.Jacobian(open_loop_J=True, state=(u, r))  # Compute ESN Jacobian
        db_din = J[np.array(self.bias_idx), np.array([self.bias_idx]).T]
        return -db_din

    def time_integrate(self, t, y=None, wash_t=None, wash_obs=None):
        if not self.trained:
            raise NotImplementedError('ESN model not trained')

        interp_flag = False
        Nt = len(t) // self.upsample
        if len(t) % self.upsample:
            Nt += 1
            interp_flag = True
        t_b = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)

        # If the time is before the washout initialization, return zeros
        if self.initialised:
            u, r = self.closedLoop(Nt)
        else:
            u = np.zeros((Nt + 1, self.N_dim, self.N_ens))
            r = np.zeros((Nt + 1, self.N_units, self.N_ens))
            if wash_t is not None:
                t1 = np.argmin(abs(t_b - wash_t[0]))
                Nt -= t1
                # Flag initialised
                self.initialised = True
                # Run washout phase in open-loop
                wash_model = interpolate(t, y, wash_t)
                washout = wash_obs - np.mean(wash_model, axis=-1)

                u_open, r_open = self.openLoop(washout)

                u[t1:t1 + self.N_wash + 1] = u_open
                r[t1:t1 + self.N_wash + 1] = r_open
                Nt -= self.N_wash

                # Run the rest of the time window in closed-loop
                if Nt > 0:
                    # Store open-loop forecast
                    self.reset_state(u=self.outputs_to_inputs(full_state=u_open[-1]), r=r_open[-1])
                    u_close, r_close = self.closedLoop(Nt)
                    u[t1 + self.N_wash + 1:] = u_close[1:]
                    r[t1 + self.N_wash + 1:] = r_close[1:]
        # Interpolate the final point if the upsample is not multiple of dt
        if interp_flag:
            u[-1] = interpolate(t_b[-Nt:], u[-Nt:], t[-1])
            r[-1] = interpolate(t_b[-Nt:], r[-Nt:], t[-1])
            t_b[-1] = t[-1]

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        return u[1:], t_b[1:]

    def train_bias_model(self,
                         plot_training=True,
                         **kwargs):
        
        data = kwargs.pop('data')

        # Set the provided input_parameters if they are already initialized

        train_data = kwargs.copy()
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, train_data.pop(key))
                
        #  Train the network
        self.train(data,
                   plot_training=plot_training,
                   add_noise=train_data['add_noise'])
        # Set trained flag
        self.trained = True
        # if self.bayesian_update:
        #     self.update_history(b=np.zeros((self.N_dim, self.m)), reset=True)
        #     self.initialise_state(data=data, N_ens=self.m)

    def get_ML_state(self, concat_reservoir_state=False):
        u, r = self.get_reservoir_state()
        if concat_reservoir_state:
            return np.concatenate([u, r], axis=0)
        else:
            return u

    @property
    def bias_idx(self):
        if self.biased_observations:
            return [a for a in np.arange(self.N_dim) if a not in self.observed_idx]
        else:
            return self.observed_idx

    def get_bias(self, state, mean=True):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        if state.shape[0] == self.N_dim:
            return state[self.bias_idx]
        elif state.shape[1] == self.N_dim:
            return state[:, self.bias_idx]
        else:
            raise AssertionError('state shape = {}'.format(state.shape))

    def get_innovations(self, state, mean=True):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        if state.shape[0] == self.N_dim:
            return state[self.observed_idx]
        elif state.shape[1] == self.N_dim:
            return state[:, self.observed_idx]
        else:
            raise AssertionError('state shape = {}'.format(state.shape))
        
        
    def __sample_model_states(self, 
                              fm: Model, 
                              Nt_min: int,
                              L: int = 10,
                              std_phi: float = None,
                              std_alpha: Union[float, Dict[str, Union[float,  List[float]]]] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample model states from the forecast model for training data generation.
        Inputs:
            fm: Forecast model instance
            Nt_min: Minimum number of time steps to sample
        Returns:
            y_L_model: Sampled model observable history (Nt x Nq x L)
        """
        
        fm = self.forecast_model.copy()

        
        # =================  Create ensemble for multi-parameter training data generation ===
        if fm.m != L:
            psi0 = np.mean(fm.current_state.copy(), axis=-1)  # Shape (Npsi,) 
            if std_phi is None:
                std_phi = np.std(fm.current_state[:fm.Nphi, :], axis=-1)  # Shape (Nphi,)
            if std_alpha is None:
                std_alpha = np.std(fm.current_state[fm.Nphi:fm.Nphi+fm.Na, :], axis=-1)  # Shape (Na,)
            new_phi = mean_vector_to_ensemble(mean_vec=psi0[:fm.Nphi], 
                                            std=std_phi, 
                                            m=L,
                                            method='uniform')  # Shape (Nphi, L)
            new_alpha = mean_vector_to_ensemble(mean_vec=psi0[fm.Nphi:fm.Nphi+fm.Na], 
                                                std=std_alpha, 
                                                m=L,
                                                method='uniform')  # Shape (Na, L)
            psi0_ens = np.concatenate([new_phi, new_alpha], axis=0)  # Shape (Npsi, L)

            fm.update_history(psi=psi0_ens[np.newaxis, :, :], reset=True)

        # Forecast to post-transient
        fm.forecast_step(t_end=fm.t_transient, reset=True)

        y_L_model = fm.model.get_observable_hist()  # Nt x Nq x m (L)
        psi = fm.current_state

        #   Remove and replace fixed points ------------- #
        tol = 1e-1
        N_CR = int(round(fm.t_CR / fm.dt))
        range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
        idx_FP = (range_y < tol)

        if len(np.flatnonzero(idx_FP)) / len(idx_FP) >= 0.2:
            allowed_FPs = np.flatnonzero(idx_FP)[0:int(0.2 * len(idx_FP)) + 1]
            idx_FP[allowed_FPs] = 0
            psi0 = psi[:, ~idx_FP]  # non-fixed point ICs (keeping one)
            print(f'There are {len(np.flatnonzero(idx_FP))}/{len(idx_FP)} fixed points')
            new_psi0 = self.rng.multivariate_normal(np.mean(psi0, axis=0),  np.cov(psi0.T),  len(np.flatnonzero(idx_FP)))
            psi0 = np.concatenate([psi0, new_psi0], axis=0)
        else:
            psi0 = psi


        #   Forecast fixed-point-free post-transient ensemble ------------- #
        fm.update_history(psi=psi0[np.newaxis, :, :], reset=True)
        fm.forecast_step(t_end=Nt_min + 2*fm.t_CR, close=True)

        return fm.model.get_observable_hist()


    def __prepare_reference_data(self, reference_data, Nt_min) -> Tuple[np.ndarray, np.ndarray]:
            """
            Prepare reference data for training data generation.
            Inputs:
                reference_data: List of Observations instances or a single instance
            Returns:
                y_raw: List of raw observation arrays (each of shape Nt x Nq x 1)
                y_true: List of clean observation arrays (each of shape Nt x Nq x 1)
            """
            if isinstance(reference_data, Observations):
                reference_data = [reference_data]

            y_raw, y_true = [], []
            for rfd in reference_data:
                # ensure ndim = 3
                yr, yt = rfd.y_raw.copy(), rfd.y_true.copy()
                if yr.ndim == 2:
                    yr = yr[:, :, np.newaxis]
                if yt.ndim == 2:
                    yt = yt[:, :, np.newaxis]

                y_raw.append(yr[-Nt_min:])
                y_true.append(yt[-Nt_min:])

            return y_raw, y_true


    def _create_bias_training_dataset(
                                    self,
                                    forecast_model: Type[Model],
                                    reference_data: Union[Type[Observations], List[Type[Observations]]], 
                                    Nt_min: int, 
                                    # Additional parameters for training data generation
                                    std_phi: float = None,
                                    std_alpha: Union[float, Dict[str, Union[float,  List[float]]]] = None,
                                    ):

        """
        Multi-parameter data generation for ESN training.
        - If the observations are biased, the bias estimator must predict  
            (1) the innovations, i.e., the difference between the raw data and the model (observable)
            (2) the difference between the truth and the model, which is the actual model bias (non observable)
        - If there is data augmentation, the training data are augmented by 
            (a) scaling the innovations by different factors (only if not correlation_based_training)
            (b) correlating the model outputs with the observations to create different training sets
        - If multiple experimental datasets are provided, the training data from each dataset are concatenated.

        Inputs:
            forecast_model: The forecast model instance
            reference_data: List of Observations instances or a single instance
            Nt_min: Minimum number of time steps for training data
            train_params: Additional parameters for training data generation
        Returns:
            train_data: Dictionary containing training data and relevant parameters
            training_keys = ['upsample',
                            'L',
                            'augment_data',
                            'correlation_based_training',
                            'biased_observations']
            data shape: (N_datasets * L * augment_data_length, Nt_min, N_dim)
        """

        print('Creating multi-parameter training data for ESN bias model...',
              f'...correlation_based_training = {self.correlation_based_training}',
              f'...biased_observations = {self.biased_observations}',
              f'...augment_data = {self.augment_data}', sep='\n\t')



        y_model_L = self.__sample_model_states(forecast_model.copy(), Nt_min, std_phi=std_phi, std_alpha=std_alpha) # Nt x Nq x L
        y_raw, y_true = self.__prepare_reference_data(reference_data, Nt_min) # Lists of Nt x Nq x 1
        Nq = y_model_L.shape[1] # number of observed variables

        # ========================================  GENERATE TRAINING DATA ========================================= #

        if not self.correlation_based_training:   # (Nóvoa & Magri 2023 CMAME)

            innovations_all, model_bias_all = [], []
            for yr, yt in zip(y_raw, y_true):
                innovations = (yr - y_model_L[-Nt_min:]).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                innovations_all.append(innovations)
                
                if self.augment_data:
                    innovations_all.append(innovations * 1e-1)
                    innovations_all.append(innovations * -1e-2)

                if self.biased_observations:
                    model_bias = (yt - y_model_L[-Nt_min:]).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                    model_bias_all.append(model_bias)
                    if self.augment_data:
                        model_bias_all.append(model_bias * 1e-1)
                        model_bias_all.append(model_bias * -1e-2)                

        else:  #  (Nóvoa et al. 2024 JFM) 
            # Here, we create augmented data sets based on correlation between model outputs and observations.
            # The augment_data_length determines how many sets are created (best, mid, worst correlations).
            y_model_L_corr = self.__correlate_data(yr, y_model_L, Nt_min) # shape (Nt_min x Nq x L * augment_data_length)

            innovations_all, model_bias_all = [], []
            for yr, yt in zip(y_raw, y_true):
                innovations = (yr - y_model_L_corr).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                innovations_all.append(innovations)
                if self.biased_observations:
                    model_bias = (yt - y_model_L_corr).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                    model_bias_all.append(model_bias)

        # Combine the innovations (and model biases) #

        if not self.biased_observations:
            train_data = np.concatenate(innovations_all, axis=0)
            observed_idx = np.arange(Nq)
        else:
            innovations_all = np.concatenate(innovations_all, axis=0)
            model_bias_all = np.concatenate(model_bias_all, axis=0) 
            train_data = np.concatenate([model_bias_all,
                                         innovations_all], axis=2)
            observed_idx = Nq + np.arange(Nq) # indices of innovations in the state vector            

        # =============================== Save train_data dict ================================ #
        # Save key keywords
        train_data_dict = {key:val for key, val in self.config.items()} 
        # add the extra kwargs used to create the training data
        # train_data_dict.update({key: val for key, val in kwargs.items() if key not in train_data_dict.keys()})
        # add training data and observed indices
        train_data_dict.update(train_data=train_data,
                               observed_idx=observed_idx,
                               )
        return train_data_dict
        




    def _load_bias_training_dataset(self, filename, Nt_min):

        # =========================== Load training data if available ============================== #
        if filename is not None:
            try:
                loaded_train_data = load_from_pickle_file(filename)
                try:
                    necessary_properties = self.config.copy()

                    if check_valid_file(loaded_train_data, 
                                        necessary_properties):
                        
                        _U = loaded_train_data['data']

                        if _U.shape[1] < Nt_min:
                            print('Re-run multi-parameter training data: Increase the length of the training data')
                        elif self.augment_data and _U.shape[0] == self.L:
                            print('Re-run multi-parameter training data: need data augment ')
                        else:
                            return loaded_train_data
                except TypeError:
                    print(f'File {filename} type = {type(loaded_train_data)} is not dict')
            except FileNotFoundError:
                print(f'Run multi-parameter training data: file {filename} not found')
        
        return None
    

    def _plot_training_dataset(self, plot_data):
            _L, _Nt, _Ndim = plot_data.shape

            if self.biased_observations:
                _nc = 2
            else:
                _nc = 1

            _Ndim = int(round(_Ndim // _nc))
            _nr = int(min(_Ndim, 10))

            fig, axs = plt.subplots(nrows=_nr, ncols=_nc, figsize=(8*_nc, 2*_nr), sharex=True, layout='constrained')
            if not isinstance(axs, np.ndarray):
                axs = [axs]

            if self.biased_observations:
                axs = axs.T.flatten()
                [axs[_ii].set(title=_ttl) for _ii, _ttl in zip([0, _nr], ['Model bias', 'Innovations'])]

            _t_data = np.arange(0, _Nt) * self.dt
            _Ls = np.random.randint(low=0, high=_L, size=len(axs))

            for kk, ax, Li in zip(range(len(axs)), axs, _Ls):
                if kk < _nr:
                    ax.plot(_t_data, plot_data[Li, :, kk], lw=1., color='k')
                else:
                    ax.plot(_t_data, plot_data[Li, :, kk - _nr + _Ndim], lw=1., color='k')

                times = [0, self._train, self._train + self._val, _t_data[-1]]
                [ax.axvspan(times[_ii], times[_ii+1], facecolor=_c, alpha=0.3, zorder=-100,
                            label=_lbl) for _ii, _c, _lbl in zip(range(3), ['orange', 'red', 'navy'],
                                                                [f'Train, Li{Li}', 'Validate', 'Test'])]
            axs[0].legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.5))
            plt.show()

