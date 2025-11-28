import os
import matplotlib.pyplot as plt
from integrator import DiscreteIntegrator, IVPIntegrator
from observations import Observations
from tools_ML.EchoStateNetwork import EchoStateNetwork
from typing import Type, List, Tuple, Union
from utils import save_to_pickle_file, load_from_pickle_file, check_valid_file
from model import Model
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

    keys_to_print = ['bayesian_update', 'upsample', 'N_ens', 'biased_observations']
    extra_keys_to_print = []

    def __init__(self, b, t, dt, integrator_class=IVPIntegrator, **kwargs):
        self.dt = dt
        self.precision_t = int(-np.log10(dt)) + 2
        self.integrator = integrator_class(self)

        # ===================== ASSIGN PROVIDED KWARGS ======================= ##
        [setattr(self, key, kwargs.pop(key)) for key in kwargs.keys() if hasattr(self, key)]

        # ========================== CREATE HISTORY ========================== ##
        b = np.asarray(b)

        # Ensure b has shape (nt, nb, nens)
        if b.ndim == 1:
            # (nb,) -> (1, nb, 1)
            b = b.reshape((1, b.size, 1))
        elif b.ndim == 2:
            # (nt, nb) -> (nt, nb, 1)
            b = b.reshape((*b.shape, 1))
        elif b.ndim == 3:
            # already (nt, nb, nens)
            pass
        else:
            raise AssertionError('b must have 1, 2 or 3 dimensions, got {}'.format(b.ndim))

        # If observations are biased, duplicate the state dimension (nq) for bias/innovations
        if self.biased_observations:
            b = np.concatenate([b, b], axis=1)

        # Ensure time array matches nt
        t = np.asarray(t)
        if t.ndim == 0:
            t = t.reshape((1,))
        if t.size != b.shape[0]:
            raise AssertionError('length of t ({}) must match number of time steps in b ({})'.format(t.size, b.shape[0]))

        self.hist = b
        self.hist_t = t
                                        
        # Add keys to print out
        if self.bayesian_update:
            self.keys_to_print += ['filter', 'inflation']
        
        self.keys_to_print += self.extra_keys_to_print


    @property
    def config(self):
        _config = dict()
        for key in self.keys_to_print:
             
             if hasattr(self, key):
                _config[key] = getattr(self, key)

        return _config


    @property
    def hist(self):
        return self._hist
    
    @hist.setter
    def hist(self, b):
        b = np.asarray(b)

        # Ensure b has shape (nt, nb, nens)
        if b.ndim == 1:
            # (nb,) -> (1, nb, 1)
            b = b.reshape((1, b.size, 1))
        elif b.ndim == 2:
            # (nt, nb) -> (nt, nb, 1)
            b = b.reshape((*b.shape, 1))
        elif b.ndim == 3:
            # already (nt, nb, nens)
            pass
        else:
            raise AssertionError('b must have 1, 2 or 3 dimensions, got {}'.format(b.ndim))
        assert b.ndim == 3, 'b must have 3 dimensions (nt, nb, nens), got {}'.format(b.ndim)
        
        self._hist = b


    @property
    def N_ens(self):
        return self.hist.shape[-1]

    @property
    def current_time(self):
        return self.hist_t[-1]

    @property
    def current_bias(self):
        current_state = self.hist[-1]
        return self.get_bias(state=current_state)

    @property
    def current_innovations(self):
        current_state = self.hist[-1]
        return self.get_innovations(state=current_state)

    def get_bias(self, state, **kwargs):
        return state

    def get_innovations(self, state, **kwargs):
        return state

    def get_ML_state(self, **kwargs):
        return None
    
    def get_bias_hist(self, mean=False):
        return self.get_bias(state=self.hist, mean=mean)

    @property
    def name(self):
        return self.__class__.__name__


    def print_bias_parameters(self):
        print('\n ---------------- Bias model parameters --------------- ')
        print(f'\t Bias class name: {self.__class__.__name__}')
        for key in sorted(set(self.keys_to_print)):
            if hasattr(self, key):
                val = getattr(self, key)
                if type(val) is float:
                    print('\t {} = {:.6}'.format(key, val))
                else:
                    print('\t {} = {}'.format(key, val))



    def time_integrate(self, Nt, y=None, wash_t=None, wash_obs=None):

        return self.integrator.advance(Nt=Nt)
    

    def update_history(self, b, t=None, reset=False, update_last_state=False, **kwargs):

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(b.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == b.shape[0], f"Length of t ({t.size}) must match number of time steps in b ({b.shape[0]})."
        
        self.history.update_history(b, t=t, reset=reset, update_last_state=update_last_state)
        self.update_aux_state(b, t=t, reset=reset, update_last_state=update_last_state, **kwargs)
    
    def update_aux_state(self, b, t=None, reset=False, update_last_state=False, **kwargs):
        pass


    def update_current_state(self, b, **kwargs):
        self.hist[-1] = b

    def reset_history(self, b, t):
        self.hist_t = t
        self.hist = b

    def copy(self):
        return deepcopy(self)


# ================================================================================================================== #


class NoBias(Bias):
    # name = 'NoBias'

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
    # name = 'ESN_bias'

    biased_observations = True
    update_reservoir = False

    correlation_based_training = True   

    wash_obs , wash_time = None, None

    extra_keys_to_print = ['rho', 
                           'sigma_in',
                           'N_units', 
                           'perform_test', 
                           'L', 
                           'connect', 
                           'tikh',]
    
    
    upsample=5
    N_units=50
    N_wash=10
    N_folds=4
    # Hyperparameter search ranges
    rho_range=(0.5, 1.)
    sigma_in_range=(np.log10(1e-5), np.log10(1e1))
    tikh_range=[1e-12, 1e-9]


    def __init__(self, 
                 forecast_model: Type[Model],
                 reference_data: Union[Type[Observations], List[Type[Observations]]] = None,
                 filename: str = None,
                 **kwargs):
        
        # ----------------------  Initialise parent Bias  and EchoStateNetwork  ------------------------- #

        kwargs['t_train'] = kwargs.get('t_train', forecast_model.t_transient/2)
        kwargs['t_val'] = kwargs.get('t_val', forecast_model.t_CR)
        kwargs['t_test'] = kwargs.get('t_test', None)

        kwargs_keys = list(kwargs.keys())
        for kwy in kwargs_keys:
            if hasattr(self, kwy) or kwy in self.extra_keys_to_print:
                setattr(self, kwy, kwargs.pop(kwy))

        y = kwargs.get('y', np.zeros((forecast_model.Nq, 1)))
        t = kwargs.get('t', 0.0)
        dt = kwargs.get('dt', forecast_model.dt)

        Bias.__init__(self, integrator_class=DiscreteIntegrator, **kwargs)
        EchoStateNetwork.__init__(self, y=self.current_state, **kwargs)


        # --------------------------  Load or Create ESN Bias Model  ------------------------- #
        
        loaded_bias = self.load_bias_model(filename)
        if loaded_bias is None:
            
            if forecast_model is None or reference_data is None:
                raise ValueError('forecast_model and reference_data must be provided to create a new bias model')
            
            # Create a copy of the forecast model to use for training data generation --------------------------
            
            forecast_model = forecast_model.copy()

            t_min = self.t_train + self.t_val
            if self.perform_test:
                if self.t_test is not None:
                    t_min += self.t_test
                else:
                    t_min += self.t_val

            Nt_min = int(np.ceil(t_min / self.dt)) + self.N_wash * self.upsample
            
            # Create training data------------------------------------------------------------------------------
            train_data_dict = self._load_bias_training_dataset(filename, Nt_min)
            if train_data_dict is None:
                std_alpha = kwargs.pop('std_alpha', None)
                std_phi = kwargs.pop('std_phi', None)
                train_data_dict = self._create_bias_training_dataset(forecast_model, reference_data, Nt_min, 
                                                                     std_phi=std_phi, std_alpha=std_alpha)

                if filename is not None:
                    save_to_pickle_file(filename, train_data_dict)

            self.train_data_dict = train_data_dict.copy()

            # Plot training dataset
            self._plot_training_dataset(plot_data=train_data_dict['data'])
            

            # # Train the ESN bias model --------------------------------------------------------------------------
            self._train_bias_model(data=train_data_dict['data'], 
                                   add_noise=kwargs.get('add_noise', False)
                                   )


    def _train_bias_model(self,
                          data, 
                           add_noise):
        """
        Train the EchoStateNetwork  using the provided training data.
        """
                
        #  Train the network
        self.train(data, plot_training=True,
                        add_noise=add_noise)
        
        # Set trained flag
        self.trained = True
        # if self.bayesian_update:
        #     self.update_history(b=np.zeros((self.N_dim, self.m)), reset=True)
        #     self.initialise_state(data=data, N_ens=self.m)


    @property
    def augment_data_length(self):
        if self.augment_data:
            if isinstance(self.augment_data, int):
                return self.augment_data
            return 2
        else:
            return 1

    def _correlate_data(self, y_L_model, _y_raw, Nt_min):
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


    def update_aux_state(self, reset=False, update_last_state=False, **kwargs): 
        if reset:
            self.reservoir_state *= 0.
        if update_last_state:
            self.reservoir_state = kwargs.get('r', None)
            if self.reservoir_state is None:
                raise ValueError('r must be provided to update_last_state=True in update_aux_state()')



    def state_derivative(self):
        r_mean = np.mean(self.reservoir_state, axis=-1, keepdims=True) 
        u_mean = self.reservoir_to_physical(r_mean)
        J = self.Jacobian(open_loop_J=True, state=(u_mean, r_mean))  # Compute ESN Jacobian
        db_din = J[np.array(self.bias_idx), np.array([self.bias_idx]).T]
        return -db_din
    

    def time_step(self, Nt=10, u_in=None, averaged=False):
        """
            Args:
                Nt: number of forecast steps (physical time, not dt_ESN)
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying input_parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        assert self.trained, 'ESN model not trained'
        # 1. get initial condition

        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)


        r = np.empty((Nt + 1, self.N_units, self.N_ens))
        u_out = np.empty((Nt + 1, self.N_dim, self.N_ens))

        r[0] = self.reservoir_state

        n_open = 0
        if u_in is None:
            u_out[0] = self.reservoir_to_physical(r[0])
        elif u_in.shape == (self.N_dim_in, self.N_ens):
            n_open = 1
            u_out[0, self.observed_idx] = u_in
        elif u_in.shape[1:] == (self.N_dim_in, self.N_ens):
            n_open = u_in.shape[0]
            u_out[:n_open, self.observed_idx] = u_in
        else:
            raise AssertionError('u_in has shape {}, expected ({}, {}) or ({}, {}, {})'.format(
                u_in.shape, self.N_dim_in, self.N_ens, u_in.shape[0], self.N_dim_in, self.N_ens))

        if averaged:
            # Mean state
            u_m, r_m = [np.mean(xx, axis=-1, keepdims=True) for xx in [u, r]]
            u_dev, r_dev = [xx - xm for xx, xm in zip([u, r], [u_m, r_m])]

            for i in range(Nt):
                # only store if i+1 > len uinput
                if i + 1 < n_open:
                    _, r_m[i + 1] = self._single_step(u_m[i], r_m[i])
                else:
                    u_m[i + 1], r_m[i + 1] = self._single_step(u_m[i], r_m[i])

            # copy into the ensemble members the mean + deviation
            u, r = [xm + xd for xm, xd in zip([u_m, r_m], [u_dev, r_dev])]
        else:
            for i in range(Nt):
                u, r[i + 1] = self._single_step(u[i], r[i])

                if i + 1 < n_open:
                    _, r[i + 1] = self._single_step(u[i], r[i])
                else:
                    u[i + 1], r[i + 1] = self._single_step(u[i], r[i])

        return self.build_psi(u, r), t


    def _single_step(self, u, r):
        u_input = self.outputs_to_inputs(full_state=u)
        return self.step(u_input, r)


    # def time_integrate(self, t, y=None, wash_t=None, wash_obs=None):
    #     return self.integrator.advance(Nt=Nt, averaged=False, alpha=self.get_alpha())
        # if not self.trained:
        #     raise NotImplementedError('ESN model not trained')

        # interp_flag = False
        # Nt = len(t) // self.upsample
        # if len(t) % self.upsample:
        #     Nt += 1
        #     interp_flag = True
        # t_b = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)

        # # If the time is before the washout initialization, return zeros
        # if self.initialised:
        #     u, r = self.closedLoop(Nt)
        # else:
        #     u = np.zeros((Nt + 1, self.N_dim, self.N_ens))
        #     r = np.zeros((Nt + 1, self.N_units, self.N_ens))
        #     if wash_t is not None:
        #         t1 = np.argmin(abs(t_b - wash_t[0]))
        #         Nt -= t1
        #         # Flag initialised
        #         self.initialised = True
        #         # Run washout phase in open-loop
        #         wash_model = interpolate(t, y, wash_t)
        #         washout = wash_obs - np.mean(wash_model, axis=-1)

        #         u_open, r_open = self.openLoop(washout)

        #         u[t1:t1 + self.N_wash + 1] = u_open
        #         r[t1:t1 + self.N_wash + 1] = r_open
        #         Nt -= self.N_wash

        #         # Run the rest of the time window in closed-loop
        #         if Nt > 0:
        #             # Store open-loop forecast
        #             self.reset_state(u=self.outputs_to_inputs(full_state=u_open[-1]), r=r_open[-1])
        #             u_close, r_close = self.closedLoop(Nt)
        #             u[t1 + self.N_wash + 1:] = u_close[1:]
        #             r[t1 + self.N_wash + 1:] = r_close[1:]
        # # Interpolate the final point if the upsample is not multiple of dt
        # if interp_flag:
        #     u[-1] = interpolate(t_b[-Nt:], u[-Nt:], t[-1])
        #     r[-1] = interpolate(t_b[-Nt:], r[-Nt:], t[-1])
        #     t_b[-1] = t[-1]

        # # update ESN physical and reservoir states, and store the history if requested
        # self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        # return u[1:], t_b[1:]



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
        
        
    def _sample_model_states(self, 
                              fm: Model, 
                              Nt_min: int,
                              std_phi: float = None,
                              std_alpha: Union[float, Dict[str, Union[float,  List[float]]]] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-parameter data generation for ESN training.

        Inputs:
            y_raw: List of raw observation arrays (each of shape Nt x Nq x 1)
            y_true: List of post-processed observation arrays (each of shape Nt x Nq x 1)
            Nt_min: Minimum number of time steps for training data
            train_params: Additional parameters for training data generation
        """

        
        # =================  Create ensemble for multi-parameter training data generation ===
        if not hasattr(self, 'L'):
            self.L = fm.m  # Use the forecast model ensemble size if not provided
        
        if std_phi is None:
            std_phi = np.std(fm.current_state[:fm.Nphi, :], axis=-1)  # Shape (Nphi,)
        if std_alpha is None:
            # get min, max for each parameter across the ensemble and store in dict
            std_alpha = {}
            for i, key in enumerate(fm.est_alpha):
                param = fm.current_state[fm.Nphi + i, :]
                std_alpha[key] = [min(param), max(param)]  # Shape (2,)

        def sample_ensemble(_psi0, _L):
            new_phi = mean_vector_to_ensemble(rng=fm.rng, 
                                              mean_vec=_psi0[:fm.Nphi], 
                                              std=std_phi, 
                                              m=_L,
                                              method='uniform')  # Shape (Nphi, _L)
            
            new_alpha = mean_vector_to_ensemble(rng=fm.rng, 
                                                mean_vec=_psi0[fm.Nphi:fm.Nphi+fm.Na], 
                                                std=std_alpha, 
                                                m=_L,
                                                method='uniform')  # Shape (Na, _L)
            
            return np.concatenate([new_phi, new_alpha], axis=0)  # Shape (Npsi, L)


        if fm.m != self.L:
            psi0 = np.mean(fm.current_state.copy(), axis=-1)  # Shape (Npsi,) 
            
            psi0_ens = sample_ensemble(psi0, self.L)  # Shape (Npsi, L)

            fm.update_history(psi=psi0_ens[np.newaxis, :, :], t=0., reset=True)


        # Forecast to post-transient
        Nt = int(np.round(fm.t_transient / fm.dt, fm.precision_t)) - 1
        psi, t = fm.time_integrate(Nt=Nt)
        fm.update_history(psi=psi, t=t, reset=True)
        # fm.close()

        y_L_model = fm.get_observable_hist()  # Nt x Nq x m (L)
        psi = psi[-1, :, :]  # Last forecast state (Npsi x m (L))

        #   Remove and replace fixed points ------------- #
        tol = 1e-1
        N_CR = int(round(fm.t_CR / fm.dt))
        range_y = np.max(np.max(y_L_model[-N_CR:], axis=0) - np.min(y_L_model[-N_CR:], axis=0), axis=0)
        idx_FP = (range_y < tol)
        psi0 = psi[-1, :, ~idx_FP]  # Nq x (m - #FPs)
        if len(np.flatnonzero(idx_FP)) / len(idx_FP) >= 0.2:
            # print(f'There are {len(np.flatnonzero(idx_FP))}/{len(idx_FP)} fixed points')
            
            # allowed_FPs = np.flatnonzero(idx_FP)[0:int(0.2 * len(idx_FP)) + 1]
            idx_FP[np.flatnonzero(idx_FP)[0]] = 0

            psi0 = psi[:, ~idx_FP]  # non-fixed point ICs (keeping one)

            new_psi0 = sample_ensemble(np.mean(psi0, axis=-1), len(np.flatnonzero(idx_FP)))  
            # new_psi0 = self.rng.multivariate_normal(np.mean(psi0, axis=-1),  
            #                                         np.cov(psi0),  len(np.flatnonzero(idx_FP)))
            
            psi0 = np.concatenate([psi0, new_psi0], axis=-1)  # Reconstructed ICs


        else:
            psi0 = psi


        #   Forecast fixed-point-free post-transient ensemble ------------- #
        fm.update_history(psi=psi0[np.newaxis, :, :], reset=True)
        psi, t = fm.time_integrate(Nt=Nt_min + N_CR)
        fm.update_history(psi=psi, t=t, reset=True)
        fm.close()

        return fm.get_observable_hist() # Nt_min+2N_CR x Nq x L


    def _prepare_reference_data(self, reference_data, Nt_min) -> Tuple[np.ndarray, np.ndarray]:
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

        # ========================= Generate model states and prepare reference data ========================= #

        y_model_L = self._sample_model_states(forecast_model.copy(), Nt_min, std_phi=std_phi, std_alpha=std_alpha) # Nt x Nq x L
        y_raw, y_true = self._prepare_reference_data(reference_data, Nt_min) # Lists of Nt x Nq x 1
        Nq = y_model_L.shape[1] # number of observed variables

        # # visualize model outputs vs observations
        # for yr, yt in zip(y_raw, y_true):
        #     plt.figure(figsize=(12, 2))
        #     plt.plot(yr[-Nt_min//2:, 0, 0], label='Raw observations')
        #     plt.plot(yt[-Nt_min//2:, 0, 0], label='True observations')
        #     plt.legend()
            
        #     plt.figure(figsize=(12, 2))
        #     plt.plot(y_model_L[-Nt_min//2:, 0, :], label='Model output')
        #     plt.legend()
        #     plt.show()



        # ========================= Create training data ========================= #
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
            
            innovations_all, model_bias_all = [], []
            for yr, yt in zip(y_raw, y_true):
                ym_L = self.__correlate_data(y_model_L, yr, Nt_min) # shape (Nt_min x Nq x L * augment_data_length)

                innovations = (yr - ym_L).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                innovations_all.append(innovations)
                if self.biased_observations:
                    model_bias = (yt - ym_L).transpose((2, 0, 1))  # shape (L x Nt x Nq)
                    model_bias_all.append(model_bias)

        # Combine the innovations (and model biases) #

        if not self.biased_observations:
            train_data = np.concatenate(innovations_all, axis=0)
            observed_idx = np.arange(Nq)
        else:
            model_bias_all = np.concatenate(model_bias_all, axis=0)
            train_data = dict(data=np.concatenate([model_bias_all, innovations_all], axis=2),
                              observed_idx=ensemble.Nq + np.arange(ensemble.Nq))

        # =============================== Save train_data dict ================================ #
        # Save key keywords
        train_data_dict = {key:val for key, val in self.config.items()} 
        # add the extra kwargs used to create the training data
        # train_data_dict.update({key: val for key, val in kwargs.items() if key not in train_data_dict.keys()})
        # add training data and observed indices
        train_data_dict.update(data=train_data,
                               observed_idx=observed_idx,
                               )
        return train_data_dict
        

        if filename is not None:
            save_to_pickle_file(filename, train_data)

        return train_data


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
        # If loading fails, or no filename provided, return None
        return None
    

    def _plot_training_dataset(self, plot_data):
        _L, _Nt, ndim = plot_data.shape

        if self.biased_observations:
            ncol = 2
        else:
            ncol = 1

        ndim = int(round(ndim // ncol))
        nrow = int(min(ndim, 10))
        t_data = np.arange(0, _Nt) * self.dt
        times = [0, self.t_train, self.t_train + self.t_val, t_data[-1]]

        Lis = np.sort(np.random.choice(_L, size=min(5, _L), replace=False))

        _, axs_all = plt.subplots(nrows=nrow * len(Lis), ncols=ncol, 
                                    figsize=(8*ncol+2, 1.*nrow*len(Lis)), sharex=True, 
                                    sharey='row', layout='constrained')
        #
        if not isinstance(axs_all, np.ndarray):
            axs_all = [axs_all]

        

        for row, Li in enumerate(Lis):
            axs = axs_all[nrow * row:nrow * (row + 1)]

            if self.biased_observations:
                axs = axs.T.flatten()
                if row == 0:
                    [axs[_ii].set(title=_ttl) for _ii, _ttl in zip([0, nrow], ['Model bias', 'Innovations'])]
            
            
            

            for kk, ax in enumerate(axs):
                if kk < nrow:
                    ax.plot(t_data, plot_data[Li, :, kk], lw=.8, color='k')
                else:
                    ax.plot(t_data, plot_data[Li, :, kk - nrow + ndim], lw=.8, color='k')


                [ax.axvspan(times[_ii], times[_ii+1], facecolor=_c, alpha=0.3, zorder=-100,
                            label=_lbl) for _ii, _c, _lbl in zip(range(3), ['orange', 'red', 'navy'],
                                                                ['Train', 'Validate', 'Test'])]
            axs[-1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.05), frameon=False, title=f'Li={Li}/{_L}', fontsize='small')


        plt.show()




    def visualize_hist(self, b=None, t=None):
        if b is None:
            b = self.get_bias()
        if t is None:
            t = self.hist_t[-len(b):]

        lbl = [f'$b_{{{i}}}$' for i in range(self.Nq)]

        # Plot the time evolution of the observables
        t_zoom = int(self.t_CR / self.dt)

        fig = plt.figure(figsize=(8, self.Nq+1), layout="constrained")
        plt.suptitle('Observables time evolution')
        axs = fig.subplots(self.Nq, 2, sharey='row', sharex='col')
        if self.Nq == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            ax[0].plot(t, b[:, ii])
            ax[1].plot(t[-t_zoom:], b[-t_zoom:, ii])
            ax[0].set(ylabel=lbl[ii])
            if ii == self.Nq-1:
                ax[0].set(xlabel='$t$', xlim=[t[0], t[-t_zoom]])
                ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])
        