
import numpy as np
from copy import deepcopy
from history import HistoryTracker
from integrator import Integrator
from model import Model

    


class Bias:
    '''
    Docstring for Bias
    Base class for bias models used in data assimilation.
    Attributes:
        upsample (int): Factor to upsample bias model time step relative to data assimilation cycle
        L (int): Length of bias state vector
        augment_data (bool): Whether to augment data with bias information
        bayesian_update (bool): Whether to perform Bayesian update to state
        biased_observations (bool): Whether observations are biased
    Methods:
        __init__: Initializes the Bias model with given parameters
        get_bias: Extracts bias from the full state
        get_innovations: Extracts innovations from the full state
        time_integrate: Advances the bias state in time
        update_history: Updates the history of the bias state
    Properties:
        name: Name of the bias model class
        bias_idx: Indices of the bias components in the state vector
        forecaster: The forecasting model used for bias prediction
        history: History object storing past bias states
        integrator: Integrator used by the bias model
    '''

    upsample = 1
    L = 1
    augment_data = False

    forecaster_type = None  # This should be set in child classes to specify the expected type of the forecaster model, e.g., ESN_model for ESN_bias.
    bayesian_update = False         # Default to not perform bayesian update to state
    biased_observations = False  # Whether observations are biased or not

    keys_to_print = ['bayesian_update', 'upsample', 'biased_observations']
    extra_keys_to_print = []

    def __init__(self, innovation, t, dt, **kwargs):


        # ===================== ASSIGN PROVIDED KWARGS ======================= ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        self.keys_to_print += self.extra_keys_to_print

        # ================== Setup dimensions ================= ##

        self.precision_t = int(-np.log10(dt)) + 2
        self.dt = dt
        self.Nq = self._format_state(innovation).shape[1] 
        
        bias_state = self.build_state(innovation)
        assert bias_state.shape[1] == self.N_dim, f"Bias state shape {bias_state.shape} does not match expected N_dim = {self.N_dim}."

        # ================== Initialize Forecaster & HISTORY ================= ##

        forcaster_dict = {'state':bias_state, **kwargs}
        
        self.init_forecaster(**forcaster_dict)
        self.update_history(bias_state, t=t, reset=True)
 

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def N_ens(self):
        return self.history.current_state.shape[-1]
    
    @property
    def initialize_bias_state(self, N_ens):
        """
        Only used at initialization. If the forecaster is a model, this shoiuld be hanfdled by the child class.
        """
        if self.biased_observations:
            return np.zeros((self.Nq, self.N_ens))
        else:
            return np.zeros((2* self.Nq, self.N_ens))

    @property
    def forecaster(self):
        assert hasattr(self, '_forecaster'), 'Forecaster not initialized yet.'
        return self._forecaster # type: ignore #should be an instance of a forecaster model, e.g., ESN_model
    
    @forecaster.setter
    def forecaster(self, obj):
        if self.forecaster_type is None:
            raise ValueError('Child class must specify forecaster_type.')
        assert isinstance(obj, self.forecaster_type), f'Forecaster must be an instance of {self.forecaster_type}, but got {type(obj)}.'
        self._forecaster = obj

    @property
    def history(self) -> HistoryTracker:
        return self._forecaster.history  # type: ignore
    
    @property
    def integrator(self) -> Integrator:
        """
        This is the integrator used by the model of the bias. E.g., DiscreteIntegrator if using ESN_model as forecaster.
        """
        return self._forecaster.integrator  # type: ignore
    
    @property
    def N_dim(self):
        if not self.biased_observations:
            return self.Nq
        else:
            return 2 * self.Nq

    @property
    def bias_idx(self):
        return np.arange(self.Nq)

    @property
    def observed_idx(self):
        if self.biased_observations:
            return self.bias_idx + self.Nq
        else:
            return self.bias_idx

    def init_forecaster(self, **kwargs):
        """
        .....
        """
        raise NotImplementedError('Bias child classes must implement _init_forecaster() method.')

    def state_derivative(self):
        """
        Returns the derivative of the bias state, which is used for time integration.
        This is computed by the forecaster model.
        """
        raise NotImplementedError('Bias child classes must implement state_derivative property, typically computed by the forecaster model.')
    
    def _format_state(self, b):
        """
        Ensure b has shape (nt, nb, nens)
        """
        if b.ndim == 3:
            return b # already (nt, nb, nens)
        elif b.ndim == 1: 
            b_repeated = np.repeat(b[:, np.newaxis], self.N_ens, axis=1)  # (nb,) -> (nb, nens)
            return b_repeated[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) Add extra dimension for time
        
        elif b.ndim == 2 and b.shape[-1] == self.N_ens:
            return b[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) # Add extra dimension for time
        elif b.ndim == 2 and b.shape[-1] != self.N_ens:
            return np.repeat(b[:, :, np.newaxis], self.N_ens, axis=2)  # (nt, nb) -> (nt, nb, nens)
        else:        
            raise AssertionError(f'b must have 1, 2 or 3 dimensions, got {b.ndim}=({b.shape})')


    def build_state(self, innovation, model_bias=None) -> np.ndarray:
        """
        Build the full bias state from innovations and model bias (if applicable)
        """
        innovation = self._format_state(innovation)

        if self.biased_observations:
            if model_bias is None:
                model_bias = innovation.copy()
            else:
                model_bias = self._format_state(model_bias)
            
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
    def current_bias(self):
        """Returns the current bias computed from the current state."""
        return self.get_bias(state=self.current_state)[0, :, :]

    @property
    def current_innovations(self):
        """Returns the current innovations computed from the current state."""
        return self.get_innovations(state=self.current_state)[0, :, :]


    def get_bias(self, state, mean=False):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        state = self._format_state(state)
        return state[:, self.bias_idx, :]
    

    def get_innovations(self, state, mean=False):
        if mean:
            state = np.mean(state, axis=-1, keepdims=True)

        state = self._format_state(state)
        return state[:, self.observed_idx, :]

    
    def get_bias_hist(self, mean=False):
        return self.get_bias(state=self.hist, mean=mean)

    def time_integrate(self, Nt, y=None, wash_t=None, wash_obs=None):
        return self.integrator.advance(Nt=Nt)
    

    def update_history(self, b, t=None, reset=False, update_last_state=False, **kwargs):
        b = self._format_state(b)

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(b.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == b.shape[0], f"Length of t ({t.size}) must match number of time steps in b ({b.shape[0]})."
        
        self.history.update_history(b, t=t, reset=reset, update_last_state=update_last_state)
        self._update_history_aux(reset=reset, update_last_state=update_last_state, **kwargs)
    
    def _update_history_aux(self, **kwargs):
        """Auxiliary method to update any additional history attributes in child classes if needed."""
        pass


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
        if isinstance(self.forecaster, Model):
            self.forecaster.print_parameters(show_header=False) 


    def copy(self):
        return deepcopy(self)

