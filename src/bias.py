
from data_assimilation import EnSRKF
import numpy as np
from copy import deepcopy
from history import HistoryTracker
from integrator import Integrator
from model import Model
from typing import Optional, Tuple
    
from plotting import categorical_cmap
import matplotlib.pyplot as plt


class Bias:
    """
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
    """

    upsample = 1
    L = 1
    augment_data = False

    forecaster_type = None  # This should be set in child classes to specify the expected type of the forecaster model, e.g., ESN_model for ESN_bias.
    bayesian_update = False         # Default to not perform bayesian update to state
    biased_observations = False  # Whether observations are biased or not
    force_retrain = False
    
    keys_to_print = ['bayesian_update', 'upsample', 'biased_observations', 'L', 'augment_data_length']
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
        

        # ================== Initialize Forecaster & HISTORY ================= ##
        
        self.init_forecaster(**kwargs)

        bias_state = self.initialize_bias_state

        assert bias_state.shape[-2] == self.N, f"Bias state shape {bias_state.shape} does not match expected (Nt, N = {self.N}, Nens)."
        self.update_history(bias_state, t=t, reset=True)
 

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def N_ens(self):
        if not hasattr(self, '_N_ens'):
            return 1
        return self._N_ens
    
    @N_ens.setter
    def N_ens(self, value):
        if value <= 0:
            raise ValueError("Number of ensemble members must be positive.")
        if hasattr(self, 'history'):
            Warning("Changing N_ens after initialization may lead to inconsistencies in the history.")
        self._N_ens = value

    @property
    def augment_data_length(self):
        augment_data = self.augment_data
        if augment_data:
            if isinstance(augment_data, int) and augment_data > 1:
                return augment_data
            return 2
        return 1

    @property
    def initialize_bias_state(self):
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
    def N(self):
        return self.N_dim + self.N_hidden

    @property
    def N_hidden(self):
        # Number of hidden units in the bias model, e.g., for ESN bias model. This is added to the state dimension N_dim to get the total state dimension N.
        return 0
    

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
    
    def washout_phase(self, d_wash, t_wash, **kwargs) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Optional method to initialize the bias model if needed, e.g., by running a washout phase with given data.
        By default, does nothing, but can be implemented in child classes if needed.
        """
        return None

    @property
    def washout_data(self):
        """
        Returns the washout data used for initializing the bias model, which is typically obtained from the washout phase using the validation data. This property can be used to access the washout data for further processing or analysis.

        Returns:
            Tuple of (washout_data, washout_time) where:
                - washout_data: np.ndarray - Washout data used for initializing the bias model.
                - washout_time: np.ndarray - Time points corresponding to the washout data.
        """
        return getattr(self, '_washout_data', (None, None))
    

    
    @washout_data.setter
    def washout_data(self, value):
        assert isinstance(value, tuple) and len(value) == 2, "Washout data must be a tuple of (washout_data, washout_time)."
        self._washout_data = value
        
    def _format_state(self, b):
        """
        Ensure b has shape (nt, nb, nens)
        """
        if b.ndim == 3:
            return b # already (nt, nb, nens)
        elif b.ndim == 1: 
            b_repeated = np.repeat(b[:, np.newaxis], self.N_ens, axis=1)  # (nb,) -> (nb, nens)
            return b_repeated[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) Add extra dimension for time
        
        elif b.ndim == 2 and (b.shape[-1] == self.N_ens or b.shape[0] == self.N_dim):
            return b[np.newaxis, :, :]  # (nb, nens) -> (1, nb, nens) # Add extra dimension for time
        elif b.ndim == 2 and b.shape[-1] != self.N_ens:
            return np.repeat(b[:, :, np.newaxis], self.N_ens, axis=2)  # (nt, nb) -> (nt, nb, nens)
        else:        
            raise AssertionError(f'b must have 1, 2 or 3 dimensions, got {b.ndim}=({b.shape})')


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
        return self.history.current_state.copy()

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

    def time_integrate(self, Nt):
        return self.integrator.advance(Nt=Nt)
    

    def update_history(self, state, t=None, reset=False, update_last_state=False, **kwargs):
        state = self._format_state(state) # Ensure shape (nt, nstate, nens)

        # Ensure time array matches nt
        if t is None:
            t = (np.arange(state.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == state.shape[0], f"Length of t ({t.size}) must match number of time steps in state ({state.shape[0]})."
        
        if update_last_state:
            assert state.shape[0] == 1, "When update_last_state is True, state must have only one time step (shape[0] == 1)."

        self.history.update_history(state, t=t, reset=reset, update_last_state=update_last_state)
        self.update_history_aux(state=state, reset=reset, update_last_state=update_last_state, **kwargs)
    
    def update_history_aux(self, **kwargs):
        """Auxiliary method to update any additional history attributes in child classes if needed."""
        pass

    
    def update_state_from_innovation(self, input_innovation):
        """
        Optional method to perform a Bayesian update to the state using the bias model. This can be implemented in child classes if needed, e.g., for ESN bias model.
        By default, does nothing, but can be implemented in child classes if needed.

        Args:
            input_data: The input data for the Bayesian update.
            method: The method to use for the Bayesian update.
            **kwargs: Additional keyword arguments that may be needed for the Bayesian update.
        """
        input_innovation = self._format_state(input_innovation) # Ensure shape (nt, nstate, nens)
        assert input_innovation.shape[0] == 1, "Input innovation must have only one time step (shape[0] == 1) for state_from_innovation method."

        forecast_state = self.current_state


        if self.bayesian_update:
            mean_innovation = np.mean(input_innovation[0], axis=-1)  # Average innovation across ensemble (obs_dim, Nens) -> (obs_dim,)
            # cov_innovation = (inn_uncertainty * np.max(np.abs(mean_innovation)))**2 * np.eye(mean_innovation.shape[0])  # Diagonal covariance of innovation (obs_dim, obs_dim)
            # cov_innovation = np.cov(input_innovation[0], rowvar=True)  # Full covariance of innovation (obs_dim, obs_dim)
            # cov_innovation = np.atleast_2d(cov_innovation)
            conv_inn = input_innovation[0] - mean_innovation[:, np.newaxis]  # Centered innovations (obs_dim, Nens)
            cov_innovation = np.cov(conv_inn, rowvar=True)  # Full covariance of centered innovations (obs_dim, obs_dim)

            updated_state = self.DA_method(Af=forecast_state, d=mean_innovation, Cdd=cov_innovation)
        else:
            updated_state = forecast_state.copy()
            mean_innovation = np.mean(input_innovation[0], axis=-1, keepdims=True)  # Average innovation across ensemble (obs_dim, Nens) -> (obs_dim, 1)
            if input_innovation.shape[-1] == 1:  # assign same mean innovation to all ensemble members
                updated_state[self.observed_idx, :] = np.repeat(mean_innovation, self.N_ens, axis=-1)

            else:# input_innovation.shape[1] != forecast_state.shape[0] => resample
                cov_innovation = np.cov(input_innovation[0], rowvar=True)
                mean_innovation = mean_innovation.flatten()
                cov_innovation = np.atleast_2d(cov_innovation)
                resampled_innovation = np.random.multivariate_normal(mean_innovation, cov_innovation, size=self.N_ens).T  # Resample innovations for each ensemble member (obs_dim, Nens)
                updated_state[self.observed_idx, :] = resampled_innovation
                
            # Run 1 open loop step to propagate the updated observed components to the bias components if needed, e.g., for ESN bias model.
            if hasattr(self, 'forecaster') and self.forecaster is not None:
                raise(NotImplementedError("Time integration after state update is not implemented yet."))
                esn = self.forecaster # type: ESN_model
                updated_state = esn.step(updated_state)  

        return updated_state


    @property
    def DA_method(self):
        if not self.bayesian_update:
            raise ValueError("Why is this being accessed when bayesian_update is False?")
        elif not hasattr(self, '_DA_method'):
            observation_operator = np.zeros((len(self.observed_idx), self.N))
            observation_operator[:, self.observed_idx] = np.eye(len(self.observed_idx))
            self._DA_method = EnSRKF(M=observation_operator)
            print(f"Initialized DA method {self._DA_method.__class__.__name__} for Bayesian update in bias model. M shape: {observation_operator.shape}")

        return self._DA_method
    
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

    def close(self):
        """Close any resources used by the bias model, e.g., forecaster model."""
        if hasattr(self, '_forecaster') and hasattr(self._forecaster, 'close'):
            self._forecaster.close()






    # ================= Visualization methods ================== #
    def visualize_bias_and_innovations(self, state=None, t=None, plot_members=True, ylims=None):
        if state is None:
            state = self.hist

        inn = self.get_innovations(state=state, mean=not plot_members)
        b = self.get_bias(state=state, mean=not plot_members)

        if t is None:
            t = self.hist_t[-len(inn):]

        lbls =( [f'inn_{ii}' for ii in range(inn.shape[1])], 
              [f'b_{ii}' for ii in range(b.shape[1])])
        ttls = ['Innovations', 'Bias']


        t_mid = len(t) - len(t)//8
        lims = [[0, t_mid ],  [t_mid, len(t)-1]  ]

        Nq = inn.shape[1]
        nens = inn.shape[-1]
        cols = categorical_cmap(nc=2, nsc=nens, continuous=False)
        cols = [cols[ii * nens:(ii + 1) * nens] for ii in range(2)]
        ci = 0
        # Plot the time evolution of the observables
        for y, lbl, ttl, cs in zip([inn, b], lbls, ttls, cols):

            fig = plt.figure(figsize=(8, Nq+1), layout="constrained")
            plt.suptitle(f'{ttl} time evolution')

            axs = fig.subplots(Nq, 2, sharey='row', sharex='col')
            if self.Nq == 1:
                axs = [axs]


            for ii, ax in enumerate(axs):
                for jj, lim in enumerate(lims):    
                    lines = ax[jj].plot(t[lim[0]:lim[1]], y[lim[0]:lim[1], ii])
                    for line, color in zip(lines, cs):
                        line.set_color(color)
                    if ii == self.Nq-1:
                        ax[jj].set(xlabel='$t$', xlim=[t[lim[0]], t[lim[1]]])
                
                ax[0].set(ylabel=lbl[ii])
                if ylims is not None:
                    ax[0].set(ylim=ylims)
    
            ci += 1


def plot_train_data(truth, bias_data, t_CR):

    L, _, _ = bias_data['data'].shape

    Nt = int(t_CR / truth.dt)
    Nq = truth.y_true.shape[1]

    # Build a common valid time window and select the segment before first observation.
    n_common = min(
        len(truth.t_true),
        truth.y_true.shape[0],
        truth.b_true.shape[0],
        bias_data['y_model'].shape[0],
        bias_data['data'].shape[1],
    )


    t0 = bias_data['y_model'].shape[0]

    yt = truth.y_raw[-t0:-t0+Nt]
    bt = truth.b_true[-t0:-t0+Nt]
    tt = truth.t_true[-t0:-t0+Nt]


    yr = bias_data['y_model'][:Nt].transpose(2, 0, 1)
    br = bias_data['data'][:, :Nt, :Nq]

    if len(tt) == 0:
        raise ValueError('Selected plotting window is empty. Check t_CR and training_data dimensions.')


    RS = []
    for ii in range(L):
        RS.append(np.linalg.norm(br[ii][:, 0]) / np.sqrt(len(yt)))

    RS = np.asarray(RS, dtype=float)
    true_RMS = np.linalg.norm(bt[:, 0]) / np.sqrt(len(yt))

    # Plot training data (single row) --------------------------
    fig = plt.figure(figsize    =[12, 2.7], layout='constrained')
    axs = fig.subplots(1, 2)
    
    # Robust color mapping: clip outliers; if RMS are nearly equal, force distinct member colors.
    if np.ptp(RS) < 1e-12:
        color_values = np.linspace(0.0, 1.0, L)
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
        cbar_extend = 'neither'
        cbar_title = 'Member'
    else:
        lo, hi = np.percentile(RS, [5, 95])
        if np.isclose(lo, hi):
            lo = float(np.min(RS))
            hi = float(np.max(RS))
        color_values = np.clip(RS, lo, hi)
        norm = Normalize(vmin=float(lo), vmax=float(hi))
        cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
        cbar_extend = 'both'
        cbar_title = '$\\mathrm{RMS}$'

    xlim = [tt[0], tt[-1]]

    axs[0].plot(tt, yt[:, 0], color='silver', linewidth=6, alpha=.8)
    axs[1].plot(tt, bt[:, 0], color='silver', linewidth=4, alpha=.8)

    for ii in range(L):
        clr = cmap.to_rgba(color_values[ii])
        axs[0].plot(tt, yr[ii][:, 0], color=clr, alpha=0.7)
        axs[1].plot(tt, br[ii][:, 0], color=clr, alpha=0.7)

    axs[0].legend(['Truth'], bbox_to_anchor=(0., 0.25), loc='upper left')
    axs[1].legend(['True RMS $={0:.3f}$'.format(true_RMS)], bbox_to_anchor=(0., 0.25), loc='upper left')
    axs[0].set(xlabel='$t$', ylabel='$\\eta$', xlim=xlim)
    axs[1].set(xlabel='$t$', ylabel='$b$', xlim=xlim)

    clb = fig.colorbar(cmap, ax=axs, orientation='vertical', extend=cbar_extend)
    clb.ax.set_title(cbar_title)

