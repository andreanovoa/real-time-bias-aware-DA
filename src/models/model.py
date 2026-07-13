# from scipy.integrate import solve_ivp
# from functools import partial
from copy import deepcopy

import numpy as np
import warnings
import matplotlib.pyplot as plt

from .integrator import IVPIntegrator, Integrator
from .history import HistoryTracker
from typeguard import typechecked, value


from typing import List, Optional, Type, Union

__all__ = ['Model']



# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model(object):
    r"""Base class for all forecast models.

    A `Model` couples three ingredients:

    - a **state history** ([`HistoryTracker`][romda.models.history.HistoryTracker])
      with pre-allocated storage of shape $(N_t, N, m)$ — time, state, ensemble
      members;
    - a **time-integration strategy**
      ([`Integrator`][romda.models.integrator.Integrator]) selected at construction;
    - the **observation operator** ``M`` mapping the (augmented) state to the
      observables.

    Physical models implement ``time_derivative(t, psi, **params)`` (continuous) or
    ``time_step(Nt)`` (discrete maps) and declare their estimable parameters in
    ``params`` with bounds in ``alpha_lims``.

    Parameters
    ----------
    psi0 : np.ndarray or list
        Initial state, shape $(N_\phi,)$ or $(N_\phi, m)$.
    dt : float
        Output time step.
    integrator_class : type[Integrator]
        Time-integration strategy (default `IVPIntegrator`).
    **kwargs
        Model-parameter overrides (any attribute defined by the child class).
    """

    params = []  # List of parameter names that can be varied in the model
    fixed_params = []
    extra_print_params = []
    governing_eqns_params = dict()

    t = 0.
    t_transient = 0.
    t_CR = 10 * 0.01
    

    Nq = 1
    alpha = None

    initialized = False
    results_folder = None

    @typechecked
    def __init__(self, 
                 psi0: Union[np.ndarray, List], 
                 dt: float, 
                 integrator_class: Type[Integrator] = IVPIntegrator, 
                 **kwargs):

        # ================= INITIALISE PHYSICAL MODEL ================== ##
        keys = list(kwargs.keys())
        [setattr(self, key, kwargs.pop(key)) for key in keys if hasattr(self, key)]

        if len(kwargs.keys()) > 1:
            print('Model key(s) {} not assigned'.format(kwargs.keys()))

        # ====================== SET INITIAL CONDITIONS ====================== ##

        # Ensure psi0 is ndarray with ndim=2
        if psi0 is None:
            raise ValueError("Initial state psi0 must be provided during Model initialization.")
        elif (isinstance(psi0, np.ndarray) and psi0.ndim == 1) or isinstance(psi0, list):
            psi0 = np.array([psi0]).T
            
        self.psi0 = psi0
        self.dt = dt
        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()

        # ========================== CREATE HISTORY ========================== ##
        # self._initial_capacity = int(self.t_transient / self.dt) # Initial capacity of history arrays
        # self._current_ti = 1  # Current time index in history arrays
        
        self.history = HistoryTracker()
        self.history._initial_capacity = int(self.t_transient / self.dt)*2 if self.t_transient > 0 else 1000
        self.update_history(psi=self.psi0[np.newaxis, :, :], 
                            t=np.array([0.]), 
                            reset=True)
        
        # ======================== SET RNG ================================== ##
        self.print_params = self.define_print_params()
        self.set_fixed_params()
        self.initialized = True

        # ================= INITIALISE INTEGRATOR STRATEGY ================== ##
        # The model holds an instance of the specific Integrator
        self.integrator = integrator_class(self)


    def update_history(self, psi: np.ndarray, t=None, reset=False, update_last_state=False):
        psi = self.__format_state(psi)
        if t is None:
            t = (np.arange(0, psi.shape[0]) * self.dt).round(self.precision_t) + self.current_time
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == psi.shape[0], f"Length of t ({t.size}) must match number of time steps in psi ({psi.shape[0]})."
        self.history.update_history(psi, t=t, reset=reset, update_last_state=update_last_state)
    #     self.update_history_aux(psi, reset=reset, update_last_state=update_last_state)
    
    # def update_history_aux(self, psi, reset=False, update_last_state=False):
    #     pass


    @property
    def state_labels(self):
        return  [f'$\\phi_{{{kk}}}$' for kk in range(self.Nphi)]
    
    @property
    def obs_labels(self):
        raise NotImplementedError("obs_labels property must be implemented in the child class.")


    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def alpha_lims(self):
        if not hasattr(self, '_alpha_lims'):
            self._alpha_lims = {key: (None, None) for key in sorted(self.params)}
        
        return self._alpha_lims
    
    @alpha_lims.setter
    def alpha_lims(self, value: dict):
        assert set(value.keys()) - set(self.params) == set(), f"Keys of alpha_lims must be a subset of {self.params}, but got {value.keys()}"
        if hasattr(self, '_alpha_lims'):
            self._alpha_lims.update(value)
        else:
            self._alpha_lims = value
            if len(self._alpha_lims) < len(self.params):
                missing_keys = set(self.params) - set(self._alpha_lims.keys())
                self._alpha_lims.update({key: (None, None) for key in missing_keys})


    @property
    def alpha_labels(self):
        if not hasattr(self, '_alpha_labels'):
            self._alpha_labels = {f'$\\alpha_{ii}$': val for ii, val in enumerate(sorted(self.params))}
        return self._alpha_labels
    
    @alpha_labels.setter
    def alpha_labels(self, value: dict):
        assert set(list(value.keys())) - set(sorted(self.params)) == set(), f"Keys of alpha_labels must be a subset of {sorted(self.params)}, but got {value.keys()}"
        if hasattr(self, '_alpha_labels'):
            self._alpha_labels.update(value)
        else:
            self._alpha_labels = value
            if len(self._alpha_labels) < len(self.params):
                missing_keys = set(sorted(self.params)) - set(self._alpha_labels.keys())
                self._alpha_labels.update({f'$\\alpha_{ii}$': val for ii, val in enumerate(missing_keys)})


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
        return self.history.current_state

    @property
    def current_time(self):
        return self.history.current_time
    
    @property
    def filename(self):
        if not hasattr(self, '_filename'):
            suffix = ''
            for key, val in self.alpha0.items(): 
                if val != getattr(self.__class__, key):
                    if np.log10(abs(val)) < -3:
                        suffix += f'_{key}{val:.2e}'
                    else:
                        suffix += f'_{key}{val}'
            if len(suffix) == 0:
                suffix = '_default'
            self._filename = f"{self.name}{suffix}"

        return self._filename
    
    @filename.setter
    def filename(self, value):
        self._filename = value
            

    def __format_state(self, psi: np.ndarray) -> np.ndarray:
        """Ensure psi has the correct shape (Nt, N, m) for history storage.
        Parameters
        ----------
        psi
            State array to format.
        Returns
        -------
            Formatted state array with shape (Nt, N, m).
        """ 
        if psi.ndim == 1:
            psi = psi[np.newaxis, :, np.newaxis]  # (N,) -> (1, N, 1)
        elif psi.ndim == 2:
            psi = psi[np.newaxis, :, :]  # (N, m) -> (1, N, m)
        elif psi.ndim == 3:
            pass  # Already in correct shape (Nt, N, m)
        else:
            raise ValueError(f"State array psi has invalid number of dimensions: {psi.ndim}={psi.shape}. Expected 1, 2, or 3.")
        return psi

    def define_print_params(self):
        return [*self.params, *self.extra_print_params]

    @property
    def psi0(self):
        return self._psi0
    
    @psi0.setter
    def psi0(self, value):

        if hasattr(self, '_psi0'):
            warnings.warn(f"psi0 is being re-assigned. Previous shape {self._psi0.shape},"
                          f" new shape {np.array(value).shape}. This is not recommended.", UserWarning)
            
            if isinstance(value, np.ndarray) and value.ndim == 1:
                value = np.array([value]).T
            self._psi0 = np.array(value)

        self._psi0 = np.array(value)

    @property
    def alpha0(self):
        return self._alpha0

    @alpha0.setter
    def alpha0(self, dict_params):
        """
        Set the initial input parameters dictionary. Initial parameters are unchanged during the model run. 
        """
        if hasattr(self, '_alpha0'):
            raise AttributeError("alpha0 is read-only and cannot be modified after initialization.")
        self._alpha0 = dict_params

    @property
    def dt(self):
        return self._dt


    @dt.setter
    def dt(self, value):
        """Setter for the time step."""
        if value <= 0:
            raise ValueError("Time step must be positive.")
        self._precision_t = int(np.ceil(-np.log10(value) + 2))  # Set precision based on dt
        # print(f'Setting time step dt={value} with precision_t={self._precision_t}')
        self._dt = np.round(value, self.precision_t)

    @property
    def precision_t(self):
        if not hasattr(self, '_precision_t'):
            if not hasattr(self, '_dt'):
                raise AttributeError("dt must be set before accessing precision_t.")
        return self._precision_t
    
    
    @property
    def dt_step(self):
        """ Getter for the integrator time step. This is the time step used by the integrator. 
            Some models (specifically Discrete models) may have a different time step for output (dt) and for integration.
        """
        return self.dt

    @property
    def Nphi(self):
        return len(self.psi0)

    @property
    def Na(self):
        if isinstance(self.ensemble, dict):
            return self.ensemble.get('Na', 0)
        else:
            return 0

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    @property
    def m(self): 
        return self.hist.shape[-1]

    
    def set_fixed_params(self):
        fixed_params = dict((key, getattr(self, key)) for key in self.fixed_params)
        # Create an instance-level dict: the class-level default must not be mutated,
        # otherwise fixed parameters leak across different Model subclasses.
        self.governing_eqns_params = {**self.governing_eqns_params, **fixed_params}


    def create_long_timeseries(self, Nt=None):
        if Nt is None:
            Nt = int(self.t_transient * 10 / self.dt)
        state, t = self.time_integrate(Nt=Nt)
        self.update_history(state, t)
        self.close()

    
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


    def copy(self):
        return deepcopy(self)


    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, :self.Nq, :]
        else:
            return self.hist[-Nt:, :self.Nq, :]

    def get_observable_hist(self, Nt=0, **kwargs):
        return self.get_observables(Nt, **kwargs)


    def print_parameters(self, show_header=True):
        if show_header:
            print('\n ------------------ Model Parameters ------------------ ')
        print(f'\t Model class name: {self.__class__.__name__}')
        for key in sorted(self.print_params):
            val = getattr(self, key)
            print(f'\t {key} = {val:.6f}' if isinstance(val, float) else f'\t {key} = {val}')

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##

    @property
    def M(self):
        if not hasattr(self, '_M'):
            self.M = None # This will trigger the setter to create the default M matrix
        return self._M
    
    @M.setter
    def M(self, M=None):
        if M is None:
            # M matrix is constructed by horizontally stacking a zero matrix of shape (Nq, Na + Nphi)
            # and an identity matrix of shape (Nq, Nq)
            M = np.hstack((np.zeros([self.Nq, self.Na + self.Nphi]), 
                           np.eye(self.Nq)))
        else:
            assert M.shape == (self.Nq, self.N), f"Shape of M must be ({self.Nq, self.N}), but got {M.shape}"

        self._M = M


    @property
    def Ma(self):
        if not hasattr(self, '_Ma'):
            self._Ma = np.hstack((np.zeros([self.Na, self.Nphi]),
                                            np.eye(self.Na),
                                            np.zeros([self.Na, self.Nq])))
        return self._Ma

    # ------------------------- Functions for update/initialise the model --------------------------- #

    def reset_model(self, psi0=None, **kwargs):

        if psi0 is None:
            psi0 = self.current_state

        Model.__init__(self, psi0=psi0, **kwargs)


    def modify_settings(self):
        pass


    def close(self):
        self.integrator.close()

    @property
    def ensemble(self) -> Union[dict, bool]:
        """Public accessor for ensemble configuration (preferred over touching _ensemble_config)."""
        return getattr(self, '_ensemble_config', False)
    
    @ensemble.setter
    def ensemble(self, config: dict):
        """Setter for ensemble configuration."""
        self._ensemble_config = config
    

    @property
    def est_alpha(self):
        if isinstance(self.ensemble, dict):
            return self.ensemble.get('est_alpha', [])
        else:
            return []
        
    @est_alpha.setter
    def est_alpha(self, value):
        if not isinstance(self.ensemble, dict):
            raise AttributeError("Cannot set est_alpha when ensemble is not configured.")
        self._ensemble_config['est_alpha'] = value


    def get_alpha(self, psi=None):
        # if not isinstance(self.ensemble, dict):
        #     return [self.alpha0.copy()]
            
        if psi is None:
            psi = self.current_state

        if psi.shape[0] == self.Nphi:
            # print('using the same get_alpha')
            return [self.alpha0.copy()] * psi.shape[-1]

        # ensure psi has members on last axis
        if psi.ndim == 1:
            psi = psi[:, np.newaxis]

        alpha_list = []
        for mi in range(psi.shape[-1]):
            alph = self.alpha0.copy()
            alph.update(zip(self.est_alpha, psi[-self.Na:, mi]))
            alpha_list.append(alph)

        return alpha_list


    # ================= Main Time Integration Method ================= #

    def time_integrate(self, Nt=100, averaged=False):
        """
        Delegates the integration task to the currently configured Integrator strategy.
        The Model's time_integrate is now just a wrapper for the Strategy's advance method.
        Some Models may override this method if they need special handling.
        Parameters
        ----------
        Nt
            number of forecast steps
        averaged : bool
            if true, each member in the ensemble is forecast individually. If false,
            the ensemble is forecast as a mean, i.e., every member is the mean forecast.
        Returns
        -------
            psi: forecasted state (Nt x N x m)
            t: time of the propagated psi
        """
        return self.integrator.advance(Nt=Nt, averaged=averaged, alpha=self.get_alpha())
    



    # ============================== Visualization methods ============================== #
    def visualize_history(self):
        self.visualize_state_hist()
        self.visualize_observable_hist()
        self.visualize_spatiotemporal_hist()



    def visualize_state_hist(self, psi=None, t=None, max_modes=10, t_zoom=None):
        if psi is None:
            psi = self.hist[:, :self.Nphi]
        if t is None:
            t = self.hist_t[-len(psi):]

        lbl = self.state_labels

        # Plot the time evolution of the observables
        if t_zoom is None:
            t_zoom = int(self.t_CR / self.dt)
        nrows = min(self.Nphi, max_modes)

        fig = plt.figure(figsize=(8, nrows+1), layout="constrained")
        plt.suptitle('State time evolution')
        axs = fig.subplots(nrows, 2, sharey='row', sharex='col')
        if nrows == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            # if complex, plot real part and imag part in the same axis
            ax[0].plot(t, psi[:, ii].real,  label='Real part')
            ax[1].plot(t[-t_zoom:], psi[-t_zoom:, ii].real,  label='Real')
            if np.iscomplexobj(psi[:, ii]):
                ax[0].plot(t, psi[:, ii].imag, label='Imag part')
                ax[1].plot(t[-t_zoom:], psi[-t_zoom:, ii].imag, label='Imag')
                ax[1].legend(fontsize='x-small', ncol=2)
            ax[0].set(ylabel=lbl[ii])
            if ii == nrows-1:
                ax[0].set(xlabel='$t$', xlim=[t[0], t[-t_zoom]])
                ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])
    

    def visualize_observable_hist(self, y=None, t=None, t_zoom=None):
        if y is None:
            y = self.get_observable_hist()
        if t is None:
            t = self.hist_t[-len(y):]

        lbl = self.obs_labels

        # Plot the time evolution of the observables
        if t_zoom is None:
            t_zoom = int(self.t_CR / self.dt)

        fig = plt.figure(figsize=(8, self.Nq+1), layout="constrained")
        plt.suptitle('Observables time evolution')
        axs = fig.subplots(self.Nq, 2, sharey='row', sharex='col')
        if self.Nq == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            ax[0].plot(t, y[:, ii])
            ax[1].plot(t[-t_zoom:], y[-t_zoom:, ii])
            ax[0].set(ylabel=lbl[ii])
            if ii == self.Nq-1:
                ax[0].set(xlabel='$t$', xlim=[t[0], t[-t_zoom]])
                ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])
    
    
    def visualize_spatiotemporal_hist(self, **kwargs):
        pass 


    def visualize_config(self):
        pass

