# from scipy.integrate import solve_ivp
# from functools import partial
from copy import deepcopy

import numpy as np
import warnings

from integrator import IVPIntegrator
import matplotlib.pyplot as plt



class HistoryTracker:
    """ Mixin class to add history tracking functionality to models.
    """
    
    # ________________________ History accessors ________________________ #

    @property
    def hist(self):
        """Returns only the valid (non-empty) portion of the history buffer."""
        return self._hist[:self.current_ti]

    @property
    def hist_t(self):
        """Returns only the valid portion of the time history."""
        return self._hist_t[:self.current_ti]
    
    @property
    def capacity(self):
        return self._hist_t.shape[0]

    @property
    def current_state(self):
        return self.hist[self.current_ti - 1]

    @property
    def current_time(self):
        return self.hist_t[self.current_ti - 1]
    
    @property
    def current_ti(self):
        return self._ti

    @current_ti.setter
    def current_ti(self, value: int):
        self._ti = value


    def _reset_history(self, psi_reset, t_reset):
        """Resets the history arrays to the provided psi_reset and t_reset.
        Args:
            psi_reset: New state history to set (Nt, N, m)
            t_reset: New time history to set (Nt,)
        """

        Nt = max(psi_reset.shape[0], self._initial_capacity)

        # Initialize the history arrays
        self._hist = np.empty((Nt, psi_reset.shape[1], psi_reset.shape[2]))
        self._hist_t = np.empty((Nt,))
        # Store the reset history

        self._hist[:psi_reset.shape[0]] = psi_reset
        self._hist_t[:t_reset.shape[0]] = t_reset
        self.current_ti = psi_reset.shape[0]

    def _reset_last_state(self, psi_new, t=None):
        """Resets only the last state in the history arrays to the provided psi_new and t."""
        if psi_new.shape[0]> 1:
            raise ValueError("psi_new must contain only one time step to reset the last state.")
        else:
            self._hist[self.current_ti - 1] = psi_new[0]
        if t is not None:
            self._hist_t[self.current_ti - 1] = t[-1]


    def update_history(self, psi: np.ndarray, t: np.ndarray, reset=False, update_last_state=False):
        assert psi.shape[0] == t.shape[0], f"Length of t ({t.shape}) must match number of time steps in psi ({psi.shape})."
        if reset: # Reset the full history 
            self._reset_history(psi, t)
        
        elif update_last_state: # Update only the last state in history
            self._reset_last_state(psi, t=t)
        else:
            t0 = self.current_ti
            t1 = t0 + psi.shape[0]

            if t1 > self.capacity:
                # print(f'History capacity exceeded: {t1} > {self.capacity}. Increasing history size.'
                #       f' Current time index: {self.current_ti}.'
                #       f' psishape: {psi.shape}')
                self._increase_hist_size(Nt=psi.shape[0]*10, Ndim=psi.shape[1])

            self._hist[t0:t1] = psi
            self._hist_t[t0:t1] = t
            self.current_ti = t1


    def _increase_hist_size(self, Nt=None, Ndim=None):
        """
        With this I avoid np.concatenate every time I want to add new data to history. 
        """
        
        if Nt is None: 
            Nt = self._initial_capacity

        new_capacity = self.capacity + Nt

        print(f'Increasing history size from {self.capacity} to {new_capacity} time steps.')

        # Create new, larger arrays

        new_hist = np.empty((new_capacity, self._hist.shape[1], self._hist.shape[2]))
        new_hist_t = np.empty((new_capacity,))

        # Copy existing data (expensive operation, but done rarely)
        new_hist[:self.capacity] = self._hist
        new_hist_t[:self.capacity] = self._hist_t

        # Update attributes
        self._hist = new_hist
        self._hist_t = new_hist_t




# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model(object):
    """ Parent Class with the general model properties and methods definitions.
    """

    alpha_labels: dict = dict()
    alpha_lims: dict = dict()

    state_labels: list = []
    fixed_params = []
    extra_print_params = []
    governing_eqns_params = dict()

    t = 0.
    t_transient = 0.
    t_CR = 10 * 0.01

    Nq = 1
    seed = 6
    alpha = None
    filename = ''

    initialized = False


    def __init__(self, integrator_class=IVPIntegrator, psi0=None, **kwargs):

        # ================= INITIALISE PHYSICAL MODEL ================== ##
        model_dict = kwargs.copy()
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, model_dict.pop(key))


        if len(model_dict.keys()) > 1:
            print('Model {} not assigned'.format(model_dict.keys()))

        # ====================== SET INITIAL CONDITIONS ====================== ##

        # Ensure psi0 is ndarray with ndim=2
        if psi0 is None:
            raise ValueError("Initial state psi0 must be provided during Model initialization.")
        elif isinstance(psi0, np.ndarray) and psi0.ndim == 1:
            psi0 = np.array([psi0]).T
        self.psi0 = psi0

        self.params = list([*self.alpha_labels])
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
        self.rng = 10
        self.print_params = self.define_print_params()
        self.initialized = True

        # ================= INITIALISE INTEGRATOR STRATEGY ================== ##
        # The model holds an instance of the specific Integrator
        self.integrator = integrator_class(self)


    def update_history(self, psi: np.ndarray, t=None, reset=False, update_last_state=False):
        psi = self.__format_state(psi)
        if t is None:
            t = (np.arange(psi.shape[0]) * self.dt).round(self.precision_t)
        if isinstance(t, float):
            t = np.array([t])
        assert t.size == psi.shape[0], f"Length of t ({t.size}) must match number of time steps in psi ({psi.shape[0]})."
        self.history.update_history(psi, t=t, reset=reset, update_last_state=update_last_state)


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
    


    def __format_state(self, psi: np.ndarray) -> np.ndarray:
        """Ensure psi has the correct shape (Nt, N, m) for history storage.
        Args:
            psi: State array to format.
        Returns:
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
        return [*self.alpha_labels, *self.extra_print_params]

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
        self._dt = value

    @property
    def precision_t(self):
        return int(-np.log10(self.dt)) + 2
    
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
        if not self.ensemble:
            return 0
        else:
            return self.ensemble.get('Na', 0)

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    @property
    def m(self): 
        return self.hist.shape[-1]

    
    def set_fixed_params(self):
        fixed_params = dict((key, getattr(self, key)) for key in self.fixed_params)
        self.governing_eqns_params.update(fixed_params)



    def create_long_timeseries(self, Nt=None):
        if Nt is None:
            Nt = int(self.t_transient * 10 / self.dt)
        state, t = self.time_integrate(Nt=Nt)
        self.update_history(state, t)
        self.close()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed):
        self._rng = np.random.default_rng(seed)

    def copy(self):
        return deepcopy(self)


    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, :self.Nq, :]
        else:
            return self.hist[-Nt:, :self.Nq, :]

    def get_observable_hist(self, Nt=0, **kwargs):
        return self.get_observables(Nt, **kwargs)


    def print_model_parameters(self):
        print('\n ------------------ {} Model Parameters ------------------ '.format(self.name))
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
            setattr(self, '_Ma', np.hstack((np.zeros([self.Na, self.Nphi]),
                                            np.eye(self.Na),
                                            np.zeros([self.Na, self.Nq]))))
        return self._Ma

    # ------------------------- Functions for update/initialise the model --------------------------- #

    def reset_model(self, psi0=None, **kwargs):

        if psi0 is None:
            psi0 = self.current_state

        Model.__init__(self, psi0=psi0, **kwargs)


    def modify_settings(self):
        pass

    # def is_not_physical(self, print_=False):
    #     if not hasattr(self, '_physical'):
    #         self._physical = 0
    #     if print_:
    #         print(f'Number of non-physical analysis = {self._physical}/{self.number_of_analysis_steps}')
    #     else:
    #         self._physical += 1

    def close(self):
        self.integrator.close()

    @property
    def ensemble(self):
        """Public accessor for ensemble configuration (preferred over touching _ensemble_config)."""
        return getattr(self, '_ensemble_config', False)
    
    @ensemble.setter
    def ensemble(self, config: dict):
        """Setter for ensemble configuration."""
        self._ensemble_config = config
    
    @property
    def est_alpha(self):
        if not self.ensemble:
            return []
        else:
            return self.ensemble.get('est_alpha', [])
        
    @est_alpha.setter
    def est_alpha(self, value):
        if not self.ensemble:
            raise AttributeError("Cannot set est_alpha when ensemble is not configured.")
        self._ensemble_config['est_alpha'] = value


    def get_alpha(self, psi=None):
        if not self.ensemble:
            return self.alpha0.copy()

        if psi is None:
            psi = self.current_state

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
        Args:
            Nt: number of forecast steps
            averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                            the ensemble is forecast as a mean, i.e., every member is the mean forecast.
        Returns:
            psi: forecasted state (Nt x N x m)
            t: time of the propagated psi
        """
        return self.integrator.advance(Nt=Nt, averaged=averaged, alpha=self.get_alpha())
    



    # ============================== Visualization methods ============================== #
    def visualize_history(self):
        self.visualize_state_hist()
        self.visualize_observables_hist()
        self.visualize_spatiotemporal_hist()



    def visualize_state_hist(self, psi=None, t=None, max_modes=10):
        if psi is None:
            psi = self.hist[:, :self.Nphi]
        if t is None:
            t = self.hist_t[-len(psi):]

        lbl = self.state_labels

        # Plot the time evolution of the observables
        t_zoom = int(self.t_CR / self.dt)
        nrows = min(self.Nq, max_modes)

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
    

    def visualize_observables_hist(self, y=None, t=None):
        if y is None:
            y = self.get_observable_hist()
        if t is None:
            t = self.hist_t[-len(y):]

        lbl = self.obs_labels

        # Plot the time evolution of the observables
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

