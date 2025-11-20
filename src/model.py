# from scipy.integrate import solve_ivp
# from functools import partial
from copy import deepcopy

import numpy as np
import warnings

from integrator import IVPIntegrator
import matplotlib.pyplot as plt



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
        self.hist = np.reshape(self.psi0, (-1, self.Nphi, 1))
        self.hist_t = np.array([0.])
        
        # ======================== SET RNG ================================== ##
        self.rng = 10
        self.print_params = self.define_print_params()
        self.initialized = True

        # ================= INITIALISE INTEGRATOR STRATEGY ================== ##
        # The model holds an instance of the specific Integrator
        self.integrator = integrator_class(self)


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
        # Set the initial input parameters dictionary
        # Initial parameters are unchanged during the model run
        # Ensure the dictionary is unmutable
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

    @property
    def default_params(self):
        return dict((key, getattr(self.__class__, key)) for key in self.params)

    @property
    def current_state(self):
        return self.hist[-1]

    @property
    def current_time(self):
        return self.hist_t[-1]

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


    def update_history(self, psi=None, t=None, reset=False, update_last_state=False):
        if type(t) is float:
            t = np.array([t])

        if not reset and not update_last_state:
            self.hist = np.concatenate((self.hist, psi), axis=0)
            self.hist_t = np.hstack((self.hist_t, t))
        elif update_last_state:
            if psi is not None:
                if psi.shape[0] != self.Nphi + self.Na:
                    psi = psi[-1]

                self.reset_last_state(psi, t=t)
            else:
                raise ValueError('psi must be provided')
        else:
            if psi is None:
                psi = np.array(np.array([self.psi0]).T)
            if psi.ndim == 2:
                psi = np.array([psi])
                if t is None:
                    t = np.array([0.])
            elif t is None:
                t = np.arange(psi.shape[0]) * self.dt
                t = np.array([t])
                if t.ndim > 1:
                    t = t[..., 0]

            self.reset_history(psi, t)

    def reset_history(self, psi, t):
        self.hist = psi
        self.hist_t = t

    def reset_last_state(self, psi, t=None):
        self.hist[-1] = psi
        if t is not None:
            self.hist_t[-1] = t

    def is_not_physical(self, print_=False):
        if not hasattr(self, '_physical'):
            self._physical = 0
        if print_:
            print(f'Number of non-physical analysis = {self._physical}/{self.number_of_analysis_steps}')
        else:
            self._physical += 1

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

