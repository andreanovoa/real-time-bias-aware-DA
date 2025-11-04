from scipy.integrate import solve_ivp
from functools import partial
from copy import deepcopy

import numpy as np

from bias import NoBias

from sys import platform

if platform == "darwin" or platform == "ios":
    import multiprocess as mp
else:
    import multiprocessing as mp


from integrator import *

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
    m = 1
    seed = 6
    psi0 = None
    alpha0 = None
    alpha = None
    filename = ''

    initialized = False
    ensemble = False

    defaults_ens: dict = dict(filter='EnKF',
                              constrained_filter=False,
                              bias_bayesian_update=False,
                              regularization_factor=1.,
                              m=10,
                              dt_obs=None,
                              est_a=[],
                              est_s=True,
                              est_b=False,
                              inflation=1.002,
                              reject_inflation=1.002,
                              std_psi=0.001,
                              std_a=0.001,
                              alpha_distr='uniform',
                              phi_distr='normal',
                              ensure_mean=False,
                              num_DA_blind=0,
                              num_SE_only=0,
                              start_ensemble_forecast=0.
                              )

    def __init__(self, integrator_class=IVPIntegrator, **kwargs):

        # ================= INITIALISE PHYSICAL MODEL ================== ##
        model_dict = kwargs.copy()
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, model_dict[key])
                del model_dict[key]

        for key, val in Model.defaults_ens.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
                del model_dict[key]

        if len(model_dict.keys()) > 1:
            print('Model {} not assigned'.format(model_dict.keys()))

        # ====================== SET INITIAL CONDITIONS ====================== ##
        self.params = list([*self.alpha_labels])
        self.alpha0 = {par: getattr(self, par) for par in self.params}
        # Ensure psi0 is ndarray with ndim=2
        if self.psi0.ndim < 2:
            self.psi0 = np.array([self.psi0]).T

        self.alpha = self.alpha0.copy()
        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi0])

        if self.ensemble:
            self.hist = self.hist.reshape(-1, self.N - self.Nq, self.m)
        else:
            self.hist = self.hist.reshape(-1, self.Nphi, 1)

        self.hist_t = np.array([0.])
        # ========================== DEFINE LENGTHS ========================== ##
        self.precision_t = int(-np.log10(self.dt)) + 2
        self.bias = None
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
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        """Setter for the time step."""
        if value <= 0:
            raise ValueError("Time step must be positive.")
        self._dt = value

    @property
    def Nphi(self):
        return len(self.psi0)

    @property
    def Na(self):
        if not hasattr(self, 'est_a'):
            return 0
        else:
            return len(self.est_a)

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    @property
    def get_default_params(self):
        return dict((key, getattr(self.__class__, key)) for key in self.params)

    @property
    def get_current_state(self):
        return self.hist[-1]

    @property
    def get_current_time(self):
        return self.hist_t[-1]

    def set_fixed_params(self):
        fixed_params = dict((key, getattr(self, key)) for key in self.fixed_params)
        self.governing_eqns_params.update(fixed_params)

    @property
    def bias_type(self):
        if hasattr(self, 'bias'):
            return type(self.bias)
        else:
            return NoBias

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
            psi0 = self.get_current_state()

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

    # -------------- Functions required for the forecasting ------------------- #

    def close(self):
        self.integrator.close()

    def get_alpha(self, psi=None):
        alpha = []
        if psi is None:
            psi = self.get_current_state
        for mi in range(psi.shape[-1]):
            ii = -self.Na
            alph = self.alpha0.copy()
            for param in self.est_a:
                alph[param] = psi[ii, mi]
                ii += 1
            alpha.append(alph)
        return alpha
    

    # ================= Main Time Integration Method ================= #

    def time_integrate(self, Nt=100, averaged=False, alpha=None):
        """
        Delegates the integration task to the currently configured Integrator strategy.
        The Model's time_integrate is now just a wrapper for the Strategy's advance method.
        Some Models may override this method if they need special handling.
        Args:
            Nt: number of forecast steps
            averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                            the ensemble is forecast as a mean, i.e., every member is the mean forecast.
            alpha: possibly-varying input_parameters
        Returns:
            psi: forecasted state (Nt x N x m)
            t: time of the propagated psi
        """
        return self.integrator.advance(Nt=Nt, averaged=averaged, alpha=alpha)

