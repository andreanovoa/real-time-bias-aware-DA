from scipy.integrate import solve_ivp
from functools import partial
from copy import deepcopy

import numpy as np

from src.bias_models import NoBias

from sys import platform

if platform == "darwin" or platform == "ios":
    import multiprocess as mp
else:
    import multiprocessing as mp


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general model properties and methods definitions.
    """

    alpha_labels: dict = dict()
    alpha_lims: dict = dict()

    state_labels: list = []
    fixed_params = []
    extra_print_params = []
    governing_eqns_params = dict()

    t = 0.
    dt = 0.01
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

    def __init__(self, **kwargs):

        # ================= INITIALISE PHYSICAL MODEL ================== ##
        model_dict = kwargs.copy()
        for key, val in kwargs.items():
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

    def define_print_params(self):
        return [*self.alpha_labels, *self.extra_print_params]

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

    def reshape_ensemble(self, m=None, reset=True):
        model = self.copy()
        if m is None:
            m = model.m
        psi = model.get_current_state
        if m == 1:
            psi = np.mean(psi, -1, keepdims=True)
            model.ensemble = False
        else:
            model.ensemble = True
            psi = model.add_uncertainty(self.rng, np.mean(psi, -1, keepdims=True), np.std(psi, -1, keepdims=True), m)
        model.update_history(psi=psi, t=0., reset=reset)
        return model

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
    # @property
    # def M(self):
    #     if not hasattr(self, '_M'):
    #         setattr(self, '_M', np.hstack((np.zeros([self.Nq, self.Na + self.Nphi]), np.eye(self.Nq))))
    #     return self._M
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
            M = np.hstack((np.zeros([self.Nq, self.Na + self.Nphi]), np.eye(self.Nq)))
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


    @staticmethod
    def add_uncertainty(rng, mean, std, m, method='uniform', ensure_mean=False):
        if method not in ['uniform', 'normal']:
            raise ValueError('Distribution {} not recognised'.format(method))

        if type(std) is dict:
            if method == 'uniform':
                ensemble_ = np.array([rng.uniform(low=sa[0], high=sa[1], size=m) for sa in std.values()])
            else:
                ensemble_ = [rng.normal(loc=np.mean(sa), scale=np.mean(sa) / 2, size=m) for sa in std.values()]
        elif type(std) is float:
            if method == 'uniform':
                ensemble_ = np.array([ma * (1. + rng.uniform(-std, std, m)) for ma in mean])
            else:
                ensemble_ = rng.multivariate_normal(mean, np.diag((mean * std) ** 2), m).T
        else:
            raise TypeError('std in normal distribution must be float not {}'.format(type(std)))

        if ensure_mean:
            ensemble_[:, 0] = mean

        return ensemble_

    def modify_settings(self):
        pass

    def init_ensemble(self, seed=None, ensemble_psi0=None, **kwargs):
        if seed is not None:
            self.rng(seed=seed)

        self.ensemble = True

        DAdict = kwargs.copy()
        for key, val in Model.defaults_ens.items():
            if key in DAdict.keys():
                setattr(self, key, DAdict[key])
            else:
                setattr(self, key, val)

        self.filename += '{}_ensemble_m{}'.format(self.name, self.m)
        self.modify_settings()

        # --------------- RESET INITIAL CONDITION AND HISTORY --------------- ##
        if ensemble_psi0 is None:
            mean_psi0 = np.mean(self.get_current_state, -1)
            ensemble_psi0 = Model.add_uncertainty(self.rng, mean_psi0, self.std_psi,
                                                 self.m, method=self.phi_distr, ensure_mean=self.ensure_mean)

        for var in [self.est_a, self.std_a]:
            if type(var) is dict:
                self.std_a = var.copy()
                self.est_a = [*var]

        if self.est_a:  # Augment ensemble with estimated input_parameters: psi = [psi; alpha]
            mean_a = np.array([getattr(self, pp) for pp in self.est_a])
            new_alpha0 = Model.add_uncertainty(self.rng, mean_a, self.std_a, self.m,
                                              method=self.alpha_distr, ensure_mean=self.ensure_mean)
            ensemble_psi0 = np.vstack((ensemble_psi0, new_alpha0))
        else:
            self.est_a = []

        # RESET ENSEMBLE HISTORY
        self.update_history(psi=ensemble_psi0, reset=True)

        if self.bias is None:
            self.init_bias()

    def init_bias(self, bias_model=NoBias, **Bdict):

        if 'bias_type' in Bdict.keys():
            raise 'redefine to bias_model'
        elif 'model' in Bdict.keys():
            raise 'redefine to bias_model'

        # Initialise bias. Note: self.bias is now an instance of the bias class
        if 'y0' in Bdict.keys():
            y0 = Bdict['y0']
        else:
            y0 = np.mean(self.get_observables(), axis=-1)
            if y0.ndim > 2:
                y0 = y0.squeeze(axis=-1)

        self.bias = bias_model(y=y0, t=self.get_current_time, dt=self.dt, **Bdict)

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
    @property
    def __pool(self):
        if not hasattr(self, '_pool'):
            N_pools = min(self.m, mp.cpu_count())
            self._pool = mp.Pool(N_pools)
        return self._pool

    def close(self):
        if hasattr(self, '_pool'):
            self.__pool.close()
            self.__pool.join()
            delattr(self, "_pool")
        else:
            pass

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

    @staticmethod
    def __forecast(y0, fun, t, params):
        # SOLVE IVP ========================================
        assert len(t) > 1

        part_fun = partial(fun, **params)

        out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45')
        psi = out.y.T

        # ODEINT =========================================== THIS WORKS AS IF HARD CODED
        # psi = odeint(fun, y0, t_interp, (params,))
        #
        # HARD CODED RUGGE KUTTA 4TH from Util ========================
        # psi = RK4(t_interp, y0, fun, params)
        return psi

    def time_integrate(self, Nt=100, averaged=False, alpha=None):
        """
            Integrator of the model. If the model is forcast as an ensemble, it uses parallel computation.
            Args:
                Nt: number of forecast steps
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying input_parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        t = np.round(self.get_current_time + np.arange(0, Nt + 1) * self.dt, self.precision_t)
        args = self.governing_eqns_params

        psi0 = self.get_current_state
        if not self.ensemble:
            psi = [Model.__forecast(y0=psi0[:, 0], fun=self.time_derivative, t=t, params={**self.alpha0, **args})]

        else:
            if not averaged:
                alpha = self.get_alpha()
                forecast_part = partial(Model.__forecast, fun=self.time_derivative, t=t)
                sol = [self.__pool.apply_async(forecast_part,
                                               kwds={'y0': psi0[:, mi].T, 'params': {**args, **alpha[mi]}})
                       for mi in range(self.m)]
                psi = [s.get() for s in sol]
            else:
                psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
                psi_deviation = psi0 - psi_mean0

                if alpha is None:
                    alpha = self.get_alpha(psi_mean0)[0]
                psi_mean = Model.__forecast(y0=psi_mean0[:, 0], fun=self.time_derivative, t=t, params={**alpha, **args})
                psi = [psi_mean + psi_deviation[:, ii] for ii in range(self.m)]

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError:
            print(alpha)
        return psi[1:], t[1:]