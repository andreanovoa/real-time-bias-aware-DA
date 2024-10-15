from scipy.interpolate import splrep, splev
from scipy.integrate import solve_ivp
import multiprocessing as mp
from functools import partial
from copy import deepcopy

import numpy as np
import os

from essentials.bias_models import NoBias
from essentials.Util import Cheb

num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)


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
                              est_a=False,
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
            self.hist = self.hist.reshape(-1, self.N-self.Nq, self.m)
        else:
            self.hist = self.hist.reshape(-1, self.N-self.Nq, 1)
        self.hist_t = np.array([0.])
        # ========================== DEFINE LENGTHS ========================== ##
        self.precision_t = int(-np.log10(self.dt)) + 2
        self.bias = None
        # ======================== SET RNG ================================== ##
        self.rng = np.random.default_rng(self.seed)
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
    def get_current_state(self):
        return self.hist[-1]

    @property
    def get_current_time(self):
        return self.hist_t[-1]

    def set_fixed_params(self):
        for key in self.fixed_params:
            self.governing_eqns_params[key] = getattr(self, key)

    @property
    def bias_type(self):
        if hasattr(self, 'bias'):
            return type(self.bias)
        else:
            return NoBias

    def set_rng(self, seed):
        self.rng = np.random.default_rng(seed)

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
            if type(val) is float:
                print('\t {} = {:.6}'.format(key, val))
            else:
                print('\t {} = {}'.format(key, val))

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##
    @property
    def M(self):
        if not hasattr(self, '_M'):
            setattr(self, '_M', np.hstack((np.zeros([self.Nq, self.Na + self.Nphi]), np.eye(self.Nq))))
        return self._M

    @property
    def Ma(self):
        if not hasattr(self, '_Ma'):
            setattr(self, '_Ma', np.hstack((np.zeros([self.Na, self.Nphi]),
                                            np.eye(self.Na),
                                            np.zeros([self.Na, self.Nq]))))
        return self._Ma

    # ------------------------- Functions for update/initialise the model --------------------------- #
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
            self.set_rng(seed=seed)

        DAdict = kwargs.copy()
        self.ensemble = True

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
            ensemble_psi0 = self.add_uncertainty(self.rng, mean_psi0, self.std_psi,
                                                 self.m, method=self.phi_distr, ensure_mean=self.ensure_mean)

        for var in [self.est_a, self.std_a]:
            if type(var) is dict:
                self.std_a = var.copy()
                self.est_a = [*var]

        if self.est_a:  # Augment ensemble with estimated parameters: psi = [psi; alpha]
            mean_a = np.array([getattr(self, pp) for pp in self.est_a])
            new_alpha0 = self.add_uncertainty(self.rng, mean_a, self.std_a, self.m,
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
                    t = t[...,0]

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
    def pool(self):
        if not hasattr(self, '_pool'):
            self._pool = mp.Pool()
        return self._pool

    def close(self):
        if hasattr(self, '_pool'):
            self.pool.close()
            self.pool.join()
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
    def forecast(y0, fun, t, params):
        # SOLVE IVP ========================================
        assert len(t) > 1

        part_fun = partial(fun, **params)

        out = solve_ivp(part_fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45')
        psi = out.y.T

        # ODEINT =========================================== THIS WORKS AS IF HARD CODED
        # psi = odeint(fun, y0, t_interp, (params,))
        #
        # HARD CODED RUGGE KUTTA 4TH ========================
        # psi = RK4(t_interp, y0, fun, params)
        return psi

    def time_integrate(self, Nt=100, averaged=False, alpha=None):
        """
            Integrator of the model. If the model is forcast as an ensemble, it uses parallel computation.
            Args:
                Nt: number of forecast steps
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        t = np.round(self.get_current_time + np.arange(0, Nt + 1) * self.dt, self.precision_t)
        args = self.governing_eqns_params

        psi0 = self.get_current_state
        if not self.ensemble:
            psi = [Model.forecast(y0=psi0[:, 0], fun=self.time_derivative, t=t, params={**self.alpha0, **args})]

        else:
            if not averaged:
                alpha = self.get_alpha()
                forecast_part = partial(Model.forecast, fun=self.time_derivative, t=t)
                sol = [self.pool.apply_async(forecast_part,
                                             kwds={'y0': psi0[:, mi].T, 'params': {**args, **alpha[mi]}})
                       for mi in range(self.m)]
                psi = [s.get() for s in sol]
            else:
                psi_mean0 = np.mean(psi0, axis=1, keepdims=True)
                psi_deviation = psi0 - psi_mean0

                if alpha is None:
                    alpha = self.get_alpha(psi_mean0)[0]
                psi_mean = Model.forecast(y0=psi_mean0[:, 0], fun=self.time_derivative, t=t,
                                          params={**alpha, **args})

                # if np.mean(np.std(self.psi[:len(self.psi0)] / np.array([self.psi0]).T, axis=0)) < 2.:
                # psi_deviation /= psi_mean0
                # psi = [psi_mean * (1 + psi_deviation[:, ii]) for ii in range(self.m)]
                # else:
                psi = [psi_mean + psi_deviation[:, ii] for ii in range(self.m)]

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        try:
            psi = np.array(psi).transpose((1, 2, 0))
        except ValueError:
            print(alpha)
        return psi[1:], t[1:]


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
    t_transient = 1.5
    t_CR = 0.04
    # defaults: dict = dict(Nq=1, dt=1e-4,
    #                       omega=2 * np.pi * 120., law='tan',
    #                       zeta=60., beta=70., kappa=4.0, gamma=1.7)  # beta, zeta [rad/s]

    Nq = 1
    dt = 1e-4
    law = 'tan'

    beta = 70.
    kappa = 4.0
    gamma = 1.7
    omega = 2 * np.pi * 120.
    zeta = 60.

    alpha_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')
    alpha_lims = dict(zeta=(5, 120), kappa=(0.1, 20), beta=(5, 120))

    state_labels: list = ['$\\eta$', '$\\mu$']

    fixed_params = ['law', 'omega']
    extra_print_params = ['law', 'omega']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = np.array([0.1, 0.1])  # initialise eta and mu

        super().__init__(**model_dict)

        #  Add fixed parameters
        self.set_fixed_params()

    # _______________ VdP specific properties and methods ________________ #
    @property
    def obs_labels(self):
        return ["$\\eta$"]

    @staticmethod
    def time_derivative(t, psi, beta, zeta, kappa, law, omega):
        eta, mu = psi[:2]
        dmu_dt = - omega ** 2 * eta + mu * (beta - zeta)
        # Add nonlinear term
        if law == 'cubic':  # Cubic law
            dmu_dt -= mu * kappa * eta ** 2
        elif law == 'tan':  # arc tan model
            dmu_dt -= mu * (kappa * eta ** 2) / (1. + kappa / beta * eta ** 2)

        return (mu, dmu_dt) + (0,) * (len(psi) - 2)


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretization and gain-delay sqrt heat release law.
    """

    name: str = 'Rijke'
    t_transient = 1.
    t_CR = 0.02

    Nm = 10
    Nc = 10
    Nq = 6
    dt = 1e-4

    beta, tau = 4.0, 1.5E-3
    C1, C2 = 0.05, 0.01
    kappa = 1E5
    xf, L = 0.2, 1.
    law = 'sqrt'

    alpha_labels = dict(beta='$\\beta$', tau='$\\tau$', C1='$C_1$', C2='$C_2$', kappa='$\\kappa$')
    alpha_lims = dict(beta=(0.01, 5), tau=[1E-6, None], C1=(0., 1.), C2=(0., 1.), kappa=(1E3, 1E8))

    fixed_params = ['cosomjxf', 'Dc', 'gc', 'jpiL', 'L',
                    'law', 'meanFlow', 'Nc', 'Nm', 'tau_adv', 'sinomjxf']

    extra_print_params = ['law', 'Nm', 'Nc', 'xf', 'L']

    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            if 'Nm' in model_dict.keys():
                Nm = model_dict['Nm']
            else:
                Nm = self.Nm
            if 'Nc' in model_dict.keys():
                Nc = model_dict['Nc']
            else:
                Nc = self.Nc
            model_dict['psi0'] = .05 * np.hstack([np.ones(2 * Nm), np.zeros(Nc)])

        super().__init__(**model_dict)

        self.tau_adv = self.tau
        self.alpha_lims['tau'][-1] = self.tau_adv

        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nq + 1)[:-1]

        # Define modes frequency of each mode and sin cos etc
        jj = np.arange(1, self.Nm + 1)
        self.jpiL = jj * np.pi / self.L
        self.sinomjxf = np.sin(self.jpiL * self.xf)
        self.cosomjxf = np.cos(self.jpiL * self.xf)

        # Mean Flow Properties
        def weight_avg(y1, y2):
            return self.xf / self.L * y1 + (1. - self.xf / self.L) * y2

        self.meanFlow = dict(u=weight_avg(10, 11.1643), p=101300.,
                             gamma=1.4, T=weight_avg(300, 446.5282), R=287.1)
        self.meanFlow['rho'] = self.meanFlow['p'] / (self.meanFlow['R'] * self.meanFlow['T'])
        self.meanFlow['c'] = np.sqrt(self.meanFlow['gamma'] * self.meanFlow['R'] * self.meanFlow['T'])

        self.set_fixed_params()

        # Wave parameters ############################################################################################
        # c1: 347.2492    p1: 1.0131e+05      rho1: 1.1762    u1: 10          M1: 0.0288          T1: 300
        # c2: 423.6479    p2: 101300          rho2: 0.7902    u2: 11.1643     M2: 0.0264          T2: 446.5282
        # Tau: 0.0320     Td: 0.0038          Tu: 0.0012      R_in: -0.9970   R_out: -0.9970      Su: 0.9000
        # Q_bar: 5000     R_gas: 287.1000     gamma: 1.4000
        ##############################################################################################################

    def modify_settings(self):
        if 'tau' in self.est_a:
            extra_Nc = 50 - self.Nc
            self.tau_adv, self.Nc = 1E-2, 50
            self.alpha_lims['tau'][-1] = self.tau_adv
            psi = self.get_current_state
            self.psi0 = np.hstack([np.mean(psi, -1),
                                   np.zeros(extra_Nc)])
            self.Dc, self.gc = Cheb(self.Nc, getg=True)
            self.update_history(reset=True)
            self.set_fixed_params()

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def obs_labels(self, loc=None):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0]]

    @property
    def state_labels(self):
        lbls0 = [f"$\\eta_{j}$" for j in np.arange(self.Nm)]
        lbls1 = ["$\\dot{\\eta}$" + f"$_{j}$" for j in np.arange(self.Nm)]
        lbls2 = [f"$\\nu_{j}$" for j in np.arange(self.Nc)]
        return lbls0 + lbls1 + lbls2

    def get_observables(self, Nt=1, loc=None, **kwargs):
        if loc is None:
            loc = self.x_mic
        loc = np.expand_dims(loc, axis=1)
        om = np.array([self.jpiL])
        mu = self.hist[-Nt:, self.Nm:2 * self.Nm, :]

        # Compute acoustic pressure and velocity at locations
        p_mic = -np.dot(np.sin(np.dot(loc, om)), mu)
        p_mic = p_mic.transpose(1, 0, 2)
        if Nt == 1:
            p_mic = p_mic[0]
        return p_mic

    @staticmethod
    def time_derivative(t, psi,
                        C1, C2, beta, kappa, tau,
                        cosomjxf, Dc, gc, jpiL, L, law, meanFlow, Nc, Nm, tau_adv, sinomjxf):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
                C1, C2, beta, kappa, tau: Possibly-inferred parameters
                cosomjxf, Dc, gc, jpiL, L, law, meanFlow, Nc, Nm, tau_adv, sinomjxf:  fixed parameters
            Returns:
                concatenation of the state vector time derivative
        """
        eta, mu, v = psi[:Nm], psi[Nm: 2 * Nm], psi[2 * Nm: 2 * Nm + Nc]

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, cosomjxf), v))

        # Evaluate u(t_interp-tau) i.e. velocity at the flame at t_interp - tau
        x_tau = tau / tau_adv
        if x_tau < 1:
            f = splrep(gc, v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            raise Exception("tau = {} can't_interp be larger than tau_adv = {}".format(tau, tau_adv))

        # Compute damping and heat release law
        zeta = C1 * (jpiL * L / np.pi) ** 2 + C2 * (jpiL * L / np.pi) ** .5

        MF = meanFlow.copy()  # Physical properties
        if law == 'sqrt':
            q_dot = MF['p'] * MF['u'] * beta * (
                    np.sqrt(abs(1. / 3 + u_tau / MF['u'])) - np.sqrt(1. / 3))  # [W/m2]=[m/s3]
        elif law == 'tan':
            q_dot = beta * np.sqrt(beta / kappa) * np.arctan(np.sqrt(beta / kappa) * u_tau)  # [m / s3]
        else:
            raise ValueError('Law "{}" not defined'.format(law))
        q_dot *= -2. * (MF['gamma'] - 1.) / L * sinomjxf  # [Pa/s]

        # governing equations
        deta_dt = jpiL / MF['rho'] * mu
        dmu_dt = - jpiL * MF['gamma'] * MF['p'] * eta - MF['c'] / L * zeta * mu + q_dot
        dv_dt = - 2. / tau_adv * np.dot(Dc, v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(len(psi) - (2 * Nm + Nc))))


# %% =================================== LORENZ 63 MODEL ============================================== %% #
class Lorenz63(Model):
    """ Lorenz 63 Class
    """
    name: str = 'Lorenz63'

    t_lyap = 0.9056 ** (-1)
    t_transient = 10 * t_lyap
    t_CR = 4 * t_lyap

    Nq = 3
    dt = 0.02
    rho = 28.
    sigma = 10.
    beta = 8. / 3.

    observe_dims = range(3)

    alpha_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')
    alpha_lims = dict(rho=(None, None), sigma=(None, None), beta=(None, None))

    extra_print_params = ['observe_dims', 'Nq', 't_lyap']

    state_labels = ['$x$', '$y$', '$z$']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = np.array([1.0, 1.0, 1.0])  # initialise x, y, z

        super().__init__(**model_dict)

        if 'observe_dims' in model_dict:
            self.Nq = len(self.observe_dims)

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obs_labels(self):
        return [self.state_labels[kk] for kk in self.observe_dims]

    def get_observables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, self.observe_dims, :]
        else:
            return self.hist[-Nt:, self.observe_dims, :]

    @staticmethod
    def time_derivative(t, psi, sigma, rho, beta):
        x1, x2, x3 = psi[:3]
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)


# %% =================================== 2X VAN DER POL MODEL ============================================== %% #
class Annular(Model):
    """ Annular combustor model, which consists of two coupled oscillators
    """

    name: str = 'Annular'

    t_transient = 0.5
    t_CR = 0.01

    ER = 0.5
    nu_1, nu_2 = 633.77, -331.39
    c2b_1, c2b_2 = 258.3, -108.27  # values in Matlab codes

    # defaults: dict = dict(Nq=4, n=1., ER=ER_0, dt=1. / 51200,
    #                       theta_b=0.63, theta_e=0.66, omega=1090 * 2 * np.pi, epsilon=2.3E-3,
    #                       nu=nu_1 * ER_0 + nu_2, c2beta=c2b_1 * ER_0 + c2b_2, kappa=1.2E-4)

    Nq = 4
    theta_mic = np.radians([0, 60, 120, 240])

    dt = 1. / 51200
    theta_b = 0.63
    theta_e = 0.66
    omega = 1090 * 2 * np.pi
    epsilon = 2.3E-3

    nu = nu_1 * ER + nu_2
    c2beta = c2b_1 * ER + c2b_2
    kappa = 1.2E-4

    # defaults['nu'], defaults['c2beta'] = 30., 5.  # spin
    # defaults['nu'], defaults['c2beta'] = 1., 25.  # stand
    # defaults['nu'], defaults['c2beta'] = 20., 18.  # mix

    alpha_labels = dict(omega='$\\omega$', nu='$\\nu$', c2beta='$c_2\\beta $', kappa='$\\kappa$',
                        epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')
    alpha_lims = dict(omega=(1000 * 2 * np.pi, 1300 * 2 * np.pi),
                      nu=(-60., 60.), c2beta=(0., 60.), kappa=(None, None),
                      epsilon=(None, None), theta_b=(0, 2 * np.pi), theta_e=(0, 2 * np.pi))

    state_labels = ['$\\eta_{a}$', '$\\dot{\\eta}_{a}$', '$\\eta_{b}$', '$\\dot{\\eta}_{b}$']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        if 'psi0' not in model_dict.keys():
            C0, X0, th0, ph0 = 10, 0, 0.63, 0  # %initial values
            # Conversion of the initial conditions from the quaternion formalism to the AB formalism
            Ai = C0 * np.sqrt(np.cos(th0) ** 2 * np.cos(X0) ** 2 + np.sin(th0) ** 2 * np.sin(X0) ** 2)
            Bi = C0 * np.sqrt(np.sin(th0) ** 2 * np.cos(X0) ** 2 + np.cos(th0) ** 2 * np.sin(X0) ** 2)
            phai = ph0 + np.arctan2(np.sin(th0) * np.sin(X0), np.cos(th0) * np.cos(X0))
            phbi = ph0 - np.arctan2(np.cos(th0) * np.sin(X0), np.sin(th0) * np.cos(X0))

            # %initial conditions for the fast oscillator equations
            psi0 = [Ai * np.cos(phai),
                    -self.omega * Ai * np.sin(phai),
                    Bi * np.cos(phbi),
                    -self.omega * Bi * np.sin(phbi)]

            model_dict['psi0'] = np.array(psi0)  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}

        super().__init__(**model_dict)

    # _______________  Specific properties and methods ________________ #
    @property
    def obs_labels(self, loc=None, measure_modes=False):
        if measure_modes:
            return ["$\\eta_1$", '$\\eta_2$']
        else:
            if loc is None:
                loc = self.theta_mic
            return ["$p(\\theta={}^\\circ)$".format(int(np.round(np.degrees(th)))) for th in np.array(loc)]

    @staticmethod
    def nu_from_ER(ER):
        return Annular.nu_1 * ER + Annular.nu_2

    @staticmethod
    def c2beta_from_ER(ER):
        return Annular.c2b_1 * ER + Annular.c2b_2

    def get_observables(self, Nt=1, loc=None, measure_modes=False, **kwargs):
        """
        pressure measurements at theta = [0º, 60º, 120º, 240º`]
        p(θ, t) = η1(t) * cos(nθ) + η2(t) * sin(nθ).
        """
        if loc is None:
            loc = self.theta_mic

        if measure_modes:
            return self.hist[-Nt:, [0, 2], :]
        else:
            eta1, eta2 = self.hist[-Nt:, 0, :], self.hist[-Nt:, 2, :]
            if max(loc) > 2 * np.pi:
                raise ValueError('Theta must be in radians')

            p_mics = np.array([eta1 * np.cos(th) + eta2 * np.sin(th) for th in np.array(loc)])
            p_mics = p_mics.transpose(1, 0, 2)
            if Nt == 1:
                return p_mics.squeeze(axis=0)
            else:
                return p_mics

    @staticmethod
    def time_derivative(t, psi, nu, kappa, c2beta, theta_b, omega, epsilon, theta_e):
        y_a, z_a, y_b, z_b = psi[:4]  # y = η, and z = dη/dt

        def k1(y1, y2, sign):
            return (2 * nu - 3. / 4 * kappa * (3 * y1 ** 2 + y2 ** 2) +
                    sign * c2beta / 2. * np.cos(2. * theta_b))

        k2 = c2beta / 2. * np.sin(2. * theta_b) - 3. / 2 * kappa * y_a * y_b

        def k3(y1, y2, sign):
            return omega ** 2 * (y1 * (1 + sign * epsilon / 2. * np.cos(2. * theta_e)) +
                                 y2 * epsilon / 2. * np.sin(2. * theta_e))

        dz_a = z_a * k1(y_a, y_b, sign=1) + z_b * k2 - k3(y_a, y_b, sign=1)
        dz_b = z_b * k1(y_b, y_a, sign=-1) + z_a * k2 - k3(y_b, y_a, sign=-1)

        return (z_a, dz_a, z_b, dz_b) + (0,) * (len(psi) - 4)
