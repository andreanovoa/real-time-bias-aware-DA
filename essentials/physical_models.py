from scipy.interpolate import splrep, splev
from scipy.integrate import solve_ivp
import multiprocessing as mp
from functools import partial
from copy import deepcopy

import numpy as np
import os

import essentials.bias_models
from essentials.Util import Cheb

num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general model properties and methods definitions.
    """
    defaults: dict = dict(t=0., seed=0,
                          psi0=np.empty(1), alpha0=np.empty(1), psi=None, alpha=None,
                          ensemble=False, filename='', governing_eqns_params=dict())
    defaults_ens: dict = dict(filter='EnKF',
                              constrained_filter=False,
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
                              alpha_distr='normal',
                              ensure_mean=False,
                              num_DA_blind=0,
                              num_SE_only=0,
                              start_ensemble_forecast=0.,
                              get_cost=False,
                              Na=0
                              )

    # __slots__ = list(defaults.keys()) + list(defaults_ens.keys()) + ['hist', 'hist_t', 'hist_J', '_pool', '_M', '_Ma']

    def __init__(self, child_defaults, **model_dict):

        # ================= INITIALISE THERMOACOUSTIC MODEL ================== ##
        default_values = {**child_defaults, **Model.defaults}
        for key, val in default_values.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
                del model_dict[key]
            else:
                setattr(self, key, val)

        for key, val in Model.defaults_ens.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
                del model_dict[key]

        if len(model_dict.keys()) != 0:
            print('Model {} not assigned'.format(model_dict.keys()))

        # ====================== SET INITIAL CONDITIONS ====================== ##
        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()
        self.psi = np.array([self.psi0]).T
        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi])
        self.hist_t = np.array([self.t])
        self.hist_J = []
        # ========================== DEFINE LENGTHS ========================== ##
        self.precision_t = int(-np.log10(self.dt)) + 2
        self.bias = None
        # ======================== SET RNG ================================== ##
        self.rng = np.random.default_rng(self.seed)

    @property
    def Nphi(self):
        return len(self.psi0)

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    @property
    def biasType(self):
        if hasattr(self, 'bias'):
            return type(self.bias)
        else:
            return essentials.bias_models.NoBias

    def copy(self):
        return deepcopy(self)

    def reshape_ensemble(self, m=None, reset=True):
        model = self.copy()
        if m is None:
            m = model.m
        psi = model.psi
        if m == 1:
            psi = np.mean(psi, -1, keepdims=True)
            model.ensemble = False
        else:
            model.ensemble = True
            psi = model.addUncertainty(self.rng, np.mean(psi, -1, keepdims=True),
                                       np.std(psi, -1, keepdims=True), m)
        model.updateHistory(psi=psi, t=0., reset=reset)
        return model

    def getObservables(self, Nt=1, **kwargs):
        if Nt == 1:
            return self.hist[-1, :self.Nq, :]
        else:
            return self.hist[-Nt:, :self.Nq, :]

    def getObservableHist(self, Nt=0, **kwargs):
        return self.getObservables(Nt, **kwargs)

    def print_model_parameters(self):
        print('\n ------------------ {} Model Parameters ------------------ '.format(self.name))
        for key in sorted(self.defaults.keys()):
            try:
                print('\t {} = {:.6}'.format(key, getattr(self, key)))
            except ValueError:
                print('\t {} = {}'.format(key, getattr(self, key)))

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
    def addUncertainty(rng, mean, std, m, method='normal', param_names=None, ensure_mean=False):
        if method == 'normal':
            if isinstance(std, float):
                cov = np.diag((mean * std) ** 2)
            else:
                raise TypeError('std in normal distribution must be float not {}'.format(type(std)))
            ens = rng.multivariate_normal(mean, cov, m).T
        elif method == 'uniform':
            ens = np.zeros((len(mean), m))
            if isinstance(std, float):
                for pi, pp in enumerate(mean):
                    if abs(std) <= .5:
                        ens[pi, :] = pp * (1. + rng.uniform(-std, std, m))
                    else:
                        ens[pi, :] = rng.uniform(pp - std, pp + std, m)
            elif isinstance(std, dict):
                if param_names is not None:
                    for pi, key in enumerate(param_names):
                        ens[pi, :] = rng.uniform(std[key][0], std[key][1], m)
                else:
                    for pi, _ in enumerate(mean):
                        ens[pi, :] = rng.uniform(std[pi][0], std[pi][1], m)
            else:
                raise TypeError('std in normal distribution must be float or dict')
        else:
            raise ValueError('Parameter distribution {} not recognised'.format(method))

        if ensure_mean:
            ens[:, 0] = mean
        return ens

    def initEnsemble(self, **DAdict):
        DAdict = DAdict.copy()
        self.ensemble = True

        for key, val in Model.defaults_ens.items():
            if key in DAdict.keys():
                setattr(self, key, DAdict[key])
            else:
                setattr(self, key, val)

        self.filename += '{}_ensemble_m{}'.format(self.name, self.m)
        if hasattr(self, 'modify_settings'):
            self.modify_settings()
        # --------------- RESET INITIAL CONDITION AND HISTORY --------------- ##
        # Note: if est_a and est_b psi = [psi; alpha; biasWeights]
        ensure_mean = self.ensure_mean
        mean_psi0 = np.mean(self.psi, -1)

        new_psi0 = self.addUncertainty(self.rng, mean_psi0, self.std_psi, self.m,
                                       method='normal', ensure_mean=ensure_mean)

        if self.est_a:  # Augment ensemble with estimated parameters
            mean_a = np.array([getattr(self, pp) for pp in self.est_a])
            new_alpha0 = self.addUncertainty(self.rng, mean_a, self.std_a, self.m, method=self.alpha_distr,
                                             param_names=self.est_a, ensure_mean=ensure_mean)
            new_psi0 = np.vstack((new_psi0, new_alpha0))
            self.Na = len(self.est_a)

        # RESET ENSEMBLE HISTORY
        self.updateHistory(psi=new_psi0, t=0., reset=True)
        if self.bias is None:
            self.initBias()

    def initBias(self, **Bdict):

        if 'biasType' in Bdict.keys():
            biasType = Bdict['biasType']
        else:
            biasType = essentials.bias_models.NoBias

        # Initialise bias. Note: self.bias is now an instance of the bias class
        self.bias = biasType(y=self.getObservables(), t=self.t, dt=self.dt, **Bdict)
        # Create bias history
        b = self.bias.getBias()
        self.bias.updateHistory(b, self.t, reset=True)

    def updateHistory(self, psi=None, t=None, reset=False):
        if not reset:
            self.hist = np.concatenate((self.hist, psi), axis=0)
            self.hist_t = np.hstack((self.hist_t, t))
        else:
            if psi is None:
                psi = np.array([self.psi0]).T
            if t is None:
                t = self.t
            self.hist = np.array([psi])
            self.hist_t = np.array([t])

        self.psi = self.hist[-1]
        self.t = self.hist_t[-1]

    def is_not_physical(self, print_=False):
        if not hasattr(self, '_physical'):
            self._physical = 0
        if print_:
            print('Number of non-physical analysis = ', self._physical)
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

    def getAlpha(self, psi=None):
        alpha = []
        if psi is None:
            psi = self.psi
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

    def timeIntegrate(self, Nt=100, averaged=False, alpha=None):
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

        t = np.round(self.t + np.arange(0, Nt + 1) * self.dt, self.precision_t)
        args = self.governing_eqns_params

        if not self.ensemble:
            psi = [Model.forecast(y0=self.psi[:, 0], fun=self.timeDerivative, t=t, params={**self.alpha0, **args})]

        else:
            if not averaged:
                alpha = self.getAlpha()
                forecast_part = partial(Model.forecast, fun=self.timeDerivative, t=t)
                sol = [self.pool.apply_async(forecast_part,
                                             kwds={'y0': self.psi[:, mi].T, 'params': {**args, **alpha[mi]}})
                       for mi in range(self.m)]
                psi = [s.get() for s in sol]
            else:
                psi_mean0 = np.mean(self.psi, axis=1, keepdims=True)
                psi_deviation = self.psi - psi_mean0

                if alpha is None:
                    alpha = self.getAlpha(psi_mean0)[0]
                psi_mean = Model.forecast(y0=psi_mean0[:, 0], fun=self.timeDerivative, t=t,
                                          params={**alpha, **args})

                # if np.mean(np.std(self.psi[:len(self.psi0)] / np.array([self.psi0]).T, axis=0)) < 2.:
                # psi_deviation /= psi_mean0
                # psi = [psi_mean * (1 + psi_deviation[:, ii]) for ii in range(self.m)]
                # else:
                psi = [psi_mean + psi_deviation[:, ii] for ii in range(self.m)]

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        psi = np.array(psi).transpose((1, 2, 0))
        return psi[1:], t[1:]


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
    defaults: dict = dict(Nq=1, t_transient=1.5, t_CR=0.04, dt=1e-4,
                          omega=2 * np.pi * 120., law='tan',
                          zeta=60., beta=70., kappa=4.0, gamma=1.7)  # beta, zeta [rad/s]

    params_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')
    params_lims = dict(zeta=(5, 120), kappa=(0.1, 20), beta=(5, 120))
    params: list = [*params_labels]  # ,'omega', 'gamma']

    fixed_params = ['law', 'omega']

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = [0.1, 0.1]  # initialise eta and mu

        super().__init__(child_defaults=VdP.defaults, **model_dict)

        # _________________________ Add fixed parameters  ________________________ #
        self.set_fixed_params()

    def set_fixed_params(self):
        for key in VdP.fixed_params:
            self.governing_eqns_params[key] = getattr(self, key)

    # _______________ VdP specific properties and methods ________________ #

    @property
    def obsLabels(self):
        return ["$\\eta$"]

    @staticmethod
    def timeDerivative(t, psi, beta, zeta, kappa, law, omega):
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
        Rijke tube model with Galerkin discretisation and gain-delay sqrt heat release law.
    """

    name: str = 'Rijke'
    defaults: dict = dict(Nm=10, Nc=10, Nq=6, t_transient=1., t_CR=0.02, dt=1e-4,
                          beta=4.0, tau=1.5E-3, C1=.05, C2=.01, kappa=1E5, xf=0.2, L=1., law='sqrt')
    params_labels = dict(beta='$\\beta$', tau='$\\tau$', C1='$C_1$', C2='$C_2$', kappa='$\\kappa$')
    params_lims = dict(beta=(0.01, 5), tau=[1E-6, None], C1=(0., 1.), C2=(0., 1.), kappa=(1E3, 1E8))
    params: list = list([*params_labels])

    fixed_params = ['cosomjxf', 'Dc', 'gc', 'jpiL', 'L', 'law', 'meanFlow', 'Nc', 'Nm', 'tau_adv', 'sinomjxf']

    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            Nm, Nc = [Rijke.defaults[key] for key in ['Nm', 'Nc']]
            if 'Nm' in model_dict.keys():
                Nm = model_dict['Nm']
            if 'Nc' in model_dict.keys():
                Nc = model_dict['Nc']
            model_dict['psi0'] = .05 * np.hstack([np.ones(2 * Nm), np.zeros(Nc)])

        super().__init__(child_defaults=Rijke.defaults, **model_dict)

        self.tau_adv = self.tau
        self.params_lims['tau'][-1] = self.tau_adv

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
        # Qbar: 5000      R_gas: 287.1000     gamma: 1.4000
        ##############################################################################################################

    def modify_settings(self):
        if self.est_a and 'tau' in self.est_a:
            extra_Nc = 50 - self.Nc
            self.tau_adv, self.Nc = 1E-2, 50
            self.psi0 = np.hstack([np.mean(self.psi, -1), np.zeros(extra_Nc)])
            self.Dc, self.gc = Cheb(self.Nc, getg=True)
            self.updateHistory(reset=True)
            self.set_fixed_params()

    # _________________________ Governing equations ________________________ #
    def set_fixed_params(self):
        for key in Rijke.fixed_params:
            self.governing_eqns_params[key] = getattr(self, key)

    # _______________ Rijke specific properties and methods ________________ #

    @property
    def obsLabels(self, loc=None):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0]]

    def getObservables(self, Nt=1, loc=None):
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
    def timeDerivative(t, psi,
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
        eta = psi[:Nm]
        mu = psi[Nm: 2 * Nm]
        v = psi[2 * Nm: 2 * Nm + Nc]

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

    t_lyap = 0.906 ** (-1)
    defaults: dict = dict(Nq=3, t_lyap=t_lyap, t_transient=t_lyap * 20., t_CR=t_lyap * 4., dt=0.02,
                          rho=28., sigma=10., beta=8. / 3.)

    params_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')
    params_lims = dict(rho=(None, None), sigma=(None, None), beta=(None, None))
    params = list([*params_labels])

    fixed_params = []

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):
        if 'psi0' not in model_dict.keys():
            model_dict['psi0'] = [1.0, 1.0, 1.0]  # initialise x, y, z

        super().__init__(child_defaults=Lorenz63.defaults, **model_dict)

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obsLabels(self):
        return ['$x$', '$y$', '$z$']

    @staticmethod
    def timeDerivative(t, psi, sigma, rho, beta):
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

    ER_0 = 0.5
    nu_1, nu_2 = 633.77, -331.39
    c2b_1, c2b_2 = 258.3, -108.27

    defaults: dict = dict(Nq=4, n=1., ER=ER_0, t_transient=0.5, t_CR=0.01, dt=1. / 51200,
                          theta_b=0.63, theta_e=0.66, omega=1090 * 2 * np.pi, epsilon=2.3E-3,
                          nu=nu_1 * ER_0 + nu_2, beta_c2=c2b_1 * ER_0 + c2b_2, kappa=1.2E-4)  # values in Matlab codes

    # defaults['nu'], defaults['beta_c2'] = 30., 5.  # spin
    # defaults['nu'], defaults['beta_c2'] = 1., 25.  # stand
    # defaults['nu'], defaults['beta_c2'] = 20., 18.  # mix

    params_labels = dict(omega='$\\omega$', nu='$\\nu$', beta_c2='$c_2\\beta $', kappa='$\\kappa$',
                         epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')
    params_lims = dict(omega=(1000 * 2 * np.pi, 1300 * 2 * np.pi), nu=(1., 60.), beta_c2=(1., 60.), kappa=(None, None),
                       epsilon=(None, None), theta_b=(0, 2 * np.pi), theta_e=(0, 2 * np.pi))
    params = list([*params_labels])
    fixed_params = []

    # __________________________ Init method ___________________________ #
    def __init__(self, **model_dict):

        if 'psi0' not in model_dict.keys():
            if 'omega' not in model_dict.keys():
                omega = Annular.defaults['omega']
            else:
                omega = model_dict['omega']

            C0, X0, th0, ph0 = 10, 0, 0.63, 0  # %initial values
            # %Conversion of the initial conditions from the quaternion formalism to the AB formalism
            Ai = C0 * np.sqrt(np.cos(th0) ** 2 * np.cos(X0) ** 2 + np.sin(th0) ** 2 * np.sin(X0) ** 2)
            Bi = C0 * np.sqrt(np.sin(th0) ** 2 * np.cos(X0) ** 2 + np.cos(th0) ** 2 * np.sin(X0) ** 2)
            phai = ph0 + np.arctan2(np.sin(th0) * np.sin(X0), np.cos(th0) * np.cos(X0))
            phbi = ph0 - np.arctan2(np.cos(th0) * np.sin(X0), np.sin(th0) * np.cos(X0))
            # %initial conditions for the fast oscillator equations
            psi0 = [Ai * np.cos(phai),
                    -omega * Ai * np.sin(phai),
                    Bi * np.cos(phbi),
                    -omega * Bi * np.sin(phbi)]

            model_dict['psi0'] = psi0  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}

        super().__init__(child_defaults=Annular.defaults, **model_dict)

        self.theta_mic = np.radians([0, 60, 120, 240])

    # _______________  Specific properties and methods ________________ #
    @property
    def obsLabels(self, loc=None, measure_modes=False):
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
    def beta_c2_from_ER(ER):
        return Annular.c2b_1 * ER + Annular.c2b_1

    def getObservables(self, Nt=1, loc=None, measure_modes=False):
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

            p_mics = np.array([eta1 * np.cos(th) + eta2 * np.sin(th)
                               for th in np.array(loc)])

            p_mics = p_mics.transpose(1, 0, 2)
            if Nt == 1:
                return p_mics.squeeze(axis=0)
            else:
                return p_mics

    @staticmethod
    def timeDerivative(t, psi, nu, kappa, beta_c2, theta_b, omega, epsilon, theta_e):
        y_a, z_a, y_b, z_b = psi[:4]  # y = η, and z = dη/dt

        def k1(y1, y2, sign):
            return (2 * nu - 3. / 4 * kappa * (3 * y1 ** 2 + y2 ** 2) +
                    sign * beta_c2 / 2. * np.cos(2. * theta_b))

        k2 = beta_c2 / 2. * np.sin(2. * theta_b) - 3. / 2 * kappa * y_a * y_b

        def k3(y1, y2, sign):
            return omega ** 2 * (y1 * (1 + sign * epsilon / 2. * np.cos(2. * theta_e)) +
                                 y2 * epsilon / 2. * np.sin(2. * theta_e))

        dz_a = z_a * k1(y_a, y_b, 1) + z_b * k2 - k3(y_a, y_b, 1)
        dz_b = z_b * k1(y_b, y_a, -1) + z_a * k2 - k3(y_b, y_a, -1)

        return (z_a, dz_a, z_b, dz_b) + (0,) * (len(psi) - 4)

