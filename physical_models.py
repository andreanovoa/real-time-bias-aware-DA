from scipy.interpolate import splrep, splev
import pylab as plt

import bias_models
from Util import Cheb, RK4
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
from functools import partial
from copy import deepcopy

from matplotlib.animation import FuncAnimation

# os.environ["OMP_NUM_THREADS"] = '1'
num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)

rng = np.random.default_rng(6)

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general model properties and methods definitions.
    """
    attr_model: dict = dict(t=0.,
                            psi0=np.empty(1),
                            )
    attr_ens: dict = dict(filter='EnKF',
                          constrained_filter=False,
                          regularization_factor=1.,
                          m=10,
                          dt_obs=None,
                          est_a=(),
                          est_s=True,
                          est_b=False,
                          biasType=bias_models.NoBias,
                          inflation=1.002,
                          reject_inflation=1.002,
                          std_psi=0.001,
                          std_a=0.001,
                          alpha_distr='normal',
                          ensure_mean=False,
                          num_DA_blind=0,
                          num_SE_only=0,
                          start_ensemble_forecast=0.,
                          get_cost=False
                          )

    def __init__(self, **model_dict):
        # ================= INITIALISE THERMOACOUSTIC MODEL ================== ##
        for key, val in self.attr.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
            else:
                setattr(self, key, val)

        for key, val in Model.attr_model.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
            else:
                setattr(self, key, val)

        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()
        self.psi = np.array([self.psi0]).T
        self.ensemble = False
        self.filename = ""
        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi])
        self.hist_t = np.array([self.t])
        self.hist_J = []
        # ========================== DEFINE LENGTHS ========================== ##
        self.Na = 0
        self.precision_t = int(-np.log10(self.dt)) + 2

    @property
    def Nphi(self):
        return len(self.psi0)
    @property
    def Nq(self):
        return np.shape(self.getObservables())[0]

    @property
    def N(self):
        return self.Nphi + self.Na + self.Nq

    def copy(self):
        return deepcopy(self)

    def getObservableHist(self, Nt=0, **kwargs):
        return self.getObservables(Nt, **kwargs)

    def print_model_parameters(self):
        print('\n ------------------ {} Model Parameters ------------------ '.format(self.name))
        for k in self.attr.keys():
            print('\t {} = {}'.format(k, getattr(self, k)))

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##
    @property
    def M(self):
        if not hasattr(self, '_M'):
            setattr(self, '_M', np.hstack((np.zeros([self.Nq, self.Na+self.Nphi]),
                                           np.eye(self.Nq))))
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
    def addUncertainty(mean, std, m, method='normal', param_names=None, ensure_mean=False):
        if method == 'normal':
            if isinstance(std, float):
                cov = np.diag((mean * std) ** 2)
            else:
                raise TypeError('std in normal distribution must be float not {}'.format(type(std)))
            ens = rng.multivariate_normal(mean, cov, m).T
        elif method == 'uniform':
            ens = np.zeros((len(mean), m))
            if isinstance(std, float):
                for ii, pp in enumerate(mean):
                    if abs(std) <= .5:
                        ens[ii, :] = pp * (1. + rng.uniform(-std, std, m))
                    else:
                        ens[ii, :] = rng.uniform(pp - std, pp + std, m)
            elif isinstance(std, dict):
                if param_names is not None:
                    for ii, key in enumerate(param_names):
                        ens[ii, :] = rng.uniform(std[key][0], std[key][1], m)
                else:
                    for ii, _ in enumerate(mean):
                        ens[ii, :] = rng.uniform(std[ii][0], std[ii][1], m)
            else:
                raise TypeError('std in normal distribution must be float or dict')
        else:
            raise ValueError('Parameter distribution {} not recognised'.format(method))

        if ensure_mean:
            ens[:, 0] = mean
        return ens

    def getOutputs(self):
        out = dict(name=self.name,
                   hist_y=self.getObservableHist(),
                   y_lbls=self.obsLabels,
                   bias=self.bias.getOutputs(),
                   hist_t=self.hist_t,
                   hist=self.hist,
                   hist_J=self.hist_J,
                   alpha0=self.alpha0
                   )
        for key in self.attr.keys():
            out[key] = getattr(self, key)
        if self.ensemble:
            for key in self.attr_ens.keys():
                out[key] = getattr(self, key)
        return out

    def initEnsemble(self, **DAdict):
        DAdict = DAdict.copy()
        self.ensemble = True
        for key, val in Model.attr_ens.items():
            if key in DAdict.keys():
                setattr(self, key, DAdict[key])
            else:
                setattr(self, key, val)
        self.filename = '{}_ensemble_m{}'.format(self.name, self.m)
        if hasattr(self, 'modify_settings'):
            self.modify_settings()
        # --------------- RESET INITIAL CONDITION AND HISTORY --------------- ##
        # Note: if est_a and est_b psi = [psi; alpha; biasWeights]
        ensure_mean = self.ensure_mean
        mean_psi = np.array(self.psi0) * rng.uniform(0.9, 1.1, len(self.psi0))
        new_psi0 = self.addUncertainty(mean_psi, self.std_psi, self.m,
                                       method='normal', ensure_mean=ensure_mean)

        if self.est_a:  # Augment ensemble with estimated parameters
            mean_a = np.array([getattr(self, pp) for pp in self.est_a])
            new_alpha0 = self.addUncertainty(mean_a, self.std_a, self.m, method=self.alpha_distr,
                                             param_names=self.est_a, ensure_mean=ensure_mean)
            new_psi0 = np.vstack((new_psi0, new_alpha0))
            self.Na = len(self.est_a)

        # RESET ENSEMBLE HISTORY
        self.updateHistory(psi=new_psi0, reset=True)

    def initBias(self, **Bdict):

        if 'biasType' in Bdict.keys():
            self.biasType = Bdict['biasType']

        # Assign some required items
        for key, default_value in zip(['t_val', 't_train', 't_test'],
                                      [self.t_CR, self.t_transient, self.t_CR]):
            if key not in Bdict.keys():
                Bdict[key] = default_value

        # Initialise bias. Note: self.bias is now an instance of the bias class
        self.bias = self.biasType(y=self.getObservables(),
                                  t=self.t, dt=self.dt, **Bdict)
        # Create bias history
        b = self.bias.getBias
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

    @staticmethod
    def forecast(y0, fun, t, params, alpha=None):
        # SOLVE IVP ========================================
        assert len(t) > 1
        out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params, alpha))
        psi = out.y.T

        # ODEINT =========================================== THIS WORKS AS IF HARD CODED
        # psi = odeint(fun, y0, t_interp, (params,))
        #
        # HARD CODED RUGGE KUTTA 4TH ========================
        # psi = RK4(t_interp, y0, fun, params)

        return psi

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
        # t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
        t = np.round(self.t + np.arange(0, Nt+1) * self.dt, self.precision_t)

        self_dict = self.govEqnDict()

        if not self.ensemble:
            psi = [Model.forecast(self.psi[:, 0], self.timeDerivative, t, params=self_dict, alpha=self.alpha0)]

        else:
            if not averaged:
                alpha = self.getAlpha()
                fun_part = partial(Model.forecast, fun=self.timeDerivative, t=t, params=self_dict)
                sol = [self.pool.apply_async(fun_part, kwds={'y0': self.psi[:, mi].T, 'alpha': alpha[mi]})
                       for mi in range(self.m)]
                psi = [s.get() for s in sol]
            else:
                psi_mean0 = np.mean(self.psi, 1, keepdims=True)
                psi_deviation = self.psi - psi_mean0

                if alpha is None:
                    alpha = self.getAlpha(psi_mean0)[0]
                psi_mean = Model.forecast(y0=psi_mean0[:, 0], fun=self.timeDerivative, t=t,
                                          params=self_dict, alpha=alpha)

                if np.mean(np.std(self.psi[:len(self.psi0)]/np.array([self.psi0]).T, axis=0)) < 2.:
                    psi_deviation /= psi_mean0
                    psi = [psi_mean * (1 + psi_deviation[:, ii]) for ii in range(self.m)]
                else:
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
    attr: dict = dict(dt=1.0E-4, t_transient=1.5, t_CR=0.04,
                      omega=2 * np.pi * 120., law='tan',
                      zeta=60., beta=70., kappa=4.0, gamma=1.7)  # beta, zeta [rad/s]

    params: list = ['beta', 'zeta', 'kappa']  # ,'omega', 'gamma']

    param_labels = dict(beta='$\\beta$', zeta='$\\zeta$', kappa='$\\kappa$')

    # __________________________ Init method ___________________________ #
    def __init__(self, **TAdict):
        super().__init__(**TAdict)
        if 'psi0' not in TAdict.keys():
            self.psi0 = [0.1, 0.1]  # initialise eta and mu
            self.updateHistory(reset=True)

    # _______________ VdP specific properties and methods ________________ #
    @property
    def param_lims(self):
        return dict(zeta=(5, 120),
                    kappa=(0.1, 20),
                    beta=(5, 120),
                    gamma=(0., 5.)
                    )

    @property
    def obsLabels(self):
        return "$\\eta$"

    def getObservables(self, Nt=1):
        if Nt == 1:  # required to reduce from 3 to 2 dimensions
            return self.hist[-1, :1, :]
        else:
            return self.hist[-Nt:, :1, :]

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        return dict(law=self.law, omega=self.omega)

    @staticmethod
    def timeDerivative(t, psi, P, A):
        eta, mu = psi[:2]
        dmu_dt = - P['omega'] ** 2 * eta + mu * (A['beta'] - A['zeta'])
        # Add nonlinear term
        if P['law'] == 'cubic':  # Cubic law
            dmu_dt -= mu * A['kappa'] * eta ** 2
        elif P['law'] == 'tan':  # arc tan model
            dmu_dt -= mu * (A['kappa'] * eta ** 2) / (1. + A['kappa'] / A['beta'] * eta ** 2)

        return (mu, dmu_dt) + (0,) * (len(psi) - 2)


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretisation and gain-delay sqrt heat release law.
        Args:
            TAdict: dictionary with the model parameters. If not defined, the default value is used.
                > Nm - Number of Galerkin modes
                > Nc - Number of Chebyshev modes
                > beta - Heat source strength [-]
                > tau - Time delay [s]
                > C1 - First damping constant [-]
                > C2 - Second damping constant [-]
                > xf - Flame location [m]
                > L - Tube length [m]
    """

    name: str = 'Rijke'
    attr: dict = dict(dt=1E-4, t_transient=1., t_CR=0.02,
                      Nm=10, Nc=10, Nmic=6,
                      beta=4.0, tau=1.5E-3, C1=.05, C2=.01, kappa=1E5,
                      xf=0.2, L=1., law='sqrt')
    params: list = ['beta', 'tau', 'C1', 'C2', 'kappa']

    param_labels = dict(beta='$\\beta$', tau='$\\tau$', C1='$C_1$', C2='$C_2$', kappa='$\\kappa$')

    def __init__(self, **TAdict):
        super().__init__(**TAdict)

        self.tau_adv = self.tau
        if 'psi0' not in TAdict.keys():
            self.psi0 = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])
            # self.resetInitialConditions()
            self.updateHistory(reset=True)
        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

        # ------------------------------------------------------------------------------------- #

        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nmic + 1)[:-1]

        # Define modes frequency of each mode and sin cos etc
        self.j = np.arange(1, self.Nm + 1)
        self.jpiL = self.j * np.pi / self.L
        self.sinomjxf = np.sin(self.jpiL * self.xf)
        self.cosomjxf = np.cos(self.jpiL * self.xf)

        # Mean Flow Properties
        def weight_avg(y1, y2):
            return self.xf / self.L * y1 + (1. - self.xf / self.L) * y2

        self.meanFlow = dict(u=weight_avg(10, 11.1643),
                             p=101300.,
                             gamma=1.4,
                             T=weight_avg(300, 446.5282),
                             R=287.1
                             )
        self.meanFlow['rho'] = self.meanFlow['p'] / (self.meanFlow['R'] * self.meanFlow['T'])
        self.meanFlow['c'] = np.sqrt(self.meanFlow['gamma'] * self.meanFlow['R'] * self.meanFlow['T'])

        # Wave parameters ############################################################################################
        # c1: 347.2492    p1: 1.0131e+05      rho1: 1.1762    u1: 10          M1: 0.0288          T1: 300
        # c2: 423.6479    p2: 101300          rho2: 0.7902    u2: 11.1643     M2: 0.0264          T2: 446.5282
        # Tau: 0.0320     Td: 0.0038          Tu: 0.0012      R_in: -0.9970   R_out: -0.9970      Su: 0.9000
        # Qbar: 5000      R_gas: 287.1000     gamma: 1.4000
        ##############################################################################################################

    def modify_settings(self):
        if self.est_a and 'tau' in self.est_a:
            extra_Nc = self.Nc - 50
            self.tau_adv, self.Nc = 1E-2, 50
            self.psi0 = np.hstack([self.psi0, np.zeros(extra_Nc)])
            # self.resetInitialConditions()
            self.updateHistory(reset=True)

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def param_lims(self):
        return dict(beta=(0.01, 5),
                    tau=(1E-6, self.tau_adv),
                    C1=(0., 1.),
                    C2=(0., 1.),
                    kappa=(1E3, 1E8)
                    )

    @property
    def obsLabels(self, loc=None, velocity=False):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        if not velocity:
            return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]
        else:
            return [["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()],
                    ["$u'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]]

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

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(Nm=self.Nm,
                 Nc=self.Nc,
                 N=self.N,
                 Na=self.Na,
                 j=self.j,
                 jpiL=self.jpiL,
                 cosomjxf=self.cosomjxf,
                 sinomjxf=self.sinomjxf,
                 tau_adv=self.tau_adv,
                 meanFlow=self.meanFlow,
                 Dc=self.Dc,
                 gc=self.gc,
                 L=self.L,
                 law=self.law
                 )
        return d

    @staticmethod
    def timeDerivative(t, psi, P, A):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
                P: dictionary with all the case parameters
                A: dictionary of varying parameters
            Returns:
                concatenation of the state vector time derivative
        """

        eta = psi[:P['Nm']]
        mu = psi[P['Nm']:2 * P['Nm']]
        v = psi[2 * P['Nm']:P['N'] - P['Na']]

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, P['cosomjxf']), v))

        # Evaluate u(t_interp-tau) i.e. velocity at the flame at t_interp - tau
        x_tau = A['tau'] / P['tau_adv']
        if x_tau < 1:
            f = splrep(P['gc'], v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            raise Exception("tau = {} can't_interp be larger than tau_adv = {}".format(A['tau'], P['tau_adv']))

        # Compute damping and heat release law
        zeta = A['C1'] * P['j'] ** 2 + A['C2'] * P['j'] ** .5

        MF = P['meanFlow']  # Physical properties
        if P['law'] == 'sqrt':
            qdot = MF['p'] * MF['u'] * A['beta'] * (
                    np.sqrt(abs(1. / 3 + u_tau / MF['u'])) - np.sqrt(1. / 3))  # [W/m2]=[m/s3]
        elif P['law'] == 'tan':
            qdot = A['beta'] * np.sqrt(A['beta'] / A['kappa']) * np.arctan(
                np.sqrt(A['beta'] / A['kappa']) * u_tau)  # [m / s3]

        qdot *= -2. * (MF['gamma'] - 1.) / P['L'] * P['sinomjxf']  # [Pa/s]

        # governing equations
        deta_dt = P['jpiL'] / MF['rho'] * mu
        dmu_dt = - P['jpiL'] * MF['gamma'] * MF['p'] * eta - MF['c'] / P['L'] * zeta * mu + qdot
        dv_dt = - 2. / P['tau_adv'] * np.dot(P['Dc'], v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(P['Na'])))


# %% =================================== LORENZ 63 MODEL ============================================== %% #
class Lorenz63(Model):
    """ Lorenz 63 Class
    """

    name: str = 'Lorenz63'
    attr: dict = dict(rho=28., sigma=10., beta=8. / 3., dt=0.02)

    params: list = ['rho', 'sigma', 'beta']
    param_labels = dict(rho='$\\rho$', sigma='$\\sigma$', beta='$\\beta$')

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict, DAdict=None):

        super().__init__(**TAdict)

        # self.t_transient = 200.
        # self.dt = 0.02
        self.t_CR = 5.

        if 'psi0' not in TAdict.keys():
            self.psi0 = [1.0, 1.0, 1.0]  # initialise x, y, z
            # self.resetInitialConditions()
            self.updateHistory(reset=True)

        # set limits for the parameters
        self.param_lims = dict(rho=(None, None), beta=(None, None), sigma=(None, None))

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obsLabels(self):
        return ["$x$", '$y$', '$z$']

    def getObservables(self, Nt=1):
        if Nt == 1:
            return self.hist[-1, :, :]
        else:
            return self.hist[-Nt:, :, :]

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        return None

    @staticmethod
    def timeDerivative(t, psi, params, alpha):
        x1, x2, x3 = psi[:3]
        dx1 = alpha['sigma'] * (x2 - x1)
        dx2 = x1 * (alpha['rho'] - x3) - x2
        dx3 = x1 * x2 - alpha['beta'] * x3
        return (dx1, dx2, dx3) + (0,) * (len(psi) - 3)


# %% =================================== 2X VAN DER POL MODEL ============================================== %% #
class Annular(Model):
    """ Annular combustor model, wich consists of two coupled oscillators
    """

    name: str = 'Annular'
    attr: dict = dict(dt=1 / 51.2E3, t_transient=0.5, t_CR=0.03,
                      n=1., theta_b=0.63, theta_e=0.66, omega=1090., epsilon=0.0023,
                      nu=17., beta_c2=17., kappa=1.2E-4)  # values in Fig.4

    # attr['nu'], attr['beta_c2'] = 30., 5.  # spin
    # attr['nu'], attr['beta_c2'] = 1., 25.  # stand
    attr['nu'], attr['beta_c2'] = 20., 18.  # mix

    params: list = ['omega', 'nu', 'beta_c2', 'kappa', 'epsilon', 'theta_b', 'theta_e']

    param_labels = dict(omega='$\\omega$', nu='$\\nu$', beta_c2='$c_2\\beta $', kappa='$\\kappa$',
                        epsilon='$\\epsilon$', theta_b='$\\Theta_\\beta$', theta_e='$\\Theta_\\epsilon$')

    # __________________________ Init method ___________________________ #
    def __init__(self, **TAdict):

        super().__init__(**TAdict)

        self.theta_mic = np.radians([0, 60, 120, 240])

        if 'psi0' not in TAdict.keys():
            self.psi0 = [100, -10, -100, 10]  # initialise \eta_a, \dot{\eta_a}, \eta_b, \dot{\eta_b}
            # self.resetInitialConditions()
            self.updateHistory(reset=True)

    # set limits for the parameters ['omega', 'nu', 'beta_c2', 'kappa']
    @property
    def param_lims(self):
        return dict(omega=(1000, 1300),
                    nu=(-40., 60.),
                    beta_c2=(1., 60.),
                    kappa=(None, None),
                    epsilon=(None, None),
                    theta_b=(0, 2 * np.pi),
                    theta_e=(0, 2 * np.pi)
                    )

    # _______________  Specific properties and methods ________________ #
    # @property
    # def obsLabels(self):
    #     return ["\\eta_1", '\\eta_2']

    def getObservables(self, Nt=1, loc=None, modes=False):
        """
        :return: pressure measurements at theta = [0º, 60º, 120º, 240º`]
        p(θ, t) = η1(t) * cos(nθ) + η2(t) * sin(nθ).
        """
        if loc is None:
            loc = self.theta_mic

        if modes:
            self.obsLabels = ["$\\eta_1$", '$\\eta_2$']
            return self.hist[-Nt:, [0, 2], :]

        self.obsLabels = ["$p(\\theta={})$".format(int(np.round(np.degrees(th)))) for th in loc]

        loc = np.array(loc)
        eta1, eta2 = self.hist[-Nt:, 0, :], self.hist[-Nt:, 2, :]

        if max(loc) > 2 * np.pi:
            raise ValueError('Theta must be in radians')

        p_mics = np.array([eta1 * np.cos(th) + eta2 * np.sin(th) for th in loc]).transpose(1, 0, 2)

        if Nt == 1:
            return p_mics.squeeze(axis=0)
        else:
            return p_mics

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict()
        return d

    @staticmethod
    def timeDerivative(t, psi, params, alpha):
        y_a, z_a, y_b, z_b = psi[:4]  # y = η, and z = dη/dt

        def k1(y1, y2, sign):
            return 2 * alpha['nu'] - 3. / 4 * alpha['kappa'] * (3 * y1 ** 2 + y2 ** 2) + \
                   sign / 2. * alpha['beta_c2'] * np.cos(2. * alpha['theta_b'])

        k2 = 0.5 * alpha['beta_c2'] * np.sin(2. * alpha['theta_b']) - 3. / 2 * alpha['kappa'] * y_a * y_b

        def k3(y1, y2, sign):
            return alpha['omega'] ** 2 * (y1 + alpha['epsilon'] / 2. * (sign * y1 * np.cos(2. * alpha['theta_e']) +
                                                                        y2 * np.sin(2. * alpha['theta_e'])))

        dz_a = z_a * k1(y_a, y_b, 1) + z_b * k2 - k3(y_a, y_b, 1)
        dz_b = z_b * k1(y_b, y_a, -1) + z_a * k2 - k3(y_b, y_a, -1)

        return (z_a, dz_a, z_b, dz_b) + (0,) * (len(psi) - 4)


if __name__ == '__main__':
    MyModel = Annular
    paramsTA = dict(dt=1 / 51.2E3)

    animate = 0
    anim_name = 'mov_mix_epsilon.gif'

    # Non-ensemble case =============================
    t1 = time.time()
    case = MyModel(paramsTA)
    state, t_ = case.timeIntegrate(int(case.t_transient * 3 / case.dt))
    case.updateHistory(state, t_)

    print(case.dt)
    print('Elapsed time = ', str(time.time() - t1))

    fig1 = plt.figure(figsize=[12, 3], layout="constrained")
    subfigs = fig1.subfigures(1, 2, width_ratios=[1.2, 1])

    ax = subfigs[0].subplots(1, 2)
    ax[0].set_title(MyModel.name)

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    # State evolution
    y, lbl = case.getObservableHist(modes=True), case.obsLabels

    ax[0].scatter(t_h, y[:, 0], c=t_h, label=lbl, cmap='Blues', s=10, marker='.')

    ax[0].set(xlabel='$t$', ylabel=lbl[0])
    i, j = [0, 1]

    if len(lbl) > 1:
        # ax[1].scatter(y[:, 0]/np.max(y[:,0]), y[:, 1]/np.max(y[:,1]), c=t_h, s=3, marker='.', cmap='Blues')
        # ax[1].set(xlabel='$'+lbl[0]+'/'+lbl[0]+'^\mathrm{max}$', ylabel='$'+lbl[1]+'/'+lbl[1]+'^\mathrm{max}$')
        ax[1].scatter(y[:, 0], y[:, 1], c=t_h, s=3, marker='.', cmap='Blues')
        ax[1].set(xlabel=lbl[0], ylabel=lbl[1])
    else:
        ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='green')

    ax[1].set_aspect(1. / ax[1].get_data_ratio())

    if not animate:
        ax2 = subfigs[1].subplots(2, 1)

        y, lbl = case.getObservableHist(modes=False), case.obsLabels
        y = np.mean(y, axis=-1)
        # print(np.min(y, axis=0))
        # sorted_id = np.argsort(np.max(abs(y[-1000:]), axis=0))
        # print(sorted_id)
        # y = y[:, sorted_id]
        # lbl = [lbl[idx] for idx in sorted_id]

        for ax in ax2:
            ax.plot(t_h, y / 1E3)
        ax2[0].set_title('Acoustic Pressure')
        ax2[0].legend(lbl, bbox_to_anchor=(1., 1.), loc="upper left", ncol=1, fontsize='small')
        ax2[0].set(xlim=[t_h[0], t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
        ax2[1].set(xlim=[t_h[-1] - case.t_CR, t_h[-1]], xlabel='$t$', ylabel='$p$ [kPa]')
    else:
        ax2 = subfigs[1].subplots(1, 1, subplot_kw={'projection': 'polar'})
        angles = np.linspace(0, 2 * np.pi, 200)  # Angles from 0 to 2π
        y, lbl = case.getObservableHist(modes=False, loc=angles), case.obsLabels
        y = np.mean(y, axis=-1)

        radius = [0, 0.5, 1]
        theta, r = np.meshgrid(angles, radius)

        # Remove radial tick labels
        ax2.set_yticklabels([])
        ax2.grid(False)

        # Add a white concentric circle
        circle_radius = 0.5
        ax2.plot(angles, [circle_radius] * len(angles), color='black', lw=1)

        idx_max = np.argmax(y[:, 0])
        polar_mesh = ax2.pcolormesh(theta, r, [y[idx_max].T] * len(radius), shading='auto', cmap='RdBu')

        ax2.set_theta_zero_location('S')  # Set zero angle to the north (top)
        ax2.set_title('Acoustic Pressure')
        ax2.set_theta_direction(1)  # Set clockwise rotation

        start_i = np.argmin(abs(t_h[-1] - (t_h[-1] - case.t_CR)))

        print((t_h[-1]))
        start_i = int((t_h[-1] - .03) // case.dt)

        print(start_i, t_h[start_i])

        dt_gif = 10

        t_gif = t_h[start_i::dt_gif]

        y_gif = y[start_i::dt_gif]


        def update(frame):
            ax2.fill(angles, [circle_radius] * len(angles), color='white')
            polar_mesh.set_array([y_gif[frame].T] * len(radius))
            ax2.set_title('Acoustic Pressure $t$ = {:.3f}'.format(t_gif[frame]))  # , fontsize='small')#, labelpad=50)


        plt.colorbar(polar_mesh, label='Pressure', shrink=0.75)
        anim = FuncAnimation(fig1, update, frames=len(t_gif))
        dt = t_gif[1] - t_gif[0]
        anim.save(anim_name, fps=dt_gif * 10)

    plt.show()

    # # Ensemble case =============================
    # paramsDA = dict(m=10, est_a=['beta'])
    # case = MyModel(paramsTA, paramsDA)
    #
    # t1 = time.time()
    # for _ in range(1):
    #     state, t_ = case.timeIntegrate(int(1. / case.dt))
    #     case.updateHistory(state, t_)
    # for _ in range(5):
    #     state, t_ = case.timeIntegrate(int(.1 / case.dt), averaged=True)
    #     case.updateHistory(state, t_)
    #
    # print('Elapsed time = ', str(time.time() - t1))
    #
    # t_h = case.hist_t
    # t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])
    #
    # _, ax = plt.subplots(1, 3, figsize=[15, 5])
    # plt.suptitle('Ensemble case')
    # # State evolution
    # y, lbl = case.getObservableHist(), case.obsLabels
    # lbl = lbl[0]
    # ax[0].plot(t_h, y[:, 0], color='blue', label=lbl)
    # i, j = [0, 1]
    # ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='blue')
    #
    # ax[0].set(xlabel='t', ylabel=lbl, xlim=[t_h[0], t_h[-1]])
    # ax[1].set(xlabel='t', xlim=[t_h[-t_zoom], t_h[-1]])
    #
    # # Params
    #
    # ai = - case.Na
    # max_p, min_p = -1000, 1000
    # c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    # mean = np.mean(case.hist, -1, keepdims=True)
    # for p in case.est_a:
    #     superscript = '^\mathrm{init}$'
    #     # reference_p = truth['true_params']
    #     reference_p = case.alpha0
    #
    #     mean_p = mean[:, ai].squeeze() / reference_p[p]
    #     std = np.std(case.hist[:, ai] / reference_p[p], axis=1)
    #
    #     max_p = max(max_p, max(mean_p))
    #     min_p = min(min_p, min(mean_p))
    #
    #     ax[2].plot(t_h, mean_p, color=c[-ai], label= p + '/' + p[:-1] + superscript + '$')
    #
    #     ax[2].set(xlabel='$t$', xlim=[t_h[0], t_h[-1]])
    #     ax[2].fill_between(t_h, mean_p + std, mean_p - std, alpha=0.2, color=c[-ai])
    #     ai += 1
    # ax[2].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
    # ax[2].plot(t_h[1:], t_h[1:] / t_h[1:], '-', color='k', linewidth=.5)
    # ax[2].set(ylim=[min_p - 0.1, max_p + 0.1])
    #
    # plt.tight_layout()
    # plt.show()
