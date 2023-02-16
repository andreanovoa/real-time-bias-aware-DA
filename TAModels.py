from scipy.interpolate import splrep, splev
import pylab as plt

import Bias
from Util import Cheb, RK4
import os
import time

import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
from functools import partial
from copy import deepcopy
from datetime import date

# os.environ["OMP_NUM_THREADS"] = '1'
num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)

rng = np.random.default_rng(6)


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general thermoacoustic model
        properties and methods definitions.
    """
    attr_parent: dict = dict(dt=1E-4, t=0.,
                             t_transient=2., t_CR=0.2,
                             psi0=np.empty(1), ensemble=False)

    attr_ens: dict = dict(m=10, est_p=[], est_s=True, est_b=False,
                          biasType=Bias.NoBias, inflation=1.01,
                          std_psi=0.001, std_a=0.001, alpha_distr='normal',
                          num_DA_blind=0, num_SE_only=0,
                          start_ensemble_forecast=0.)

    def __init__(self, TAdict):
        model_dict = TAdict.copy()
        # ================= INITIALISE THERMOACOUSTIC MODEL ================== ##
        for key, val in self.attr_child.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
            else:
                setattr(self, key, val)

        for key, val in Model.attr_parent.items():
            if key in model_dict.keys():
                setattr(self, key, model_dict[key])
            else:
                setattr(self, key, val)

        self.alpha0 = {par: getattr(self, par) for par in self.params}
        self.alpha = self.alpha0.copy()
        self.N, self.Na = len(self.psi0), 0
        self.psi = np.array([self.psi0]).T

        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi])
        self.hist_t = np.array([self.t])
        self.hist_J = []

    def copy(self):
        return deepcopy(self)

    def getObservableHist(self, Nt=0):
        return self.getObservables(Nt)

    def printModelParameters(self):
        print('\n ------------------ {} Model Parameters ------------------ '.format(self.name))
        for k in self.attr_child.keys():
            print('\t {} = {}'.format(k, getattr(self, k)))

    # --------------------- DEFINE OBS-STATE MAP --------------------- ##
    @property
    def M(self):
        if not hasattr(self, '_M'):
            obs = self.getObservables()
            Nq = np.shape(obs)[0]
            # if ensemble.est_b:
            #     y0 = np.concatenate(y0, np.zeros(ensemble.bias.Nb))
            y0 = np.concatenate((np.zeros(self.N), np.ones(Nq)))
            self._M = np.zeros((Nq, len(y0)))
            iq = 0
            for ii in range(len(y0)):
                if y0[ii] == 1:
                    self._M[iq, ii] = 1
                    iq += 1
        return self._M

    # -------------- Functions for update/initialise the model ------------------- #
    @staticmethod
    def addUncertainty(y_mean, y_std, m, method='normal'):
        if method == 'normal':
            cov = np.diag((y_std * np.ones(len(y_mean))) ** 2)
            return (y_mean * rng.multivariate_normal(np.ones(len(y_mean)), cov, m)).T
        elif method == 'uniform':
            ens_aug = np.zeros((len(y_mean), m))
            for ii, pp in enumerate(y_mean):
                ens_aug[ii, :] = pp * (1. + rng.uniform(-y_std, y_std, m))
            return ens_aug
        else:
            raise 'Parameter distribution not recognised'

    def resetInitialConditions(self):
        self.psi = np.array([self.psi0]).T
        self.hist = np.array([self.psi])
        self.N = len(self.psi0)

    def initEnsemble(self, DAdict):
        DAdict = DAdict.copy()
        self.ensemble = True
        for key, val in Model.attr_ens.items():
            if key in DAdict.keys():
                setattr(self, key, DAdict[key])
            else:
                setattr(self, key, val)

        # ----------------------- DEFINE STATE MATRIX ----------------------- ##
        # Note: if est_p and est_b psi = [psi; alpha; biasWeights]
        # if self.m > 1:
            # if True:
        mean_psi = np.array(self.psi0)  # * rng.uniform(0.9, 1.1, len(self.psi0))
        # self.psi = self.addUncertainty(mean, self.std_psi, self.m, method=self.alpha_distr)
        cov = np.diag((self.std_psi ** 2 * abs(mean_psi)))
        self.psi = rng.multivariate_normal(mean_psi, cov, self.m).T
        if 'ensure_mean' in DAdict.keys() and DAdict['ensure_mean']:
            print(self.psi.shape)
            print('chiky')
            self.psi[:, 0] = np.array(self.psi0)
            print(self.psi.shape)

        if len(self.est_p) > 0:  # Augment ensemble with estimated parameters
            self.Na = len(self.est_p)
            self.N += self.Na
            mean_a = np.array([getattr(self, pp) for pp in self.est_p])  # * rng.uniform(0.9, 1.1, len(self.psi0))
            ens_a = self.addUncertainty(mean_a, self.std_a, self.m, method=self.alpha_distr)

            if 'ensure_mean' in DAdict.keys() and DAdict['ensure_mean']:
                ens_a[:, 0] = mean_a

            self.psi = np.vstack((self.psi, ens_a))

        # ------------------------ INITIALISE BIAS ------------------------ ##
        if 'Bdict' not in DAdict.keys():
            DAdict['Bdict'] = {}
        Bdict = DAdict['Bdict'].copy()
        self.initBias(Bdict)

        # ========================== RESET ENSEMBLE HISTORY ========================== ##
        self.hist = np.array([self.psi])

    def initBias(self, Bdict):
        # Assign some required items
        Bdict['est_b'] = self.est_b
        Bdict['dt'] = self.dt
        if 'filename' not in Bdict.keys():  # default bias file name
            Bdict['filename'] = self.name + '_' + str(date.today())
        # Initialise bias. Note: self.bias is now an instance of the bias class
        yb = self.getObservables()
        self.bias = self.biasType(yb, self.t, Bdict)
        # # Augment state matrix if you want to infer bias weights

        # Create bias history
        b = self.bias.getBias(yb)
        self.bias.updateHistory(b, self.t, reset=True)

    def updateHistory(self, psi, t):
        self.hist = np.concatenate((self.hist, psi), axis=0)
        self.hist_t = np.hstack((self.hist_t, t))
        self.psi = psi[-1]
        self.t = t[-1]

    # -------------- Functions required for the forecasting ------------------- #
    @property
    def pool(self):
        if not hasattr(self, '_pool'):
            self._pool = mp.Pool()
        return self._pool

    def close(self):
        self.pool.close()
        self.pool.join()
        delattr(self, "_pool")

    @staticmethod
    def forecast(y0, fun, t, params, alpha=None):
        # SOLVE IVP ========================================
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
            for param in self.est_p:
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
                psi: forecasted ensemble state
                t: time of the propagated psi
        """
        t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
        self_dict = self.govEqnDict()

        if not self.ensemble:
            psi = [Model.forecast(self.psi[:, 0], self.timeDerivative, t, params=self_dict, alpha=self.alpha0)]
            psi = np.array(psi).transpose(1, 2, 0)
            return psi[1:], t[1:]

        if not averaged:
            alpha = self.getAlpha()
            fun_part = partial(Model.forecast, fun=self.timeDerivative, t=t, params=self_dict)
            sol = [self.pool.apply_async(fun_part, kwds={'y0': self.psi[:, mi].T, 'alpha': alpha[mi]})
                   for mi in range(self.m)]
            psi = [s.get() for s in sol]
        else:
            psi_mean = np.mean(self.psi, 1, keepdims=True)
            psi_std = (self.psi - psi_mean) / psi_mean
            if alpha is None:
                alpha = self.getAlpha(psi_mean)[0]
            psi_mean = Model.forecast(y0=psi_mean[:, 0], fun=self.timeDerivative, t=t, params=self_dict, alpha=alpha)
            psi = [psi_mean * (1 + psi_std[:, ii]) for ii in range(self.m)]

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        psi = np.array(psi).transpose(1, 2, 0)
        return psi[1:], t[1:]



# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
    attr_child: dict = dict(omega=2 * np.pi * 120., law='tan',
                            zeta=60., beta=70., kappa=4.0, gamma=1.7)  # beta, zeta [rad/s]
    params: list = ['zeta', 'kappa', 'beta']  # ,'omega', 'gamma']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None, DAdict=None):
        if TAdict is None:
            TAdict = {}

        super().__init__(TAdict)

        if 'psi0' not in TAdict.keys():
            self.psi0 = [0.1, 0.1]  # initialise eta and mu
            self.resetInitialConditions()

        # initialise model history
        if DAdict is not None:
            self.initEnsemble(DAdict)

        # set limits for the parameters
        self.param_lims = dict(zeta=(20, 120), kappa=(0.1, 10.),
                               gamma=(None, None), beta=(20, 120))

    # _______________ VdP specific properties and methods ________________ #
    @property
    def obsLabels(self):
        return "$\\eta$"

    def getObservables(self, Nt=1):
        if Nt == 1: # required to reduce from 3 to 2 dimensions
            return self.hist[-1, [0], :]
        else:
            return self.hist[-Nt:, [0], :]

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(law=self.law,
                 N=self.N,
                 Na=self.Na,
                 omega=self.omega
                 )
        if d['Na'] > 0:
            d['est_p'] = self.est_p
        return d

    @staticmethod
    def timeDerivative(t, psi, P, A):
        eta, mu = psi[:2]
        dmu_dt = - P['omega'] ** 2 * eta + mu * (A['beta'] - A['zeta'])
        # Add nonlinear term
        if P['law'] == 'cubic':  # Cubic law
            dmu_dt -= mu * A['kappa'] * eta ** 2
        elif P['law'] == 'tan':  # arc tan model
            dmu_dt -= mu * (A['kappa'] * eta ** 2) / (1. + A['kappa'] / A['beta'] * eta ** 2)

        return (mu, dmu_dt) + (0,) * P['Na']


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretisation and gain-delay sqrt heat release law.
        Args:
            TAdict: dictionary with the model parameters. If not defined, the default value is used.
                > Nm [10] - Number of Galerkin modes
                > Nc [50] - Number of Chebyshev modes
                > beta [0.4] - Heat source strength [-]
                > tau [2E-3] - Time delay [s]
                > C1 [.1] - First damping constant [-]
                > C2 [.06] - Second damping constant [-]
                > xf- Flame location [m]
                > L - Tube length [m]
    """

    name: str = 'Rijke'
    attr_child: dict = dict(Nm=10, Nc=10, Nmic=6,
                            beta=0.6, tau=2.E-3, C1=.05, C2=.01, kappa=1E5,
                            xf=0.2, L=1., law='sqrt')
    params: list = ['beta', 'tau', 'C1', 'C2', 'kappa']

    def __init__(self, TAdict=None, DAdict=None):
        if TAdict is None:
            TAdict = {}
        super().__init__(TAdict)

        if DAdict is None:
            DAdict = {}

        if 'est_p' in DAdict.keys() and 'tau' in DAdict['est_p']:
            self.tau_adv, self.Nc = 1E-2, 50
        else:
            self.tau_adv = self.tau

        if 'psi0' not in TAdict.keys():
            self.psi0 = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])
            self.resetInitialConditions()

        assert self.N == self.Nc + 2 * self.Nm

        self.param_lims = dict(beta=(0.01, 5),
                               tau=(1E-6, self.tau_adv),
                               C1=(0., 1.),
                               C2=(0., 1.),
                               kappa=(1E3, 1E8)
                               )
        # ------------------------------------------------------------------------------------- #
        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

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

        # ------------------------------------------------------------------------------------- #
        if DAdict is not None:
            self.initEnsemble(DAdict)

    # _______________ Rijke specific properties and methods ________________ #

    @property
    def obsLabels(self, loc=None, velocity=False):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        if not velocity:
            return ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]
        else:
            return [["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()],
                    ["$u'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]]

    def getObservables(self, Nt=1, loc=None, velocity=False):
        if loc is None:
            loc = np.expand_dims(self.x_mic, axis=1)
        om = np.array([self.jpiL])

        eta = self.hist[-Nt:, :self.Nm, :]
        mu = self.hist[-Nt:, self.Nm:2 * self.Nm, :]

        # Compute acoustic pressure and velocity at locations
        p_mic = -np.dot(np.sin(np.dot(loc, om)), mu)
        p_mic = p_mic.transpose(1, 0, 2)
        if Nt == 1:
            p_mic = p_mic[0]

        if velocity:
            u_mic = np.dot(np.cos(np.dot(loc, om)), eta)
            u_mic = u_mic.transpose(1, 0, 2)
            if Nt == 1:
                u_mic = u_mic[0]
            return [p_mic, u_mic]
        else:
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
        if self.Na > 0:
            d['est_p'] = self.est_p
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
        zeta = A['C1'] * P['j'] ** 2   + A['C2'] * P['j'] ** .5

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


# %% =================================== VAN DER POL MODEL ============================================== %% #
class Lorenz63(Model):
    """ Lorenz 63 Class
    """

    name: str = 'Lorenz63'
    attr_child: dict = dict(rho=28., sigma=10., beta=8. / 3.)
    # attr_child: dict = dict(rho=20., sigma=10., beta=1.8)

    params: list = ['rho', 'sigma', 'beta']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None, DAdict=None):

        if TAdict is None:
            TAdict = {}

        super().__init__(TAdict)

        self.t_transient = 0.
        self.dt = 0.01
        self.t_CR = 5.

        if 'psi0' not in TAdict.keys():
            self.psi0 = [1.0, 1.0, 1.0]  # initialise x, y, z
            self.resetInitialConditions()

        if DAdict is not None:
            self.initEnsemble(DAdict)

        # set limits for the parameters
        self.param_lims = dict(rho=(None, None), beta=(None, None), sigma=(None, None))

    # _______________ Lorenz63 specific properties and methods ________________ #
    @property
    def obsLabels(self):
        return ["x", 'y', 'z']

    def getObservables(self, Nt=1):
        if Nt == 1:
            return self.hist[-1, :, :]
        else:
            return self.hist[-Nt:, :, :]

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(N=self.N,
                 Na=self.Na)
        if d['Na'] > 0:
            d['est_p'] = self.est_p
        return d

    @staticmethod
    def timeDerivative(t, psi, params, alpha):
        x1, x2, x3 = psi[:3]
        dx1 = alpha['sigma'] * (x2 - x1)
        dx2 = x1 * (alpha['rho'] - x3) - x2
        dx3 = x1 * x2 - alpha['beta'] * x3
        return (dx1, dx2, dx3) + (0,) * params['Na']


if __name__ == '__main__':
    MyModel = Lorenz63
    paramsTA = dict(law='tan', dt=2E-4)
    t_max = 10.

    t1 = time.time()
    # Non-ensemble case =============================
    case = MyModel(paramsTA)
    state, t_ = case.timeIntegrate(int(case.t_transient*2 / case.dt))
    case.updateHistory(state, t_)

    print('Elapsed time = ', str(time.time() - t1))

    _, ax = plt.subplots(1, 2, figsize=[10, 5])
    plt.suptitle('Non-ensemble case')

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    # State evolution
    y, lbl = case.getObservableHist(), case.obsLabels
    lbl = lbl[0]
    ax[0].plot(t_h, y[:, 0], color='green', label=lbl)
    i, j = [0, 1]
    ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='green')

    # Ensemble case =============================
    paramsDA = dict(m=10, est_p=['beta'])
    case = MyModel(paramsTA, paramsDA)

    t1 = time.time()
    for _ in range(1):
        state, t_ = case.timeIntegrate(int(1. / case.dt))
        case.updateHistory(state, t_)
    for _ in range(5):
        state, t_ = case.timeIntegrate(int(.1 / case.dt), averaged=True)
        case.updateHistory(state, t_)

    print('Elapsed time = ', str(time.time() - t1))

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    _, ax = plt.subplots(1, 3, figsize=[15, 5])
    plt.suptitle('Ensemble case')
    # State evolution
    y, lbl = case.getObservableHist(), case.obsLabels
    lbl = lbl[0]
    ax[0].plot(t_h, y[:, 0], color='blue', label=lbl)
    i, j = [0, 1]
    ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='blue')

    ax[0].set(xlabel='t', ylabel=lbl, xlim=[t_h[0], t_h[-1]])
    ax[1].set(xlabel='t', xlim=[t_h[-t_zoom], t_h[-1]])

    # Params

    ai = - case.Na
    max_p, min_p = -1000, 1000
    c = ['g', 'sandybrown', 'mediumpurple', 'cyan']
    mean = np.mean(case.hist, -1, keepdims=True)
    for p in case.est_p:
        superscript = '^\mathrm{init}$'
        # reference_p = truth['true_params']
        reference_p = case.alpha0

        mean_p = mean[:, ai].squeeze() / reference_p[p]
        std = np.std(case.hist[:, ai] / reference_p[p], axis=1)

        max_p = max(max_p, max(mean_p))
        min_p = min(min_p, min(mean_p))

        ax[2].plot(t_h, mean_p, color=c[-ai], label='$\\' + p + '/\\' + p + superscript)

        ax[2].set(xlabel='$t$', xlim=[t_h[0], t_h[-1]])
        ax[2].fill_between(t_h, mean_p + std, mean_p - std, alpha=0.2, color=c[-ai])
        ai += 1
    ax[2].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)
    ax[2].plot(t_h[1:], t_h[1:] / t_h[1:], '-', color='k', linewidth=.5)
    ax[2].set(ylim=[min_p - 0.1, max_p + 0.1])

    plt.tight_layout()
    plt.show()
