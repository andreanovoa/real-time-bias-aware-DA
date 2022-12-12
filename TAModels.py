from scipy.optimize import fsolve
from scipy.interpolate import splrep, splev
import pylab as plt
from Util import Cheb, RK4
import os

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
    attr_model = dict(dt=1E-4, t=0., psi0=None)
    attr_ens = dict(m=10, est_p=[], est_s=True, est_b=False,
                    biasType=None, inflation=1.01,
                    std_psi=0.1, std_a=0.001, alpha_distr='normal',
                    num_DA_blind=0, num_SE_only=0,
                    )

    def __init__(self, TAdict, DAdict):
        TAdict = TAdict.copy()
        DAdict = DAdict.copy()
        # ================= INITIALISE THERMOACOUSTIC MODEL ================== ##
        for key, val in Model.attr_model.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
            else:
                setattr(self, key, val)
        self.alpha0 = {par: getattr(self, par) for par in self.params}

        # ================== INITIALISE ENSEMBLE IF DESIRED ================== ##
        self.N, self.Na = len(self.psi0), 0
        if DAdict is None or len(DAdict) == 0:
            self.psi = np.array([self.psi0]).T
            self.ensemble = False
            self.alpha = self.alpha0.copy()
        else:
            self.ensemble = True
            for key, val in Model.attr_ens.items():
                if key in DAdict.keys():
                    setattr(self, key, DAdict[key])
                else:
                    setattr(self, key, val)
            # ----------------------- DEFINE STATE MATRIX ----------------------- ##
            # Note: if est_p and est_b psi = [psi; alpha; biasWeights]
            if self.m > 1:
                mean = np.array(self.psi0)  # * rng.uniform(0.9, 1.1, len(self.psi0))
                # self.psi = self.addUncertainty(mean, self.std_psi, self.m, method=self.alpha_distr)
                cov = np.diag((self.std_psi ** 2 * abs(mean)))
                self.psi = rng.multivariate_normal(mean, cov, self.m).T
                if len(self.est_p) > 0:  # Augment ensemble with estimated parameters
                    self.Na = len(self.est_p)
                    self.N += self.Na
                    mean = np.array([getattr(self, p) for p in self.est_p])  # * rng.uniform(0.9, 1.1, len(self.psi0))
                    ens_a = self.addUncertainty(mean, self.std_a, self.m, method=self.alpha_distr)

                    print(np.mean(ens_a, -1), mean)
                    self.psi = np.vstack((self.psi, ens_a))
            else:
                self.psi = np.array(self.psi0)
                if len(self.est_p) > 0:
                    self.Na = len(self.est_p)
                    self.N += self.Na
                    self.psi = np.append(self.psi, np.array([getattr(self, p) for p in self.est_p]))
                    self.psi = np.expand_dims(self.psi, 1)

            # ------------------------ INITIALISE BIAS ------------------------ ##
            if self.biasType is not None:
                if 'Bdict' not in DAdict.keys():
                    DAdict['Bdict'] = {}
                Bdict = DAdict['Bdict'].copy()
                self.initBias(Bdict)

            # --------------------- DEFINE OBS-STATE MAP --------------------- ##
            obs = self.getObservables()
            Nq = np.shape(obs)[0]
            # if ensemble.est_b:
            #     y0 = np.concatenate(y0, np.zeros(ensemble.bias.Nb))
            y0 = np.concatenate((np.zeros(self.N), np.ones(Nq)))
            self.M = np.zeros((Nq, len(y0)))
            iq = 0
            for ii in range(len(y0)):
                if y0[ii] == 1:
                    self.M[iq, ii] = 1
                    iq += 1

        # ========================== CREATE HISTORY ========================== ##
        self.hist = np.array([self.psi])
        self.hist_t = np.array([self.t])
        self.hist_J = []

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
        # if self.est_b:
        #     weights, names = self.bias.getWeights()
        #     Nw = len(weights)
        #     self.N += Nw  # Increase ensemble size
        #
        #     ens_b = np.zeros((Nw, self.m))
        #     ii = 0
        #     for w in weights:
        #         low = w[0] - self.std_a
        #         high = w[0] + self.std_a
        #         ens_b[ii, :] = low.T + (high - low).T * np.random.random_sample((1, self.m))
        #         ii += 1
        #     # Update bias weights and update state matrix
        #     self.bias.updateWeights(ens_b)
        #     self.psi = np.vstack((self.psi, ens_b))

        # Create bias history
        b = self.bias.getBias(yb)
        self.bias.updateHistory(b, self.t, reset=True)
        # # Add TA training parameters
        # if 'train_TAparams' in Bdict.keys():
        #     self.bias.train_TAparams = Bdict['train_TAparams']
        # else:
        #     self.bias.train_TAparams = self.alpha0

    def copy(self):
        return deepcopy(self)

    @property
    def pool(self):
        if not hasattr(self, '_pool'):
            self._pool = mp.Pool()
        return self._pool

    def close(self):
        self.pool.close()
        self.pool.join()
        delattr(self, "_pool")

    def updateHistory(self, psi, t):
        self.hist = np.concatenate((self.hist, psi), axis=0)
        self.hist_t = np.hstack((self.hist_t, t))
        self.psi = psi[-1]
        self.t = t[-1]

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

    @staticmethod
    def addUncertainty(mean, std, m, method='normal'):
        if method == 'normal':
            cov = np.diag((std * mean) ** 2)
            return rng.multivariate_normal(mean, cov, m).T
        elif method == 'uniform':
            ens_aug = np.zeros((len(mean), m))
            for ii, p in enumerate(mean):
                if p > 0:
                    ens_aug[ii, :] = rng.uniform(p * (1. - std), p * (1. + std), m)
                else:
                    ens_aug[ii, :] = rng.uniform(p * (1. + std), p * (1. - std), m)
            return ens_aug
        else:
            raise 'Parameter distribution not recognised'

    @staticmethod
    def forecast(y0, fun, t, params, alpha=None):
        # SOLVE IVP ========================================
        out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params, alpha))
        psi = out.y.T

        # ODEINT =========================================== THIS WORKS AS IF HARD CODED
        # psi = odeint(fun, y0, t_interp, (params,))

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
            Returns:
                psi: forecasted ensemble state
                t: time of the propagated psi
        """
        t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
        self_dict = self.govEqnDict()

        if not self.ensemble:
            psi = [Model.forecast(self.psi[:, 0], self.timeDerivative, t, params=self_dict, alpha=self.alpha0)]
            psi = np.array(psi)
            psi = psi.transpose(1, 2, 0)
            return psi[1:], t[1:]

        if not averaged:
            alpha = self.getAlpha()
            # # # OPTION 1b: Process and Queue ------------------------------------
            # self.processes = []
            # time1 = time.time()
            # self.runProcesses(fun=self.timeDerivative, t=t, params=self_dict, alpha=alpha)
            # # Get results. Ensure sorted queue and return values only
            # results = [self.queueOUT.get() for _ in self.processes]
            # results.sort(key=lambda x: x[0])
            # psi = [r[-1] for r in results]
            # # print('\n 1 Processes and Queues 2: ' + str(time.time() - time1) + ' s')

            # # OPTION 2: with Pool as p ----------------------------------------------
            # time1 = time.time()
            fun_part = partial(Model.forecast, fun=self.timeDerivative, t=t, params=self_dict)
            sol = [self.pool.apply_async(fun_part, kwds={'y0': self.psi[:, mi].T, 'alpha': alpha[mi]})
                   for mi in range(self.m)]
            psi = [s.get() for s in sol]
            # # print('2 with pool: ' + str(time.time() - time1) + ' s')
        else:
            psi_mean = np.mean(self.psi, 1, keepdims=True)
            psi_std = (self.psi - psi_mean) / psi_mean
            if alpha is None:
                alpha = self.getAlpha(psi_mean)[0]
            psi_mean = Model.forecast(y0=psi_mean[:, 0], fun=self.timeDerivative, t=t, params=self_dict, alpha=alpha)
            psi = []
            for ii in range(self.m):
                psi.append(psi_mean * (1 + psi_std[:, ii]))

        # Rearrange dimensions to be Nt x N x m and remove initial condition
        psi = np.array(psi)
        psi = psi.transpose(1, 2, 0)
        return psi[1:], t[1:]


# %% ==================================== RIJKE TUBE MODEL ============================================== %% #
class Rijke(Model):
    """
        Rijke tube model with Galerkin discretisation and gain-delay sqrt heat release law.
        Args:
            TAdict: dictionary with the model parameters. If not defined, the default value is used.
                > Nm [10] - Number of Galerkin modes
                > Nc [50] - Number of Chebyshev modes
                > beta [1E6] - Heat source strength [W s^1/2 m^-2/5]
                > tau [2E-3] - Time delay [s]
                > C1 [.1] - First damping constant [?]
                > C2 [.06] - Second damping constant [?]
                > xf [.2] - Flame location [m]
                > L [1] - Tube length [m]
    """

    name: str = 'Rijke'
    attr: dict = dict(Nm=10, Nc=10, Nmic=6,
                      beta=1E6, tau=2.E-3, C1=.1, C2=.06, kappa=1E5,
                      xf=1.18, L=1.92, law='sqrt')
    params: list = ['beta', 'tau', 'C1', 'C2', 'kappa']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None, DAdict=None):

        if TAdict is None:
            TAdict = {}
        else:
            TAdict = TAdict.copy()
        if DAdict is None:
            DAdict = {}
        else:
            DAdict = DAdict.copy()


        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
            else:
                setattr(self, key, val)

        if 'est_p' in DAdict.keys() and 'tau' in DAdict['est_p']:
            self.tau_adv = 1E-2
            self.Nc = 50
        else:
            self.tau_adv = self.tau

        if 'psi0' not in TAdict.keys():  # initialise acoustic modes
            TAdict['psi0'] = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])

        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)
        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nmic)

        # Mean Flow Properties
        c1, c2 = [350., 300.]
        c = (1. - self.xf / self.L) * c1 + self.xf / self.L * c2
        self.meanFlow = dict(rho=1.20387, u=1E-4, p=101300., gamma=1.4, c1=c1, c2=c2, c=c)

        # Define modes frequency of each mode and sin cos etc
        self.j = np.arange(1, self.Nm + 1)
        xf, L, MF = [self.xf, self.L, self.meanFlow]

        def fun(om):
            return MF['c2'] * np.sin(om * xf / MF['c1']) * np.cos(om * (L - xf) / MF['c2']) + \
                   MF['c1'] * np.cos(om * xf / MF['c1']) * np.sin(om * (L - xf) / MF['c2'])

        omegaj = fsolve(fun, self.j * c / L * np.pi)  # Initial guess using a weighted averaged mean speed of sound
        self.omegaj = np.array(omegaj)

        self.sinomjxf = np.sin(self.omegaj / self.meanFlow['c'] * self.xf)
        self.cosomjxf = np.cos(self.omegaj / self.meanFlow['c'] * self.xf)

        # initialise Model parent (history)
        super().__init__(TAdict, DAdict)

        self.param_lims = dict(beta=(1E3, 1E8), tau=(0, self.tau_adv),
                               C1=(None, None), C2=(None, None),
                               kappa=(1E3, 1E8))
        print('\n -------------------- RIJKE MODEL PARAMETERS -------------------- \n',
              '\t Nm = {}  \t beta = {:.2} \t law = {} \n'.format(self.Nm, self.beta, self.law),
              '\t Nc = {}  \t tau = {:.2} \t tau_adv = {:.2}\n'.format(self.Nc, self.tau, self.tau_adv),
              '\t Nmic = {} \t xf = {:.2} '.format(self.Nmic, self.xf))

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def eta(self):
        return self.hist[:, 0:self.Nm, :]

    @property
    def mu(self):
        return self.hist[:, self.Nm:2 * self.Nm, :]

    @property
    def v(self):
        return self.hist[:, 2 * self.Nm:2 * self.Nm + self.Nc, :]

    def getObservableHist(self, Nt=0, loc=None, velocity=False):

        if np.shape(self.hist)[0] == 1:
            raise Exception('Object has no history')
        else:
            if loc is None:
                loc = np.expand_dims(self.x_mic, axis=1)
            # Define the labels
            labels_p = ["$p'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]
            labels_u = ["$u'(x = {:.2f})$".format(x) for x in loc[:, 0].tolist()]
            # Compute acoustic pressure and velocity at locations
            om = np.array([self.omegaj])
            c = self.meanFlow['c']

            p = -np.dot(np.sin(np.dot(loc, om) / c), self.mu)
            # p = -np.dot(mu, np.sin(np.dot(np.transpose(om), loc) / c))
            p = p.transpose(1, 0, 2)
            if velocity:
                u = np.dot(np.cos(np.dot(loc, om) / c), self.eta)
                u = u.transpose(1, 0, 2)
                return [p[-Nt:], u[-Nt:]], [labels_p, labels_u]
            else:
                return p[-Nt:], labels_p

    def getObservables(self, velocity=False):

        # Compute acoustic pressure and velocity at microphone locations
        om = np.array([self.omegaj])
        c = self.meanFlow['c']
        eta = self.psi[:self.Nm]
        mu = self.psi[self.Nm:2 * self.Nm]

        x_mic = np.expand_dims(self.x_mic, axis=1)
        p = -np.dot(np.sin(np.dot(x_mic, om) / c), mu)
        if velocity:
            u = np.dot(np.cos(np.dot(x_mic, om) / c), eta)
            return np.concatenate((p, u))
        else:
            return p

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(Nm=self.Nm,
                 Nc=self.Nc,
                 N=self.N,
                 Na=self.Na,
                 j=self.j,
                 omegaj=self.omegaj,
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
    def timeDerivative(t, psi, params, alpha):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
                params: dictionary with all the case parameters
                alpha: dictionary of varying parameters
            Returns:
                concatenation of the state vector time derivative
        """

        if len(psi.shape) == 0:
            psi = t

        eta = psi[:params['Nm']]
        mu = psi[params['Nm']:2 * params['Nm']]
        v = psi[2 * params['Nm']:2 * params['Nm'] + params['Nc']]

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, params['cosomjxf']), v))

        # Evaluate u(t_interp-tau) i.e. velocity at the flame at t_interp - tau
        x_tau = alpha['tau'] / params['tau_adv']

        if x_tau < 1:
            f = splrep(params['gc'], v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            print('tau = ', alpha['tau'], 'tau_adv', params['tau_adv'])
            raise Exception("tau can't_interp be larger than tau_adv")

        # Compute damping and heat release law
        zeta = alpha['C1'] * params['j'] ** 2 + alpha['C2'] * params['j'] ** .5

        MF = params['meanFlow']  # Physical properties

        if params['law'] == 'sqrt':
            qdot = alpha['beta'] * (np.sqrt(abs(MF['u'] / 3. + u_tau)) - np.sqrt(MF['u'] / 3.))  # [W/m2]=[m/s3]
        elif params['law'] == 'tan':
            qdot = alpha['beta'] * np.sqrt(alpha['beta'] / alpha['kappa']) * np.arctan(
                np.sqrt(alpha['beta'] / alpha['kappa']) * u_tau)  # [m / s3]
        else:
            raise ValueError('Undefined heat law')

        qdot *= -2. * (MF['gamma'] - 1.) / params['L'] * params['sinomjxf']  # [Pa/s]

        # governing equations
        deta_dt = params['omegaj'] / (MF['rho'] * MF['c']) * mu
        dmu_dt = - params['omegaj'] * MF['rho'] * MF['c'] * eta - MF['c'] / params['L'] * zeta * mu + qdot
        dv_dt = - 2. / params['tau_adv'] * np.dot(params['Dc'], v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(params['Na'])))


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - cubic heat release law
        - atan heat release law
            Note: gamma appears only in the higher order polynomial which is currently commented out
    """

    name: str = 'VdP'
    attr: dict = dict(omega=2 * np.pi * 120., law='tan',
                      zeta=60., beta=70., kappa=3.4, gamma=1.7)  # beta, zeta [rad/s]
    params: list = ['omega', 'zeta', 'kappa', 'beta']  # ,'omega', 'gamma']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None, DAdict=None):

        if TAdict is None:
            TAdict = {}
        else:
            TAdict = TAdict.copy()
        if DAdict is None:
            DAdict = {}
        else:
            DAdict = DAdict.copy()

        # print('Initialising Van der Pol')
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
            else:
                setattr(self, key, val)

        if 'psi0' not in TAdict.keys():
            TAdict['psi0'] = [0.1, 0.1]  # initialise eta and mu

        # initialise model history
        super().__init__(TAdict, DAdict)

        # set limits for the parameters
        self.param_lims = dict(omega=(0, None), zeta=(20, 100), kappa=(0.1, 10.),
                               gamma=(None, None), beta=(20, 100))

    # _______________ VdP specific properties and methods ________________ #
    def printModelParams(self):
        print('\n ------------------ VAN DER POL MODEL PARAMETERS ------------------ \n',
              '\t Heat law = {0}'.format(self.law))
        for k, v in self.getParameters().items():
            print('\t {} = {}'.format(k, v))

    def getObservableHist(self, Nt=0):
        if np.shape(self.hist)[0] == 1:
            return None, "$\\eta$"
        else:
            return self.hist[-Nt:, [0], :], "$\\eta$"

    def getObservables(self):
        eta = self.psi[0, :]
        return np.expand_dims(eta, axis=0)

    def getParameters(self):
        if self.law not in ['cubic', 'tan']:
            raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
        return {key: self.alpha0[key] for key in ['zeta', 'kappa', 'beta']}  # ['beta', 'zeta', 'kappa']}

    @property
    def growth_rate(self, zeta, beta):
        return .5 * (beta - zeta)

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(law=self.law,
                 N=self.N,
                 Na=self.Na,
                 psi0=self.psi0
                 )
        if d['N'] > len(d['psi0']):
            d['est_p'] = self.est_p
        return d

    @staticmethod
    def timeDerivative(t, psi, params, alpha):
        # if type(params) is not dict:
        #     params = params[0]
        # if len(psi.shape) == 0:
        #     psi = t

        eta, mu = psi[:2]
        # alpha = d['alpha'].copy()
        # Na = d['N'] - len(d['psi0'])  # number of parameters estimated
        # if Na > 0:
        #     ii = len(d['psi0'])
        #     for param in d['est_p']:
        #
        #         P[param] = psi[ii]
        #         ii += 1

        dmu_dt = - alpha['omega'] ** 2 * eta + mu * (alpha['beta'] - alpha['zeta'])
        # Add nonlinear term
        if params['law'] == 'cubic':  # Cubic law
            dmu_dt -= mu * alpha['kappa'] * eta ** 2
        elif params['law'] == 'tan':  # arc tan model
            dmu_dt -= mu * (alpha['kappa'] * eta ** 2) / (1. + alpha['kappa'] / alpha['beta'] * eta ** 2)
        else:
            raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
            # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial
        return np.hstack([mu, dmu_dt, np.zeros(params['Na'])])


if __name__ == '__main__':
    import time

    t1 = time.time()
    dt = 2E-4
    paramsTA = dict(law='tan', dt=dt)

    # Non-ensemble case =============================
    case = Rijke(paramsTA)
    state, t_ = case.timeIntegrate(int(.5 / dt))
    case.updateHistory(state, t_)

    _, ax = plt.subplots(1, 2, figsize=[10, 5])
    plt.suptitle('Non-ensemble case')

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    # State evolution
    y, lbl = case.getObservableHist()
    lbl = lbl[0]
    ax[0].plot(t_h, y[:, 0], color='green', label=lbl)
    i, j = [0, 1]
    ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='green')
    # plt.show()

    print(paramsTA)
    # Ensemble case =============================
    paramsDA = dict(m=10, est_p=['beta', 'tau'])
    case = Rijke(paramsTA, paramsDA)


    for _ in range(5):
        state, t_ = case.timeIntegrate(int(.25 / dt))
        case.updateHistory(state, t_)

    print('Elapsed time = ', str(time.time() - t1))

    t_h = case.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / case.dt)])

    _, ax = plt.subplots(1, 3, figsize=[15, 5])
    plt.suptitle('Ensemble case')
    # State evolution
    y, lbl = case.getObservableHist()
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
