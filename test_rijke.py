import numpy as np
import pylab as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import fsolve
from Util import Cheb


from scipy.integrate import odeint, solve_ivp
import multiprocessing as mp
from functools import partial
from Util import RK4
from copy import deepcopy
from datetime import date
import os as os

rng = np.random.default_rng(6)

num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc / 2.)


# %% =================================== PARENT MODEL CLASS ============================================= %% #
class Model:
    """ Parent Class with the general thermoacoustic model
        properties and methods definitions.
    """
    attr_model = dict(dt=1E-4, t=0., psi0=None)

    def __init__(self, params=None):
        if params is None:
            params = {}
        for key, val in self.attr_model.items():
            if key in params.keys():
                setattr(self, key, params[key])
            else:
                setattr(self, key, val)
        # start the
        self.psi = np.array([self.psi0]).T
        self.N = len(self.psi)
        # History
        self.hist = np.array([self.psi])
        self.hist_t = np.array(self.t)
        # Initial set of parameters
        self.alpha0 = {p: getattr(self, p) for p in self.params}

    # ____________________________ Integrator ____________________________ #
    def timeIntegrate(self, Nt=100):
        """
            Integrator of the model.
            Args:
                Nt: number of forecast steps
            Returns:
                psi: forecasted ensemble (without the initial condition)
                t: time of the propagated psi
        """
        t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
        params = self.govEqnDict()
        y0 = self.psi[:, 0].copy()

        # out = solve_ivp(self.timeDerivative, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params,))
        # psi = [out.y.T]

        psi = [RK4(t, y0, self.timeDerivative, params)]

        #        psi = [odeint(self.timeDerivative, y0, t, (params,))]

        psi = np.array(psi).transpose(1, 2, 0)

        return psi[1:], t[1:]  # Remove initial condition

    # ____________________________ MANIPULATE HISTORY  ____________________________ #
    def updateHistory(self, psi, t):
        self.hist = np.concatenate((self.hist, psi), axis=0)
        self.hist_t = np.hstack((self.hist_t, t))
        self.psi = psi[-1]
        self.t = t[-1]

    def viewHistory(self):
        # TODO I think I should create a new module for plots, not class method
        t = self.hist_t
        yy, lbl = self.getObservableHist()
        _, ax = plt.subplots(len(lbl), 2, figsize=[10, 5])
        t_zoom = min([len(t) // 2, int(0.05 / self.dt)])
        i = 0
        colors = ['blue', 'red']
        for y in yy:
            ax[i, 0].plot(t, y, color=colors[i])
            ax[i, 0].set(ylabel=lbl[i], xlim=[t[0], t[-1]])
            # zoom in
            ax[i, 1].plot(t, y, color=colors[i])
            ax[i, 1].set(xlim=[t[-t_zoom], t[-1]], yticks=[])
            i += 1
        plt.tight_layout()
        plt.show()


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
    attr = dict(Nm=10, Nc=10, Nmic=6,
                beta=5E6, tau=1.5E-3, C1=.1, C2=.06,
                xf=1.18, L=1.92)
    params = ['beta', 'tau', 'C1', 'C2']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None):

        if TAdict is None:
            TAdict = {}
        # print('Initialising Rijke')
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
            else:
                setattr(self, key, val)
        if hasattr(self, 'est_p') and 'tau' in self.est_p:
            self.tau_adv = 1E-2
            self.Nc = 50
        else:
            self.tau_adv = self.tau

        if 'psi0' not in TAdict.keys():  # initialise acoustic modes
            TAdict['psi0'] = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])

        # initialise Model parent (history)
        super().__init__(TAdict)

        print('\n -------------------- RIJKE MODEL PARAMETERS -------------------- \n',
              '\t Nm = {0}  \t Nc = {1} \t Nmic = {2} \n'.format(self.Nm, self.Nc, self.Nmic),
              '\t beta = {0:.2} \t tau = {1:.2}\n'.format(self.beta, self.tau),
              '\t xf = {0:.2} \t tau_adv = {1:.2}'.format(self.xf, self.tau_adv))

        # Microphone locations
        self.x_mic = np.linspace(self.xf, self.L, self.Nmic)
        # Chebyshev modes
        self.Dc, self.gc = Cheb(self.Nc, getg=True)

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

    def getObservableHist(self, loc=None, velocity=False):

        if np.shape(self.hist)[0] == 1:
            raise Exception('Object has no history')
        else:
            if loc is None:
                loc = np.expand_dims(self.x_mic, axis=1)
            # Define the labels
            labels_p = ["$p'(x = {:.2f})$".format(x) for x in loc[0, :].tolist()]
            labels_u = ["$u'(x = {:.2f})$".format(x) for x in loc[0, :].tolist()]
            # Compute acoustic pressure and velocity at locations
            om = np.array([self.omegaj])
            c = self.meanFlow['c']

            p = -np.dot(np.sin(np.dot(loc, om) / c), self.mu)
            # p = -np.dot(mu, np.sin(np.dot(np.transpose(om), loc) / c))
            p = p.transpose(1, 0, 2)
            if velocity:
                u = np.dot(np.cos(np.dot(loc, om) / c), self.eta)
                u = u.transpose(1, 0, 2)
                return [p, u], [labels_p, labels_u]
            else:
                return p, labels_p

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
                 j=self.j,
                 omegaj=self.omegaj,
                 cosomjxf=self.cosomjxf,
                 sinomjxf=self.sinomjxf,
                 alpha0=self.alpha0,
                 N=self.N,
                 psi0=self.psi0,
                 tau_adv=self.tau_adv,
                 meanFlow=self.meanFlow,
                 Dc=self.Dc,
                 gc=self.gc,
                 L=self.L
                 )
        if d['N'] > len(d['psi0']):
            d['est_p'] = self.est_p

        return d

    @staticmethod
    def timeDerivative(t, psi, d):
        """
            Governing equations of the model.
            Args:
                psi: current state vector
                t: current time
                d: dictionary with all the case parameters
            Returns:
                concatenation of the state vector time derivative
        """

        if type(d) is not dict:
            d = d[0]

        try:
            assert np.shape(psi)[0] > 0
        except:
            temp = psi
            psi = t
            t = temp

        eta = psi[:d['Nm']]
        mu = psi[d['Nm']:2 * d['Nm']]
        v = psi[2 * d['Nm']:2 * d['Nm'] + d['Nc']]
        # Physical properties
        MF = d['meanFlow']
        # Parameters
        P = d['alpha0'].copy()
        # Na = d['Na']
        Na = d['N'] - len(d['psi0'])
        if Na > 0:
            ii = len(d['psi0'])
            for param in d['est_p']:
                P[param] = psi[ii]
                ii += 1

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, d['cosomjxf']), v))

        # Evaluate u(t-tau) i.e. velocity at the flame at t - tau
        x_tau = P['tau'] / d['tau_adv']

        if x_tau < 1:
            f = splrep(d['gc'], v2)
            u_tau = splev(x_tau, f)
        elif x_tau == 1:  # if no tau estimation, bypass interpolation to speed up code
            u_tau = v2[-1]
        else:
            raise Exception("tau can't be larger than tau_adv")
        # Compute damping and heat release law
        zeta = P['C1'] * d['j'] ** 2 + P['C2'] * d['j'] ** .5
        qdot = P['beta'] * (np.sqrt(abs(MF['u'] / 3. + u_tau)) - np.sqrt(MF['u'] / 3.))  # [W/m2]
        qdot *= -2. * (MF['gamma'] - 1.) / d['L'] * d['sinomjxf']  # [Pa/s]

        # governing equations
        deta_dt = d['omegaj'] / (MF['rho'] * MF['c']) * mu
        dmu_dt = - d['omegaj'] * MF['rho'] * MF['c'] * eta - MF['c'] / d['L'] * zeta * mu + qdot
        dv_dt = - 2. / d['tau_adv'] * np.dot(d['Dc'], v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(Na)))


def forecast(y0, fun, t, params):
    # if y0 is None:
    #     y0 = case.psi[:, mi]

    # SOLVE IVP ========================================
    out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params,))
    psi = out.y.T

    # ODEINT =========================================== THIS WORKS AS IF HARD CODED
    # psi = odeint(fun, y0, t, (params,))

    # HARD CODED RUGGE KUTTA 4TH ========================
    # psi = RK4(t, y0, fun, params)

    return psi


# ========================================================================= #
def createEnsemble(parent, Ens_params=None, TA_params=None, Bias_params=None):
    """ Function that creates an ensemble of the class parent.
        - Example of implementation:
            import Rijke as TA_model
            model_params    =   {'dt':1E-4, 'tau':2E-3}
            filter_params   =   {'m':5, 'est_p':['beta']}
            ensemble = createEnsemble(TA_model, filter_params, model_params
    """
    print('Creating ensemble of ' + parent.name + ' model')

    global Ensemble
    class Ensemble(parent):
        attr_ens = dict(m=10, est_p=[], est_s=True,
                        est_b=False, bias=None,
                        std_psi=0.1, std_a=0.1,
                        getJ=False)

        def __init__(self, DAdict=None, TAdict=None, Bdict=None):
            """ Constructor of the Ensemble
                Inputs:
                    DAdict: dictionary of parameters related to Data Assimilation
                    TAdict: dictionary of parameters related to the Thermoacoustic model
                    Bdict: dictionary of parameters related to the model of the Bias
            """
            ## ======================= INITIALISE ENSEMBLE ======================= ##
            if DAdict is None:
                DAdict = {}
            for key, val in Ensemble.attr_ens.items():
                if key in DAdict.keys():
                    setattr(self, key, DAdict[key])
                else:
                    setattr(self, key, val)

            ## ================ INITIALISE THERMOACOUSTIC MODEL ================== ##
            if TAdict is None:
                TAdict = {}
            super().__init__(TAdict)

            ## =========================== INITIALISE J ========================== ##
            if self.getJ:
                self.hist_J = []

            ## ======================= DEFINE STATE MATRIX ======================= ##
            # Note: if est_p and est_b psi = [psi; alpha; biasWeights]
            mean = np.array(self.psi0)  # * rng.uniform(0.9,1.1, len(self.psi0))
            cov = np.diag(self.std_psi ** 2 * abs(mean))
            self.psi = rng.multivariate_normal(mean, cov, self.m).T
            if len(self.est_p) > 0:
                self.N += len(self.est_p)  # Increase ensemble size
                i = 0
                ens_a = np.zeros((len(self.est_p), self.m))
                for p in self.est_p:
                    p = np.array([getattr(self, p)])
                    # p *=  rng.uniform(0.5,2, len(p))
                    ens_a[i, :] = rng.uniform(p * (1. - self.std_a), p * (1. + self.std_a), self.m)
                    i += 1
                self.psi = np.vstack((self.psi, ens_a))

            ## ========================= INITIALISE BIAS ========================= ##
            if self.bias is not None:
                if Bdict is None:
                    Bdict = {}
                # Assign some required items
                Bdict['est_b'] = self.est_b
                Bdict['dt'] = self.dt
                if 'filename' not in Bdict.keys():  # default bias file name
                    Bdict['filename'] = parent.name + '_' + str(date.today())
                # Initialise bias. Note: self.bias is now an instance of the bias class
                y = self.getObservables()
                self.bias = self.bias(y, self.t, Bdict)
                # Augment state matrix if you want to infer bias weights
                if self.est_b:
                    weights, names = self.bias.getWeights()
                    Nw = len(np.concatenate(weights[:]))
                    self.N += Nw  # Increase ensemble size

                    ens_b = np.zeros((Nw, self.m))
                    ii = 0
                    for w in weights:
                        low = w[:, 0] * (1. - self.std_a)
                        high = w[:, 0] * (1. + self.std_a)
                        ens_b[ii:ii + len(w), :] = low.T + (high - low).T * np.random.random_sample((len(w), self.m))
                        ii += 1
                    # Update bias weights and update state matrix
                    self.bias.updateWeights(ens_b)
                    self.psi = np.vstack((self.psi, ens_b))
                # Create bias history
                b = self.bias.getBias(y)
                self.bias.updateHistory(b, self.t)

            ## ========================== CREATE HISTORY ========================== ##
            self.hist = np.array([self.psi])
            self.hist_t = np.array([self.t])

            ## ======================= DEFINE OBS-STATE MAP ======================= ##
            y0 = np.zeros(self.N)
            obs = self.getObservables()
            Nq = np.shape(obs)[0]
            # if ensemble.est_b:
            #     y0 = np.concatenate(y0, np.zeros(ensemble.bias.Nb))
            y0 = np.concatenate((y0, np.ones(Nq)))
            self.M = np.zeros((Nq, len(y0)))
            iq = 0
            for i in range(len(y0)):
                if y0[i] == 1:
                    self.M[iq, i] = 1
                    iq += 1

        # ____________________________ Integrator ____________________________ #
        def timeIntegrate(self, Nt=100, averaged=False):
            """
                Integrator of the ensemble. Rewrites the integrator of the model to account for m.
                (Uses Parallel computation if self is an ensemble with m > 1. #TODO find why not working)
                Args:
                    Nt: number of forecast steps
                    averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                    the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                Returns:
                    psi: forecasted ensemble (without the initial condition)
                    t: time of the propagated psi
            """
            t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
            self_dict = self.govEqnDict()
            if not averaged:
                ## OG ## results = [odeint(self.timeDerivative, self.psi[:, mi], t, (self_dict,)) for mi in range(self.m)]
                #                if len(t) > 1000:
                #                else:
                #                 if self.name == 'VdP':
                #                     psi = [forecast(mi, self, t=t, params=self_dict) for mi in range(self.m)]
                #                 else:
                with mp.Pool() as p:
                    psi = p.map(partial(forecast, fun=self.timeDerivative, t=t, params=self_dict), self.psi.T)
                    p.close()
                    p.join()
                # psi = [forecast(psi_j, self.timeDerivative, t=t, params=self_dict) for psi_j in self.psi.T]
            else:
                mean = np.mean(self.psi, 1)
                # psi = [odeint(self.timeDerivative, mean, t, (self_dict,))]
                psi = [forecast(mean, self, t=t, params=self_dict)]
                psi = np.repeat(psi, self.m, axis=0)

            psi = np.array(psi).transpose(1, 2, 0)

            return psi[1:], t[1:]  # Remove initial condition

        # ____________________________ Copy ensemble ____________________________ #
        def copy(self):
            return deepcopy(self)

    return Ensemble(Ens_params, TA_params, Bias_params)

#%%

if __name__ == '__main__':
    from Util import plotHistory
    import time


    dt = 1E-5
    paramsTA = dict(dt=dt,
                    L=1.92,
                    xf=1.18)
    Nt = int(.5/dt)


    c='red'

    # %% ========================= ENSEMBLE TEST =============================== #

    paramsDA = dict(m=100,
                    std_psi=.01)
    model = createEnsemble(Rijke, paramsDA, paramsTA)

    t1 = time.time()
    state, t_ = model.timeIntegrate(Nt=Nt, averaged=False)
    model.updateHistory(state, t_)

    print('Elapsed time = ', str(time.time() - t1))

    #
    t = model.hist_t
    t_zoom = min([len(t) - 1, int(0.05 / model.dt)])

    _, ax = plt.subplots(1, 2, figsize=[10, 5])

    # State evolution
    y, lbl = model.getObservableHist()
    lbl = lbl[0]


    y_mean = np.mean(y, -1)
    std = np.std(y[:, 0, :], axis=-1)
    ax[0].plot(t, y_mean[:, 0], '-', color=c)
    ax[0].fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.5, color=c)
    ax[1].plot(t, y_mean[:, 0], '-', color=c)
    ax[1].fill_between(t, y_mean[:, 0] + std, y_mean[:, 0] - std, alpha=0.5, color=c)

    ax[0].set(xlabel='$t$', ylabel=lbl, xlim=[t[0], t[-1]])
    ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])
    plt.tight_layout()


    # %% ========================= SINGLE TEST =============================== #
    case = Rijke(paramsTA)
    t1 = time.time()
    state, t_ = case.timeIntegrate(Nt=Nt)
    case.updateHistory(state, t_)
    print('Elapsed time = ', str(time.time() - t1))

    t = case.hist_t
    t_zoom = min([len(t) - 1, int(0.05 / case.dt)])


    # State evolution
    y, lbl = case.getObservableHist()
    lbl = lbl[0]

    c='blue'
    ax[0].plot(t, y[:, 0], '-', color=c)
    ax[1].plot(t, y[:, 0], '-', color=c)

    ax[0].set(xlabel='$t$', ylabel=lbl, xlim=[t[0], t[-1]])
    ax[1].set(xlabel='$t$', xlim=[t[-t_zoom], t[-1]])
    plt.tight_layout()

    plt.show()