# import os
# os.environ["OMP_NUM_THREADS"]= '1'


from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import splrep, splev

import numpy as np
import pylab as plt
# import time

from Util import Cheb, RK4

rng = np.random.default_rng(6)


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

        out = solve_ivp(self.timeDerivative, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params,))
        psi = [out.y.T]

        # psi = [RK4(t, y0, self.timeDerivative, params)]

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
                beta=1E6, tau=2.E-3, C1=.1, C2=.06,
                kappa=1E5,
                xf=1.18, L=1.92,
                law='sqrt')
    params = ['beta', 'tau', 'C1', 'C2', 'kappa']

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
              '\t Nm = {}  \t beta = {:.2} \t law = {} \n'.format(self.Nm, self.beta, self.law),
              '\t Nc = {}  \t tau = {:.2} \t tau_adv = {:.2}\n'.format(self.Nc, self.tau, self.tau_adv),
              '\t Nmic = {} \t xf = {:.2} '.format(self.Nmic, self.xf))

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

        self.param_lims = dict(beta=(1E3, 1E8), tau=(0, self.tau_adv),
                               C1=(None, None), C2=(None, None),
                               kappa=(1E3, 1E8))

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
                 L=self.L,
                 law=self.law
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
            print('tau = ', P['tau'], 'tau_adv', d['tau_adv'])
            raise Exception("tau can't be larger than tau_adv")

        # Compute damping and heat release law
        zeta = P['C1'] * d['j'] ** 2 + P['C2'] * d['j'] ** .5

        if d['law'] == 'sqrt':
            qdot = P['beta'] * (np.sqrt(abs(MF['u'] / 3. + u_tau)) - np.sqrt(MF['u'] / 3.))  # [W/m2]=[m/s3]
        elif d['law'] == 'tan':
            qdot = P['beta'] * np.sqrt(P['beta'] / P['kappa'] ) * np.arctan(np.sqrt(P['beta'] / P['kappa']) * u_tau)  # [m / s3]
        else:
            raise ValueError('Undefined heat law')

        qdot *= -2. * (MF['gamma'] - 1.) / d['L'] * d['sinomjxf']  # [Pa/s]

        # governing equations
        deta_dt = d['omegaj'] / (MF['rho'] * MF['c']) * mu
        dmu_dt = - d['omegaj'] * MF['rho'] * MF['c'] * eta - MF['c'] / d['L'] * zeta * mu + qdot
        dv_dt = - 2. / d['tau_adv'] * np.dot(d['Dc'], v2)

        return np.concatenate((deta_dt, dmu_dt, dv_dt[1:], np.zeros(Na)))


# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(Model):
    """ Van der Pol Oscillator Class
        - Low order model: cubic heat release law [omega, nu, kappa]
        - High order model: atan heat release law [omega, nu, kappa, beta]
            Note: gamma appears only in the higher order polynomial which is
                  currently commented out
    """

    name = 'VdP'
    attr = dict(omega=2 * np.pi * 120., law='atan',
                nu=7., kappa=3.4, gamma=1.7, beta=70.)
    params = ['omega', 'nu', 'kappa', 'gamma', 'beta']

    # __________________________ Init method ___________________________ #
    def __init__(self, TAdict=None):
        if TAdict is None:
            TAdict = {}
        # print('Initialising Van der Pol')
        for key, val in self.attr.items():
            if key in TAdict.keys():
                setattr(self, key, TAdict[key])
            else:
                setattr(self, key, val)

        if 'psi0' not in TAdict.keys():
            TAdict['psi0'] = [0.1, 0.1]  # initialise eta and mu

        # initialise model history
        super().__init__(TAdict)

        self.param_lims = dict(omega=(0, None), nu=(None, None), kappa=(None, None),
                               gamma=(None, None), beta=(None, None))

        print('\n ------------------ VAN DER POL MODEL PARAMETERS ------------------ \n',
              '\t Heat law = {0} \n'.format(self.law),
              '\t nu = {0} \n\t kappa = {1:.2}'.format(self.nu, self.kappa))

    # _______________ VdP specific properties and methods ________________ #
    def getObservableHist(self, Nt=0):
        if np.shape(self.hist)[0] == 1:
            raise Exception('Case has no psi history')
        else:
            return self.hist[-Nt:, [0], :], ["$\\eta$"]  # , "$\\dot{\\eta}$"]

    def getObservables(self):
        eta = self.psi[0, :]
        return np.expand_dims(eta, axis=0)

    # _________________________ Governing equations ________________________ #
    def govEqnDict(self):
        d = dict(law=self.law,
                 N=self.N,
                 psi0=self.psi0,
                 alpha0=self.alpha0
                 )
        if d['N'] > len(d['psi0']):
            d['est_p'] = self.est_p
        return d

    @staticmethod
    def timeDerivative(t, psi, d):
        if type(d) is not dict:
            d = d[0]
        try:
            assert np.shape(psi)[0] > 0
        except:
            temp = psi
            psi = t
            t = temp

        eta, mu = psi[:2]
        P = d['alpha0'].copy()
        Na = d['N'] - len(d['psi0'])  # number of parameters estimated
        if Na > 0:
            ii = len(d['psi0'])
            for param in d['est_p']:
                P[param] = psi[ii]
                ii += 1
        deta_dt = mu
        dmu_dt = - P['omega'] ** 2 * eta

        if d['law'] == 'cubic':  # Cubic law
            dmu_dt += mu * (2. * P['nu'] - P['kappa'] * eta ** 2)
        elif d['law'] == 'tan':  # arc tan model
            dmu_dt += mu * (P['beta'] ** 2 / (P['beta'] + P['kappa'] * eta ** 2) - P['beta'] + 2 * P['nu'])
        else:
            raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
            # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial

        return np.hstack([deta_dt, dmu_dt, np.zeros(Na)])


# %%


if __name__ == '__main__':
    # import cProfile

    import time

    t1 = time.time()
    paramsTA = dict(law='atan', dt=2E-4)
    model = Rijke(paramsTA)

    # pr = cProfile.Profile()
    # pr.enable()
    state, t_ = model.timeIntegrate(1000)
    model.updateHistory(state, t_)

    print('Elapsed time = ', str(time.time() - t1))
    # pr.disable()
    # pr.print_stats(sort="calls")

    t_h = model.hist_t
    t_zoom = min([len(t_h) - 1, int(0.05 / model.dt)])

    _, ax = plt.subplots(1, 2, figsize=[10, 5])

    # State evolution
    y, lbl = model.getObservableHist()
    lbl = lbl[0]
    ax[0].plot(t_h, y[:, 0], color='blue', label=lbl)
    i, j = [0, 1]
    ax[1].plot(t_h[-t_zoom:], y[-t_zoom:, 0], color='blue')

    ax[0].set(xlabel='$t$', ylabel=lbl, xlim=[t_h[0], t_h[-1]])
    ax[1].set(xlabel='$t$', xlim=[t_h[-t_zoom], t_h[-1]])
    plt.tight_layout()
    plt.show()
