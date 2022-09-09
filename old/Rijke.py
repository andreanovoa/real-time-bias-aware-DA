# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:55:49 2022

@author: an553
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.interpolate import splrep, splev
import pylab as plt
import multiprocessing as mp


#

class Case():
    name = 'Rijke'
    attr = {'beta': 1E6, 'tau': 2E-3, 'C1': .1, 'C2': .06,
            'psi0': None, 'xf': 0.2, 'L': 1.,
            'Nm': 10, 'Nc': 50, 'tau_adv': 1E-2, 't': 0., 'dt': 1E-4}
    params = ['beta', 'tau', 'C1', 'C2']

    def __init__(self, TA_params=None):
        # Evaluate default parameters
        if TA_params is None: TA_params = {}
        for key, val in self.attr.items():
            if key in TA_params.keys():
                setattr(self, key, TA_params[key])
            else:
                setattr(self, key, val)

        if self.psi0 is None:  # initialise acoustic modes
            self.psi0 = .05 * np.hstack([np.ones(2 * self.Nm), np.zeros(self.Nc)])

        assert len(self.alpha0) == len(self.params)

    # _________________________ General properties _________________________ #
    @property
    def psi0(self):  # State initial condition
        return self._psi0

    @psi0.setter
    def psi0(self, val):
        self._psi0 = val

    @property
    def alpha0(self):  # Parameters initial condition (true values)
        return self._alpha0

    @property
    def dt(self):  # Integration time step
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val

    @property
    def N(self):  # State vector size
        if not hasattr(self, '_N'):
            self._N = len(self.psi)
        return self._N

    @property
    def psi(self):  # State vector at current time
        if not hasattr(self, '_psi'):
            self._psi = np.array([self.psi0]).T
        return self._psi

    @psi.setter
    def psi(self, val):
        self._psi = val

    @property
    def hist(self):  # State vector history
        if not hasattr(self, '_hist'):
            self._hist = np.array([self.psi])
        return self._hist

    @hist.setter
    def hist(self, val):
        self._hist = val
        # self._hist = np.vstack((self.hist, psi))

    @property
    def hist_t(self):  # Time history
        if not hasattr(self, '_hist_t'):
            self._hist_t = np.array(self.t)
        return self._hist_t

    @hist_t.setter
    def hist_t(self, val):
        self._hist_t = val

    # __________________________ General methods ___________________________ #

    def updateHistory(self, psi, t):
        self.hist = np.vstack((self.hist, psi))
        self.hist_t = np.hstack((self.hist_t, t))
        self.psi = psi[-1]
        self.t = t[-1]

    def viewHistory(self):
        t = self.hist_t
        yy, lbl = self.getObservables()
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

    # _______________ Rijke specific properties and methods ________________ #
    @property
    def meanFlow(self):  # Mean flow properties
        if not hasattr(self, '_meanFlow'):
            c1 = 300.
            c2 = 350.
            c = (1. - self.xf / self.L) * c2 + self.xf / self.L * c1
            self._meanFlow = {'L': self.L,
                              'rho': 1.20387,
                              'u': 1E-4,
                              'p': 101300.,
                              'gamma': 1.4,
                              'c1': c1,
                              'c2': c2,
                              'c': c
                              }
        return self._meanFlow

    def getParams(self):
        return [getattr(self, p) for p in self.params]

    def getObservables(self, locs=[]):
        if np.shape(self.hist)[0] == 1:
            print('Case has no psi history')
            p, u = [None, None]
        else:
            if not locs:
                locs = np.array([self.xf])
            # Compute acoustic pressure and velocity at locs
            om = np.array([self.omegaj])
            c = self.meanFlow['c']
            eta = np.transpose(self.eta, (0, 2, 1))
            mu = np.transpose(self.mu, (0, 2, 1))
            #
            p = -np.dot(mu, np.sin(np.dot(locs, om) / c))
            u = np.dot(eta, np.cos(np.dot(locs, om) / c))
        return [p, u], ["$p'$", "$u'$"]

    @property
    def eta(self):
        return self.hist[:, 0:self.Nm, :]

    @property
    def mu(self):
        return self.hist[:, self.Nm:2 * self.Nm, :]

    @property
    def v(self):
        return self.hist[:, 2 * self.Nm:2 * self.Nm + self.Nc, :]

    @property
    def Dc(self):
        if not hasattr(self, '_Dc'):
            self._Dc, self._gc = Cheb(self.Nc)
        return self._Dc

    @property
    def gc(self):
        if not hasattr(self, '_Dc'):
            self._Dc, self._gc = Cheb(self.Nc)
        return self._gc

    @property
    def j(self):
        if not hasattr(self, '_j'):
            self._j = np.arange(1, self.Nm + 1)
        return self._j

    @property
    def omegaj(self):
        if not hasattr(self, '_omegaj'):
            xf = self.xf
            L = self.meanFlow['L']
            c1 = self.meanFlow['c1']
            c2 = self.meanFlow['c2']
            c = self.meanFlow['c']

            def fun(om):
                return c2 * np.sin(om * xf / c1) * np.cos(om * (L - xf) / c2) + \
                       c1 * np.cos(om * xf / c1) * np.sin(om * (L - xf) / c2)

            # Initial guess using a weighted averaged mean speed of sound
            initial_guess = self.j * c / L * np.pi
            # Solve fun(om)
            omegaj = fsolve(fun, initial_guess)
            self._omegaj = np.array(omegaj)
        return self._omegaj

    @property
    def sinomjxf(self):
        if not hasattr(self, '_sinomjxf'):
            self._sinomjxf = np.sin(self.omegaj / self.meanFlow['c'] * self.xf)
        return self._sinomjxf

    @property
    def cosomjxf(self):
        if not hasattr(self, '_cosomjxf'):
            self._cosomjxf = np.cos(self.omegaj / self.meanFlow['c'] * self.xf)
        return self._cosomjxf

    # _________________________ Governing equations ________________________ #
    @staticmethod
    def timeDerivative(psi, t, case):
        eta = psi[:case.Nm]
        mu = psi[case.Nm:2 * case.Nm]
        v = psi[2 * case.Nm:2 * case.Nm + case.Nc]
        # Physical properties
        MF = case.meanFlow.copy()
        # Parameters
        P = case.alpha0.copy()
        if len(psi) > len(case.psi0):
            ii = len(case.psi0)
            for param in case.est_p:
                P[param] = psi[ii]
                ii += 1

        # Advection equation boundary conditions
        v2 = np.hstack((np.dot(eta, case.cosomjxf), v))

        # Evaluate u(t-tau) i.e. velocity at the flame at t - tau
        x_tau = P['tau'] / case.tau_adv

        assert x_tau <= 1

        f = splrep(case.gc, v2)
        u_ftau = splev(x_tau, f)  # u_ftau  =   v2[-1]

        # Compute damping and heat release law
        zeta = P['C1'] * case.j ** 2 + P['C2'] * case.j ** .5
        qdot = P['beta'] * (np.sqrt(abs(MF['u'] / 3. + u_ftau))
                            - np.sqrt(MF['u'] / 3.))  # [W/m2]
        qdot *= -2. * (MF['gamma'] - 1.) / MF['L'] * case.sinomjxf  # [Pa/s]

        # governing equations
        deta_dt = case.omegaj / (MF['rho'] * MF['c']) * mu

        dmu_dt = - (case.omegaj * MF['rho'] * MF['c'] * eta
                    + MF['c'] / MF['L'] * zeta * mu - qdot)
        dv_dt = - 2. / case.tau_adv * np.dot(case.Dc, v2)

        return np.hstack([deta_dt, dmu_dt, dv_dt[1:],
                          np.zeros(case.N - len(case.psi0))])

    def timeIntegrate(self, Nt=1000, averaged=False):  # __________________________________________
        t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
        if not averaged:
            # TODO MODIFY THIS TO RUN IN PARALLEL
            # psi = [odeint(self.timeDerivative, self.psi[:, mi], t, (self,)) for mi in range(self.m)]

            with mp.Pool(mp.cpu_count()) as p:
                # p.map(forecast, list(range(ens.m)))
                #
                # psi = p.map(self.func, list(range(self.m)))
                # psi = [r.get() for r in psi]
                results = [p.apply_async(self.forecast, args=(self, i, t)) for i in range(self.m)]
                results = [r.get() for r in results]
            psi = np.array(results).transpose(1, 2, 0)

            psi = np.array(psi).transpose(1, 2, 0)
        else:
            mean = np.mean(self.psi, 1)
            psi = [odeint(self.timeDerivative, mean, t, (self,))]
            psi = np.array(psi).transpose(1, 2, 0)
            psi = np.repeat(psi, self.m, axis=2)

        psi = psi[1:]  # Remove initial condition
        t = t[1:]
        return psi, t

    @staticmethod
    def forecast(self, mi, t):
        return odeint(self.timeDerivative, self.psi[:, mi], t, (self,))


# ========================================================================= #

# def timeIntegrate(case, Nt=1000):  # __________________________________________
#     t = np.linspace(case.t, case.t + Nt * case.dt, Nt + 1)
#     psi = [odeint(case.timeDerivative, case.psi[:, 0], t, (case,))]
#     psi = np.array(psi).transpose(1, 2, 0)
#     psi = psi[1:]  # Remove initial condition
#     t = t[1:]
#
#     return psi, t


def Cheb(Nc, lims=[0, 1]):  # __________________________________________________
    ''' Compute the Chebyshev collocation derivative matrix (D) 
        and the Chevyshev grid of (N + 1) points in [ [0,1] ] / [-1,1]
    '''
    g = - np.cos(np.pi * np.arange(Nc + 1, dtype=float) / Nc)
    c = np.hstack([2., np.ones(Nc - 1), 2.]) * (-1) ** np.arange(Nc + 1)
    X = np.outer(g, np.ones(Nc + 1))
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(Nc + 1))
    D -= np.diag(D.sum(1))

    # Mopdify 
    if lims[0] == 0:
        g = (g + 1.) / 2.

    return D, g


# %% ======================================================================= #
if __name__ == '__main__':
    rijke = Case()
    state, time = rijke.timeIntegrate(averaged=True)
    rijke.updateHistory(state, time)
    rijke.viewHistory()
