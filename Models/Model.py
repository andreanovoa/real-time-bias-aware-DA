
#import os
#os.environ["OMP_NUM_THREADS"]= '1'

from os import cpu_count

from scipy.integrate import odeint, solve_ivp

import numpy as np
import pylab as plt
# import time

#from Util import Cheb

rng = np.random.default_rng(6)

num_proc = cpu_count()
if num_proc > 1:
    num_proc = int(num_proc / 2.)



# %% =================================== PARENT MODEL CLASS ============================================= %% #
class ModelParent:
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
        params = self.__dict__.copy()
        y0 = self.psi[:,0].copy()
        
        out = solve_ivp(self.timeDerivative, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='LSODA', args=(params,))
        psi = [out.y.T]
        
#        psi = [odeint(self.timeDerivative, y0, t, (params,))]
        
        psi = np.array(psi).transpose(1, 2, 0)

        return psi[1:], t[1:]  # Remove initial condition

    # ____________________________ MANIPULATE HISTORY  ____________________________ #
    def updateHistory(self, psi, t):
        self.hist = np.concatenate((self.hist, psi), axis=0)
        self.hist_t = np.hstack((self.hist_t, t))
        self.psi = psi[-1]
        self.t = t[-1]



# %% =================================== VAN DER POL MODEL ============================================== %% #
class VdP(ModelParent):
    """ Van der Pol Oscillator Class
        - Low order model: cubic heat release law [omega, nu, kappa]
        - High order model: atan heat release law [omega, nu, kappa, beta]
            Note: gamma appears only in the higher order polynomial which is
                  currently commented out
    """

    name = 'VdP'
    attr = dict(omega=2 * np.pi * 120., law='cubic',
                nu=10., kappa=3.4, gamma=1.7, beta=70.)
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

        print('\n ------------------ VAN DER POL MODEL PARAMETERS ------------------ \n',
              '\t Heat law = {0} \n'.format(self.law),
              '\t nu = {0} \n\t kappa = {1:.2}'.format(self.nu, self.kappa))

    # _______________ VdP specific properties and methods ________________ #
    def getObservableHist(self, Nt=0):
        if np.shape(self.hist)[0] == 1:
            raise Exception('Case has no psi history')
        else:
            return self.hist[-Nt:, 0, :], ["$\\eta$"]  # , "$\\dot{\\eta}$"]

    def getObservables(self):
        eta = self.psi[0, :]
        return np.expand_dims(eta, axis=0)

    #        return self.psi[0]

    # _________________________ Governing equations ________________________ #
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
        N = d['N']  # state vector length
        Na = N - len(d['psi0'])  # number of parameters estimated
        if Na > 0:
            ii = len(d['psi0'])
            for param in d['est_p']:
                P[param] = psi[ii]
                ii += 1
        deta_dt = mu
        dmu_dt = - P['omega'] ** 2 * eta

        if d['law'] == 'cubic':  # Cubic law
            dmu_dt += mu * (2. * P['nu'] - P['kappa'] * eta ** 2)
        elif d['law'] == 'atan':  # arc tan model
            dmu_dt += mu * (P['beta'] ** 2 / (P['beta'] + P['kappa'] * eta ** 2) - P['beta'] + 2 * P['nu'])
        else:
            raise TypeError("Undefined heat release law. Choose 'cubic' or 'tan'.")
            # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial

        return np.hstack([deta_dt, dmu_dt, np.zeros(Na)])


if __name__ == '__main__':
    # import cProfile

    paramsTA = dict(law='atan', dt=2E-4)
    model = Rijke(paramsTA)

    # pr = cProfile.Profile()
    # pr.enable()
    state, t_ = model.timeIntegrate(1000)

    # pr.disable()
    # pr.print_stats(sort="calls")

    model.updateHistory(state, t_)
    model.viewHistory()
