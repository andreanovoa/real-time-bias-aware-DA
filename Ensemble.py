# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:55:49 2022

@author: an553
"""
import os

import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = '1'

num_proc = os.cpu_count()
if num_proc > 1:
    num_proc = int(num_proc)

import numpy as np
import time
from scipy.integrate import odeint, solve_ivp
import multiprocessing as mp
from functools import partial
from Util import RK4
from copy import deepcopy
from datetime import date

rng = np.random.default_rng(6)


def globalforecast(y0, fun, t, params):
    # print(y0[-2:])
    # SOLVE IVP ========================================
    out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK23', args=(params,))
    if not out.success:
        out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params,))
        print('RK23 failed, switched to RK45')
        print(y0[-2:])
    psi = out.y.T

    # ODEINT =========================================== THIS WORKS AS IF HARD CODED
    # psi = odeint(fun, y0, t_interp, (params,))

    # HARD CODED RUGGE KUTTA 4TH ========================
    # psi = RK4(t_interp, y0, fun, params)

    return psi


globalp = mp.Pool()


# ========================================================================= #
def createEnsemble(parent, DA_params=None, TA_params=None, Bias_params=None):
    """ Function that creates an ensemble of the class parent.
        - Example of implementation:
            import Rijke as TA_model
            TA_params    =   {'dt':1E-4, 'tau':2E-3}
            DA_params   =   {'m':5, 'est_p':['beta']}
            ensemble = createEnsemble(TA_model, DA_params, TA_params)
    """
    # p = mp.Pool(10)
    # new_p = Multiprocessor()

    global Ensemble
    print('Creating ensemble of ' + parent.name + ' model')

    class Ensemble(parent):
        attr_ens = dict(m=10, est_p=[], est_s=True,
                        est_b=False, bias=None,
                        std_psi=0.1, std_a=0.1,
                        getJ=False, inflation=1.01,
                        num_DA_blind=0, num_SE_only=0,
                        alpha_distr='normal')

        def __init__(self, DAdict=None, TAdict=None, Bdict=None):
            """ Constructor of the Ensemble
                Inputs:
                    DAdict: dictionary of parameters related to Data Assimilation
                    TAdict: dictionary of parameters related to the Thermoacoustic model
                    Bdict: dictionary of parameters related to the model of the Bias
            """

            # ======================= INITIALISE ENSEMBLE ======================= ##
            if DAdict is None:
                DAdict = {}
            for key, val in Ensemble.attr_ens.items():
                if key in DAdict.keys():
                    setattr(self, key, DAdict[key])
                else:
                    setattr(self, key, val)

            # ================ INITIALISE THERMOACOUSTIC MODEL ================== ##
            if TAdict is None:
                TAdict = {}
            super().__init__(TAdict)

            # =========================== INITIALISE J ========================== ##
            if self.getJ:
                self.hist_J = []

            # ======================= DEFINE STATE MATRIX ======================= ##
            # Note: if est_p and est_b psi = [psi; alpha; biasWeights]
            mean = np.array(self.psi0)  # * rng.uniform(0.9, 1.1, len(self.psi0))
            cov = np.diag(self.std_psi ** 2 * abs(mean))
            self.psi = rng.multivariate_normal(mean, cov, self.m).T
            if len(self.est_p) > 0:
                self.N += len(self.est_p)  # Increase ensemble size
                ens_a = np.zeros((len(self.est_p), self.m))
                if self.alpha_distr == 'uniform':
                    for i, p in enumerate(self.est_p):
                        p = np.array([getattr(self, p)])
                        # p *=  rng.uniform(0.5,2, len(p))
                        if p > 0:
                            ens_a[i, :] = rng.uniform(p * (1. - self.std_a), p * (1. + self.std_a), self.m)
                        else:
                            ens_a[i, :] = rng.uniform(p * (1. + self.std_a), p * (1. - self.std_a), self.m)
                elif self.alpha_distr == 'normal':
                    mean = np.array([getattr(self, p) for p in self.est_p])  # * rng.uniform(0.9, 1.1, len(self.psi0))
                    cov = np.diag(self.std_a ** 2 * abs(mean))
                    ens_a = rng.multivariate_normal(mean, cov, self.m).T
                else:
                    raise 'Parameter distribution not recognised'

                # plt.figure()
                # plt.hist(ens_a[0,:])
                # plt.show()

                self.psi = np.vstack((self.psi, ens_a))

            # ========================= INITIALISE BIAS ========================= ##
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
                    Nw = len(weights)
                    self.N += Nw  # Increase ensemble size

                    ens_b = np.zeros((Nw, self.m))
                    ii = 0
                    for w in weights:
                        low = w[0] - self.std_a
                        high = w[0] + self.std_a
                        ens_b[ii, :] = low.T + (high - low).T * np.random.random_sample((1, self.m))
                        ii += 1
                    # Update bias weights and update state matrix
                    self.bias.updateWeights(ens_b)
                    self.psi = np.vstack((self.psi, ens_b))

                # Create bias history
                b = self.bias.getBias(y)
                self.bias.updateHistory(b, self.t)

                # Add TA training parameters
                if 'train_TAparams' in Bdict.keys():
                    self.bias.train_TAparams = Bdict['train_TAparams']
                else:
                    self.bias.train_TAparams = self.alpha0

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

        @staticmethod
        def _wrapper(mi, queueOUT, **kwargs):
            psi = Ensemble.forecast(**kwargs)
            queueOUT.put((mi, psi))

        @staticmethod
        def forecast(y0, fun, t, params):
            # ret = Ensemble.forecast(*args, **kwargs)

            # SOLVE IVP ========================================
            out = solve_ivp(fun, t_span=(t[0], t[-1]), y0=y0, t_eval=t, method='RK45', args=(params,))
            psi = out.y.T

            # ODEINT =========================================== THIS WORKS AS IF HARD CODED
            # psi = odeint(fun, y0, t_interp, (params,))

            # HARD CODED RUGGE KUTTA 4TH ========================
            # psi = RK4(t_interp, y0, fun, params)

            return psi

        def runProcesses(self, **kwargs):
            kwargs['queueOUT'] = self.queueOUT
            for mi in range(self.m):
                kwargs['y0'] = self.psi[:, mi].T
                kwargs['mi'] = mi
                p = mp.Process(target=self._wrapper, kwargs=kwargs)
                self.processes.append(p)
                p.start()

        def startPool(self):
            self.pool = mp.Pool()

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
                    t_interp: time of the propagated psi
            """

            t = np.linspace(self.t, self.t + Nt * self.dt, Nt + 1)
            self_dict = self.govEqnDict()
            if not averaged:
                # # OPTION 1: Process and Queue ------------------------------------
                # self.processes = []
                # if ~hasattr(self, 'queueOUT'):
                #     self.queueOUT = mp.Queue()
                #
                # # time1 = time.time()
                # self.runProcesses(fun=self.timeDerivative, t_interp=t_interp, params=self_dict)
                # # Get results. Ensure sorted queue and return values only
                # results = [self.queueOUT.get() for _ in self.processes]
                # results.sort(key=lambda y: y[0])
                # psi = [r[-1] for r in results]
                # # print('\n Processes and Queues: ' + str(time.time() - time1) + ' s')
                # # ---------------------------------------------------------------------------------------------------

                # # OPTION 2: with Pool as p ----------------------------------------
                # # time1 = time.time()
                # with mp.Pool() as p:
                #     psi = p.map(partial(self.forecast, fun=self.timeDerivative, t_interp=t_interp, params=self_dict),
                #                          self.psi[:, range(self.m)].T)
                # # print('\n with pool: ' + str(time.time() - time1) + ' s')
                # # ---------------------------------------------------------------------------------------------------

                # # OPTION 3: attribute pool + map ----------------------------------------
                # if ~hasattr(self, 'pool'):
                #     self.startPool()
                # # time1 = time.time()
                # psi = self.pool.map(partial(self.forecast, fun=self.timeDerivative, t_interp=t_interp, params=self_dict),
                #                      self.psi[:, range(self.m)].T)
                # # psi = list(psi)
                # # print('\n map: ' + str(time.time() - time1) + ' s')
                # # ---------------------------------------------------------------------------------------------------

                # # OPTION 4: attribute pool + imap ----------------------------------------
                # if ~hasattr(self, 'pool'):
                #     self.startPool()
                # # time1 = time.time()
                # psi = self.pool.imap(partial(self.forecast, fun=self.timeDerivative, t_interp=t_interp, params=self_dict),
                #                      self.psi[:, range(self.m)].T)
                # psi = list(psi)
                # # print('\n imap: ' + str(time.time() - time1) + ' s')
                # # ---------------------------------------------------------------------------------------------------

                # # OPTION 5: global pool + map ----------------------------------------
                # # time1 = time.time()
                # psi = globalp.map(partial(globalforecast, fun=self.timeDerivative, t_interp=t_interp, params=self_dict),
                #                    self.psi[:, range(self.m)].T)
                # # print('\n with pool: ' + str(time.time() - time1) + ' s')
                # # ---------------------------------------------------------------------------------------------------

                # OPTION 6: global pool + imap ---------------------------------------- [FASTEST BUT NEEDS GLOBAL POOL]
                # time1 = time.time()
                psi = globalp.imap(partial(globalforecast, fun=self.timeDerivative, t=t, params=self_dict),
                                   self.psi[:, range(self.m)].T)
                psi = list(psi)
                # print('\n with pool: ' + str(time.time() - time1) + ' s')
                # ---------------------------------------------------------------------------------------------------
            else:
                mean = np.mean(self.psi, 1)
                # psi = [odeint(self.timeDerivative, mean, t_interp, (self_dict,))]
                psi = [Ensemble.forecast(y0=mean, fun=self.timeDerivative, t=t, params=self_dict)]
                psi = np.repeat(psi, self.m, axis=0)

            psi = np.array(psi)
            psi = psi.transpose(1, 2, 0)

            return psi[1:], t[1:]  # Remove initial condition

        # ____________________________ Copy ensemble ____________________________ #
        def copy(self):

            return deepcopy(self)

    return Ensemble(DA_params, TA_params, Bias_params)


# %% ======================================================================= #
if __name__ == '__main__':
    import TAModels
    from Util import plotHistory

    model = TAModels.Rijke

    paramsTA = dict(dt=2E-4)
    paramsDA = dict(m=20, est_p=['beta'])

    ens, _ = createEnsemble(model, paramsDA, paramsTA)
    state, time = ens.timeIntegrate(Nt=500, averaged=False)
    ens.updateHistory(state, time)
    state, time = ens.timeIntegrate(Nt=300, averaged=True)
    ens.updateHistory(state, time)

    print(ens.getObservables)
    print(ens.viewHistory)
    print(ens.__class__)
    print(ens.alpha0)

    plotHistory(ens)
