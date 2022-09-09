# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:04:55 2022

@author: an553
"""
import numpy as np
from scipy.integrate import odeint
import pylab as plt
from copy import deepcopy
import multiprocessing as mp

rng = np.random.default_rng(6)


#
def createEnsemble(Model, Ens_params=None, TA_params=None):
    """ Function that creates the ensemble.
        - Example of implementation:
            import Rijke as TA_model
            model_params    =   {'dt':1E-4, 'tau':2E-3}
            filter_params   =   {'m':5, 'est_p':['beta']}
            ensemble = createEnsemble(TA_model, filter_params, model_params
    """
    print('Creating ensemble of ' + Model.Case.name)
    # Force input parameters to be empty dictionaries if undefined
    if Ens_params is None:
        Ens_params = {}
    if TA_params is None:
        TA_params = {}
    global Ensemble
    # Define Ensemble class as a child of Model.Case class
    class Ensemble(Model.Case):
        """ Child class that creates an ensemble from the parent class Model
        """
        attr_ens = {'m': 10, 'std_psi': 0.1, 'std_a': 0.1,
                    'est_p': [], 'est_s': True, 'est_b': False
                    }

        def __init__(self, Ens_p, TA_p):
            # Load thermoacosutic model
            super().__init__(TA_p)
            # Evaluate attributes
            for key, val in Ensemble.attr_ens.items():
                if Ens_p is None: Ens_p = {}
                if key in Ens_p.keys():
                    setattr(self, key, Ens_p[key])
                else:
                    setattr(self, key, val)
            # Create state matrix. Note: if est_p then psi = [psi; alpha]
            mean = np.array(self.psi0)
            # mean        *=  rng.uniform(0.9,1.1, len(self.psi0))
            cov = np.diag(self.std_psi ** 2 * abs(mean))
            self.psi = rng.multivariate_normal(mean, cov, self.m).T
            if len(self.est_p) > 0:
                i = 0
                ens_a = np.zeros((len(self.est_p), self.m))
                for p in self.est_p:
                    p = np.array([getattr(self, p)])
                    # p *=  rng.uniform(0.5,2, len(p))  ### THIS MIGHT BE TOO MUCH UNCERTAINTY AT THE BEGINNING
                    ens_a[i, :] = rng.uniform(p * (1. - self.std_a),
                                              p * (1. + self.std_a), self.m)
                    i += 1
                self.psi = np.vstack((self.psi, ens_a))
            # Create history.
            self.hist = np.array([self.psi])
            self.hist_t = np.array([self.t])


        def plotHistory(self, truth=None):  # _________________________________________________________________________
            """ Function that plots the history of the observables and the
                parameters with a zoomed region in the state.
            """

            def plotwithshade(x, y, c, yl=None):
                mean = np.mean(y, 1)
                std = np.std(y, 1)
                ax[i, j].plot(x, mean, color=c, label=lbl)
                ax[i, j].fill_between(x, mean + std, mean - std, alpha=.2, color=c)
                if yl is True:
                    ax[i, j].set(ylabel=yl)
                ax[i, j].set(xlabel='$t$', xlim=[x[0], x[-1]])
                ax[i, j].legend(bbox_to_anchor=(1., 1.), loc="upper left", ncol=1)

            t = self.hist_t
            t_zoom = min([len(t) - 1, int(0.05 / self.dt)])

            _, ax = plt.subplots(2, 2, figsize=[10, 5])
            # Truth
            if truth is not None:
                y, _ = truth.getObservableHist()
                y = np.squeeze(y)
                ax[0, 0].plot(t, y, color='k', alpha=.2, label='Truth', linewidth=4)
                ax[0, 1].plot(t, y, color='k', alpha=.2, label='Truth', linewidth=4)

            i, j = [0, 0]
            # State evolution
            y, lbl = self.getObservables()
            lbl = lbl[0]
            plotwithshade(t, y[0], 'blue', yl=lbl[0])
            i, j = [0, 1]
            plotwithshade(t[-t_zoom:], y[0][-t_zoom:], 'blue')
            # Paramerter evolution
            params = self.hist[:, self.N - len(self.est_p):, :]
            c = ['g', 'sandybrown', 'mediumpurple']
            i, j = [1, 0]
            p_j = 0
            for p in self.est_p:
                lbl = '$\\' + p + '/\\' + p + '^t$'
                plotwithshade(t, params[:, p_j] / self.alpha0[p], c[p_j])
                p_j += 1

            plt.tight_layout()
            plt.show()

        def getCovariance(self, mean=None):
            if mean is None:
                mean = np.mean(self.hist, -1, keepdims=True)
            Psi = self.hist[:, :self.N - len(self.est_p)] - mean
            Cpp = [np.dot(Psi[ti], Psi[ti].T) / (self.m - 1.) \
                   for ti in range(len(mean))]
            return Cpp

        def copy(self):
            return deepcopy(self)

    print('Done')
    return Ensemble(Ens_params, TA_params)
    # %%


if __name__ == '__main__':
    import Rijke as TA_model
    import time

    model_params = {'dt': 1E-4}  # dictionary with parameters to the model
    filter_params = {'m': 10, 'est_p': ['beta']}  # dictionary with DA parameters

    ens = createEnsemble(TA_model, filter_params, model_params)

    # t = np.linspace(ens.t, ens.t + 100 * ens.dt, 100 + 1)

    # start = time.time()
    # psi = [odeint(ens.timeDerivative, ens.psi[:, mi], t, (ens,)) for mi in range(ens.m)]
    # psi = np.array(psi).transpose(1, 2, 0)
    # end = time.time()
    #
    # print('For loop time = ' + str(end-start))

    start = time.time()

    # def forecast(run_id, t_):
    #     global ens
    #     return odeint(ens.timeDerivative, ens.psi[:, run_id], t_, (ens,))
    #
    # with mp.Pool(mp.cpu_count()) as p:
    #     # p.map(forecast, list(range(ens.m)))
    #     #
    #     results = [p.apply_async(forecast, args=(i, t)) for i in range(ens.m)]
    #     results = [r.get() for r in results]
    # psi_parallel = np.array(results).transpose(1, 2, 0)

    state, time = ens.timeIntegrate(100)

    end = time.time()

    print('Parallel loop time = ' + str(end - start))

    # ens.updateHistory(results, t)

    # state, time = ens.timeIntegrate(5000)
    # ens.updateHistory(state, time)

    # psi, t = ens.timeIntegrate(ens, 100)   
    # ens.updateHistory(psi, t)

    # %%
    # ens.plotHistory()
