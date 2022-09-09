# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:55:49 2022

@author: an553
"""


import numpy as np
from Util import Rijke

# ========================================================================= #

rng = np.random.default_rng(6)


def createEnsembleClass(parent):
    global Ensemble

    class Ensemble(parent):
        attr_ens = {'m': 10, 'std_psi': 0.1, 'std_a': 0.1,
                    'est_p': [], 'est_s': True, 'est_b': False
                    }

        def __init__(self, Ens_p=None, TAdict=None):
            if Ens_p is None:
                Ens_p = {}
            # Evaluate attributes
            for key, val in Ensemble.attr_ens.items():
                if key in Ens_p.keys():
                    setattr(self, key, Ens_p[key])
                else:
                    setattr(self, key, val)
            # Initialise thermoacoustic model
            super().__init__(TAdict)
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

    return Ensemble


# %% ======================================================================= #
# if __name__ == '__main__':
paramsTA = dict(dt=2E-4)
paramsDA = dict(m=20)

Class = createEnsembleClass(Rijke)
rijke = Class(paramsDA, paramsTA)
state, time = rijke.timeIntegrate(averaged=False)

print(rijke.getObservables)
print(rijke.viewHistory)
print(rijke.__class__)

rijke.updateHistory(state, time)
rijke.viewHistory()
