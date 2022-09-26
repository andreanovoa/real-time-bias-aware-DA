# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""


'''   
    TODO: 
        - Understand the difference between a fnc inside or outside the class
        - set est_p = [False]*4 and covariance non zero when 
                True so that only the true parameters are assimilated
'''



import numpy as np
from scipy.integrate import odeint
import pylab as plt

rng = np.random.default_rng()



class Case:
    name    =   'Van der Pol model'
    attr    =   ['psi0', 't','dt',
                 'm', 'var_psi','var_p', 
                 'DA','est_p', 'est_s', 'est_b']
    params  =   ['omega', 'nu', 'kappa', 'gamma', 'beta']
    
    def __init__(self, psi0=[0.1,0.1], HOM=False, omega = 2*np.pi*120, 
                 nu = 10.0, kappa = 3.4, gamma = 1.7, beta = 70., 
                 dt = 0.0001, t = 0,
                 m = 1, var_psi = 0.1**2, var_p =  0.2**2, 
                 DA = False, est_s = True, est_p = False, est_b=False):
        
        if psi0 is None: psi0=[0.1,0.1] # initialise eta and mu
        
        for attr in Case.attr:       setattr(self, attr, eval(attr))
        for attr in Case.params:     setattr(self, attr, eval(attr))
            
        if HOM: self.model = 'HOM'
        else:   self.model = 'LOM'
        
        mean        =   np.array(self.psi0)
        mean_p      =   np.array([getattr(self, p) for p in Case.params])
        
        self.N      =   len(mean)
        cov         =   np.diag(0. * mean)   
        cov_p       =   np.diag(0. * mean_p)   
        
        if DA is True:
            mean    *=  1.0
            cov     =   np.diag(var_psi * mean)   
            if est_p is True:
                '''TODO: set est_p = [False]*4 and covariance non zero when 
                True so that only the true parameters are assimilated'''
                mean_p  *=  1.0
                cov_p   =  np.diag(var_psi * mean_p)
        
        self.psi    =   rng.multivariate_normal(mean, cov, m).T
        self.alpha  =   rng.multivariate_normal(mean_p, cov_p, m).T
        self.bias   =   [0.] # 'TODO the shape should be q'
        
        self.hist   =   np.array([self.psi]) 
        self.hist_t =   np.array([self.t])
        self.hist_a =   np.array([self.alpha])
        self.hist_b =   np.array([self.bias])
   
    def getObservables(self):
        return self.psi[0]  + self.bias
    
    def updateHistory(self, psi, t, b=None):        
        self.hist   =   np.vstack((self.hist, psi))
        self.hist_t =   np.hstack((self.hist_t, t))
        self.psi    =   psi[-1]
        self.t      =   t[-1]
        if self.est_p is True:
            self.hist_a =   np.vstack((self.hist_a, 
                                       np.tile(self.alpha, (len(t),1,1))))
        if self.est_b is True:
            self.bias   =   b[-1]
            self.hist_b =   np.vstack((self.hist_b, b))

    def viewHistory(self):
        if self.m > 1:
            psi     =   np.mean(self.hist,2)
        else:
            psi     =    self.hist            
        labels  =   ['$\eta$', '$\mu$']
        colors  =   ['blue', 'red']
        
        t      =    self.hist_t          
        t_zoom =    min([len(t)-1, int(0.05 / self.dt)])
        
        fig, ax = plt.subplots(2, 2); fig.suptitle(self.model + ' Van der Pol')
        for i in range(2):
            ax[i,0].plot(t,psi[:,i], color = colors[i])
            ax[i,0].set(ylabel=labels[i], xlim = [t[0], t[-1]])
            ax[i,1].plot(t,psi[:,i], color = colors[i])
            ax[i,1].set(xlim = [t[-t_zoom], t[-1]], yticks = [])
            if self.m > 1:
                std = np.std(self.hist[:,i,:])
                ax[i,0].fill_between(t, psi[:,i]+std, psi[:,i]-std, \
                                     alpha = 0.2, color = colors[i])
                ax[i,1].fill_between(t, psi[:,i]+std, psi[:,i]-std, \
                                     alpha = 0.2, color = colors[i])     
        plt.tight_layout(); 
        plt.show()
        
    


def timeDerivative(psi, t, vdp, mi=0): #______________________________________     
    eta, mu                         =   psi[:]
    omega, nu, kappa, gamma, beta   =   vdp.hist_a[-1,:,mi]    
    deta_dt     =   mu
    dmu_dt      =  - omega**2 * eta
    if vdp.model == 'LOM':
        dmu_dt  +=  mu * (2.*nu - kappa * eta**2)
    elif vdp.model == 'HOM':
        dmu_dt  +=  mu * (beta**2 / (beta + kappa*eta**2) - beta + 2*nu) # atan model
        # dmu_dt  +=  mu * (2.*nu + kappa * eta**2 - gamma * eta**4) # higher order polinomial
        
    return np.hstack([deta_dt, dmu_dt])





def timeIntegrate(case, Nt=10000): #__________________________________________
    t       =   np.linspace(case.t, case.t + Nt * case.dt, Nt+1)
    psi     =   [odeint(timeDerivative, case.psi[:,mi], t, (case,mi)) \
                  for mi in range(case.m)]
    psi     =   np.array(psi).transpose(1,2,0)
    
    psi     =   psi[1:] # Remove initial condition
    t       =   t[1:] 
    
    return psi, t
    


