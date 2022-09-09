# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:06:25 2022

@author: an553
"""
import numpy as np
from scipy.integrate import odeint
import pylab as plt
#
#
class Case:
    ''' Van der Pol Oscillator Class
        - Low order model: cubic heat release law [omega, nu, kappa]
        - High order model: atan heat release law [omega, nu, kappa, beta]
            Note: gamma appears only in the higher order polynomial which is 
                  currently commented out
    '''
    name    =   'VdP'
    attr    =   {'omega':2*np.pi*120, 'nu':10., 'kappa':3.4, 
                 'gamma':1.7, 'beta':70., 
                 'psi0':None, 'LOM':True, 
                 't':0.,'dt':1E-4}
    params  =   ['omega', 'nu', 'kappa', 'gamma', 'beta']
    
    def __init__(self, TA_params=None):
        if TA_params is None: TA_params = {}
        for key, val in self.attr.items():
            if key in TA_params.keys(): 
                setattr(self, key, TA_params[key])
            else:                       
                setattr(self, key, val)
            
        if self.psi0 is None: 
            self.psi0 = [0.1,0.1] # initialise eta and mu
            
        assert len(self.alpha0) == len(self.params)
        # self.alpha0    =   {p:getattr(self, p) for p in self.params}
    # _________________________ General properties _________________________ #
    @property 
    def psi0(self):         # State initial condition 
        return self._psi0
    @psi0.setter
    def psi0(self, val):
        self._psi0    =   val
    
    @property 
    def alpha0(self):       # Parameters initial condition (true values)
        if not hasattr(self, '_alpha0'):
            self._alpha0    =   {p:getattr(self, p) for p in self.params}
        return self._alpha0
    
    
    @property 
    def dt(self):           # Integration time step
        return self._dt
    @dt.setter
    def dt(self, val):
        self._dt    =   val
        
    @property 
    def t(self):           # Current time
        return self._t
    @t.setter
    def t(self, val):
        self._t    =   val
    
    @property
    def N(self):           # State vector size
        if not hasattr(self, '_N'):
            self._N    =   len(self.psi) 
        return self._N
        
    @property 
    def psi(self):           # State vector at current time
        if not hasattr(self, '_psi'):
            self._psi    =   np.array([self.psi0]).T
        return self._psi
    @psi.setter
    def psi(self, val):     
        self._psi = val
        
    @property 
    def hist(self):           # State vector history
        if not hasattr(self, '_hist'):    
            self._hist   =   np.array([self.psi]) 
        return self._hist
    @hist.setter
    def hist(self, val):     
        self._hist = val
        # self._hist = np.vstack((self.hist, psi))
    
    @property 
    def hist_t(self):       # Time history
        if not hasattr(self, '_hist_t'):    
            self._hist_t   =   np.array(self.t) 
        return self._hist_t    
    @hist_t.setter
    def hist_t(self, val):     
        self._hist_t = val
    # __________________________ General methods ___________________________ #
    
    def updateHistory(self, psi, t):        
        self.hist   =   np.vstack((self.hist, psi))
        self.hist_t =   np.hstack((self.hist_t, t))
        self.psi    =   psi[-1]
        self.t      =   t[-1]
        

    def viewHistory(self):
        psi     =    self.hist   
        t       =    self.hist_t                   
        labels  =   ['$\\eta$', '$\\mu$']
        colors  =   ['blue', 'red']
        t_zoom  =    min([len(t)-1, int(0.05 / self.dt)])        
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Van der Pol')
        for i in range(2):
            ax[i,0].plot(t,psi[:,i], color = colors[i])
            ax[i,0].set(ylabel=labels[i], xlim = [t[0], t[-1]])
            ax[i,1].plot(t,psi[:,i], color = colors[i])
            ax[i,1].set(xlim = [t[-t_zoom], t[-1]], yticks = [])
        plt.tight_layout() 
        plt.show()
        
    # _____________________ VdP specific methods ___________________________ #
    def getParams(self):
        return np.array([getattr(self, p) for p in self.params])
    
    def getObservables(self):
        if np.shape(self.hist)[0] == 1:
            print('Case has no psi history')
            eta = None
        else:
            eta     =   self.hist[:,0,:]
        return [eta], ["$\\eta$"]
    
    
    # _________________________ Governing equations ________________________ #
    @staticmethod
    def timeDerivative(psi, t, case):   
        eta, mu     =   psi[:2]
        P           =   case.alpha0.copy()
        
        if len(psi) > len(case.psi0):
            ii = len(case.psi0)
            for param in case.est_p:
                # exec(param + " = " + str(psi[ii]))
                P[param] = psi[ii]
                ii += 1
            
            
        deta_dt     =   mu
        dmu_dt      =  - P['omega']**2 * eta
        
        if case.LOM: # Cubic law
            dmu_dt  +=  mu * (2.*P['nu'] - P['kappa'] * eta**2)
        else:   # atan model
            dmu_dt  +=  mu * (P['beta']**2 / (P['beta'] + P['kappa']*eta**2) - 
                              P['beta'] + 2*P['nu']) 
            
            # dmu_dt  +=  mu * (2.*P['nu'] + P['kappa'] * eta**2 - P['gamma'] * eta**4) # higher order polinomial
        

        return np.hstack([deta_dt, dmu_dt, np.zeros(len(psi)-2)])
        

 # ========================================================================= #
def timeIntegrate(case, Nt=10000): #__________________________________________
    t       =   np.linspace(case.t, case.t + Nt * case.dt, Nt+1)
    psi     =   [odeint(case.timeDerivative, case.psi[:,0], t, (case, ))]
    psi     =   np.array(psi).transpose(1,2,0)    
    psi     =   psi[1:] # Remove initial condition
    t       =   t[1:] 
    
    return psi, t
    

if __name__ == '__main__':
    vdp     =   Case()
    state, time  =   timeIntegrate(vdp)
    vdp.updateHistory(state, time)    
    vdp.viewHistory()
    

