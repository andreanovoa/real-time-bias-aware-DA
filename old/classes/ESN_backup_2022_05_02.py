# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:14:52 2022

@author: an553
"""

'''   
    TODO: 
        - Add a function to train and create the ESN from AR codes in Bias
        - Familiarise with @property stuff
'''

import os
os.environ["OMP_NUM_THREADS"] = '1' # imposes only one core
import numpy as np
import matplotlib.pyplot as plt
# import skopt
# from skopt.space import Real
# from skopt.learning import GaussianProcessRegressor as GPR
# from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
# import time


plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

class ESN(object):
    attrs   =  ['norm', 'Win', 'Wout', 'W', 'dt',
                'N_wash', 'N_unit', 'N_dim', 
                'bias_in', 'rho', 'sigma_in', 'bias_out',
                'upsample', 'hyperparameters', 'training_time']  
    
    def __init__(self, filename, train=False, *args):   
        
        if not train:        
            file = np.load(filename)
            for attr in file.files:       
                setattr(self, attr, file[attr])
        # else:
        #     run 
            
    # @classmethod
    # def trainESN(self):
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    @property
    def U(self):
        if not hasattr(self, '_U'):
            self._U = np.zeros(self.N_dim)     # initialise input to ESN
        return self._U
    
    @property
    def r(self):
        if not hasattr(self, '_r'):
            self._r = np.zeros(self.N_unit)    # initialise reservoir to zeros
        return self._r
    
    @U.setter
    def U(self, new_U):     self._U = new_U
        
    @r.setter
    def r(self, new_r):     self._r = new_r
          
        
    def updateReservoir(self, new_U, new_r):
        self.U  =   new_U
        self.r  =   new_r
    
        
        
    def step(self):
        """ Advances one ESN time step.
            Returns:
                new augmented reservoir state (new state with bias_out)
        """
        # input is normalized and input bias added
        U_aug   =   np.hstack((self.U/self.norm, np.array([self.bias_in]))) 
        # hyperparameters are explicit here
        r_post  =   np.tanh(np.dot(U_aug * self.sigma_in, self.Win) + \
                              self.rho * np.dot(self.r, self.W)) 
        # output bias added
        r_aug   =   np.hstack((np.squeeze(r_post), np.array([self.bias_out])))
        
        return r_aug

    def openLoop(self, U_wash):
        """ Initialises ESN in open-loop.
            Input:
                - U_wash: washout input time series
            Returns:
                - U:  prediction from ESN during open loop
                - ra: time series of augmented reservoir states
        """
        
        
        N       =   U_wash.shape[0] -1 # AN: I added the -1 because we don't want forecast, just initialisation
        ra      =   np.empty((N+1, self.N_unit+1))
        U       =   np.empty((N+1, self.N_dim))
        ra[0]   =   np.concatenate((self.r, np.array([self.bias_out])))
        for i in np.arange(1,N+1):
            self.updateReservoir(U_wash[i-1], ra[i-1,:self.N_unit])
            ra[i]   =   self.step()
            U[i]    =   np.dot(ra[i], self.Wout)
            
        # update reservoir with predicted step
        self.updateReservoir(U[i], ra[i,:self.N_unit])
        
        return U, ra

    def closedLoop(self, Nt):
        """ Advances ESN in closed-loop.
            Input:
                - Nt: number of forecast time steps
            Returns:
                - U:  forecast time series
                - ra: time series of augmented reservoir states
        """
        Nt      =   int(Nt)
        ra      =   np.empty((Nt+1, self.N_unit+1))
        U       =   np.empty((Nt+1, self.N_dim))
        ra[0]   =   np.concatenate((self.r, np.array([self.bias_out])))
        U[0]    =   np.dot(ra[0], self.Wout)        
        if U[0] == self.U:
            for i in np.arange(1,Nt+1):
                self.updateReservoir(U[i-1], ra[i-1,:self.N_unit])
                ra[i] = self.step()
                U[i] = np.dot(ra[i], self.Wout)
        else:
            print('closed loop - sowething is wrong')
        return U, ra


#%% ===========================================================================
if __name__ == '__main__':
    
    esn     =   ESN(os.getcwd() + '\\data\\bias_VdP_ESN.npz')    
    washout =   np.load(os.getcwd() + '\\data\\bias_VdP.npz')
    
    
    washout =   washout['bias']
    
    U, r = esn.openLoop(washout[10000:10000+int(esn.N_wash)])
    
    Uc, rc = esn.closedLoop(2000)

    plt.figure()
    
    
    plt.plot(washout[10000:], 
             '-o', color='grey', linewidth=5, alpha =0.4, label = 'truth')
    
    plt.plot(U, '-',marker='.', label='open loop')
    
    
    
    
    plt.plot(np.arange(len(U), len(U)+len(Uc), 1), 
             Uc, '-',marker='+', label='closed loop')
    
    
    
    plt.legend(loc='best')
    plt.ylim([min(washout)*1.1, max(washout)*1.1])
    plt.xlim([0, len(U)+len(Uc)])
    
    
    
    
    