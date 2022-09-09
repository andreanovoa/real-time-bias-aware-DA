# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:36:33 2022

@author: an553
"""


'''   
    TODO: 
        - Define a function getObservables in classType to generalise this code
'''



from VdP_backup_2022_05_04 import VdP, timeIntegrate

import pylab as plt
import numpy as np
from scipy.signal import find_peaks



plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

def bifurcationDiagram(classType, sweep_param, range_param, HOM=False):
    
    psi = []
    
    for param in range_param:
        case    =   classType(HOM=HOM)
        for attr in [sweep_param]:    setattr(case, attr, param)
        case.alpha      =   np.array([getattr(case, p) for p in case.params])
        case.hist_a[-1] =   case.alpha[:].reshape(len(case.params),1)
            
        psi_i, t      =   timeIntegrate(case, Nt = int(8./case.dt))
        case.updateHistory(psi_i, t)      
        psi.append(case.hist[-int(1./case.dt):])
        
    return psi

if __name__ == '__main__':
    param       =   'nu'
    HOM         =   False
    range_param =   np.arange(-5,20,1.)
    psi         =   bifurcationDiagram(VdP, param, range_param, HOM=HOM)
    
    plt.figure()
    for i in range(len(range_param)):
        y = np.squeeze(psi[i][:,0,:].copy())
        for k in [1,-1]:
            peaks = find_peaks(y*k)
            peaks = y[peaks[0]]
            plt.plot(np.ones(len(peaks))*range_param[i], peaks, '.', color='b')
            plt.xlabel('$\\'+param+'$')
            plt.ylabel('$\\eta$')
            if HOM: plt.title('HOM')
            else:   plt.title('LOM')
                
    plt.figure()
    plt.plot(y)
