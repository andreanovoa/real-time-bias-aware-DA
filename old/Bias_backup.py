# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:06:13 2022

@author: an553
"""

'''   
    TODO: 
        - Understand the code lol
        - Make it a function with allowed inputs: tr
'''

import os as os
os.environ["OMP_NUM_THREADS"] = '1' # imposes only one core
import numpy as np
import matplotlib.pyplot as plt
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
import time

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=20)
plt.rc('legend', facecolor='white', framealpha=1, edgecolor='white')

figs_folder =os.getcwd()+'\\figs\\'
data_folder =os.getcwd()+'\\data\\'

# %% FUNCTIONS
    

def RVC_Noise(x):
    # chaotic Recycle Validation    
    global rho, sigma_in, tikh_opt, k, ti, noise_opt
    rho      = x[0]
    sigma_in = 10**x[1]
        
        
    lenn     = tikh.size
    len1     = noises.size
    Mean     = np.zeros((lenn, len1))
    
    #Train using tv: training+val, Wout is passed with all the combinations of tikh and target noise
    Xa_train, Wout, LHS0, RHS0 = train_n(U_wash, U_tv, Y_tv, tikh, sigma_in, rho)

    #Different validation folds
    for i in range(N_fo):

        p      = N_in + i*N_fw
        Y_val  = U[N_wash + p : N_wash + p + N_val].copy() #data to compare the cloop prediction with
        
        
        for jj in range(len1):
            for j in range(lenn):
                #cloop for each tikh-noise combinatio
                Yh_val      =   closed_loop(N_val-1, Xa_train[p], \
                                            Wout[j, jj], sigma_in, rho)[0] 
                Mean[j,jj] +=   np.log10(np.mean((Y_val-Yh_val)**2)/\
                                         np.mean(norm**2))
                # prevent from diverging to infinity: put MSE equal to 10^10 
                # (useful for hybrid and similar architectures)
                if np.isnan(Mean[j,jj]) or np.isinf(Mean[j,jj]):
                    Mean[j,jj] = 10*N_fo
                
    # select and save the optimal tikhonov and noise level in the targets
    a           =   np.unravel_index(Mean.argmin(), Mean.shape)
    tikh_opt[k] =   tikh[a[0]]
    noise_opt[k]=   noises[a[1]]
    k          +=1
    print(k,'Par :', rho,sigma_in, tikh[a[0]], noises[a[1]], Mean[a]/N_fo)

    return Mean[a]/N_fo

## ESN with bias architecture

def step(x_pre, u, sigma_in, rho):
    """ Advances one ESN time step.
        Args:
            x_pre: reservoir state
            u: input
        Returns:
            new augmented state (new state with bias_out appended)
    """
    # input is normalized and input bias added
    u_augmented = np.hstack((u/norm, np.array([bias_in]))) 
    # hyperparameters are explicit here
    x_post      = np.tanh(np.dot(u_augmented*sigma_in, Win) + rho*np.dot(x_pre, W)) 
    # output bias added
    x_augmented = np.concatenate((x_post, np.array([bias_out])))
    return x_augmented

def open_loop(U, x0, sigma_in, rho):
    """ Advances ESN in open-loop.
        Args:
            U: input time series
            x0: initial reservoir state
        Returns:
            time series of augmented reservoir states
    """
    N_units = len(x0)
    bias_out = 1.
    
    
    N  = U.shape[0]
    Xa = np.empty((N+1, N_units+1))
    Xa[0] = np.concatenate((x0, np.array([bias_out])))#, U[0]/norm))
    for i in np.arange(1,N+1):
        Xa[i] = step(Xa[i-1,:N_units], U[i-1], sigma_in, rho)

    return Xa

def closed_loop(N, x0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N+1, dim))
    Yh[0] = np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa

def train_n(U_washout, U_train, Y_train, tikh, sigma_in, rho):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf_washout = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]

    ## open-loop train phase
    Xa = open_loop(U_train, xf_washout, sigma_in, rho)
    
    ## Ridge Regression
    LHS  = np.dot(Xa[1:].T, Xa[1:])

    Wout = np.zeros((len(tikh),len(noises),N_units+1,dim))
    RHS  = np.zeros((len(noises),N_units+1,dim))
    for jj in range(len(noises)):
        
        RHS[jj]  = np.dot(Xa[1:].T, Y_train[jj])
        
        for j in range(len(tikh)):
                Wout[j, jj] = np.linalg.solve(LHS + tikh[j]*np.eye(N_units+1), RHS[jj])
    

    return Xa, Wout, LHS, RHS

def train_save_n(U_washout, U_train, Y_train, tikh, sigma_in, rho, noise):
    """ Trains ESN.
        Args:
            U_washout: washout input time series
            U_train: training input time series
            tikh: Tikhonov factor
        Returns:
            time series of augmented reservoir states
            optimal output matrix
    """

    ## washout phase
    xf_washout = open_loop(U_washout, np.zeros(N_units), sigma_in, rho)[-1,:N_units]

    ## open-loop train phase
    Xa = open_loop(U_train, xf_washout, sigma_in, rho)
    
    ## Ridge Regression
    LHS  = np.dot(Xa[1:].T, Xa[1:])
    sh_0 = Y_train.shape[0]
    
    for i in range(N_lat):
            Y_train[:,i] = Y_train[:,i] + rnd.normal(0, noise*data_std[i], sh_0)
    RHS  = np.dot(Xa[1:].T, Y_train)
    
    Wout = np.linalg.solve(LHS + tikh*np.eye(N_units+1), RHS)

    return Wout

def closed_loop_test(N, x0, Y0, Wout, sigma_in, rho):
    """ Advances ESN in closed-loop.
        Args:
            N: number of time steps
            x0: initial reservoir state
            Wout: output matrix
        Returns:
            time series of prediction
            final augmented reservoir state
    """
    xa = x0.copy()
    Yh = np.empty((N+1, dim))
    Yh[0] = Y0 #np.dot(xa, Wout)
    for i in np.arange(1,N+1):
        xa = step(xa[:N_units], Yh[i-1], sigma_in, rho)
        Yh[i] = np.dot(xa, Wout)

    return Yh, xa
    
#%%


def main(data, 
         upsample = 1,
         run_test=False, 
         plot_search=True):
    
    
    dt_data     =   1./10000            # integration time step
    # upsample    =   1                   # upsample of the ESN from dt
    dt          =   dt_data * upsample  # ESN time step
    
    U           =   data['bias'][::upsample]
    N_lat       =   U.shape[1] #number of dimensions
    
    t_lyap      =   .1                  # Lyapunov time (AN: not really?)
    N_lyap      =   int(t_lyap/dt)      # number of time steps in one Lyapunov time
    
    # number of time steps for washout, train, validation, test
    N_wash      =   int(0.1/dt)
    N_train     =   int(1.8/dt)
    N_val       =   int(0.2/dt)  #length of the data used is train+val
    
    N_tv        =   N_train + N_val
    N_wtv       =   N_wash + N_tv   # useful for compact code later
    
    N_test      =   len(U) - N_wtv -10
        
    
    
    print('')
    print('Training time :',(N_train)*dt, 's')
    print('Validation time :',(N_val)*dt, 's')
    
    #compute norm (normalize inputs by component range)
    U_data  =   U[:N_wash+N_train+N_val]
    m       =   U_data.min(axis=0)
    M       =   U_data.max(axis=0)
    norm    =   M - m
    
    ## ========================== DIVIDE THE DATA ========================= ##
    U_wash  =   U[:N_wash]
    U_tv    =   U[N_wash:N_wtv-1]
    Y_tv    =   U[N_wash+1:N_wtv].reshape(1,N_tv-1,N_lat)
    
    ## ============================ ADD NOISE ============================= ##
    ## Add noise to inputs and targets during training. Larger noise_level  
    ## promote stability in long term, but hinder time accuracy
    noisy       =   True
    noise_level =   0.03 
    noises      =   np.array([noise_level]) #target noise (if multiple, optimize)
    data_std    =   np.std(U,axis=0)
    
    seed        =   0                        
    rnd         =   np.random.RandomState(seed)
    
    if noisy: #input noise (it is not optimized)
        for i in range(N_lat):
            U_tv[:,i]   =   U_tv[:,i].copy() + \
                            rnd.normal(0, noise_level*data_std[i], N_tv-1)
    
        Y_tv  = np.zeros((len(noises), N_tv-1, N_lat))
        for jj in range(noises.size):
            for i in range(N_lat):
                Y_tv[jj,:,i] =  U[N_wash+1:N_wtv,i].copy() + \
                                rnd.normal(0, noises[jj]*data_std[i], N_tv-1)
    
    
    ## ======================= INIT ESN HYPERPARAMS ======================= ##
    
    bias_in     =   .1          # input bias
    bias_out    =   1.0         # output bias 
    N_units     =   100         # units in the reservoir
    dim         =   U.shape[1]  # dimension of inputs (and outputs) 
    connect     =   5           # average connections/row in the state matrix 
    sparse      =   1 - connect/(N_units-1)     
    tikh        =   np.array([1e-4,1e-8,1e-12,1e-16])  # Tikhonov factor evals    
    
    ## ================ GRID SEARCH AND BAYESIAN PARAMS =================== ##
    
    # Set the parameters for Grid Search and Bayesian Optimization    
    n_tot       =   20   #Total Number of Function Evaluatuions
    n_in        =   0    #Number of Initial random points    
    spec_in     =   .5   #range for hyperp (spectral radius and input scaling)
    spec_end    =   1.2    
    in_scal_in  =   np.log10(1e-3)
    in_scal_end =   .3
        
    # In case we want to start from a grid_search, the first n_grid^2 points 
    # are from grid search if n_grid^2 = n_tot then it is pure grid search
    n_grid = 4  # (with n_grid**2 < n_tot you get Bayesian Optimization)
    
    # computing the points in the grid
    if n_grid > 0:
        x1    = [[None] * 2 for i in range(n_grid**2)]
        k     = 0
        for i in range(n_grid):
            for j in range(n_grid):
                x1[k] = [spec_in + (spec_end - spec_in)/(n_grid-1)*i,
                         in_scal_in + (in_scal_end - in_scal_in)/(n_grid-1)*j]
                k   += 1
    
    # range for hyperparameters
    search_space =  [Real(spec_in, spec_end, name='spectral_radius'),
                     Real(in_scal_in, in_scal_end, name='input_scaling')]
    
    # ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
    kernell     =   ConstantKernel(constant_value=1.0, 
                                   constant_value_bounds=(1e-1, 3e0))* \
                                   Matern(length_scale=[0.2,0.2], \
                                   nu=2.5, length_scale_bounds=(1e-2, 1e1)) 
    
    #Hyperparameter Optimization using either Grid Search or Bayesian Optimization
    def g(val):
        
        #Gaussian Process reconstruction
        b_e = GPR(kernel = kernell,
                normalize_y = True, # true => mean = avg of the obj func data, else = 0
                n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
                noise = 1e-10, # only for numerical stability
                random_state = 10) # seed
        
        
        #Bayesian Optimization
        res = skopt.gp_minimize(val,                         # the function to minimize
                          search_space,                      # the bounds on each dimension of x
                          base_estimator       = b_e,        # GP kernel
                          acq_func             = "gp_hedge", # the acquisition function
                          n_calls              = n_tot,      # total number of evaluations of f
                          x0                   = x1,         # Initial grid search points to be evaluated at
                          n_random_starts      = n_in,       # the number of additional random initialization points
                          n_restarts_optimizer = 3,          # number of tries for each acquisition
                          random_state         = 10,         # seed
                               )   
        return res
    
    
    ## ===================== TRAIN & VALIDATE NETWORK ===================== ##
    
    
    ens     =   1           # Number of Networks in the ensemble    
    val     =   RVC_Noise   # Which validation strategy
    N_fo    =   15          # number of folds
    N_in    =   0           # interval before the first fold
    N_fw    =   N_lyap//2   # NUM Steps forward the val interval is shifted
                            # (N_fw*N_fo has to be smaller than N_train)
    
    #Quantities to be saved
    par      = np.zeros((ens, 4))      # GP parameters
    x_iters  = np.zeros((ens,n_tot,2)) # coordS in hp space where f has been evaluated
    f_iters  = np.zeros((ens,n_tot))   # values of f at those coordinates
    minimum  = np.zeros((ens, 5))      # minima found per each member of the ensemble
    
    tikh_opt = np.zeros(n_tot) #optimal tikhonov
    noise_opt= np.zeros(n_tot) #optimal noise  (both then saved in minimum)
    Woutt    = np.zeros(((ens, N_units+1,dim))) #to save output matrix
    Winn     = np.zeros((ens,dim+1, N_units))   #to save input matrix
    Ws       = np.zeros((ens,N_units, N_units)) #to save state matrix
    
    # save the final gp reconstruction for each network
    gps      = [None]*ens
    
    # to check time
    ti       = time.time()
    
    print('')
    print('HYPERPARAMETER SEARCH:')
    print(str(n_grid) + 'x' +str(n_grid) + ' grid points plus ' + 
          str(n_tot-n_grid**2) + ' points with Bayesian Optimization')
    print('')
    
    
    ## ======================== Win & W GENERATION ======================== ##
    for i in range(ens):       
        print('Realization    :', i+1)
        k   =   0
        seed    =   i+1
        rnd     =   np.random.RandomState(seed)
    
        Win     =   np.zeros((dim+1, N_units))
        for j in range(N_units):
            #only one element different from zero per row
            Win[rnd.randint(0, dim+1),j] = rnd.uniform(-1, 1) 
        
        
        W   =   rnd.uniform(-1, 1, (N_units, N_units)) * \
                (rnd.rand(N_units, N_units) < (1-sparse)) # set the sparseness
        spectral_radius     =   np.max(np.abs(np.linalg.eigvals(W)))
        W   /=  spectral_radius #scaled to have unitary spec radius
        
        # Bayesian Optimization
        res        =    g(val)
        
        
        #Saving Quantities for post_processing anmd future use of the network
        gps[i]     =    res.models[-1]    
        gp         =    gps[i]
        x_iters[i] =    np.array(res.x_iters)
        f_iters[i] =    np.array(res.func_vals)
        minimum[i] =    np.append(res.x,[tikh_opt[np.argmin(f_iters[i])],
                                      noise_opt[np.argmin(f_iters[i])],res.fun])
        params     =    gp.kernel_.get_params()
        key        =    sorted(params)
        par[i]     =    np.array([params[key[2]],params[key[5]][0],\
                                  params[key[5]][1], gp.noise_])
        
        Woutt[i]   =    train_save_n(U_wash, U_tv, U[N_wash+1:N_wtv],\
                         minimum[i,2],10**minimum[i,1], \
                         minimum[i,0], minimum[i,3])
        
        Winn[i]    =    Win.copy()
        Ws[i]      =    W.copy()
        
        #Plotting Optimization Convergence for each network
        print('')
        print('Time per hyperparameter evaluation:', -(ti - time.time())/n_tot)
        print('Best Results: x', minimum[i,0], 10**minimum[i,1], minimum[i,2], minimum[i,3],
              ', f', -minimum[i,-1])
        print('')
    
    
    #%%
    if run_test:
        #### Quick test
        # Running the networks in the test set.
        
        subplots = 5 #number of plotted intervals
        plt.rcParams["figure.figsize"] = (15,2*subplots)
        plt.subplots(subplots,1)
        
        N_test  =  25 #number of test set intervals
        N_t1    =  N_wtv + 95 #when the first interval starts
        NNN     =  5    #length of the test set interval
        loops   =  5*NNN #number of updates of the input with correct data inside the test interval
        N_intt  =  NNN*N_val//loops #length of each subinterval before correct input data is given
        
        for k in range(ens):
            
            # load matrices and hyperparameters
            Win      =  Winn[k].copy()
            W        =  Ws[k].copy()
            Wout     =  Woutt[k].copy()
            rho      =  minimum[k,0].copy()
            sigma_in =  10**minimum[k,1].copy()       
            
            errors   = np.zeros(N_test)
            #Different intervals in the test set
            for i in range(N_test):
                        
                # data for washout and target in each interval
                U_wash  = U[N_t1 - N_wash+i*N_val : N_t1 + i*N_val]
                Y_t     = U[N_t1 + i*N_val: N_t1 + i*N_val + N_intt*loops] 
                
                #washout for each interval
                xa1        =    open_loop(U_wash, np.zeros(N_units), 
                                          sigma_in, rho)[-1]
                
                Yh_t, xa1  =    closed_loop_test(N_intt-1, xa1, Y_t[0],
                                                 Wout, sigma_in, rho)
                
                # Do the multiple subloops inside each test interval
                if loops > 1:
                    for j in range(loops-1):
                        Y_start    =     Y_t[(j+1)*N_intt-1].copy() #
        #                 Y_start    =  Yh_t[-1].copy()# #uncomment this to not update input
                        Y1, xa1    =    closed_loop_test(N_intt, xa1, Y_start, Wout, sigma_in, rho)
                        Yh_t       =    np.concatenate((Yh_t,Y1[1:]))
                
                errors[i] = np.log10(np.mean((Yh_t-Y_t)**2)/np.mean(norm**2))
                
        #         print(np.log10(np.mean((Yh_t-Y_t)**2)/np.mean(norm**2)))
                if i < subplots:
                    plt.subplot(subplots,1,i+1)
                    plt.plot(np.arange(N_intt*loops)*dt,Y_t[:,:2]/norm[:2], 'b')
                    plt.plot(np.arange(N_intt*loops)*dt,Yh_t[:,:2]/norm[:2], '--r')
                
        print('Median and max error in test:', np.median(errors), errors.max())  
        plt.tight_layout()
        plt.savefig(figs_folder + file + 'test_run.pdf')
        plt.close()
    

    
    #%% OUTPUTS 
    
    # Plot Gaussian Process reconstruction for each network in the ensemble afte n_tot evaluations
    # The GP reconstruction is based on the n_tot function evaluations decided in the search
    
    
    n_length    = 100 # points to evaluate the GP at
    xx, yy      = np.meshgrid(np.linspace(spec_in, spec_end,n_length),
                              np.linspace(in_scal_in, in_scal_end,n_length))
    x_x         = np.column_stack((xx.flatten(),yy.flatten()))
    x_gp        = res.space.transform(x_x.tolist())  ##gp prediction needs this normalized format 
    y_pred      = np.zeros((ens,n_length,n_length))
    
    fig = plt.figure(figsize=[10, 5],tight_layout=True)    
    for i in range(ens):
        # retrieve the gp reconstruction
        gp         = gps[i]
        
        plt.subplot(ens, 1, 1+i)
        
        amin = np.amin([10,f_iters.max()])
        
        y_pred[i] = np.clip(-gp.predict(x_gp), a_min=-amin,
                            a_max=-f_iters.min()).reshape(n_length,n_length) 
                            # Final GP reconstruction for each realization at the evaluation points
            
        # plt.title('Mean GP of realization \#'+ str(i+1))
        
        #Plot GP Mean
        plt.xlabel('Spectral Radius')
        plt.ylabel('Input Scaling (log-scale)')
        CS      = plt.contourf(xx, yy, y_pred[i],levels=20,cmap='Blues')
        cbar = plt.colorbar()
        cbar.set_label('-$\log_{10}$(MSE)',labelpad=15)
        CSa     = plt.contour(xx, yy, y_pred[i],levels=20,colors='black',
                              linewidths=1, linestyles='solid', alpha=0.3)
        
        #   Plot the n_tot search points
        plt.plot(x_iters[i,:n_grid**2,0],x_iters[i,:n_grid**2,1], 'v', c='w',\
                    alpha=0.8, markeredgecolor='k',markersize=10)
        plt.plot(x_iters[i,n_grid**2:,0],x_iters[i,n_grid**2:,1], 's', c='w',\
                  alpha=0.8, markeredgecolor='k',markersize=8) 
        # plt.scatter(x_iters[i,n_grid**2:,0],x_iters[i,n_grid**2:,1],
        #             c=np.arange(n_tot-n_grid**2), marker='s', cmap='Reds') #bayesian Optimization ones are plotted with sequential color indicating their orderind
        
    plt.savefig(filename + '_Hyperparameter_search.pdf')
    plt.close()
    
    
    np.savez(filename+'_ESN',
             norm       =   norm,
             Win        =   Winn,
             Wout       =   Woutt,
             W          =   Ws,
             dt         =   dt,
             N_wash     =   int(N_wash),
             N_unit     =   int(N_units),
             N_dim      =   int(dim),
             bias_in    =   bias_in,
             bias_out   =   bias_out,
             rho        =   minimum[0,0],
             sigma_in   =   10**minimum[0,1],
             upsample   =   int(upsample),
             hyperparameters    =   [minimum[0,0], 10**minimum[0,1],bias_in],\
             training_time      =   (N_train+N_val)*dt
             )

#%% ====================================================================== %%#


if __name__ == 'builtins':
    main(data)