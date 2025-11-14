from pdb import pm
from model import *
from tools_ML.EchoStateNetwork import EchoStateNetwork
import matplotlib.pyplot as plt
import scipy.linalg as sla

import inspect

from utils import interpolate


class ESN_model(EchoStateNetwork, Model):
    """ ESN model Class
        - Use a ESN as a data-driven forecast model
        - Note: training data is a mandatory input to the initialization
    """

    name: str = 'ESN_model'
    figs_folder: str = 'figs/ESN_model/'

    update_reservoir = True
    update_state = True

    Wout_svd = False

    t_train, t_val, t_test = None, None, 0.

    perform_test = True
    save_ESN_training = False

    upsample = 1

    N_wash = 5  # Number of washout steps i.e., open-loop initialization
    N_units = 50  # Number of neurons
    N_func_evals = 40
    N_grid = 5
    noise = 1e-2
    Win_type = 'sparse'
    N_folds = 8
    N_split = 5

    # Hyperparameter optimization ranges
    rho_range = (.2, .8)
    sigma_in_range = (np.log10(0.5), np.log10(50.))
    tikh_range = [1E-6, 1E-9, 1E-12]

    extra_print_params = ['rho', 'sigma_in', 'N_units', 'N_wash', 'upsample', 
                          'update_reservoir', 'update_state']


    def __init__(self,
                 data,
                 dt,
                 plot_training=True, 
                 **kwargs):
        """
        Arguments:
        - data: data to train the ESN (train + validate + test). The data shape must be [Na x Nt x Ndim].
        - psi0: initial state of the ESN prediction (not including the reservoir state).
        - plot_training: whether to plot or not the training data and training convergence.
        """

        self.dt = dt

        # Increase ndim if there is only one set of parameters
        if data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]
        elif data.ndim == 2:
            data = data[np.newaxis, :]
        

        # Check that the times are provided and not in time steps
        Nt = data.shape[1]
        for key in ["train", "val", "test"]:
            if f"N_{key}" in kwargs: 
                setattr(self, f"t_{key}", kwargs.pop(f"N_{key}") * self.dt)
                # print('setting t_{key} to {getattr(self, f"t_{key}")}, self.dt={self.dt}')s

        # Set other ESN_model attributes provided
        for key in list(kwargs.keys()):
            if key in vars(ESN_model):
                setattr(self, key, kwargs.pop(key))

        # _________________________ Set time attributes _________________________ #
        t_total = Nt * self.dt
        self.t_train = self.t_train or t_total * 0.8
        self.t_val = self.t_val or self.t_train * 0.2

        if self.perform_test:
            self.t_test = self.t_test or t_total - self.t_train - self.t_val

            assert abs((ts := sum([self.t_train, self.t_val, self.t_test])) - t_total) <= self.dt / 2., \
                f"t_train + t_val + t_test {ts} <= t_total {t_total}"


        # _________________________ Init EchoStateNetwork _______________________ #

        ESN_dict = dict()
        for key in list(kwargs.keys()):
            if key in vars(EchoStateNetwork):
                ESN_dict[key] = kwargs.pop(key)


        EchoStateNetwork.__init__(self,
                                  y=data[0, 0],
                                  dt=self.dt,
                                  **ESN_dict)

        self.t_CR = self.t_val

        # ______________________ Train the EchoStateNetwork _______________________ #
        # Train the network
        self.train_network(data=data, 
                           plot_training=plot_training, 
                           **kwargs)

        # Initialise SVD Wout terms if required
        if self.Wout_svd:
            [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
            self.Wout_Sigma = self.Wout_Sigma0

        # ________________________________ Init Model _______________________________ #
        
        kwargs['psi0'] = self.build_psi(*self.get_reservoir_state())[0]

        # print('Initializing Model with psi0 shape:', kwargs['psi0'].shape)
        Model.__init__(self, integrator_class=DiscreteIntegrator, **kwargs)



    # ______________________ New class attributes ______________________ #
    def modify_settings(self, **kwargs):
        # Modify the settings of the ESN_model
        print('Modifyig settings...')
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f'Key {key} not in ESN_model class')
        

        if self.ensemble:
            print(self.ensemble.keys())
            est_alpha = self.est_alpha.copy()

            if 'Wout' in est_alpha:
                if not self.Wout_svd:
                    self.Wout_svd = True
                    [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
                    self.Wout_Sigma = self.Wout_Sigma0

                # Replace wout for the SVD components to estimate them
                est_alpha.remove('Wout')

                for qj in np.arange(self.N_dim):
                    key = f'svd_{qj}'
                    setattr(self, key, self.Wout_Sigma0[qj])
                    est_alpha.append(key)
            print('Updated est_alpha:', est_alpha)
            self.est_alpha = est_alpha
            print('New est_alpha in config:', self.est_alpha)

        # Set the M matrix to None to force re-computation
        self.M = None 

    @property
    def dt_step(self):
        return self.dt_ESN
    
    @property
    def alpha_labels(self):
        lbls = {}
        if len(self.est_alpha) > 0:
            for key in self.est_alpha:
                if 'svd' in key:
                    _j = key.split('_')[1]
                    lbls[key] = f'$\\sigma_{_j}$'
                else:
                    lbls[key] = key
        return lbls

    @property
    def alpha_lims(self):
        return {key: (None, None) for key in self.est_alpha}
    


    @property
    def Wout_U(self):
        """Dimensions N_dim x N_dim"""
        return self._Wout_U
    
    @Wout_U.setter
    def Wout_U(self, U):
        assert U.shape == self.Wout.shape, f"Expected shape {self.Wout.shape}, got {U.shape}"
        self._Wout_U = U

    @property
    def Wout_Vh(self):
        """Dimensions N_dim x N_dim"""
        return self._Wout_Vh
    
    @Wout_Vh.setter
    def Wout_Vh(self, Vh):
        assert Vh.shape == (self.N_dim, self.N_dim), \
        f"Expected shape ({self.N_dim}, {self.N_dim}), got {Vh.shape}"
        self._Wout_Vh = Vh


    @property
    def Wout_Sigma(self):
        if self.Wout_svd:
            self.Wout_Sigma = self.alpha_to_Sigma
        return self._Wout_Sigma

    @property
    def alpha_to_Sigma(self):
        alpha_matrix = self.get_alpha_matrix
        alpha_labels = self.est_alpha

        eigs = np.zeros((self.m, self.N_dim, self.N_dim))

        for qi in range(self.N_dim):
            key = f'svd_{qi}'
            if key in alpha_labels:
                ai = alpha_labels.index(key)
                vals = alpha_matrix[ai]
            else:
                vals = self.Wout_Sigma0[qi] * np.ones(self.m)

            eigs[:, qi, qi] = vals

        return eigs
    
    @property
    def get_alpha_matrix(self):
        alpha = np.empty((len(self.est_alpha), self.m))
        for aj, param in enumerate(self.est_alpha):
            for mi, alpha_dict in enumerate(self.get_alpha()):
                alpha[aj, mi] = alpha_dict[param]
        return alpha


    @property
    def Wout_Sigma0(self):
        return self._Wout_Sigma0
    
    @Wout_Sigma0.setter
    def Wout_Sigma0(self, eigs):
        self._Wout_Sigma0 = eigs


    @Wout_Sigma.setter
    def Wout_Sigma(self, eigs):
        if eigs.ndim == 1:
            assert eigs.shape[0] == self.N_dim , \
                f"Expected shape ({self.N_dim},) got {eigs.shape}"
            eigs = np.diag(eigs)
        elif eigs.ndim == 2:
            assert eigs.shape[-1] == self.N_dim , \
                f"Expected shape ({self.N_dim},) got {eigs.shape}"
            if eigs.shape[0] != self.m:
                assert eigs.shape[0] == self.N_dim and np.allclose(eigs, np.diag(np.diagonal(eigs))), \
                    f"Expected diagonal matrix, got {eigs.shape}"
            else:
                eigs = np.array([np.diag(e) for e in eigs])  ## this will be needed for the parameter estimation
                assert eigs.shape == (self.m, self.N_dim, self.N_dim), \
                    f"Expected shape ({self.m}, {self.N_dim},{self.N_dim}) got {eigs.shape}"
        else:
            assert eigs.shape == (self.m, self.N_dim, self.N_dim), \
                f"Expected shape ({self.m}, {self.N_dim},{self.N_dim}) got {eigs.shape}"
    
        self._Wout_Sigma = eigs



    # ______________________ Changed EchoStateNetwork class attributes ______________________ #


    def initialise_state(self, data, N_ens=1, seed=0):
        if hasattr(self, 'seed'):
            seed = self.seed
        rng0 = np.random.default_rng(seed)
        # initialise state with a random sample from test data
        u_init, r_init = np.empty((self.N_dim, N_ens)), np.empty((self.N_units, N_ens))
        
        # Random time windows and dimension
        if data.shape[0] == 1:
            dim_ids = [0] * N_ens
        else:
            # Choose a random dimension from the data
            replace = N_ens <= data.shape[0]
            dim_ids = rng0.choice(data.shape[0], size=N_ens, replace=replace)

        # Open loop for each ensemble member
        t_ids = rng0.choice(data.shape[1] - self.N_wash, size=N_ens, replace=False)
        for ii, ti, dim_i in zip(range(N_ens), t_ids, dim_ids):
            self.reset_state(u=np.zeros((self.N_dim, 1)),
                              r=np.zeros((self.N_units, 1)))
            u_open, r_open = self.openLoop(data[dim_i, ti: ti + self.N_wash])
            u_init[:, ii], r_init[:, ii] = u_open[-1], r_open[-1]

        # Set physical and reservoir states as ensembles
        self.reset_state(u=u_init, r=r_init)


    def train_network(self, data, plot_training=True, **kwargs):

        if plot_training:
            ESN_model.plot_training_data(train_data=data, case=self)
        
        # Get the arguments of interest
        possible_args = inspect.getfullargspec(self.train)[0]
        train_args = {key: val for key, val in kwargs.items() if key in possible_args}
        # Train network        
        self.train(train_data=data, plot_training=plot_training, **train_args)


    def reset_ESN(self, data, u0=None, **kwargs):

        if u0 is None:
            u0 = self.get_reservoir_state()[0]  # self.u

        EchoStateNetwork.__init__(self,
                                  y=u0,
                                  dt=self.dt,
                                  figs_folder=self.figs_folder,
                                  **kwargs)

        self.train_network(data, **kwargs)

        # Reset model class
        kwargs['psi0'] = self.build_psi(*self.get_reservoir_state())
        self.reset_model(**kwargs) 


    # ______________________ Changed Model class attributes ______________________ #
        

    @property
    def Nq(self):
        return self.N_dim


    @property
    def state_labels(self):
        labels = []
        if self.update_state:
            labels += self.obs_labels
        if self.update_reservoir:
            labels += [f'$r_{{{j+1}}}$' for j in np.arange(self.N_units)]
            
        return labels

    @property
    def obs_labels(self):
        return [f'$u_{j+1}$' for j in np.arange(self.N_dim)]
        

    def reset_history(self, hist, t):
        print('Resetting history with shape:', hist.shape)
        # printz
        
        assert hist.shape[1] == self.N_dim + self.N_units + self.Na, \
        f'psi.shape ={hist.shape}; Ndim, Nunit, Na = {self.N_dim}, {self.N_units}, {self.Na}'

        # Reset state and time history
        self.hist = hist
        self.hist_t = t
        # Reset EchoStateNetwork states
        u, r = self.unbuild_psi()
        self.reset_state(u=u, r=r)

    def reset_last_state(self, psi, t=None):
        
        self.hist[-1] = psi
        if t is not None:
            self.hist_t[-1] = t
            
        u, r = self.unbuild_psi()
        self.reset_state(u=u, r=r)


    def reservoir_to_physical(self, r_aug=None):
        if not self.Wout_svd:
            return np.dot(r_aug.T, self.Wout).T
        else:
            Wout = np.einsum('ij,kjl,lm->imk', self.Wout_U, self.Wout_Sigma, self.Wout_Vh)
            
            return np.einsum('ij,ikj->kj', r_aug, Wout)
        

    
    
    def time_step(self, Nt=10, averaged=False):
        """
            Args:
                Nt: number of forecast steps (physical time, not dt_ESN)
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying input_parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        assert self.trained, 'ESN model not trained'
        # 1. get initial condition


        interp_flag = False
        Nt = Nt // self.upsample
        if Nt % self.upsample:
            Nt += 1
            interp_flag = True

        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)


        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))
        u[0, :], r[0] = self.get_reservoir_state()


        if averaged:
            # Mean state
            u_m, r_m = [np.mean(xx, axis=-1, keepdims=True) for xx in [u, r]]
            # deviations from the mean
            u_dev, r_dev = [xx - xm for xx, xm in zip([u, r], [u_m, r_m])]

            for i in range(Nt):
                u_m[i + 1], r_m[i + 1] = self._single_step(u_m[i], r_m[i])

            # copy into the ensemble members the mean + deviation
            u, r = [xm + xd for xm, xd in zip([u_m, r_m], [u_dev, r_dev])]
        else:

            for i in range(Nt):
                u[i + 1], r[i + 1] = self._single_step(u[i], r[i])

        return self.build_psi(u, r), t


    def _single_step(self, u, r):
        u_input = self.outputs_to_inputs(full_state=u)
        return self.step(u_input, r)
    

    def build_psi(self, u, r):
        """ Build the full state vector psi from physical states u and reservoir states r
         Returns:
            psi: full state vector (Nt x (Nphi + Na) x m)
        """
        if u.ndim == 2:
            u = u[np.newaxis, :, :]
        if r.ndim == 2:
            r = r[np.newaxis, :, :]

        if u.shape[0] != r.shape[0]:
            raise ValueError(f'Incompatible time dimension for u ({u.shape[0]}) and r ({r.shape[0]})')
    
        if self.update_state and self.update_reservoir:
            phi = np.concatenate((u, r), axis=1)
        elif self.update_state:
            phi = u
        elif self.update_reservoir:
            phi = r
        else:
            raise ValueError(f'Incompatible Nphi={self.Nphi} for ESN model with N_dim={self.N_dim} and N_units={self.N_units}')
        
        if self.Na > 0:
            alph = self.get_alpha_matrix
            alph = np.tile(alph, reps=(u.shape[0], 1, 1)) # repeat for all time steps (alpha is constant in time)
            return np.concatenate((phi, alph), axis=1) 
        else:
            return phi

    def unbuild_psi(self, psi=None):
        """ Extract physical states u and reservoir states r from the full state vector psi
            Args:
                psi: full state vector (N x m). If None, use the current_state
            Returns:
                u: physical states (N_dim x m) (or None if not updated)
                r: reservoir states (N_units x m) (or None if not updated)
        """
        if psi is None:
            psi = self.current_state

        if self.update_state and self.update_reservoir:
            u = psi[:self.N_dim]
            r = psi[self.N_dim:self.N_dim+self.N_units]
        elif  self.update_state:
            u = psi[:self.N_dim]
            r = None
        elif self.update_reservoir:
            r = psi[:self.N_units]
            u = None
        else:
            raise ValueError('Both update_state and update_reservoir are False')
        return u, r



    # ______________________________ Plotting functions ______________________________ #
    @staticmethod
    def plot_training_data(case, train_data):
        if train_data.ndim == 1:
            train_data = train_data[np.newaxis, :, np.newaxis]
        elif train_data.ndim == 2:
            train_data = train_data[np.newaxis, :]

        L, Nt, Ndim = train_data.shape
        t_data = np.arange(0, Nt) * case.dt

        nrows = min(Ndim, 20)
        for data_l in train_data:

            fig, axs = plt.subplots(nrows=nrows, ncols=1,
                                    figsize=(8, nrows), sharex=True,
                                    layout='constrained')
            axs = axs.T.flatten()

            for kk, ax in enumerate(axs):
                ax.plot(t_data, data_l[:, kk], lw=1., color='k')
                ax.axvspan(0, case.t_train, facecolor='orange',
                           alpha=0.3, zorder=-100, label='Train')
                ax.axvspan(case.t_train, case.t_train + case.t_val,
                           facecolor='red', alpha=0.3, zorder=-100, label='Validation')
                ax.axvspan(case.t_train + case.t_val,
                           case.t_train + case.t_val + case.t_test, facecolor='navy',
                           alpha=0.2, zorder=-100, label='Test')
            axs[0].legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.5))


    def visualize_config(self):
        self.plot_Wout()

        pm = self  # shorthand

        if pm.hist.shape[0] > 1:

            # Find global min and max for the color scale
            vmin, vmax = np.min(pm.hist[:, pm.Nq:pm.Nq+pm.N_units, :]), np.max(pm.hist[:, pm.Nq:pm.Nq+pm.N_units, :])


            fig1 = plt.figure(figsize=(8, 4), layout="constrained")
            axs1 = fig1.subplots(pm.Nq, 1, sharey=True, sharex=True)
            y = pm.get_observable_hist() # history of the model observables 
            lbl = pm.obs_labels

            norm_u = np.max(np.max(y[100:], axis=0, keepdims=True), axis=-1, keepdims=True).T - np.min(np.min(y[100:], axis=0, keepdims=True), axis=-1, keepdims=True).T
            u = (y - np.mean(y, axis=0, keepdims=True)) / (0.5*norm_u)


            # Choose a colormap
            cmap = plt.get_cmap('tab10', pm.m)  

            for ii, ax in enumerate(axs1):
                [ax.plot(pm.hist_t, u[:, ii, mi], c=cmap(mi)) for mi in range(pm.m)]
                ax.set(ylabel=lbl[ii])
            
            fig1.legend([f'$mi={mi}$' for mi in range(pm.m)], loc='center left', bbox_to_anchor=(1.0, .5), ncol=1, frameon=False)

            for ti in [10, 50, 75, 100]:
                for ax in axs1:
                    ax.set(xlim=[-.01, pm.hist_t[ti]+.01], ylim=[-1, 1])
                    ax.axvline(pm.hist_t[ti], c='k', ls='--')

                fig = plt.figure(figsize=(12, 8), layout="constrained")
                axs = fig.subplots(1, 2, width_ratios=[pm.Nq, pm.N_units], sharey=True)
                
                im1 = axs[0].imshow(u[ti].T, cmap='RdBu', vmin=-1, vmax=1)
                axs[0].set(title=f'physical state', ylabel='m_i', xlabel='u_i norm.')
                im2 = axs[1].imshow(pm.hist[ti, pm.Nq:pm.Nq+pm.N_units, :].T, cmap='PuOr', vmin=vmin, vmax=vmax)
                axs[1].set(title=f'reservoir state', xlabel='r_i')
                cbar = fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.2)
                cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.2)





    def visualize_spatiotemporal_hist(self,  y_hist=None, t=None, nrows=None, averaged=False):
        
        if y_hist is None:
            n_t = int(self.t_CR // self.dt)
            y_hist = self.hist[-n_t:, :self.Nphi]
            
        if t is None:
            t = self.hist_t[-len(y_hist):]

        if y_hist.shape[1] > self.N_dim:
            y_hist_list = [y_hist[:, :self.N_dim], y_hist[:, self.N_dim:self.N_dim + self.N_units]]
            titles = ['Physical state', 'Reservoir state']
            labels = [self.state_labels[:self.N_dim], self.state_labels[self.N_dim:self.N_dim + self.N_units]]
            cmaps = ['RdBu_r', 'PRGn']
        else:
            y_hist_list = [y_hist]
            titles = ['Physical state']
            labels = [self.state_labels[:self.N_dim]]
            cmaps = ['RdBu_r']
        
        if not averaged:
            if nrows is None:
                nrows = min(10, y_hist.shape[-1])
            
            for y_hist, ttl, lbl, cmap in zip(y_hist_list, titles, labels, cmaps):
                fig = plt.figure(figsize=(10, 1.5 * nrows))
                axs = fig.subplots(nrows=nrows, sharey=True, sharex=True)
                if nrows == 1:
                    axs = [axs]
                lim = np.max(abs(y_hist))

                for mi, ax in enumerate(axs):
                    im = ax.imshow(y_hist[:, :, mi].T, 
                                aspect='auto', origin='lower', 
                                cmap=cmap, vmin=-lim, vmax=lim,
                                # extent=[t[0], t[-1], 0, y_hist.shape[1]])  # TRANSPOSE
                                )
                        
                    
                axs[0].set(title=rf"ESN_model {ttl} spatiotemporal evolution. $N_\text{{units}}={self.N_units}$")
                axs[-1].set(xlabel="$t$")
                ytx = np.arange(len(lbl))+.5
                if len(lbl) > 6:
                    lbl, ytx = [zz[::len(lbl)//5] for zz in (lbl, ytx)]
                    
                axs[1].set(xlabel="$t$")
                [ax.set(yticks=ytx, yticklabels=lbl) for ax in axs] 
                fig.colorbar(im, ax=axs, orientation='vertical', shrink=1/nrows) 
        else:

            for y_hist, ttl, lbl, cmap in zip(y_hist_list, titles, labels, cmaps):
                # Averaged ensemble visualization
                y_mean_hist = np.mean(y_hist, axis=-1)

                fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)

                # Mean evolution
                lim_mean = np.max(abs(y_mean_hist))
                im0 = axs[0].imshow(y_mean_hist.T, 
                                    aspect='auto', origin='lower', 
                                    cmap=cmap, vmin=-lim_mean, vmax=lim_mean,
                                    # extent=[t[0], t[-1], 0, y_hist.shape[1]]
                                    )
                axs[0].set(title=rf"{ttl} spatiotemporal evolution (mean and std). $N_\text{{units}}={self.N_units}$")
                fig.colorbar(im0, ax=axs[0], orientation='vertical') 

                # Deviation covariance evolution

                var_ensemble = np.var(y_hist, axis=-1, ddof=1).T            # (Nt, Nx)
                var_ensemble = np.sqrt(var_ensemble)                     # Standard deviation

                lim_dev = np.max(abs(var_ensemble))
                im1 = axs[1].imshow(var_ensemble,  # Plot covariance of deviations
                                    aspect='auto', origin='lower', 
                                    cmap='magma', vmin=0, vmax=lim_dev,
                                    # extent=[t[0], t[-1], 0, y_hist.shape[1]]
                                    )
                                    
                fig.colorbar(im1, ax=axs[1], orientation='vertical')
                # Add ticks and labels

                ytx = np.arange(len(lbl))+.5
                if len(lbl) > 6:
                    lbl, ytx = [zz[::len(lbl)//5] for zz in (lbl, ytx)]
                    
                axs[1].set(xlabel="$t$")
                [ax.set(yticks=ytx, yticklabels=lbl) for ax in axs] 






    def plot_Wout(self):
        
        if not self.Wout_svd:
            # Visualize the output matrix
            fig, ax = plt.subplots()
            im = ax.matshow(self.Wout.T, cmap="PRGn", aspect=4., vmin=-np.max(self.Wout), vmax=np.max(self.Wout))
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            plt.colorbar(im, orientation='horizontal', extend='both')
            ax.set(ylabel='$N_u$', xlabel='$N_r$', title='$\\mathbf{W}_\\mathrm{out}$')

        else:
            fig, axs = plt.subplots(1, 4, figsize=(15, 15), width_ratios=[1, 1, 1, 1])
            eigs = self.Wout_Sigma
            if eigs.ndim >2:
                eigs = np.mean(eigs, axis=0)
            
            Wout = np.dot(self.Wout_U, np.dot(eigs, self.Wout_Vh))

            for W, ax, title in zip([Wout, self.Wout_U, eigs, self.Wout_Vh], axs, 
                                    ['$\\bar{\\mathbf{W_{out}}} = $', '$\\mathbf{U}$', '$\\bar{\\Sigma}$', '$\\mathbf{V}^\\mathrm{T}$']):
                cmap = 'PuOr'
                im = ax.imshow(W, cmap=cmap, vmin=-np.max(W), vmax=np.max(W))
                ax.set(title=title)
                # set the same colorbar for all the matrices
                fig.colorbar(im, ax=ax, shrink=.9, orientation='horizontal')




