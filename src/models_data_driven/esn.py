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
        
        kwargs['psi0'] = np.concatenate(self.get_reservoir_state(), axis=0)
        Model.__init__(self, integrator_class=DiscreteIntegrator, **kwargs)



    # ______________________ New class attributes ______________________ #
    def modify_settings(self, **kwargs):
        # Modify the settings of the ESN_model
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f'Key {key} not in ESN_model class')
            
        if self.est_a and 'Wout' in self.est_a:
            if not self.Wout_svd:
                self.Wout_svd = True
                [self.Wout_U, self.Wout_Sigma0, self.Wout_Vh] = sla.svd(self.Wout, full_matrices=False)
                self.Wout_Sigma = self.Wout_Sigma0

            
            self.est_a.remove('Wout')

            for qj in np.arange(self.N_dim):
                key = f'svd_{qj}'
                self.est_a.append(key)
                setattr(self, key, self.Wout_Sigma0[qj])

        self.M = None 
        # print(f'[ESN_model] after M.shape={self.M.shape}')
    

    @property
    def alpha_labels(self):
        if not hasattr(self, 'est_a'):
            return  dict()
        else:
            lbls = dict()
            for key in self.est_a:
                if 'svd' in key:
                    _j = key.split('_')[1]
                    lbls[key] = f'$\\sigma_{_j}$'
                else:
                    lbls[key] = key
            return lbls

    @property
    def alpha_lims(self):
        return {key: (None, None) for key in self.est_a}

    # @property
    # def Wout_svd_to_estimate(self):
    #     """TODO: Modify settings to not estimate all  SVDs"""
    #     return [key for key in self.est_a if 'svd' in key]


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
        alpha_labels = self.est_a.copy()

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
        alpha = np.empty((len(self.est_a), self.m))
        for aj, param in enumerate(self.est_a):
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
        kwargs['psi0'] = np.concatenate(self.get_reservoir_state(), axis=0)
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
            labels += [f'$r_{j+1}$' for j in np.arange(self.N_units)]
            
        return labels

    @property
    def obs_labels(self):
        return [f'$u_{j+1}$' for j in np.arange(self.N_dim)]
        
    def set_states_to_update(self, reset=False):
        u, r = None, None

        psi = self.current_state

        # print(f'[m_dd] psi.shape {psi.shape}')

        if self.update_state and self.update_reservoir or reset:
            u = psi[:self.N_dim]
            r = psi[self.N_dim:self.N_dim+self.N_units]
        elif  self.update_state:
            u = psi[:self.N_dim]
        elif self.update_reservoir:
            r = psi[:self.N_units]
        else:
            raise ValueError
        return u, r

    def reset_history(self, hist, t):

        assert hist.shape[1] == self.N_dim + self.N_units + self.Na, \
        f'psi.shape ={hist.shape}; Ndim, Nunit, Na = {self.N_dim}, {self.N_units}, {self.Na}'

        # Reset state and time history
        self.hist = hist
        self.hist_t = t
        # Reset EchoStateNetwork states
        u, r = self.set_states_to_update(reset=True)
        self.reset_state(u=u, r=r)

    def reset_last_state(self, psi, t=None):
        
        self.hist[-1] = psi
        if t is not None:
            self.hist_t[-1] = t
            
        u, r = self.set_states_to_update()
        self.reset_state(u=u, r=r)


    def reservoir_to_physical(self, r_aug=None):
        if not self.Wout_svd:
            return np.dot(r_aug.T, self.Wout).T
        else:
            Wout = np.einsum('ij,kjl,lm->imk', self.Wout_U, self.Wout_Sigma, self.Wout_Vh)
            
            return np.einsum('ij,ikj->kj', r_aug, Wout)
        


    def time_integrate(self, Nt=10, averaged=False, alpha=None):
        # Call the generic integrator to get the raw ESN steps
        psi_raw, t_raw = self.integrator.advance(Nt=Nt, averaged=averaged, alpha=alpha)
        
        # --- ESN-specific Post-Processing ---
        # (This is where the interpolation and state reset from your original code goes)
        
        t_physical = t_raw # Placeholder: The actual interpolation logic must go here
        
        # Update ESN state (assuming state is in psi_raw[-1])
        # self.reset_state(...)
        
        return psi_raw, t_physical
    
    
    def time_step(self, psi0, Nt=10, averaged=False):
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
        # 1. get initiall condition


        interp_flag = False
        Nt = Nt // self.upsample
        if Nt % self.upsample:
            Nt += 1
            interp_flag = True

        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)


        r = np.empty((Nt + 1, self.N_units, self.u.shape[-1]))
        u = np.empty((Nt + 1, self.N_dim, self.u.shape[-1]))
        u[0, self.observed_idx], r[0] = self.get_reservoir_state()


        for i in range(Nt):
            u_input = self.outputs_to_inputs(full_state=u[i])
            u[i + 1], r[i + 1] = self.step(u_input, r[i])



    def time_integrate(self, Nt=10, averaged=False, alpha=None):
        """
            NB: No parallel computation here
            Args:
                Nt: number of forecast steps
                averaged (bool): if true, each member in the ensemble is forecast individually. If false,
                                the ensemble is forecast as a mean, i.e., every member is the mean forecast.
                alpha: possibly-varying input_parameters
            Returns:
                psi: forecasted state (Nt x N x m)
                t: time of the propagated psi
        """

        interp_flag = False
        Nt = Nt // self.upsample
        if Nt % self.upsample:
            Nt += 1
            interp_flag = True

        t = np.round(self.current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)

        if averaged:
            u_m, r_m = [np.mean(xx, axis=-1, keepdims=True) for xx in self.get_reservoir_state()]
            for i in range(Nt):
                self.input_parameters = [self.alpha0[key] for key in self.est_a]

                u_input = self.outputs_to_inputs(full_state=u_m[i])
                u_m[i + 1], r_m[i + 1] = self.step(u_input, r_m[i])
            
            # Copy the same state into all ensemble members
            u, r = [np.repeat(xx, self.m, axis=-1) for xx in [u_m, r_m]]
            assert u.shape == ()
        else:
            u, r = self.closedLoop(Nt)

        # Interpolate if the upsample is not multiple of dt or if upsample > 1
        if self.upsample > 1 or interp_flag:
            t_physical = np.round(self.current_time + np.arange(0, Nt * self.upsample + 1) * self.dt, self.precision_t)
            u, r = [interpolate(t, xx, t_eval=t_physical) for xx in [u, r]]
        else:
            t_physical = t.copy()

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        psi = np.concatenate((u, r), axis=1)

        if hasattr(self, 'std_a'):
            alph = self.get_alpha_matrix
            alph = np.tile(alph, reps=(psi.shape[0], 1, 1))
            psi = np.concatenate((psi, alph), axis=1)

        return psi[1:], t_physical[1:]

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




