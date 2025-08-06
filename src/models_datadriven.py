from src.model import *

from src.ML_models.EchoStateNetwork import EchoStateNetwork
from src.ML_models.POD import POD

import matplotlib.backends.backend_pdf as plt_pdf
from src.utils import interpolate, add_pdf_page, get_figsize_based_on_domain

import inspect
import numpy as np
import os as os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as sla

from time import time

from modulo_vki.modulo import ModuloVKI as Modulo


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
        Model.__init__(self, **kwargs)



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

        return 
    

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
        alpha_matrix = self.get_alpha_matrix()
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

        psi = self.get_current_state

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

        assert self.trained, 'ESN model not trained'

        interp_flag = False
        Nt = Nt // self.upsample
        if Nt % self.upsample:
            Nt += 1
            interp_flag = True

        t = np.round(self.get_current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)

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
            t_physical = np.round(self.get_current_time + np.arange(0, Nt * self.upsample + 1) * self.dt, self.precision_t)
            u, r = [interpolate(t, xx, t_eval=t_physical) for xx in [u, r]]
        else:
            t_physical = t.copy()

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        psi = np.concatenate((u, r), axis=1)

        if hasattr(self, 'std_a'):
            alph = self.get_alpha_matrix()
            alph = np.tile(alph, reps=(psi.shape[0], 1, 1))
            psi = np.concatenate((psi, alph), axis=1)

        return psi[1:], t_physical[1:]

    def get_alpha_matrix(self):
        alpha = np.empty((len(self.est_a), self.m))
        for aj, param in enumerate(self.est_a):
            for mi, alpha_dict in enumerate(self.get_alpha()):
                alpha[aj, mi] = alpha_dict[param]
        return alpha
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



class POD_ESN(ESN_model, POD):
    """ Performs POD for a data matrix and trains an ESN to forecast the POD
        temporal coefficients.

        
        D(x,t) = Σ_j σ_j φ_j(x) ψ_j(t)    for  j = 0, ..., N_modes-1

        [latex]
        D(x,t) = \sum_j \sigma_j \phi_j(x) \psi_j(t) \quad \\text{for} \quad j = 0, \ldots, N_\mathrm{modes}-1

        POD properties:
            - Psi: temporal basis [N, N_modes], with N = Ndim x Nx x Ny
            - Phi: spatial basis [Nt, N_modes]
            - Sigma: POD Sigmas [N_modes, ],  note: Lambdas can be computed as: Sigma = np.sqrt(Lambda)
    """

    name: str = 'POD-ESN'
    figs_folder: str = 'figs/POD-ESN/'

    Nq = 10
    t_CR = 0.5
    

    measure_modes = False  # Wether measurements are the POD coefficients
    sensor_locations = None
    qr_selection = True

    perform_test = False # Wether to perform testing of the ESN model

    extra_print_params = [*ESN_model.extra_print_params, 'Nq', 'measure_modes', 'N_modes']

    def __init__(self,
                 data,
                 dt,
                 plot_case=False,
                 pdf_file=None, 
                 skip_sensor_placement=False,
                 train_ESN=True,
                 domain_of_measurement=None,
                 down_sample_measurement=None,
                 **kwargs):
        """
        Initialize the POD-ESN model.
        
        Args:
            - data  (np.ndarray): Data to be used for the POD decomposition and ESN training  [Nu x ... x Nt]
            - plot_case (bool, optional): Whether to plot the case. Defaults to True.
            - pdf_file (None or str, optional): Whether to save the plot case. If a string is provided, it is used as the filename. Defaults to None.
            - skip_sensor_placement (bool, optional): Whether to skip sensor placement. Defaults to False.
            - train_ESN (bool, optional): Whether to train the ESN. Defaults to True.
            - **kwargs: Additional keyword arguments to configure the parent classes Model/ESN/POD.
                e.g.,   domain (list): Domain of the data.
                        grid_shape (tuple): Shape of the grid.
                        t_CR (float): Time constant for the ESN.
                        Nq (int): Number of measurements or sensors.
                        sensor_locations (list): Locations of the sensors.
                        etc.
        """

        for key in list(kwargs.keys()):
            if key in vars(POD_ESN):
                setattr(self, key, kwargs.pop(key))

        # __________________________ Init POD ___________________________ #
        POD.__init__(self,
                     X=data,
                     **kwargs)  # Initialize POD class and run decomposition

        # __________________________ Init ESN ___________________________ #
        # Initialize ESN to forecast the POD coefficients
        if train_ESN:
            ESN_model.__init__(self,
                               psi0=self.Phi[0],
                               data=self.Phi,
                               dt = dt,
                               plot_training=plot_case,
                               **kwargs)

        # __________________________ Select sensors ___________________________ #
        if self.measure_modes or skip_sensor_placement:
            self.Nq = self.N_modes
        elif self.sensor_locations is None:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            self.sensor_locations = self.define_sensors(N_sensors=self.Nq)
            self.Nq = len(self.sensor_locations)
        else:
            # If the sensors are already defined, use them
            self.Nq = len(self.sensor_locations)



        if plot_case:
            POD.plot_POD_modes(case=self, num_modes=self.N_modes, cmap='viridis')
            POD.plot_time_coefficients(case=self)
            POD.plot_spectrum(case=self)
            display_sensors = self.sensor_locations is not None
            POD.plot_flows_rms(case=self, datasets=[data], names=['original'], 
                               display_sensors=display_sensors)

            if pdf_file is not None:
                self.pdf_file = pdf_file
                if isinstance(self.pdf_file, str):
                    self.pdf_file = plt_pdf.PdfPages(f'{self.pdf_file}.pdf')

                figs = [plt.figure(ii) for ii in plt.get_fignums()]
                for fig in figs:
                    add_pdf_page(self.pdf_file, fig_to_add=fig, close_figs=True)


        print('========= POD-ESN model complete =========')

    @property
    def obs_labels(self):
        if self.measure_modes:
            obs_labels = [f"$\\Phi_{j+1}$" for j in np.arange(self.N_modes)]
        else:
            ux_labels = ["${u_x}" + f"_{j}$" for j in np.arange(self.N_sensors)]
            uy_labels = ["${u_y}" + f"_{j}$" for j in np.arange(self.N_sensors)]
            obs_labels = [*ux_labels, *uy_labels]
        assert len(obs_labels) == self.Nq
        return obs_labels
    

    @property
    def state_labels(self):
        labels = []
        if self.update_state:
            labels +=[f'$\\Phi_{j+1}$' for j in np.arange(self.N_modes)]
        if self.update_reservoir:
            labels += [f'$r_{j+1}$' for j in np.arange(self.N_units)]
        return labels

    @property
    def N_sensors(self):
        if self.measure_modes:
            return 0 
        else:
            return int(self.Nq / 2)

    def get_POD_coefficients(self, Nt=1):
        if Nt == 1:
            Phi = self.hist[-1, :self.N_modes]
        else:
            Phi = self.hist[-Nt:, :self.N_modes]
        return Phi

    def get_observables(self, Nt=1, Phi=None, **kwargs):
        if self.measure_modes:
            obs = self.get_POD_coefficients(Nt=Nt)
        else:
            Psi = self.Psi[self.sensor_locations]
            Q_mean = self.Q_mean[self.sensor_locations]
            if Phi is None:
                Phi = self.get_POD_coefficients(Nt=Nt)

            obs = self.reconstruct(Phi=Phi, 
                                   Psi=Psi, 
                                   Q_mean=Q_mean, 
                                   reshape=False)
            if obs.ndim == 4:
                obs = obs[0]
            if obs.ndim == 3:
                obs = obs.transpose(1, 0, 2)
            
        return obs

    def reset_case(self, reset_POD=False, reset_ESN=False, Phi0=None, **kwargs):
        if reset_POD:
            self.rerun_POD_decomposition(**kwargs)
            reset_ESN = True  # The ESN must be reset to account for the change in POD modes

        if reset_ESN:
            if Phi0 is None:
                Phi0 = self.Phi[0]
            self.reset_ESN(psi0=Phi0, **kwargs)


    def reset_sensors(self, measure_modes=False, 
                      domain_of_measurement=None, down_sample_measurement=None, N_sensors=None, qr_selection=False):
        
        self.measure_modes = measure_modes
        if measure_modes:
            self.Nq = self.N_modes
            self.sensor_locations = None
        else:
            self.domain_of_measurement = domain_of_measurement
            self.down_sample_measurement = down_sample_measurement
            self.qr_selection = qr_selection
            self.sensor_locations = self.define_sensors(N_sensors=N_sensors)
            self.Nq = len(self.sensor_locations)


    @property
    def domain_of_measurement(self):
        if not hasattr(self, '_domain_of_measurement'):
            self._domain_of_measurement = None
        return self._domain_of_measurement

    @domain_of_measurement.setter
    def domain_of_measurement(self, dom):
        if dom is None:
            dom = self.domain_of_interest
        self._domain_of_measurement = dom

    @property
    def down_sample_measurement(self):
        if not hasattr(self, '_down_sample_measurement'):
            self.down_sample_measurement = None
        return self._down_sample_measurement
        
    
    @down_sample_measurement.setter
    def down_sample_measurement(self, dsm):
        if dsm is None:
            dsm = self.down_sample.copy()
        elif isinstance(dsm, int):
            dsm = [dsm, dsm]
        elif self.down_sample is not None:
            dsm = [int(x / y) for x, y in zip(dsm, self.down_sample)]

        # print(f'_down_sample_measurement set to {dsm}')

        self._down_sample_measurement = dsm


    def define_sensors(self, N_sensors=None):

        # Define the measurement grid
        grid_idx = POD.domain_mesh(domain=self.domain, 
                                   grid_shape=self.grid_shape,
                                   domain_of_interest=self.domain_of_measurement,
                                   down_sample=self.down_sample_measurement,
                                   ravel=True)[-1]

        # Select a number N_sensors of the grid wither randomly or according to qr =selection scheme
        if N_sensors is None:
            N_sensors = self.N_sensors

        if self.qr_selection:
            dom = self.Psi[grid_idx]
            # t1 = time.time()
            qr_idx = sla.qr(dom.T, pivoting=True)[-1]
            # print('qr decomposition time :', time.time() - t1)
            idx = grid_idx[qr_idx[:N_sensors]]
        else:
            rng = self.rng
            if N_sensors < len(grid_idx):
                idx = np.sort(rng.choice(grid_idx, size=N_sensors, replace=False), axis=None)
            else:
                idx = grid_idx.copy()

        if N_sensors > len(grid_idx):
            print(f'Requested number of sensors {N_sensors} >= grid size in domain of measurement ({len(grid_idx)})')

        n_grid = self.grid_shape[1] * self.grid_shape[2]
        sensor_idx = [idx + (n_grid * ii) for ii in range(self.grid_shape[0])]

        return np.array(sensor_idx).reshape((-1,))

    # ========================================== PLOTS =======================================================


    @staticmethod
    def plot_case(case, datasets=None, names=None, num_modes=None):

        if num_modes is None:
            num_modes = case.N_modes

        POD.plot_POD_modes(case=case, num_modes=num_modes, cmap='viridis')
        POD.plot_time_coefficients(case=case,  num_modes=num_modes)
        POD.plot_spectrum(case=case,  max_mode=num_modes)

        if datasets is not None:
            display_sensors = case.sensor_locations is not None
            POD.plot_flows_rms(case=case, datasets=datasets, names=names, display_sensors=display_sensors)

        if case.trained:
            if num_modes is None:
                train_data=case.Phi
            else:
                train_data=case.Phi[:, :num_modes]
            ESN_model.plot_training_data(case=case, train_data=train_data)



    @staticmethod
    def plot_sensors(case, background_data=None):
        if background_data is None:
            background_data = case.reconstruct(Phi=case.Phi[-1], reshape=True)

        if background_data.ndim > 3:
            background_data = background_data[..., -1].copy()

        # Original domain of the data
        Ux = background_data.copy()

        # subset = not np.array_equal(case.domain_og, case.domain_of_interest)
        # if subset:
        #     X1_og, X2_og = POD.domain_mesh(domain=case.domain_og,
        #                                    grid_shape=case.grid_shape_og,
        #                                    ravel=False)[:2]

        # Domain of interest, i.e., cut version of the original
        Ux_focus = case.original_to_domain_of_interest(original_data=Ux)

        X1, X2, grid_idx = POD.domain_mesh(domain=case.domain_og,
                                           grid_shape=case.grid_shape_og,
                                           down_sample=case.down_sample,
                                           domain_of_interest=case.domain_of_interest,
                                           ravel=False)
        # Domain of measurement
        idx_s = case.sensor_locations[case.sensor_locations < len(X1.ravel())]

        down_sampled = case.down_sample != case.down_sample_measurement
        if down_sampled:
            grid_idx_m = POD.domain_mesh(domain=case.domain,
                                         grid_shape=case.grid_shape,
                                         down_sample=case.down_sample_measurement,
                                         domain_of_interest=case.domain_of_measurement,
                                         ravel=True)[-1]


        figsize, ncols, nrows = get_figsize_based_on_domain(domain=case.domain_of_interest, total_subplots=2)
    

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharey=True, sharex=True)

        norms = [mpl.colors.Normalize(vmin=np.min(u), vmax=np.max(u)) for u in [Ux[0], Ux[1]]]

        # windowed = case.domain_og != case.domain_of_measurement
        windowed = not np.array_equal(case.domain_of_interest, case.domain_of_measurement)

        for ii, ax in enumerate(axs):
            # if subset:
            #     ax.pcolormesh(X1_og, X2_og, Ux[ii],
            #                   cmap=mpl.colormaps['Greys'], norm=norms[ii], rasterized=True)
            ax.pcolormesh(X1, X2, Ux_focus[ii],
                          cmap=mpl.colormaps['viridis'], norm=norms[ii], rasterized=True)
            if windowed:
                dom = case.domain_of_measurement.copy()
                square = mpl.patches.Rectangle((dom[0], dom[2]), dom[1] - dom[0], dom[3] - dom[2],
                                               edgecolor='k', facecolor='none', lw=2,
                                               label="Domain of measurement", zorder=-10)

                ax.add_patch(square)

            if down_sampled:
                ax.plot(X1.ravel()[grid_idx_m], X2.ravel()[grid_idx_m], 'x', color='w', ms=5,
                        label="Possible sensor locations")

            ax.scatter(X1.ravel()[idx_s], X2.ravel()[idx_s],
                       c=np.arange(len(X2.ravel()[idx_s])),
                       cmap='YlOrRd', edgecolors='k', s=3.5 ** 2, lw=.5, label="Sensor locations")







    # @staticmethod
    # def plot_MSE_evolution(case,
    #                        original_data,
    #                        N_modes_cases=None,
    #                        N_modes_cases_plot=None):
    #     MSE = []
    #     MSE_sensors = []
    #
    #     if N_modes_cases is None:
    #         N_modes_cases = np.arange(case.N_modes, step=20)[1:]
    #     if N_modes_cases_plot is None:
    #         N_modes_cases_plot = [0, int(case.N_modes // 2), case.N_modes]
    #
    #     for _N_modes in N_modes_cases:
    #         _case = case.copy()
    #         _case.rerun_POD_decomposition(N_modes=_N_modes)
    #
    #         if _N_modes in N_modes_cases_plot:
    #             _MSE = POD_ESN_v2.plot_reconstruct(case=_case, original_data=original_data)
    #         else:
    #             _reconstructed_data = _case.reconstruct(Phi=_case.Phi[-1], reshape=True)
    #             if _reconstructed_data.ndim > 3:
    #                 _reconstructed_data = _reconstructed_data[..., -1]
    #             _MSE = POD_ESN_v2.compute_MSE(_reconstructed_data, original_data)
    #             plt.savefig(case.figs_folder + f'reconstruct_POD{_N_modes}', dpi=300)
    #
    #         MSE.append(_MSE)
    #
    #     plt.figure()
    #     plt.plot(N_modes_cases, MSE)
    #     plt.gca().set(xlabel='Number of modes in the decomposition', ylabel='MSE')