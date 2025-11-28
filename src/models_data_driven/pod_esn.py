
from tools_ML import POD
from models_data_driven import ESN_model
from model import *
import scipy.linalg as sla
from utils import *


class POD_ESN(ESN_model, POD):
    """ Performs POD for a data matrix and trains an ESN to forecast the POD
        temporal coefficients.

        
        D(x,t) = Σ_j sigma_j φ_j(x) ψ_j(t)    for  j = 0, ..., N_modes-1

        [latex]
        D(x,t) = \\sum_j \\sigma_j \\phi_j(x) \\psi_j(t) 
        for j = 0, ..., N_modes-1

        POD properties:
            - Psi: temporal basis [N, N_modes], with N = Ndim x Nx x Ny
            - Phi: spatial basis [Nt, N_modes]
            - Sigma: POD Sigmas [N_modes, ],  note: Lambdas can be computed as: Sigma = np.sqrt(Lambda)
    """

    # name: str = 'POD-ESN'
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

        print(f'Nq={self.Nq}')

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
            dom = self.domain
        self._domain_of_measurement = dom

    @property
    def down_sample_measurement(self):
        if not hasattr(self, '_down_sample_measurement'):
            self.down_sample_measurement = None
        return self._down_sample_measurement
        
    
    @down_sample_measurement.setter
    def down_sample_measurement(self, dsm):
        if dsm is not None:
            if isinstance(dsm, int):
                dsm = [dsm, dsm]
            elif self.down_sample is not None:
                dsm = [int(x / y) for x, y in zip(dsm, self.down_sample)]
            else:
                raise ValueError()

        self._down_sample_measurement = dsm

    @property
    def grid_of_measurement(self):


        Nx, Ny = self.grid_shape[1:]

        if self.domain == self.domain_of_measurement:
            x_idx = np.arange(Nx)
            y_idx = np.arange(Ny)
        else:
            x_min, x_max, y_min, y_max = self.domain
            doi_x_min, doi_x_max, doi_y_min, doi_y_max = self.domain_of_measurement

            # Generate 1D spatial grids for original domain
            x = np.linspace(x_min, x_max, Nx)
            y = np.linspace(y_min, y_max, Ny)

            # Find indices within domain_of_interest along each axis
            x_idx = np.where((x >= doi_x_min) & (x <= doi_x_max))[0]
            y_idx = np.where((y >= doi_y_min) & (y <= doi_y_max))[0]


        if len(x_idx) == 0 or len(y_idx) == 0:
            raise ValueError('Domain of interest does not overlap with original domain grid.')

        down_sample = self.down_sample_measurement
        if down_sample is not None:
            step_x, step_y = down_sample
            x_idx = x_idx[::step_x]
            y_idx = y_idx[::step_y]


        return np.ix_(x_idx, y_idx)



    def define_sensors(self, N_sensors=None):

        # Define the measurement grid
        # grid_idx = POD.domain_mesh(domain=self.domain, 
        #                            grid_shape=self.grid_shape,
        #                            domain_of_interest=self.domain_of_measurement,
        #                            down_sample=self.down_sample_measurement,
        #                            ravel=True)[-1]
        
        Nx, Ny = self.grid_shape[1:]  # get spatial dims shape
        grid_idx = np.ravel_multi_index(self.grid_of_measurement, dims=(Nx, Ny)).ravel()

        # grid_idx = self.grid_of_measurement.ravel()

        # Select a number N_sensors of the grid wither randomly or according to qr =selection scheme
        if N_sensors is None:
            N_sensors = self.N_sensors

        if self.qr_selection:
            dom = self.Psi[grid_idx]
            qr_idx = sla.qr(dom.T, pivoting=True)[-1]
            idx = grid_idx[qr_idx[:N_sensors]]
        else:
            if N_sensors < len(grid_idx):
                idx = np.sort(self.rng.choice(grid_idx, size=N_sensors, replace=False), axis=None)
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
            ax.set_aspect('equal')

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