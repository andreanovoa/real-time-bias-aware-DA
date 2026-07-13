
from typing import Optional

from romda.tools import POD
from .esn import ESN_model
import scipy.linalg as sla
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from romda.utils import get_figsize_based_on_domain, add_pdf_page, plt_pdf


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
        
        Parameters
        ----------
        data : np.ndarray
            Data to be used for the POD decomposition and ESN training  [ (Nu, N_t, Nx, Ny) or (N_t, Ndim*Nx*Ny) ]
        plot_case : bool, optional
            Whether to plot the case. Defaults to True.
        pdf_file : None or str, optional
            Whether to save the plot case. If a string is provided, it is used as the filename. Defaults to None.
        skip_sensor_placement : bool, optional
            Whether to skip sensor placement. Defaults to False.
        train_ESN : bool, optional
            Whether to train the ESN. Defaults to True.
        **kwargs
            Additional keyword arguments to configure the parent Model/ESN/POD
            classes, e.g. ``domain`` (physical domain of the data), ``grid_shape``,
            ``Nq`` (number of sensors), ``sensor_locations``, ``N_modes``,
            ``t_train``, ``t_val``, ``N_units``, etc.
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
            Phi = self.Phi.copy()
            if Phi.ndim == 2:
                Phi = Phi[np.newaxis, ...]

            Phi = Phi.transpose(0, 2, 1)  # must be LxNtxNdim for ESN
            ESN_model.__init__(self,
                               data=Phi, 
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
            rec = self.reconstruct(X=data[:, -1])
            rec = self._to_physical_grid(rec)
            err = np.sqrt((rec - data[:, -1])**2) / np.nanmax(data[:, -1]**2)
            datasets={'Input': data[:, -1],
                      f'POD {self.N_modes} modes': rec,
                      'Error': err}

            POD_ESN.plot_case(case=self, num_modes=self.N_modes, datasets=datasets)

                                     
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

    @property
    def sensor_rows(self):
        """
        Rows of Psi / Q_mean corresponding to the sensor locations.
        The sensor_locations are raw-grid indices (var * Nx * Ny + g), whereas the POD
        basis rows follow the masked flat ordering — this property maps between the two.
        """
        if self.sensor_locations is None:
            return None
        if not hasattr(self, '_sensor_rows') or self._sensor_rows is None:
            self._sensor_rows = self.grid_index_to_flat_rows(self.sensor_locations)
        return self._sensor_rows

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
            if Phi is None:
                Phi = self.get_POD_coefficients(Nt=Nt) # Nt x N_modes x Ndim

            og_shape = Phi.shape
            reshape = Phi.ndim == 3
            if reshape:
                Phi = Phi.transpose(1, 0, 2)  # N_modes x Nt x Ndim
                Phi = Phi.reshape(self.N_modes, -1)  # N_modes x Nt*Ndim
                
            obs = self.decode(Z=Phi, idx=self.sensor_rows) #shape (N, Nt*Ndim)

            if reshape:
                obs = obs.reshape(self.Nq, og_shape[0], og_shape[2])  # Nq x Nt x Ndim
                obs = obs.transpose(1, 0, 2)  # Nt x Nq x Ndim


        return obs # Nt x Nq x m

    def reset_case(self, reset_POD=False, reset_ESN=False, Phi0=None, **kwargs):
        if reset_POD:
            self.rerun_POD_decomposition(**kwargs)
            reset_ESN = True  # The ESN must be reset to account for the change in POD modes

        if reset_ESN:
            if Phi0 is None:
                Phi0 = self.Phi[0]
            self.reset_ESN(psi0=Phi0, **kwargs)


    def select_sensors(self, measure_modes=False, 
                      domain_of_measurement=None, 
                      down_sample_measurement=None, 
                      N_sensors=None, qr_selection=False):
        
        self.measure_modes = measure_modes
        self._sensor_rows = None  # invalidate the cached Psi-row mapping
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
            self._domain_of_measurement = self.domain
        return self._domain_of_measurement

    @domain_of_measurement.setter
    def domain_of_measurement(self, dom):
        self._domain_of_measurement = dom #type: list

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
            
            elif not (isinstance(dsm, list) and len(dsm) == 2 and all(isinstance(x, int) for x in dsm)):
                raise ValueError()

        self._down_sample_measurement = dsm

    @property
    def grid_of_measurement(self):


        Nx, Ny = self.grid_shape[1:]

        if self.domain == self.domain_of_measurement or self.domain_of_measurement is None:
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

        grid = np.ravel_multi_index(np.ix_(x_idx, y_idx), dims=(Nx, Ny))
        # remove the idx not on fluid
        grid_idx_fluid = np.where(self.fluid_mask_flat)[0]
        grid_idx_fluid = np.intersect1d(grid_idx_fluid, grid)

        # append the idx for the other variables (e.g., uy) if needed
        if self.grid_shape[0] > 1:
            grid_idx_fluid = np.concatenate([grid_idx_fluid + Nx*Ny*i for i in range(self.grid_shape[0])], axis=None)

        return grid_idx_fluid



    def define_sensors(self, N_sensors=None):

        # Define the measurement grid

        Nu, Nx, Ny = self.grid_shape
        measure_grid_idx = np.asarray(self.grid_of_measurement)

        one_dom = measure_grid_idx[measure_grid_idx < Nx * Ny]  # only the first variable (e.g., ux) for sensor placement



        if N_sensors is None:
            N_sensors = self.N_sensors


        if self.qr_selection:

            Psi = self._to_physical_grid(self.Psi).transpose(1, 0, 2, 3)  # (r, Nu, Nx, Ny)

            Psi = np.nan_to_num(Psi, nan=0.0)
            Psi = Psi.reshape(Psi.shape[0], Nu, Nx * Ny)

            # choose one variable block for placement, e.g. variable 0
            A = Psi.reshape(Psi.shape[0], -1)  # shape (n_candidates, r)
            A = A[:, measure_grid_idx].T # shape (r, n_candidates)
            

            if N_sensors > A.shape[1]:
                A = np.dot(A, A.T)  # shape (n_candidates, n_candidates)

            qr_idx = sla.qr(A.T, pivoting=True)[-1]
            
            sensor_idx = measure_grid_idx[qr_idx[:N_sensors]]
            sensor_idx = sensor_idx.ravel() % (Nx * Ny)  # only the first variable (e.g., ux) for sensor placement    

            if np.unique(sensor_idx).size < N_sensors:
                print(f'Warning: QR selection returned {np.unique(sensor_idx).size} unique sensors, less than requested {N_sensors}.')
                sensor_idx = np.unique(sensor_idx)
                extra_needed = N_sensors - sensor_idx.size
                if extra_needed > 0:
                    extra_sensor_idx = measure_grid_idx[qr_idx[N_sensors:N_sensors + extra_needed]]
                    extra_sensor_idx = extra_sensor_idx.ravel() % (Nx * Ny) 
                    sensor_idx = np.concatenate([sensor_idx, extra_sensor_idx])
        else:
            if N_sensors < len(one_dom):
                sensor_idx = np.sort(self.rng.choice(one_dom, size=N_sensors, replace=False), axis=None)
            else:
                sensor_idx = one_dom.copy()

        if N_sensors > len(measure_grid_idx):
            print(f'Requested number of sensors {N_sensors} >= grid size in domain of measurement ({len(measure_grid_idx)})')


        #plot the sensors against the og grid and measurement grid for debugging
        plt.figure()
        # original grid
        x_idx, y_idx = np.unravel_index(np.arange(Nx*Ny), (Nx, Ny))
        plt.scatter(x_idx, y_idx, label='Original grid', alpha=0.01
                    )
        #measurement grid
        x_idx, y_idx = np.unravel_index(measure_grid_idx[:len(measure_grid_idx)//2], (Nx, Ny))
        plt.scatter(x_idx, y_idx, label='Measurement grid')

        #sensors
        x_idx, y_idx = np.unravel_index(sensor_idx, (Nx, Ny))
        plt.scatter(x_idx, y_idx, label='Sensors')
        plt.legend()
        plt.show()

        # sensors fro all u
        sensor_idx = [sensor_idx + Nx*Ny*i for i in range(self.grid_shape[0])]

        return np.array(sensor_idx).reshape((-1,))

    # ========================================== PLOTS =======================================================
    

    @staticmethod
    def plot_case(case, datasets: Optional[dict]=None, num_modes=None):

        from romda.plotting.pod import plot_modes, plot_time_coefficients, plot_spectrum, plot_flows_rms

        if num_modes is None:
            num_modes = case.N_modes

        plot_modes(case=case, num_modes=num_modes, cmap='viridis', n_col=2)
        plot_time_coefficients(case=case, num_modes=num_modes)
        plot_spectrum(case=case, max_mode=num_modes)


        if datasets is not None:
            display_sensors = case.sensor_locations is not None
            plot_flows_rms(case=case, datasets=datasets, display_sensors=display_sensors)
