from src.utils import save_figs_to_pdf, get_figsize_based_on_domain, crop_data_to_domain_of_interest
from copy import deepcopy

import numpy as np
import os as os

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import RecurrencePlot
from modulo_vki.modulo import ModuloVKI as Modulo



class POD:
    """ Performs POD for a data matrix using MODULO
        temporal coefficients.

        D(x,t) = Σ_j σ_j φ_j(x) ψ_j(t)    for  j = 0, ..., N_modes-1

        POD properties:
            - Psi: temporal basis [N, N_modes], with N = Ndim x Nx x Ny
            - Phi: spatial basis [Nt, N_modes]
            - Sigma: POD Sigmas [N_modes, ],  note: Lambdas can be computed as: Sigma = np.sqrt(Lambda)
    """

    # POD input_parameters
    N_modes = 20
    method = 'snapshot'
    eig_solver = 'svd_sklearn_randomized'
    Q_mean = None

    field_labels = ['$u_x$', '$u_y$']
    POD_attrs = ['Phi', 'Psi', 'Sigma', 'grid_shape', 'Q_mean', 'domain']


    domain = [-1, 1, -1, 1]
    indices_to_original_grid = None

    name = 'POD'
    figs_folder = 'figs/POD/'


    def __init__(self,
                 X=None,
                 plot_decomposition=False,
                 save_decomposition=False,
                 domain_of_interest = None,   ## TODO: if provided, cut the data and down sample it
                 down_sample = None,
                 **kwargs):
        """
        Initialize the POD class.

        Args:
            - X (numpy.ndarray, optional): Snapshot data tensor with time as the last dimension: (Nu, Nx, Ny, Nt)
                If X is None, all POD attributes must be provided in kwargs.
            - domain (tuple, optional): Domain of the data.
            - plot_decomposition: Boolean flag to plot the decomposition.
            - save_decomposition: Boolean flag to save decomposition plots in a .pdf file.

        Raises:
            - TypeError: If X is not None and is not a numpy array.
            - KeyError: If required POD attributes are missing when X is None.
            - ValueError: If the domain is not provided.
        """

        model_dict = kwargs.copy()
        for key, val in kwargs.items():
            if hasattr(POD, key):
                setattr(self, key, val)
                del model_dict[key]

        # __________________________ Init POD ___________________________ #
        # Validate input data
        if X is not None and not isinstance(X, np.ndarray):
            raise TypeError('X must be either None or a np.ndarray.')

        elif X is None:
            try:
                [setattr(self, attr, model_dict[attr]) for attr in self.POD_attrs]
            except KeyError:
                raise KeyError('POD attributes not provided.')
            self.N_modes = self.Phi.shape[-1]

            for attr in ['domain', 'grid_shape']:
                if getattr(self, f'{attr}_og') is None:
                    setattr(self, f'{attr}_og', getattr(self, attr))

        elif isinstance(X, np.ndarray):
            # Remove NaN values and the mean fields
            X = X.copy()
            X[np.isnan(X)] = 0.

            # Downsample and cut the domain of interest
            if domain_of_interest is not None or down_sample is not None:

                # Set the downsample as x-y downsample list
                if isinstance(down_sample, int):
                    down_sample = [down_sample, down_sample]
                else:
                    assert isinstance(down_sample, list)

                # (Nu, Nx, Ny, Nt) = X.shape
                original_data = X.transpose(0, 3, 1, 2) # (Nu, Nt, Nx, Ny)
                cropped_data, cropped_grid_indices = crop_data_to_domain_of_interest(original_data, 
                                                                                     original_domain=self.domain,
                                                                                     domain_of_interest=domain_of_interest,
                                                                                     down_sample=down_sample)
                # Store new domain and the indices to the original dataset
                Q = cropped_data.transpose(0, 2, 3, 1) # (Nu, Nx, Ny, Nt)
                self.domain = domain_of_interest
                self.grid_shape = Q.shape[:-1]
                self.indices_to_original_grid = cropped_grid_indices
            else:
                Q = X

            # Concantenate all dimensions and remove the mean field
            Q = Q.reshape((-1, Q.shape[-1]))

            self.Q_mean = np.mean(Q, axis=-1, keepdims=True)
            Q -= self.Q_mean

            # Compute Total kinetic energy in the data
            u, v = self.restore_shape(Q)
            self._TKE = 0.5 * np.sum((np.mean(u ** 2, axis=2) + np.mean(v ** 2, axis=2)))

            # Perform POD decomposition using modulo_vki [https://github.com/mendezVKI/MODULO/]
            self.Psi, self.Phi, self.Sigma = self.run_modulo_POD(data=Q)


        # __________________________ Plot case ___________________________ #
        if plot_decomposition:
            POD.plot_POD_modes(case=self, num_modes=self.N_modes, cmap='viridis')
            POD.plot_time_coefficients(case=self)
            POD.plot_spectrum(case=self)
            POD.plot_flows_rms(case=self, datasets=X.copy(), names=['Original data'])

            if save_decomposition:
                os.makedirs(self.figs_folder, exist_ok=True)
                if self.name == POD.name:
                    filename = f'POD_{self.N_modes}-Grid{self.domain}.pdf'
                else:
                    filename = f'{self.name}.pdf'
                # Save all figures into a pdf
                save_figs_to_pdf(pdf_name=self.figs_folder + filename)



    def run_modulo_POD(self, data):
        """
        Perform POD decomposition using MODULO.

        Args:
            - data (numpy.ndarray): Input data matrix for POD decomposition. 
        
        Returns:
            - Psi: Temporal basis functions.
            - Phi: Spatial basis functions.
            - Sigma: Singular values.

        Raises:
            - NotImplementedError: If the method is not 'snapshot'
        """
        # Initialize modulo instance
        m = Modulo(data=data, eig_solver=self.eig_solver, n_Modes=self.N_modes)

        if self.method == 'snapshot':
            return m.POD()
        else:
            raise NotImplementedError(f'Method {self.method} not implemented')


    def project_data_onto_Psi(self, data, remove_mean=True):
        """
        Project input data onto the temporal basis Psi.  D(x, t) = Φ(t) Σ Ψ(x)
        
        Args:
           - data (numpy.ndarray): Input data shaped [..., Nt].
           - remove_mean (bool, optional): If True, subtracts the mean field before projection.
        
        Returns:
           - numpy.ndarray: Projection coefficients.
        """
        data = data.copy()
        if data.ndim > 3:
            data = data.reshape(-1, data.shape[-1])
        elif data.shape[0] != self.Q_mean.shape[0]:
            data = data.reshape(self.Q_mean.shape[0], -1)

        if remove_mean:
            data -= self.Q_mean

        return np.dot(np.diag(self.Sigma ** -1), np.dot(self.Psi.T, data)).T


    def reconstruct(self, Phi=None, Psi=None, Q_mean=None, reshape=True):
        """Reconstruct data with the selected number of POD modes.

        Args (all optional, default to self.Phi, self.Psi, self.Q_mean):
            - Phi: Time coefficients.
            - Psi: Spatial basis functions.
            - Q_mean: Mean field to add back.
            - reshape: Boolean flag to restore original shape.
        Returns:
            - Reconstructed flow field
        """

        if Phi is None:
            Phi = self.Phi[[-1]].T
        if Psi is None:
            Psi = self.Psi
        if Q_mean is None:
            Q_mean = self.Q_mean

        # Ensure Psi and Phi are in the correct shape
        # Phi = Phi.squeeze()
        if Phi.ndim < 2:
            Phi = Phi.reshape(self.N_modes, -1)

        # elif Phi.ndim == 2:
        #     if Phi.shape[0] != self.N_modes:
        #         Phi = Phi.transpose(1, 0)
        
        elif Phi.shape[-2] != self.N_modes: 
            shape_array = np.array(Phi.shape)
            i_mode = np.argwhere(shape_array == self.N_modes)[0]
            if len(i_mode) > 1:
                raise ValueError(f'Unsure about the dimensions.')
            else:
                # Shft the order of dimensions to ensure the modes are in the second to last dimension
                i_mode = i_mode[0]
                print(f'Changing the order of dimensions to ensure modes are in the second to last dimension: {i_mode}, {shape_array}--{Phi.shape}')

                new_order = np.arange(Phi.ndim)
                new_order[-2] = i_mode
                new_order[i_mode] = Phi.ndim - 2


                Phi = Phi.transpose(new_order)

            # shape_array = np.array(Phi.shape)
            # i_mode = np.argwhere(shape_array == self.N_modes)[0][0]
            # i_m = np.argwhere(shape_array == self.m)[0][0]
            # i_Nt = [ii for ii in np.arange(Phi.ndim) if ii not in [i_m, i_mode]][0]
            # Phi = Phi.transpose(i_Nt, i_mode, i_m)

        assert Phi.shape[-2] == self.N_modes, f'Phi must have {self.N_modes} modes.'

        new_shape = list(np.ones(Phi.ndim, dtype=int))
        new_shape[-2] = self.N_modes

        Sigma = np.reshape(self.Sigma, newshape=new_shape)


        Q_POD = np.dot(Psi, Sigma * Phi)
        
        if Q_POD.ndim > Q_mean.ndim:
            Q_mean = Q_mean[..., np.newaxis]
        
        reconstructed_data = Q_mean + Q_POD

        if reshape:
            reconstructed_data = self.restore_shape(data=reconstructed_data)

        return reconstructed_data


    def restore_shape(self, data=None):
        """
        Reshape data into the original shape of the dataset.
        
        Args:
            - data (numpy.ndarray, optional): Data to reshape. Defaults to self.Psi.
        
        Returns:
            - numpy.ndarray: Reshaped data.
        """
        if data is None:
            data = self.Psi
        if data.ndim > 1:
            if data.shape[:len(self.grid_shape)] == self.grid_shape:
                return data
            else:
                return np.reshape(data, newshape=(*self.grid_shape, data.shape[1]))
        else:
            return np.reshape(data, newshape=(*self.grid_shape,))

    # ========================================== METRICS ==================================================
    @staticmethod
    def compute_MSE(POD_data, original_data, time_evolution=False):
        """Compute the Mean Squared Error (MSE) between POD reconstructed data and the original dataset.

        Args:
            - POD_data: The reconstructed dataset from POD.
            - original_data: The original dataset.
            - time_evolution: If True, computes the MSE as a function of time; otherwise, computes a scalar MSE.

        Returns:
            - MSE value (float) or an np.ndarray if time_evolution=True.
        """
        POD_data, original_data = POD.flatten(POD_data, original_data)
        original_data[np.isnan(original_data)] = 0
        if time_evolution:
            MSE = np.mean((original_data - POD_data) ** 2, axis=0)
        else:
            MSE = np.mean((original_data - POD_data) ** 2)
        return MSE

    @staticmethod
    def compute_RMS(POD_data, original_data):
        """
        Compute the Root Mean Square (RMS) error between the reconstructed and original data.

        Args:
            - POD_data: The POD reconstructed dataset.
            - original_data: The original dataset.
        Returns:
            - RMS error (numpy array).
        """
        original_data[np.isnan(original_data)] = 0
        return np.sqrt((original_data - POD_data) ** 2)


    # ========================================== UTILITIES ==================================================
    @staticmethod
    def flatten(*args):
        """
        Flatten multidimensional arrays to 2D for processing.
        Args:
            - *args: Multiple numpy arrays.

        Returns:
            - List of reshaped arrays where the first dimension is flattened.
        """
        flat_args = []
        for arg in args:
            if arg.ndim > 2:
                flat_args.append(arg.reshape(-1, arg.shape[-1]))
            else:
                flat_args.append(arg.copy())
        return flat_args
    

    @property
    def domain_mesh(self, ravel=False):
        """
        Generate a mesh grid for the given domain.

        Args:
            - domain: Tuple specifying the domain (x_min, x_max, y_min, y_max).
            - grid_shape: The shape of the spatial grid.
            - domain_of_interest: Optional subdomain to extract.
            - down_sample: Downsampling factor (int or tuple).
            - ravel: If True, returns flattened arrays.
        Returns:
            - X1, X2: Mesh grid coordinates.
            - grid_idx_number: Indices corresponding to the grid.
        """

        # Generate the full grid
        x1 = np.linspace(*self.domain[:2], num=self.grid_shape[-2])
        x2 = np.linspace(*self.domain[2:], num=self.grid_shape[-1])

        # Calculate down sampled points
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')

        if ravel:
            return X1.ravel(), X2.ravel()
        else:
            return X1, X2



    def original_data_to_domain_of_interest(self, original_data):
        if self.indices_to_original_grid is None:
            return original_data
        else:
            original_data = original_data.copy()        
            try:
                if original_data.ndim == 2:
                    return original_data[self.indices_to_original_grid]
                else:
                    # Assume 1st dimesion is Nu 
                    return original_data[:, self.indices_to_original_grid[0], self.indices_to_original_grid[1]]
            except:
                raise ValueError('Make sure you are passing the original_data in [(Nu) x Nx x Ny x (Nt)]')
                

    
    # ========================================== METHODS ==================================================
    def rerun_POD_decomposition(self, data=None, N_modes=None):
        """
        Rerun POD decomposition with new data or mode count.

        Args:
            - data: New dataset for decomposition.
            - N_modes: Number of modes to retain.

        Raises:
            - ValueError: If neither data nor N_modes are provided.
        """
        if data is None and N_modes is None:
            raise ValueError("Either 'data' or 'N_modes' must be defined")

        if N_modes is None:
            N_modes = self.N_modes

        if N_modes < self.N_modes:
            # Truncate modes
            self.Psi = self.Psi[:, :N_modes]
            self.Phi = self.Phi[:, :N_modes]
            self.Sigma = self.Sigma[:N_modes]
            self.N_modes = N_modes

        elif data is None:
            raise ValueError(" Requested number of modes > self.N_modes. Data must be provided.")
        else:
            data[np.isnan(data)] = 0.
            data = data.reshape((-1, data.shape[-1]))
            data -= self.Q_mean

            self.Psi, self.Phi, self.Sigma = self.run_modulo_POD(data)


    def copy(self):
        """Return a deep copy of the POD instance."""
        return deepcopy(self)

    # ========================================== PLOTS =======================================================

    @staticmethod
    def plot_POD_modes(case, Psi=None, num_modes=2, save=False, cmap='viridis', dim=None):
        """
        Plot the first few POD modes.

        Arguments:
            - case: The POD instance.
            - Psi: Temporal basis functions (optional).
            - num_modes: Number of modes to display.
            - save: If True, saves the plots.
            - cmap: Colormap for visualization.
            - dim: Dimensions to plot.
        """
       
        if Psi is None:
            Psi = case.Psi.copy()

        if Psi.ndim == 2:
            Psi = case.restore_shape(Psi)

        if dim is not None:
            if isinstance(dim, int):
                dim = [dim]
        else:
            dim = np.arange(Psi.shape[0])

        POD_modes = [Psi[dim_i] for dim_i in dim]

        X1, X2 = case.domain_mesh()

        # n_col, n_row = calculate_subplot_grid(case.domain_of_interest, total_subplots=num_modes)
        figsize, n_col, n_row = get_figsize_based_on_domain(case.domain_of_interest, total_width=8, total_subplots=num_modes)

        for jj, data in enumerate(POD_modes):
            fig1 = plt.figure(figsize=figsize, layout='constrained')
            axs = fig1.subplots(nrows=n_row, ncols=n_col, sharex=True, sharey=True)
            if case.N_modes == 1:
                axs = [axs]
            else:
                axs = axs.ravel()
            norm = mpl.colors.Normalize(vmin=np.min(data[..., 0]), vmax=np.max(data[..., 0]))

            for kk, ax in zip(range(num_modes), axs):
                im0 = ax.pcolormesh(X1, X2, data[..., kk], cmap=mpl.colormaps[cmap], norm=norm, rasterized=True)

                if kk >= num_modes - n_col:
                    ax.set(xlabel='$y$')
                if kk % n_col == 0:
                    ax.set(ylabel='$x$')

                ax.set_title(f'mode {kk}', fontsize='xx-small')

            fig1.colorbar(mappable=im0, ax=axs, shrink=0.5, aspect=20)


            if num_modes < len(axs):
                for jj_del in np.arange(jj, num_modes, 1):
                    fig1.delaxes(axs[jj_del])
            if save:
                plt.savefig(f'POD_first_{num_modes}_modes_dim{jj}', dpi=300)
                

    @staticmethod
    def plot_time_coefficients(case, Phi=None, num_modes=None, plot_recurrence=False):
        """
        Plot the time evolution of POD coefficients.

        Args:
            - case: The POD instance.
            - Phi: POD time coefficients (optional).
            - num_modes: Number of modes to plot.
            - nrows: Number of rows in the subplot.
        """
        if Phi is None:
            Phi = case.Phi.copy()

        if num_modes is not None:
            Phi = Phi[:, :num_modes]
        else:
            num_modes = Phi.shape[-1]


        Nt = Phi.shape[0]


        # window = int(Nt // nrows)
        if num_modes < 10:
            window = Nt
        elif num_modes < 50:
            window = 200
        else:
            window = num_modes
                    
        nrows = max(int(Nt // window), 1)


        time_coeffs = [Phi[i0*window:(i0+1)*window] for i0 in range(nrows)]
        # time_windows = [times[i*window : min((i+1)*window, Nt)] for i in range(nrows)]


        fig1 = plt.figure(figsize=(10, 1.5*nrows), layout='tight')  #timeseries

        axs1 = fig1.subplots(nrows=nrows, ncols=1, sharey=False, sharex=False)
        norm = mpl.colors.Normalize(vmin=np.min(Phi), vmax=np.max(Phi))
        i0 = -1

        if nrows == 1:
            axs1 = [axs1]
        for ax, phi in zip(axs1, time_coeffs):
            i0 += 1
            extent = [i0*window, (i0+1)*window, 0, num_modes]
            im0 = ax.imshow(phi.T, cmap=mpl.cm.viridis, 
                            norm=norm, 
                aspect='auto',
                extent=extent,
                origin='lower')
            ax.set(xlim=(i0*window,(i0+1)*window-1))
            # POD.new_method()

        fig1.colorbar(mappable=im0, ax=axs1[0], shrink=0.75, orientation='vertical')

        if plot_recurrence:
            if num_modes < 5:
                ncols1 = 2
                nrows1 =num_modes // ncols1 + (num_modes % ncols1 > 0)
            else:
                nrows1 = int(np.ceil(num_modes / 10))
                ncols1 = num_modes // nrows1 + (num_modes % (nrows1 * 10) > 0)
            fig1.colorbar(mappable=im0, ax=axs1[0], shrink=0.75, orientation='vertical')


            # Get the recurrence plots for all the time series
            rp = RecurrencePlot(threshold='point', percentage=20)
            X_rp = rp.fit_transform(Phi.T)


            fig2 = plt.figure(figsize=(4 * ncols1, 3 * nrows1)) #recurrent plot
            gs = GridSpec(1, 1, figure=fig2)  # Define a single GridSpec slot
            grid = ImageGrid(fig2, gs[0, 0], nrows_ncols=(nrows1, ncols1), axes_pad=0.1, share_all=True)
            for i, xx in enumerate(X_rp):
                grid[i].imshow(xx, cmap='binary', origin='lower')

            grid[0].get_yaxis().set_ticks([])
            grid[0].get_xaxis().set_ticks([])
            
            if num_modes < len(grid):
                for jj_del in np.arange(i, num_modes, 1):
                    fig1.delaxes(grid[jj_del])

    @staticmethod
    def new_method():
        i0 += 100


    @staticmethod
    def plot_spectrum(case, max_mode=None):
        """
        Plot the spectrum of the POD decomposition.
        The energy of each POD mode is given by λ_i / 2, where λ_i = sigma_i^2

        Args:
            - case: The POD instance.
            - u: Optional velocity component.
            - v: Optional velocity component.
            - max_mode: Maximum number of modes to plot.

        """

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        Lambda = case.Sigma ** 2
        normalized_Lambda = Lambda / Lambda[0]

        axs[0].bar(np.arange(case.N_modes)+1, normalized_Lambda, color='C4')
        axs[0].set(xlabel='Mode number $j$', title='$\\lambda_j / \\lambda_0$',
                    xlim=[0, max(case.N_modes, 10)])
        axs[1].plot(np.arange(case.N_modes)+1, np.cumsum(Lambda) / sum(Lambda), color='C4',
                        label='$\\dfrac{\\sum_{j}{\,\\lambda_j}}{ \\sum_{k=0}^{' + f'{case.N_modes}' + '}\\lambda_k}$')

        if case._TKE is not None:
            energy = Lambda / 2 / case.Phi.shape[0]
            axs[1].plot(np.arange(case.N_modes)+1, np.cumsum(energy) / case._TKE,
                            dashes=[10, 5], color='k', label='$\\dfrac{\\sum_{j}{\,\\lambda_j}}{\mathrm{TKE}}$')

        axs[1].grid(visible=True, linestyle='--', alpha=0.5)
        axs[1].set(xlabel='Mode number $j$', title='Cumulative energy')
        axs[1].legend(ncol=1, bbox_to_anchor=[1, 1], loc='upper left')
        [ax.set(ylim=[0, 1.05], xlim=[-1, None]) for ax in axs]

        # Zoom insets
        if max_mode is not None and max_mode < case.N_modes:
            # Spectrum inset
            ax_zoom0 = fig.add_axes([0.3, 0.5, 0.15, 0.35])  # [left, bottom, width, height] in figure coordinates
            ax_zoom0.bar(np.arange(max_mode)+1, normalized_Lambda[:max_mode], color='C4')
            ax_zoom0.set_xlim([0, max_mode+1])
            ax_zoom0.set_ylim([0, None])

            # Cumulative energy inset
            ax_zoom1 = fig.add_axes([0.72, 0.2, 0.15, 0.35])
            ax_zoom1.grid(visible=True, linestyle='--', alpha=0.5)
            ax_zoom1.plot(np.arange(max_mode)+1, np.cumsum(Lambda[:max_mode]) / np.sum(Lambda), color='C4')
            if case._TKE is not None:
                ax_zoom1.plot(np.arange(max_mode)+1, np.cumsum(energy[:max_mode]) / case._TKE, dashes=[5, 2], color='k')
            ax_zoom1.set_xlim([0, max_mode+1])


    @staticmethod
    def plot_flows_rms(case,
                       datasets,
                       reconstructed_data=None,
                       display_dims=None,
                       display_RMS='all',
                       names=None,
                       norm_flow=None,
                       norm_rms=None,
                       cmap_flow='viridis',
                       cmap_rms='Reds',
                       save=False,
                       display_sensors=False):
        """
        Plot flow fields and RMS for the given case and datasets with consistent scaling.

        Args:
            - case: The POD instance.
            - datasets: List of datasets to visualize.
            - reconstructed_data: POD reconstructed dataset (optional).
            - display_dims: Indices of dimensions to display.
            - display_RMS: Whether to plot RMS error ('all' or specific indices).
            - names: Labels for the datasets.
            - norm_flow: Normalization for flow data.
            - norm_rms: Normalization for RMS plots.
            - cmap_flow: Colormap for flow visualization.
            - cmap_rms: Colormap for RMS visualization.
            - save: If True, saves the figure.
            - display_sensors: If True, overlays sensor locations.
        """

        def prepare_data(_data, _target_shape):
            """Prepare dataset by handling NaNs and reshaping if necessary."""
            _data[np.isnan(_data)] = 0
            if _data.ndim > 3:
                _data = _data[..., -1]

            if _data.shape != _target_shape:
                # return case.original_to_domain_of_interest(_data)
                raise ValueError(f'Provided data is not the correct shape, maybe it was cropped? data.shape={_data.shape}, but _target_shape-{_target_shape}')
            else:
                return _data

        def compute_global_norm(_prepared_datasets, _datasets_titles, _norm_flow, _norm_rms):
            """Compute global min and max for consistent RMS and flow scaling."""
            if _norm_rms is None:
                _rms_data = np.array([_d for _d, _t in zip(_datasets, _datasets_titles) if 'RMS' in _t])
                _norm_rms = mpl.colors.Normalize(vmin=0., vmax=_rms_data.max())

            if _norm_flow is None:
                _norm_flow = []
                for r in range(nrows):
                    _row = np.array([_yy[r] for _yy in _prepared_datasets])
                    _norm_flow.append(mpl.colors.Normalize(vmin=np.min(_row), vmax=np.max(_row)))
            return _norm_flow, _norm_rms

        datasets = datasets if isinstance(datasets, list) else [datasets]

        if display_RMS == 'all':
            display_RMS = list(np.arange(len(datasets)))
        else:
            display_RMS = display_RMS if isinstance(display_RMS, list) else [display_RMS]

        if reconstructed_data is None:
            reconstructed_data = case.reconstruct(Phi=case.Phi[-1], reshape=True)

        if reconstructed_data.ndim > 3:
            reconstructed_data = reconstructed_data[..., -1].copy()

        display_dims = display_dims or np.arange(reconstructed_data.shape[0])
        if isinstance(display_dims, float):
            display_dims = [display_dims]

        nrows = len(display_dims)
        X1, X2 = case.domain_mesh()

        if display_sensors and hasattr(case, 'sensor_locations'):
            X1_r, X2_r = X1.ravel(), X2.ravel()
            idx = case.sensor_locations[case.sensor_locations < len(X1.ravel())]
            dom = case.domain_of_measurement
            display_dom = case.domain != case.domain_of_measurement and dom is not None
        else:
            display_sensors = False

        # Prepare data titles and cmaps to plot
        _datasets = [reconstructed_data]
        _titles = [f'POD reconstruct {case.N_modes} modes']
        _cmaps = [cmap_flow]

        for ii, dataset, name in zip(range(len(datasets)), datasets, names or [None] * len(datasets)):
            dataset = prepare_data(dataset, reconstructed_data.shape)
            _datasets.append(dataset)
            _cmaps.append(cmap_flow)
            _titles.append(name or f'dataset {ii}')
            if ii in display_RMS:
                RMS = POD.compute_RMS(reconstructed_data, dataset)
                _datasets.append(RMS)
                _titles.append(f'RMS({name or f"dataset {ii}"})')
                _cmaps.append(cmap_rms)

        norm_flow, norm_rms = compute_global_norm(_datasets, _titles, norm_flow, norm_rms)

        ncols = len(_datasets)

        figsize = get_figsize_based_on_domain(case.domain_of_interest, total_width=10, total_subplots=ncols*nrows)[0]

        sub_figs = plt.figure(figsize=figsize,
                              layout='constrained').subfigures(nrows=nrows, ncols=1)

        for jj, (fig, norm_f) in enumerate(zip(sub_figs if nrows > 1 else [sub_figs], norm_flow)):
            axs = fig.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True)
            im_rms, im_flow = None, None

            for ax, dataset, title, cmap in zip(axs, _datasets, _titles, _cmaps):

                # Assign the plot to the appropriate variable
                if 'RMS' in title:
                    im_rms = ax.pcolormesh(X1, X2, dataset[jj], cmap=cmap, norm=norm_rms, rasterized=True)
                else:
                    im_flow = ax.pcolormesh(X1, X2, dataset[jj], cmap=cmap, norm=norm_f, rasterized=True)

                if display_sensors:
                    ax.scatter(X1_r[idx], X2_r[idx], c=np.arange(len(idx)), cmap='YlOrRd', edgecolors='k', s=12.25,
                               lw=.5)
                    if display_dom:
                        ax.add_patch(mpl.patches.Rectangle((dom[0], dom[2]), dom[1] - dom[0], dom[3] - dom[2],
                                                           edgecolor='orange', facecolor='none', lw=3, ls='--'))

                ax.set(ylabel='$x$', title=title if jj == 0 else None, xlabel='$y$' if jj > 0 else None)

            # Add colorbars for flow and RMS plots
            [fig.colorbar(im, ax=axs, shrink=0.5) for im in [im_rms, im_flow] if im]

        if save:
            plt.savefig(f'{case.figs_folder}{case.name}_flows_rms.png', dpi=300)

        return sub_figs


