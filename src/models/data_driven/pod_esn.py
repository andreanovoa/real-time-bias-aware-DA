
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from romda.utils import add_pdf_page, plt_pdf

from .autoencoders import POD
from .esn import ESN_model, phi_to_esn_layout


def synthetic_field(N_t=200, Nx=16, Ny=8, seed=0):
    """Two travelling-wave snapshot fields, ``(2, N_t, Nx, Ny)`` — the shape
    convention `Projector.fit` expects — low-rank enough that a handful of latent
    modes reconstructs them almost exactly. For demos and self-checks (same
    construction as ``tests/test_pod_spod.py``); real cases load actual snapshots."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, N_t)
    x = np.linspace(0, 2 * np.pi, Nx)
    y = np.linspace(0, np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = (np.sin(X)[None] * np.cos(t)[:, None, None]
         + 0.5 * np.sin(2 * X + Y)[None] * np.sin(2 * t)[:, None, None])
    v = np.cos(X + Y)[None] * np.sin(t)[:, None, None]
    u = u + 0.02 * rng.standard_normal(u.shape)  # keep the data full-rank
    v = v + 0.02 * rng.standard_normal(v.shape)
    return np.array([u, v])  # (2, N_t, Nx, Ny)


class POD_ESN(ESN_model, POD):
    r"""POD-projected echo state network: a [`POD`][romda.models.data_driven.autoencoders.POD]
    decomposition reduces the (spatial) field to a handful of temporal
    coefficients, and an [`ESN_model`][romda.models.data_driven.esn.ESN_model] is
    trained to forecast those coefficients in time.

    Following the [`POD`][romda.models.data_driven.autoencoders.POD] convention (see its
    docstring), writing $\mathbf{Q} = \mathbf{X} - \bar{\mathbf{Q}}$ for the
    zero-mean data, the field is approximated as

    $$
    \mathbf{X} \approx \boldsymbol{\Psi}\boldsymbol{\Phi} + \bar{\mathbf{Q}},
    $$

    with $\boldsymbol{\Psi}$ (`Psi`) the orthonormal *spatial* modes and
    $\boldsymbol{\Phi}$ (`Phi`) the *temporal* coefficients
    ($\boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q}$, already scaled by the
    singular values `Sigma`). `POD_ESN.__init__` runs the POD decomposition first,
    then trains the ESN on $\boldsymbol{\Phi}$ (transposed to the
    ``(L, Nt, N_modes)`` layout `ESN_model` expects) to forecast the temporal
    coefficients forward in time; `get_observables` maps the ESN's closed-loop
    `Phi` forecast back to physical sensor readings (`decode`) or returns the modal
    coefficients directly if `measure_modes`.
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
        """Run the POD decomposition, train the ESN on the resulting temporal
        coefficients, and select sensor locations for observation.

        Parameters
        ----------
        data : np.ndarray
            Data for the POD decomposition and ESN training, shape
            ``(Nu, Nt, Nx, Ny)`` or ``(Nt, Ndim*Nx*Ny)``.
        dt : float
            Time step of `data`; forwarded to `ESN_model.__init__`.
        plot_case : bool
            Whether to plot the POD modes/spectrum/reconstruction (and the ESN
            training process, if `train_ESN`). Default False.
        pdf_file : str, optional
            If given (and `plot_case`), save the resulting figures to
            ``f'{pdf_file}.pdf'``.
        skip_sensor_placement : bool
            If True, skip sensor placement and observe the POD modes directly
            (equivalent to `measure_modes`). Default False.
        train_ESN : bool
            Whether to train the ESN on the POD temporal coefficients. Default True.
        domain_of_measurement : list, optional
            Sub-domain ``[x0, x1, y0, y1]`` to restrict candidate sensor locations
            to, if `sensor_locations` is not already given. Defaults to `domain`.
        down_sample_measurement : int or (int, int), optional
            Grid down-sampling factor(s) applied to the measurement domain before
            sensor selection.
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
            ESN_model.__init__(self,
                               data=phi_to_esn_layout(self.Phi.copy()),
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
        r"""list of str: LaTeX labels for the observed quantities -- POD coefficients
        $\Phi_1, \dots, \Phi_{N_\mathrm{modes}}$ if `measure_modes`, otherwise the
        sensor readings' $u_x$/$u_y$ components."""
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
        r"""list of str: LaTeX labels for the state vector, $\Phi_1, \dots,
        \Phi_{N_\mathrm{modes}}$ (POD temporal coefficients, if `update_state`)
        followed by $r_1, \dots, r_{N_\mathrm{units}}$ (reservoir units, if
        `update_reservoir`)."""
        labels = []
        if self.update_state:
            labels +=[f'$\\Phi_{j+1}$' for j in np.arange(self.N_modes)]
        if self.update_reservoir:
            labels += [f'$r_{j+1}$' for j in np.arange(self.N_units)]
        return labels

    @property
    def N_sensors(self):
        """int: Number of sensor locations per velocity component ($u_x$ or $u_y$),
        i.e. ``Nq // 2`` (each location contributes 2 observables); 0 if
        `measure_modes`."""
        if self.measure_modes:
            return 0
        else:
            return int(self.Nq / 2)

    @property
    def sensor_rows(self):
        """np.ndarray or None: Rows of `Psi`/`Q_mean` corresponding to
        `sensor_locations` (cached after first access; invalidated by
        `select_sensors`). `sensor_locations` are raw-grid indices
        (``var * Nx * Ny + g``), whereas the POD basis rows follow the masked flat
        ordering -- this property maps between the two, via
        `grid_index_to_flat_rows`. None if `sensor_locations` is None.
        """
        if self.sensor_locations is None:
            return None
        if not hasattr(self, '_sensor_rows') or self._sensor_rows is None:
            self._sensor_rows = self.grid_index_to_flat_rows(self.sensor_locations)
        return self._sensor_rows

    def get_POD_coefficients(self, Nt=1):
        r"""Read the forecasted POD temporal coefficients off the state history.

        Parameters
        ----------
        Nt : int
            Number of trailing history steps to return. Default 1.

        Returns
        -------
        np.ndarray
            $\boldsymbol{\Phi}$-block of `hist`, shape ``(N_modes, m)`` if
            ``Nt == 1`` else ``(Nt, N_modes, m)``.
        """
        if Nt == 1:
            Phi = self.hist[-1, :self.N_modes]
        else:
            Phi = self.hist[-Nt:, :self.N_modes]
        return Phi

    def get_observables(self, Nt=1, Phi=None, **kwargs):
        """Map the (forecasted) POD coefficients to observables: the coefficients
        themselves if `measure_modes`, otherwise physical-space sensor readings
        obtained via `decode` at `sensor_rows`.

        Parameters
        ----------
        Nt : int
            Number of trailing history steps to return. Default 1.
        Phi : np.ndarray, optional
            POD coefficients to decode, shape ``(N_modes, Nt, m)``. Defaults to
            `get_POD_coefficients(Nt)`.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Observables, shape ``(Nq, m)`` (or ``(Nt, Nq, m)`` for the reshaped case).
        """
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
        """Optionally rerun the POD decomposition and/or reset (retrain) the ESN.

        Parameters
        ----------
        reset_POD : bool
            If True, rerun the POD decomposition (`rerun_POD_decomposition`) and
            force `reset_ESN` to True (since the modes it forecasts change).
            Default False.
        reset_ESN : bool
            If True, reset the `EchoStateNetwork`/`Model` state via `reset_ESN`
            (`ESN_model.reset_ESN`). Default False.
        Phi0 : np.ndarray, optional
            Passed to `reset_ESN` as ``psi0`` (defaults to `Phi[0]`); note
            `reset_ESN` itself recomputes ``psi0`` from the freshly trained network
            before resetting the model, so this value is not the final initial state.
        **kwargs
            Forwarded to `rerun_POD_decomposition` and/or `reset_ESN`.

        Returns
        -------
        None
        """
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
        """(Re)configure how the model is observed: either the raw POD coefficients
        (`measure_modes`) or physical-space point sensors placed by `define_sensors`.

        Parameters
        ----------
        measure_modes : bool
            If True, observe the POD coefficients directly (`Nq = N_modes`, no
            sensors). Default False.
        domain_of_measurement : list, optional
            Sub-domain ``[x0, x1, y0, y1]`` to restrict candidate sensor locations
            to. Defaults to `domain`.
        down_sample_measurement : int or (int, int), optional
            Grid down-sampling factor(s) for the measurement domain.
        N_sensors : int, optional
            Number of sensor locations to place. Defaults to `N_sensors`.
        qr_selection : bool
            Whether to use QR-pivoting sensor placement (`define_sensors`).
            Default False.

        Returns
        -------
        None
            Sets `measure_modes`, `sensor_locations` and `Nq` in place.
        """
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
        """list: Sub-domain ``[x0, x1, y0, y1]`` sensors may be placed in. Defaults
        to the full `domain` if never set."""
        if not hasattr(self, '_domain_of_measurement'):
            self._domain_of_measurement = self.domain
        return self._domain_of_measurement

    @domain_of_measurement.setter
    def domain_of_measurement(self, dom):
        """Set `domain_of_measurement`."""
        self._domain_of_measurement = dom #type: list

    @property
    def down_sample_measurement(self):
        """list of int or None: Grid down-sampling factors ``[step_x, step_y]``
        applied to the measurement grid before sensor placement. None (no
        down-sampling) if never set."""
        if not hasattr(self, '_down_sample_measurement'):
            self.down_sample_measurement = None
        return self._down_sample_measurement


    @down_sample_measurement.setter
    def down_sample_measurement(self, dsm):
        """Set `down_sample_measurement`, broadcasting a single int to both axes.

        Raises
        ------
        ValueError
            If `dsm` is not None, an int, or a pair of ints.
        """
        if dsm is not None:
            if isinstance(dsm, int):
                dsm = [dsm, dsm]
            elif isinstance(dsm, (list, tuple)) and len(dsm) == 2 and all(isinstance(x, int) for x in dsm):
                dsm = list(dsm)
            else:
                raise ValueError(f'down_sample_measurement must be an int or a pair of ints, got {dsm!r}')

        self._down_sample_measurement = dsm

    @property
    def grid_of_measurement(self):
        """np.ndarray: Flat-grid indices (fluid cells only, all velocity
        components) eligible for sensor placement, i.e. `domain_of_measurement`
        down-sampled by `down_sample_measurement` and intersected with
        `fluid_mask_flat`.

        Raises
        ------
        ValueError
            If `domain_of_measurement` does not overlap the model `domain`.
        """

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



    def define_sensors(self, N_sensors=None, plot=False):
        """Choose sensor grid locations within `grid_of_measurement`.

        If `qr_selection`, uses column-pivoted QR on the (physical-grid) spatial
        modes `Psi` restricted to the candidate locations, so the chosen sensors
        best condition the mode-reconstruction problem (a greedy, deterministic
        sensor-placement heuristic); otherwise picks `N_sensors` random locations
        (`rng`).

        Parameters
        ----------
        N_sensors : int, optional
            Number of sensor locations to place. Defaults to `N_sensors`.
        plot : bool, optional
            Show the debug scatter of grid/measurement domain/sensors (blocks in
            interactive backends). Default False.

        Returns
        -------
        np.ndarray
            Flat-grid sensor indices, one block per velocity component, shape
            ``(grid_shape[0] * N_sensors,)``.
        """
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


        if plot:
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
        """Plot the POD modes, temporal coefficients, spectrum and (if `datasets`
        is given) flow/reconstruction/error fields with sensor locations overlaid.

        Parameters
        ----------
        case : POD_ESN
            Instance to plot.
        datasets : dict, optional
            Named fields to pass to `romda.plotting.pod.plot_flows_rms` (e.g.
            ``{'Input': ..., 'Reconstruction': ..., 'Error': ...}``).
        num_modes : int, optional
            Number of POD modes to show. Defaults to `case.N_modes`.

        Returns
        -------
        None
        """
        from romda.plotting.pod import plot_flows_rms, plot_modes, plot_spectrum, plot_time_coefficients

        if num_modes is None:
            num_modes = case.N_modes

        plot_modes(case=case, num_modes=num_modes, cmap='viridis', n_col=2)
        plot_time_coefficients(case=case, num_modes=num_modes)
        plot_spectrum(case=case, max_mode=num_modes)


        if datasets is not None:
            display_sensors = case.sensor_locations is not None
            plot_flows_rms(case=case, datasets=datasets, display_sensors=display_sensors)
