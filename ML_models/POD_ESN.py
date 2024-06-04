import numpy as np
import os as os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf

from modulo_vki.modulo import ModuloVKI as Modulo

from ML_models.EchoStateNetwork import EchoStateNetwork
from essentials.physical_models import Model
from essentials.Util import interpolate

folder = './POD-ESN_figs/'

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=14, serif='Times New Roman')
plt.rc('mathtext', rm='times', bf='times:bold')
plt.rc('legend', framealpha=0, edgecolor=(0, 0, 1, 0.1))


class POD_ESN(EchoStateNetwork, Model):
    """ Performs POD for a data matrix and trains an ESN to forecast the POD
        temporal coefficients.

        D(x,t) = Σ_j σ_j φ_j(x) ψ_j(t)    for  j = 0, ..., N_modes-1


        POD properties:
            - Psi: temporal basis [N, N_modes], with N = Ndim x Nx x Ny
            - Phi: spatial basis [Nt, N_modes]
            - Sigma: POD Sigmas [N_modes, ],  note: Lambdas can be computed as: Sigma = np.sqrt(Lambda)
    """

    name: str = 'POD-ESN'
    Nq = 10
    domain = [-2, 2, 12, 0]
    field_labels = ['$u_x$', '$u_y$', '$p$']

    N_modes = 20
    method = 'snapshot'
    eig_solver = 'svd_sklearn_randomized'
    POD_on_perturbations = True
    Q_mean = None
    dt = 0.01
    plot_decomposition = True

    # Echo state network parameters
    upsample = 1
    N_wash = 10
    rho_range = (.2, 1.05)
    sigma_in_range = (-5, -1)
    tikh_range = [1e-8, 1e-10, 1e-12, 1e-16]
    noise = 1e-3

    POD_attrs = ['Phi', 'Psi', 'Sigma', 'grid_shape', 'Q_mean']

    sensor_locations = None
    update_reservoir = True

    def __init__(self, X=None, psi0=None, **model_dict):
        """
        Arguments:

        - X: Snapshot data tensor, the last dimension must be time.
             If X=None, the POD_attrs must be provided in model_dict.
        - psi0: initial condition for the ESN state and reservoir
        """

        for key, val in model_dict.items():
            if hasattr(self, key):
                setattr(self, key, val)

        self.state_labels = [f'$\\phi_{j}' for j in np.arange(self.N_modes)]

        # __________________________ Init POD ___________________________ #

        if X is None:
            try:
                [setattr(self, attr, model_dict[attr]) for attr in self.POD_attrs]
            except KeyError:
                raise KeyError('POD attributes not provided.')
            self.N_modes = self.Phi.shape[-1]

        elif isinstance(X, np.ndarray):
            Q = X.copy()
            Q[np.isnan(Q)] = 0.
            self.grid_shape = Q.shape[:-1]
            Q = Q.reshape((-1, Q.shape[-1]))
            if self.POD_on_perturbations:
                self.Q_mean = np.mean(Q, axis=-1, keepdims=True)
                Q -= self.Q_mean

            # Perform POD decomposition using modulo_vki [https://github.com/mendezVKI/MODULO/]
            m = Modulo(data=Q, eig_solver=self.eig_solver, n_Modes=self.N_modes)  # Initialize modulo instance
            if self.method == 'snapshot':
                self.Psi, self.Phi, self.Sigma = m.compute_POD_K()
            else:
                raise NotImplementedError('Method {} not implemented'.format(self.method))
        else:
            raise TypeError('X must be either None or a np.ndarray.')

        # __________________________ Init ESN ___________________________ #
        # Initialize ESN to model the POD coefficients
        Nt = self.Phi.shape[0]
        t_total = Nt * self.dt

        EchoStateNetwork.__init__(self,
                                  y=self.Phi[0],
                                  t_train=t_total * .7,
                                  t_val=t_total * .3,
                                  t_test=0.,
                                  figs_folder=folder,
                                  **model_dict
                                  )

        self.train_network()

        if self.plot_decomposition:
            print('========= POD-ESN model complete. Plotting POD and ESN test =========')
            os.makedirs(folder, exist_ok=True)
            self.plot_POD_modes(num_modes=self.N_modes, cmap='viridis')
            self.plot_time_coefficients()
            self.plot_spectrum()
            self.plot_reconstruct(original_data=X)

            filename = f'POD_{self.N_modes}-ESN_{self.N_units}.pdf'
            pdf_file = plt_pdf.PdfPages(folder + filename)

            for ii in plt.get_fignums():
                fig = plt.figure(ii)
                pdf_file.savefig(fig)
                plt.close(fig)
            pdf_file.close()  # Close results pdf

        # __________________________ Init Model ___________________________ #

        if psi0 is None:
            u, r = self.get_reservoir_state()
            if self.update_reservoir:
                psi0 = np.concatenate([u, r], axis=0)
            else:
                psi0 = u.copy()

        Model.__init__(self, psi0=psi0, **model_dict)

        # __________________________ Select sensors ___________________________ #
        if self.sensor_locations is None:
            self.sensor_locations = POD_ESN.define_sensors(self.Nq, self.domain, self.grid_shape, rng=self.rng)

        self.Nq = len(self.sensor_locations)


    def reset_history(self,  psi, t):
        self.hist_t = np.array([t])
        if self.hist_t.ndim > 1:
            self.hist_t = self.hist_t.squeeze()

        self.hist = np.array([psi])
        if self.hist.ndim > 3:
            self.hist = self.hist.squeeze(axis=0)

        u0 = self.hist[-1, :self.N_modes]
        if self.update_reservoir:
            r0 = self.hist[-1, self.N_modes:]
        else:
            r0 = np.tile(self.r, reps=(1, self.m))

        self.reset_state(u=u0, r=r0)

    def restore_shape(self, data=None):
        """ Reshape data into the original dimension of the data."""
        if data is None:
            data = self.Psi
        if data.ndim > 1:
            if data.shape[:len(self.grid_shape)] == self.grid_shape:
                return data
            else:
                return np.reshape(data, newshape=(*self.grid_shape, data.shape[1]))
        else:
            return np.reshape(data, newshape=(*self.grid_shape,))

    def train_network(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key):
                getattr(self, key, val)
            else:
                raise ValueError(f'Attribute {key} not recognized.')
        # Train network
        self.train(train_data=self.Phi)

    def get_observables(self, Nt=1, Phi=None, **kwargs):
        Psi = self.Psi[self.sensor_locations]
        Q_mean = self.Q_mean[self.sensor_locations]
        if Phi is None:
            if Nt == 1:
                Phi = self.get_current_state[:self.N_modes]
            else:
                Phi = self.hist[-Nt:, :self.N_modes]

        return self.reconstruct(Phi=Phi, Psi=Psi, Q_mean=Q_mean, reshape=False)

    def reconstruct(self, Phi=None, Psi=None, Q_mean=None, reshape=True):
        """Reconstruct data with the selected number of POD modes.

        Arguments:
        Returns:
            Reconstructed flow field
        """

        if Phi is None:
            Phi = self.Phi[[-1]].T
        if Psi is None:
            Psi = self.Psi
        if Q_mean is None:
            Q_mean = self.Q_mean

        Phi = np.atleast_2d(Phi)
        if Phi.shape[-2] != self.N_modes:
            if Phi.ndim > 2:
                shape_array = np.array(Phi.shape)
                i_mode = np.argwhere(shape_array == self.N_modes)[0][0]
                i_m = np.argwhere(shape_array == self.m)[0][0]
                i_Nt = [ii for ii in np.arange(Phi.ndim) if ii not in [i_m, i_mode]][0]
                Phi = Phi.transpose(i_Nt, i_mode, i_m)
            else:
                Phi = Phi.transpose()

        assert Phi.shape[-2] == self.N_modes

        new_shape = list(np.ones(Phi.ndim, dtype=int))
        new_shape[-2] = self.N_modes

        Sigma = np.reshape(self.Sigma, newshape=new_shape)

        new_shape = list(np.ones(Phi.ndim, dtype=int))
        new_shape[0] = Q_mean.shape[0]
        Q_mean = np.reshape(Q_mean, newshape=new_shape)

        Q_POD = np.dot(Psi, Sigma * Phi)
        reconstructed_data = Q_mean + Q_POD

        if reshape:
            reconstructed_data = self.restore_shape(data=reconstructed_data)

        return reconstructed_data

    @property
    def obs_labels(self):
        return ["$Q^\\mathrm{POD}" + f"_{j}$" for j in np.arange(self.Nq)]

    def time_integrate(self, Nt=10, averaged=False, alpha=None):

        if not self.trained:
            raise NotImplementedError('ESN model not trained')

        interp_flag = False
        Nt = Nt // self.upsample
        if Nt % self.upsample:
            Nt += 1
            interp_flag = True

        t = np.round(self.get_current_time + np.arange(0, Nt + 1) * self.dt, self.precision_t)

        # If the time is before the washout initialization, return zeros
        u, r = self.closedLoop(Nt)

        # Interpolate if the upsample is not multiple of dt or if upsample > 1
        if self.upsample > 1 and interp_flag:
            t_b = np.round(self.get_current_time + np.arange(0, Nt + 1) * self.dt_ESN, self.precision_t)
            u = interpolate(t_b, u, t_eval=t)
            r = interpolate(t_b, r, t_eval=t)

        # update ESN physical and reservoir states, and store the history if requested
        self.reset_state(u=self.outputs_to_inputs(full_state=u[-1]), r=r[-1])

        if self.update_reservoir:
            Af = np.concatenate((u, r), axis=1)
        else:
            Af = u

        return Af[1:], t[1:]


    @staticmethod
    def compute_MSE(POD_data, original_data):
        original_data[np.isnan(original_data)] = 0
        return np.mean((original_data - POD_data) ** 2)

    @staticmethod
    def compute_RMS(POD_data, original_data):
        original_data[np.isnan(original_data)] = 0
        return np.sqrt((original_data - POD_data) ** 2)

    @staticmethod
    def domain_mesh(domain, grid_shape, down_sample=None, ravel=False):
        x1 = np.linspace(*domain[:2], num=grid_shape[1])
        x2 = np.linspace(*domain[2:], num=grid_shape[2])
        X1, X2 = np.meshgrid(x1, x2)
        grid_idx = np.arange(0, len(X1.ravel())).reshape(X1.shape)

        if down_sample is not None:
            if isinstance(down_sample, int):
                step_x1 = down_sample
                step_x2 = down_sample
            else:
                step_x1, step_x2 = down_sample
            X1 = X1[::step_x2, ::step_x1]
            X2 = X2[::step_x2, ::step_x1]
            grid_idx = grid_idx[::step_x2, ::step_x1]

        if ravel:
            return X1.ravel(), X2.ravel(), grid_idx.ravel()
        else:
            return X1, X2, grid_idx

    @staticmethod
    def define_sensors(Nq, domain, grid_shape, down_sample=(10, 20), rng=None):

        X1_d, X2_d, grid_idx_d = POD_ESN.domain_mesh(domain, grid_shape,
                                                     down_sample=down_sample, ravel=True)

        idx = grid_idx_d[(abs(X1_d) < 1.5) & (abs(X2_d - 6) < 2)]

        if rng is None:
            rng = np.random.default_rng()

        idx = np.sort(rng.choice(idx, size=Nq, replace=False), axis=None)

        n_grid = grid_shape[1] * grid_shape[2]
        return np.array([idx + (n_grid * ii) for ii in range(grid_shape[0])]).reshape((-1,))

    # ========================================== PLOTS =======================================================

    def plot_POD_modes(self, Psi=None, num_modes=2, save=False, cmap='viridis', dim=None):

        if Psi is None:
            Psi = self.Psi.copy()

        if Psi.ndim == 2:
            Psi = self.restore_shape(Psi)

        if dim is not None:
            if isinstance(dim, int):
                dim = [dim]
        else:
            dim = np.arange(Psi.shape[0])

        POD_modes = [Psi[dim_i] for dim_i in dim]

        X1, X2 = POD_ESN.domain_mesh(self.domain, self.grid_shape)[:-1]

        n_col = min(num_modes, 5)
        n_row = int((num_modes + 1) // n_col)

        for jj, data in enumerate(POD_modes):

            fig1 = plt.figure(figsize=(1.5 * n_col, 3 * n_row), layout='constrained')
            axs = fig1.subplots(nrows=n_row, ncols=n_col, sharex=True, sharey=True)
            axs = axs.ravel()
            norm = mpl.colors.Normalize(vmin=np.min(data[..., 0]), vmax=np.max(data[..., 0]))

            for kk, ax in zip(range(num_modes), axs):
                im0 = ax.contourf(X1, X2, data[..., kk].T, 100,
                                  cmap=mpl.colormaps[cmap], norm=norm)
                if kk >= num_modes - n_col:
                    ax.set(xlabel='$y$')
                if kk % n_col == 0:
                    ax.set(ylabel='$x$')
                if (kk + 1) % n_col == 0:
                    fig1.colorbar(mappable=im0, ax=ax)

                ax.set(title=f'mode {kk}')

            if num_modes < len(axs):
                for jj_del in np.arange(jj, num_modes, 1):
                    fig1.delaxes(axs[jj_del])
            if save:
                plt.savefig(f'POD_first_{num_modes}_modes_dim{jj}', dpi=300)

    def plot_time_coefficients(self, Phi=None, num_modes=None):
        if Phi is None:
            Phi = self.Phi.copy()

        if num_modes is not None:
            Phi = Phi[:, :num_modes]

        Phi_dot = np.dot(Phi, Phi.T)

        fig2 = plt.figure(figsize=(10, 6), layout='constrained')
        axs = fig2.subplots(nrows=3, ncols=2, sharey=False, sharex=False, width_ratios=(4, 1))

        norm = mpl.colors.Normalize(vmin=np.min(Phi), vmax=np.max(Phi))
        i0 = 0
        for row in axs:
            im0 = row[0].imshow(Phi.T, cmap=mpl.cm.viridis, norm=norm)
            im1 = row[1].imshow(Phi_dot, cmap=mpl.cm.Greys)
            row[0].set(xlim=(i0, i0 + 100))
            row[1].set(xlim=(i0, i0 + 100), ylim=(i0, i0 + 100))
            i0 += 100

        fig2.colorbar(mappable=im0, ax=axs[-1, 0], shrink=0.5, orientation='horizontal')
        fig2.colorbar(mappable=im1, ax=axs[-1, 1], shrink=0.5, orientation='horizontal')

    def plot_reconstruct(self, original_data, reconstructed_data=None, clean_data=None, dim=None, titles=None, save=False):
        if reconstructed_data is None:
            reconstructed_data = self.reconstruct(Phi=self.Phi[-1], reshape=True)
        if dim is None:
            dim = np.arange(reconstructed_data.shape[0])
        elif isinstance(dim, float):
            dim = [dim]

        if original_data.ndim > 3:
            original_data = original_data[..., -1].copy()
        if reconstructed_data.ndim > 3:
            reconstructed_data = reconstructed_data[..., -1].copy()

        original_data[np.isnan(original_data)] = 0.

        nrows = len(dim)

        X1, X2 = POD_ESN.domain_mesh(self.domain, self.grid_shape)[:-1]

        if clean_data is not None:
            clean_data[np.isnan(clean_data)] = 0.
            RMS = POD_ESN.compute_RMS(reconstructed_data, clean_data)
            MSE = POD_ESN.compute_MSE(reconstructed_data, clean_data)
            datasets = [clean_data, original_data, reconstructed_data, RMS]
            if titles is None:
                titles = ['Clean', 'Noisy', 'POD on noisy', 'RMS(clean)']
        else:
            RMS = POD_ESN.compute_RMS(reconstructed_data, original_data)
            MSE = POD_ESN.compute_MSE(reconstructed_data, original_data)
            datasets = [original_data, reconstructed_data, RMS]
            if titles is None:
                titles = ['Original', 'POD', 'RMS']

        cmaps = ['viridis'] * len(datasets)
        cmaps[-1] = 'Reds'

        ncols = len(datasets)

        fig1, grid = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2.5, nrows*4),
                                  sharey=True, sharex=True, layout='constrained')

        for kk, row, data, cmap, tt in zip(range(ncols), grid.T, datasets, cmaps, titles):
            ii_ax = kk
            for jj, ax_ in zip(dim, row):
                im0 = ax_.contourf(X1, X2, data[jj].T, 100, cmap=mpl.colormaps[cmap])
                plt.colorbar(im0, ax=ax_, shrink=0.5)
                ax_.set(ylabel='$x$', title=tt, xlabel='$y$')
                ii_ax += 3

        fields = [self.field_labels[kk] for kk in dim]
        fields = ', '.join(fields)

        fig1.suptitle(f'Instantaneous flow fields {fields} \n '
                      f'POD reconstruction with {self.N_modes} modes. MSE = {MSE:.6}')
        if save:
            plt.savefig(f'POD_reconstruction_{self.N_modes}', dpi=300)

    def plot_sensors(self, background_data=None, fluctuations=False):
        if background_data is None:
            background_data = self.reconstruct(Q_mean=self.Q_mean*0., reshape=True)
            fluctuations = True
        if background_data.ndim > 3:
            background_data = background_data[..., -1].copy()

        XX1, XX2 = POD_ESN.domain_mesh(self.domain, self.grid_shape)[:-1]
        X1, X2 = XX1.ravel(), XX2.ravel()

        idx = self.sensor_locations[self.sensor_locations < len(X1)]

        X1, X2 = X1[idx], X2[idx]

        title = self.field_labels.copy()
        if fluctuations:
            title = ['$\\tilde{' + ttl[1] + '}' + ttl[2:-1] + '$' for ttl in title]

        fig1, axs = plt.subplots(figsize=(7, 4), nrows=1,ncols=background_data.shape[0],
                                 sharex=True, sharey=True, layout='tight')
        cmap = mpl.colormaps['viridis']
        axs[0].set(ylabel='$x$', xlabel='$y$')
        for jj, ax_ in enumerate(axs):
            im0 = ax_.contourf(XX1, XX2, background_data[jj].T, 100, cmap=cmap)
            ax_.set(xlabel='$y$', title=title[jj])
            plt.colorbar(im0, ax=ax_, shrink=0.5)
            ax_.plot(X1, X2, 'o', markersize=4, color='r')
            for kk, x1, x2 in zip(range(len(idx)), X1, X2):
                ax_.annotate(kk, (x1, x2))

    def plot_spectrum(self):
        plt.figure()
        plt.bar(np.arange(self.N_modes), self.Sigma, color='C4')
        plt.gca().set(xlabel='Mode number $j$', ylabel='$\\sigma_j$')

