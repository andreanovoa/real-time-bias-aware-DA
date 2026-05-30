"""
autoencoders.py
===============

Dimensionality-reduction building blocks for data-driven ROMs.

All projectors — linear or nonlinear — share the same sklearn-style API:

    p.fit(Q)         -- learn the representation from data  Q (N_x, N_t)
    p.encode(Q)      -- Q (N_x, N_t) --> Z (N_latent, N_t)
    p.decode(Z)      -- Z (N_latent, N_t) --> Q_hat (N_x, N_t)
    p.reconstruct(Q) -- full round-trip
    p.score(Q)       -- mean squared reconstruction error
    p.N_latent       -- size of the latent (bottleneck) space

Class hierarchy
---------------

    Projector (ABC)               shared interface + N_latent + reconstruct/score/copy
    ├── POD(Projector)            Proper Orthogonal Decomposition (linear)
    │     N_latent == N_modes retained
    │     Sigma, Psi, Phi         decomposition results
    │     truncate / restore_shape / plot_spectrum / …
    │
    ├── SPOD(POD)                 Spectral POD (Sieber et al. JFM 2016)
    │     inherits all POD helpers; only _decompose is overridden
    │     to apply the Toeplitz low-pass filter before the eigensolve
    │
    ├── AE(Projector)             Fully-connected Autoencoder  [stub]
    └── CAE(Projector)            Convolutional Autoencoder    [stub]

These are pure dimensionality-reduction tools — they have no temporal
forecaster.  Combine with an ESN or LSTM in models/data_driven/ to build
a complete ROM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from abc import abstractmethod
from typing import Optional, Union
from scipy.signal import get_window
from scipy.special import gammaincinv

from .pod_spod import snapshot_pod, snapshot_pod_randomized, spod_sieber


from utils import (save_figs_to_pdf,
                    get_figsize_based_on_domain,
                    crop_data_to_domain_of_interest) 

from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid



__all__ = ['Projector', 'AE', 'CAE', 'POD', 'SPOD']


class Projector(ABC):
    """
    Abstract base for all dimensionality-reduction building blocks,
    both linear (POD, SPOD) and nonlinear (AE, CAE).

    Every projector exposes:

        N_latent  -- size of the latent / bottleneck space. In POD/SPOD, this is the number of modes retained. 
        fit       -- learn the representation from data
        encode    -- map state space Q --> latent Z
        decode    -- map latent Z --> reconstructed state Q_hat
        reconstruct, score, copy -- provided as concrete methods
    """

    N_latent: int = 20   

    @abstractmethod
    def fit(self, Q: np.ndarray) -> 'Projector':
        """Learn the projection from data Q (N_x, N_t). Returns self."""

    @abstractmethod
    def encode(self, Q: np.ndarray) -> np.ndarray:
        """Map Q (N_x, N_t) to latent representation Z (N_latent, N_t)."""

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Map latent Z (N_latent, N_t) back to state space Q_hat (N_x, N_t)."""

    def reconstruct(self, Q: np.ndarray) -> np.ndarray:
        """Full round-trip: encode then decode."""
        return self.decode(self.encode(Q))

    def score(self, Q: np.ndarray) -> float:
        """Mean squared reconstruction error ||Q - reconstruct(Q)||^2 / N."""
        return float(np.mean((Q - self.reconstruct(Q)) ** 2))

    def copy(self) -> 'Projector':
        """Return a deep copy."""
        return deepcopy(self)




class AE(Projector):
    """
    Autoencoder  [stub — not yet implemented].

    Planned: PyTorch MLP encoder/decoder with MSE loss, bottleneck of size
    ``latent_dim``, trained end-to-end.
    """

    def fit(self, Q: np.ndarray) -> 'AE':
        raise NotImplementedError(
            'AE is not yet implemented.')

    def encode(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError



class CAE(Projector):
    """
    Convolutional Autoencoder  [stub — not yet implemented].

    Planned: PyTorch Conv2d encoder/decoder operating on the 2-D spatial grid.
    Requires structured (Nx, Ny) input — use ``grid_shape`` to restore the 2-D
    layout before passing to the network.
    """

    def fit(self, Q: np.ndarray) -> 'CAE':
        raise NotImplementedError(
            'CAE is not yet implemented')

    def encode(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError



# ─────────────────────────────────────────────────────────────────────────────
# POD  — absorbs the former LinearROM base
# ─────────────────────────────────────────────────────────────────────────────

class POD(Projector):
    """
    Snapshot Proper Orthogonal Decomposition.

    Inherits the shared interface from ``Projector`` (fit/encode/decode/
    reconstruct/score/copy/N_latent) and adds linear-specific attributes,
    geometry helpers, and plotting utilities.

    Two solvers are available via ``method``:

    ``'exact'``
        Full eigendecomposition of the temporal correlation matrix C = Q^T Q / N_t.
        Returns all N_t modes.  Exact but O(N_t^3).

    ``'randomized'``  (default)
        Randomized SVD (Halko, Martinsson & Tropp 2011).  Returns only the
        leading ``n_modes`` modes.  Fast and memory-efficient for large data.

    After ``fit(Q)`` the following attributes are available:

        Sigma  (N_latent,)      singular values, descending
        Psi    (N_x, N_latent)  spatial modes (orthonormal columns)
        Phi    (N_latent, N_t)  temporal coefficients from training data
        Q_mean (N_x, 1)         temporal mean

    Parameters
    ----------
    n_modes      : int         Modes / latent-space size.  Default: 20.
    method       : str         'exact' | 'randomized'.  Default: 'randomized'.
    n_iter       : int         Power-iteration steps for 'randomized'.  Default: 4.
    random_state : int | None  Seed for reproducibility.
    grid_shape   : tuple       Optional (Nu, Nx, Ny) for restore_shape().
    domain       : list        Optional [x0, x1, y0, y1] for domain_mesh.
    **kwargs     : Pre-set any instance attribute (e.g. Sigma=s, Psi=p, …).

    Examples
    --------
    ::

        pod = POD(n_modes=20).fit(Q)
        Z   = pod.encode(Q)              # (20, N_t)
        Q_r = pod.reconstruct(Q)         # (N_x, N_t)

    Loading pre-computed results::

        pod = POD(Sigma=s, Psi=p, Phi=ph, Q_mean=m, grid_shape=gs)
    """

    # ── class-level defaults ─────────────────────────────────────────────────
    Sigma:        Optional[np.ndarray] = None
    Psi:          Optional[np.ndarray] = None
    Phi:          Optional[np.ndarray] = None
    Q_mean:       Optional[np.ndarray] = None
    grid_shape:   Optional[tuple]      = None   # (Nu, Nx, Ny) for restore_shape
    domain:       Optional[list]       = None   # [x0, x1, y0, y1]
    field_labels: list                 = ['$u_x$', '$u_y$']
    _TKE:         Optional[float]      = None
    indices_to_original_grid: Optional[np.ndarray] = None
    method:       str           = 'randomized'
    n_iter:       int           = 4
    random_state: Optional[int] = None

    def __init__(self,
                 n_modes:      int            = 20,
                 method:       str            = 'randomized',
                 n_iter:       int            = 4,
                 random_state: Optional[int]  = None,
                 grid_shape:   Optional[tuple] = None,
                 domain:       Optional[list]  = None,
                 **kwargs):
        self.N_latent     = n_modes
        self.grid_shape   = grid_shape
        self.domain       = domain
        self.method       = method
        self.n_iter       = n_iter
        self.random_state = random_state
        for key, val in kwargs.items():
            if hasattr(type(self), key) or key in ('Sigma', 'Psi', 'Phi',
                                                    'Q_mean', '_TKE',
                                                    'indices_to_original_grid'):
                setattr(self, key, val)
        # infer latent size from pre-loaded Phi if provided
        if self.Phi is not None and self.N_latent == 20:
            self.N_latent = self.Phi.shape[0]
        # backward compat: auto-fit if raw data provided as kwarg 'X'
        if 'X' in kwargs and kwargs['X'] is not None:
            self.fit(kwargs['X'])

    # ── N_modes backward-compat alias ─────────────────────────────────────────

    @property
    def N_modes(self) -> int:
        """Backward-compatible alias for N_latent."""
        return self.N_latent

    @N_modes.setter
    def N_modes(self, value: int) -> None:
        self.N_latent = value

    # ── decomposition hook (overridden by SPOD) ────────────────────────────────

    def _decompose(self, Q: np.ndarray) -> tuple:
        """
        Run the SVD/eigendecomposition on zero-mean Q.

        Returns
        -------
        Sigma  : (N_latent,)
        Psi    : (N_x, N_latent)
        Phi    : (N_latent, N_t)
        """
        if self.method == 'exact':
            Sigma, Psi, Phi, C = snapshot_pod(Q)
            self._C = C
            return Sigma[:self.N_latent], Psi[:, :self.N_latent], Phi[:self.N_latent]
        elif self.method == 'randomized':
            return snapshot_pod_randomized(Q, n_modes=self.N_latent,
                                           n_iter=self.n_iter,
                                           random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method '{self.method}'. "
                             "Choose 'exact' or 'randomized'.")

    # ── Projector interface ────────────────────────────────────────────────────

    def fit(self, Q: np.ndarray) -> 'POD':
        """
        Fit the POD to data matrix Q (N_x, N_t).

        A residual mean is stored so calling fit on non-centred data also works.
        """
        self.Q_mean = Q.mean(axis=1, keepdims=True)
        Q_c = Q - self.Q_mean
        self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
        result = self._decompose(Q_c)
        self.Sigma = result[0]
        self.Psi   = result[1]
        self.Phi   = result[2]
        assert self.Sigma is not None and self.Psi is not None and self.Phi is not None, \
            "Decomposition must return Sigma, Psi, Phi."
        self.N_latent = self.Sigma.shape[0]
        return self

    def encode(self, Q: np.ndarray) -> np.ndarray:
        """
        Project Q onto the spatial modes.

            Z = Psi^T (Q - Q_mean)    shape (N_latent, N_t)
        """
        return self.Psi.T @ (Q - self.Q_mean)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct from latent coefficients.

            Q_hat = Psi Z + Q_mean    shape (N_x, N_t)
        """
        return self.Psi @ Z + self.Q_mean

    def reconstruct(self, Q: np.ndarray = None,
                    n_modes: int = None) -> np.ndarray:
        """
        Full round-trip with optional mode truncation.

        Parameters
        ----------
        Q       : ndarray (N_x, N_t).  If None, uses stored training Phi.
        n_modes : retain only the first n_modes modes.  Default: all.
        """
        Z  = self.encode(Q) if Q is not None else self.Phi
        nm = n_modes or self.N_latent
        return self.Psi[:, :nm] @ Z[:nm] + self.Q_mean

    def score(self, Q: np.ndarray) -> float:
        """Mean squared reconstruction error."""
        return float(np.mean((Q - self.reconstruct(Q)) ** 2))

    # ── utilities ─────────────────────────────────────────────────────────────

    def energy_fraction(self):
        """
        Relative and cumulative energy per mode.

        Returns
        -------
        rel : ndarray (N_latent,)  fraction of total energy per mode
        cum : ndarray (N_latent,)  cumulative fraction
        """
        lam = self.Sigma ** 2
        rel = lam / lam.sum()
        return rel, np.cumsum(rel)

    def truncate(self, n_modes: int) -> 'POD':
        """Truncate to the first n_modes modes in-place."""
        if n_modes >= self.N_latent:
            return self
        self.Psi      = self.Psi[:, :n_modes]
        self.Phi      = self.Phi[:n_modes, :]
        self.Sigma    = self.Sigma[:n_modes]
        self.N_latent = n_modes
        return self

    def restore_shape(self, data: np.ndarray = None) -> np.ndarray:
        """Reshape flat data (N_x, ...) back to the original grid shape."""
        if data is None:
            data = self.Psi
        if self.grid_shape is None:
            raise ValueError('grid_shape not set — pass grid_shape to the constructor.')
        if data.ndim > 1:
            if data.shape[:len(self.grid_shape)] == self.grid_shape:
                return data
            return np.reshape(data, (*self.grid_shape, data.shape[-1]))
        return np.reshape(data, self.grid_shape)

    @property
    def domain_mesh(self):
        """
        Meshgrid for the spatial domain.

        Returns
        -------
        X1, X2 : ndarray  Coordinate arrays shaped (Nx, Ny).
        """
        if self.domain is None or self.grid_shape is None:
            raise ValueError('Both domain and grid_shape must be set.')
        x1 = np.linspace(*self.domain[:2], num=self.grid_shape[-2])
        x2 = np.linspace(*self.domain[2:], num=self.grid_shape[-1])
        return np.meshgrid(x1, x2, indexing='ij')

    def original_data_to_domain_of_interest(self, original_data: np.ndarray):
        """Crop original-grid data to the fitted domain of interest."""
        if self.indices_to_original_grid is None:
            return original_data
        original_data = original_data.copy()
        try:
            if original_data.ndim == 2:
                return original_data[self.indices_to_original_grid]
            return original_data[:, self.indices_to_original_grid[0],
                                     self.indices_to_original_grid[1]]
        except Exception:
            raise ValueError(
                'Pass original_data in shape [(Nu) x Nx x Ny x (Nt)].')

    # ── backward-compat aliases ───────────────────────────────────────────────

    def project_data_onto_Psi(self, data: np.ndarray,
                               remove_mean: bool = True) -> np.ndarray:
        """
        Project data onto the spatial basis and normalise by Sigma.

        Equivalent to the original MODULO-based ``project_data_onto_Psi``.
        Returns shape (N_t, N_latent) for backward compatibility.
        For new code prefer ``encode(Q)`` which returns (N_latent, N_t).
        """
        data = data.copy()
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])
        elif data.shape[0] != self.Q_mean.shape[0]:
            data = data.reshape(self.Q_mean.shape[0], -1)
        if remove_mean:
            data = data - self.Q_mean
        Z = self.Psi.T @ data
        return (Z / self.Sigma[:, None]).T

    def rerun_decomposition(self, Q: np.ndarray = None,
                             n_modes: Optional[int] = None) -> 'POD':
        """Re-fit or truncate.  Drop-in for ``rerun_POD_decomposition``."""
        if Q is None and n_modes is None:
            raise ValueError("Provide Q (new data) or n_modes (truncation).")
        if n_modes is not None and (Q is None or n_modes < self.N_latent):
            return self.truncate(n_modes)
        if Q is not None:
            if n_modes is not None:
                self.N_latent = n_modes
            return self.fit(Q)
        return self

    # ── metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_MSE(ROM_data: np.ndarray, original_data: np.ndarray,
                    time_evolution: bool = False):
        """Mean Squared Error between ROM reconstruction and original data."""
        ROM_data, original_data = POD.flatten(ROM_data, original_data)
        original_data[np.isnan(original_data)] = 0.
        if time_evolution:
            return np.mean((original_data - ROM_data) ** 2, axis=0)
        return float(np.mean((original_data - ROM_data) ** 2))

    @staticmethod
    def compute_RMS(ROM_data: np.ndarray, original_data: np.ndarray):
        """Root Mean Square error (field)."""
        original_data[np.isnan(original_data)] = 0.
        return np.sqrt((original_data - ROM_data) ** 2)

    @staticmethod
    def flatten(*args):
        """Flatten multi-dimensional arrays to 2-D (space × time)."""
        return [a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a.copy()
                for a in args]

    # ── plots ─────────────────────────────────────────────────────────────────

    @staticmethod
    def plot_modes(case: 'POD', Psi: np.ndarray = None,
                   num_modes: int = 2, save: bool = False,
                   cmap: str = 'viridis', dim=None):
        """
        Plot the first ``num_modes`` spatial modes.

        Parameters
        ----------
        case      : fitted POD instance.
        Psi       : optional override for spatial modes.
        num_modes : number of modes to display.
        cmap      : matplotlib colormap name.
        dim       : which field components to plot (default: all).
        """
        if get_figsize_based_on_domain is None:
            raise ImportError('utils.get_figsize_based_on_domain not available.')
        if Psi is None:
            Psi = case.Psi.copy()
        if Psi.ndim == 2:
            Psi = case.restore_shape(Psi)
        dim = np.arange(Psi.shape[0]) if dim is None else (
              [dim] if isinstance(dim, int) else dim)
        X1, X2 = case.domain_mesh
        figsize, n_col, n_row = get_figsize_based_on_domain(
            case.domain, total_subplots=num_modes)
        figsize = (n_col * 2, n_row * figsize[1] / figsize[0] * 4)
        for jj, d in enumerate(dim):
            data = Psi[d]
            fig  = plt.figure(figsize=figsize, layout='constrained')
            axs  = fig.subplots(nrows=n_row, ncols=n_col,
                                sharex=True, sharey=True)
            axs  = [axs] if case.N_latent == 1 else axs.ravel()
            norm = colors.Normalize(vmin=data[..., 0].min(),
                                    vmax=data[..., 0].max())
            for kk, ax in zip(range(num_modes), axs):
                im = ax.pcolormesh(X1, X2, data[..., kk],
                                   cmap=mpl.colormaps[cmap], norm=norm,
                                   rasterized=True)
                ax.set_title(f'mode {kk}', fontsize='xx-small')
                ax.set_aspect('equal')
                if kk >= num_modes - n_col:
                    ax.set_xlabel('$y$')
                if kk % n_col == 0:
                    ax.set_ylabel('$x$')
            fig.colorbar(im, ax=axs, shrink=0.25, aspect=20)
            if save:
                plt.savefig(f'modes_dim{jj}.png', dpi=300)

    # keep old name as alias
    plot_POD_modes = plot_modes

    @staticmethod
    def plot_time_coefficients(case: 'POD', Phi: np.ndarray = None,
                                num_modes: int = None,
                                plot_recurrence: bool = False):
        """
        Imshow of the temporal coefficient matrix Phi (N_latent, N_t).

        Parameters
        ----------
        case            : fitted POD instance.
        Phi             : optional override, shape (N_latent, N_t).
        num_modes       : number of modes to include.
        plot_recurrence : if True, also plot recurrence plots (requires pyts).
        """
        if Phi is None:
            Phi = case.Phi.copy()
        if num_modes is not None:
            Phi = Phi[:num_modes, :]
        else:
            num_modes = Phi.shape[0]
        N_t    = Phi.shape[1]
        window = N_t if num_modes < 10 else (200 if num_modes < 50 else num_modes)
        nrows  = max(int(N_t // window), 1)
        slices = [Phi[:, i * window:(i + 1) * window] for i in range(nrows)]

        fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 1.5 * nrows),
                                 layout='tight')
        if nrows == 1:
            axs = [axs]
        norm = colors.Normalize(vmin=Phi.min(), vmax=Phi.max())
        im   = None
        for i0, (ax, sl) in enumerate(zip(axs, slices)):
            im = ax.imshow(sl, cmap=mpl.colormaps['viridis'], norm=norm,
                           aspect='auto',
                           extent=[i0 * window, (i0 + 1) * window,
                                   0, num_modes],
                           origin='lower')
        if im is not None:
            fig.colorbar(im, ax=axs[0], shrink=0.75, orientation='vertical')

        if plot_recurrence:
            if not _HAS_PYTS:
                raise ImportError('pyts is required for recurrence plots.')
            ncols1 = min(2, num_modes)
            nrows1 = int(np.ceil(num_modes / max(ncols1, 1)))
            rp     = RecurrencePlot(threshold='point', percentage=20)
            X_rp   = rp.fit_transform(Phi)
            fig2   = plt.figure(figsize=(4 * ncols1, 3 * nrows1))
            grid   = ImageGrid(fig2, GridSpec(1, 1)[0, 0],
                               nrows_ncols=(nrows1, ncols1),
                               axes_pad=0.1, share_all=True)
            for ii, xx in enumerate(X_rp):
                grid[ii].imshow(xx, cmap='binary', origin='lower')
            grid[0].get_yaxis().set_ticks([])
            grid[0].get_xaxis().set_ticks([])

    @staticmethod
    def plot_spectrum(case: 'POD', max_mode: Optional[int] = None):
        """
        Bar chart of eigenvalue spectrum + cumulative energy.

        Parameters
        ----------
        case     : fitted POD instance.
        max_mode : if set, adds a zoom inset up to this mode number.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        Lambda            = case.Sigma ** 2
        normalised_Lambda = Lambda / Lambda[0]

        axs[0].bar(np.arange(case.N_latent) + 1, normalised_Lambda, color='C4')
        axs[0].set(xlabel='Mode $j$', title='$\\lambda_j / \\lambda_0$',
                   xlim=[0, max(case.N_latent, 10)])

        cum_energy = np.cumsum(Lambda) / Lambda.sum()
        axs[1].plot(np.arange(case.N_latent) + 1, cum_energy, 'o-', color='C4',
                    label='$\\Sigma \\lambda_j / \\Sigma_k \\lambda_k$')

        if case._TKE is not None:
            energy_frac = Lambda / 2 / (case.Phi.shape[1] if case.Phi is not None else 1)
            axs[1].plot(np.arange(case.N_latent) + 1,
                        np.cumsum(energy_frac) / case._TKE,
                        dashes=[10, 5], color='k', label='TKE fraction')

        axs[1].grid(visible=True, linestyle='--', alpha=0.5)
        axs[1].set(xlabel='# modes', title='Cumulative energy')
        axs[1].legend(ncol=1, bbox_to_anchor=[1, 1], loc='upper left')
        for ax in axs:
            ax.set(ylim=[0, 1.05], xlim=[-1, None])

        if max_mode is not None and max_mode < case.N_latent:
            ax0 = fig.add_axes((0.30, 0.50, 0.15, 0.35))
            ax0.bar(np.arange(max_mode) + 1, normalised_Lambda[:max_mode],
                    color='C4')
            ax0.set_xlim(0, max_mode + 1)
            ax1 = fig.add_axes((0.72, 0.20, 0.15, 0.35))
            ax1.grid(visible=True, linestyle='--', alpha=0.5)
            ax1.plot(np.arange(max_mode) + 1,
                     cum_energy[:max_mode], color='C4')
            ax1.set_xlim(0, max_mode + 1)
        return fig

    @staticmethod
    def plot_flows_rms(case: 'POD',
                       datasets,
                       reconstructed_data=None,
                       display_dims=None,
                       display_RMS: Union[str, int, list] = 'all',
                       names=None,
                       norm_flow=None,
                       norm_rms=None,
                       cmap_flow: str = 'viridis',
                       cmap_rms:  str = 'Reds',
                       save: bool = False,
                       display_sensors: bool = False):
        """
        Side-by-side flow fields + RMS error panels.

        Parameters
        ----------
        case               : fitted POD instance.
        datasets           : list of raw data arrays to compare against.
        reconstructed_data : precomputed reconstruction (default: case.reconstruct()).
        display_dims       : which field components to show.
        display_RMS        : 'all' or list of dataset indices to include RMS for.
        names              : labels for each dataset.
        norm_flow/norm_rms : matplotlib Normalize instances.
        cmap_flow/cmap_rms : colormap names.
        save               : if True, saves to PNG.
        display_sensors    : overlay sensor locations if available.
        """
        if get_figsize_based_on_domain is None:
            raise ImportError('utils.get_figsize_based_on_domain not available.')

        def _prep(d, target):
            d[np.isnan(d)] = 0.
            if d.ndim > 3:
                d = d[..., -1]
            if d.shape != target:
                raise ValueError(
                    f'Data shape {d.shape} does not match target {target}.')
            return d

        def _global_norms(prepared, titles, nf, nr):
            if nr is None:
                rms_data = np.array([d for d, t in zip(prepared, titles)
                                     if 'RMS' in t])
                nr = colors.Normalize(vmin=0., vmax=rms_data.max())
            if nf is None:
                nf = [colors.Normalize(vmin=np.min([y[r] for y in prepared]),
                                       vmax=np.max([y[r] for y in prepared]))
                      for r in range(nrows)]
            return nf, nr

        datasets = datasets if isinstance(datasets, list) else [datasets]

        if display_RMS == 'all':
            rms_ids = list(range(len(datasets)))
        else:
            rms_ids = [display_RMS] if isinstance(display_RMS, int) else list(display_RMS)

        if reconstructed_data is None:
            reconstructed_data = case.reconstruct()
        if reconstructed_data.ndim > 3:
            reconstructed_data = reconstructed_data[..., -1].copy()

        display_dims = display_dims if display_dims is not None else \
                       np.arange(reconstructed_data.shape[0])
        if isinstance(display_dims, float):
            display_dims = [display_dims]
        nrows = len(display_dims)
        X1, X2 = case.domain_mesh

        idx, display_sensors = [], display_sensors
        if display_sensors and hasattr(case, 'sensor_locations'):
            idx = case.sensor_locations[
                case.sensor_locations < len(X1.ravel())]

        _datasets = [reconstructed_data]
        _titles   = [f'ROM {case.N_latent} modes']
        _cmaps    = [cmap_flow]

        for ii, (ds, name) in enumerate(
                zip(datasets, names or [None] * len(datasets))):
            ds = _prep(ds, reconstructed_data.shape)
            _datasets.append(ds)
            _cmaps.append(cmap_flow)
            _titles.append(name or f'dataset {ii}')
            if ii in rms_ids:
                _datasets.append(POD.compute_RMS(reconstructed_data, ds))
                _titles.append(f'RMS({name or f"dataset {ii}"})')
                _cmaps.append(cmap_rms)

        norm_flow, norm_rms = _global_norms(_datasets, _titles,
                                            norm_flow, norm_rms)
        ncols   = len(_datasets)
        figsize = get_figsize_based_on_domain(case.domain,
                                              total_subplots=ncols * nrows)[0]
        figsize = (ncols * 2, nrows * figsize[1] / figsize[0] * 4)
        sub_figs = plt.figure(figsize=figsize, layout='constrained') \
                      .subfigures(nrows=nrows, ncols=1)

        for jj, (fig, nf) in enumerate(
                zip(sub_figs if nrows > 1 else [sub_figs], norm_flow)):
            axs = fig.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True)
            im_rms = im_flow = None
            for ax, ds, title, cm in zip(axs, _datasets, _titles, _cmaps):
                if 'RMS' in title:
                    im_rms = ax.pcolormesh(X1, X2, ds[jj], cmap=cm,
                                           norm=norm_rms, rasterized=True)
                else:
                    im_flow = ax.pcolormesh(X1, X2, ds[jj], cmap=cm,
                                            norm=nf, rasterized=True)
                if display_sensors and len(idx):
                    ax.scatter(X1.ravel()[idx], X2.ravel()[idx],
                               c=np.arange(len(idx)),
                               cmap='YlOrRd', edgecolors='k', s=12.25, lw=.5)
                ax.set_aspect('equal')
            for im in [im_rms, im_flow]:
                if im is not None:
                    plt.colorbar(im, ax=axs, shrink=0.5)
        if save:
            plt.savefig('rom_flows_rms.png', dpi=300)
        return sub_figs


# ─────────────────────────────────────────────────────────────────────────────
# SPOD (Sieber)  — inherits from POD, only overrides _decompose
# ─────────────────────────────────────────────────────────────────────────────

class SPOD(POD):
    """
    Spectral POD — Sieber, Paschereit & Oberleithner (JFM 2016).

    Inherits the full POD interface.  The only difference is that the snapshot
    correlation matrix C is replaced by a low-pass filtered version

        C_tilde = G^T C G

    where G is a banded symmetric Toeplitz filter matrix.  Setting ``Nf=0``
    exactly recovers snapshot POD.

    Parameters
    ----------
    Nf          : int   Filter half-width (0 = POD limit, N_t/2 = DFT limit).
    filter_kind : str   'gaussian' | 'box' | 'hann'.  Default: 'gaussian'.
    n_modes     : int   Modes to retain.  Default: all (N_t).
    grid_shape  : tuple Optional (Nu, Nx, Ny) for restore_shape().
    domain      : list  Optional [x0, x1, y0, y1] for domain_mesh.

    Attributes
    ----------
    C_tilde : ndarray (N_t, N_t)  Filtered correlation matrix (stored after fit).
    """

    def __init__(self,
                 Nf:          int            = 0,
                 filter_kind: str            = 'gaussian',
                 n_modes:     int            = 20,
                 grid_shape:  Optional[tuple] = None,
                 domain:      Optional[list]  = None,
                 **kwargs):
        super().__init__(n_modes=n_modes, 
                         grid_shape=grid_shape,
                         domain=domain, 
                         **kwargs)
        self.Nf                 = Nf
        self.filter_kind        = filter_kind
        self._n_modes_requested = n_modes   # None = keep all after fit

    def _decompose(self, Q: np.ndarray) -> tuple:
        Sigma, Psi, Phi, C_tilde = spod_sieber(Q, self.Nf, self.filter_kind)
        self.C_tilde = C_tilde
        if self._n_modes_requested is not None:
            nm = min(self._n_modes_requested, Sigma.shape[0])
            return Sigma[:nm], Psi[:, :nm], Phi[:nm]
        self.N_latent = Sigma.shape[0]
        return Sigma, Psi, Phi

