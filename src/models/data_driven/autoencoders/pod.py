"""
pod.py
======
Linear projectors: `POD` and `SPOD` (Sieber). Thin sklearn-style wrappers
around the decomposition algorithms in `.pod_utils`.
"""

from __future__ import annotations

import numpy as np

from . import Projector
from .pod_utils import snapshot_pod, snapshot_pod_randomized, spod_sieber

__all__ = ['POD', 'SPOD']


class POD(Projector):
    r"""
    Snapshot POD.

    Inherits the shared interface from `Projector` (`fit`/`encode`/`decode`/
    `reconstruct`/`score`/`N_latent`) and adds linear-specific attributes,
    geometry helpers, and plotting utilities.

    Two solvers are available via ``method``:

    - ``'exact'`` — full eigendecomposition of the temporal correlation matrix
      $\mathbf{C} = \mathbf{Q}^\mathrm{T}\mathbf{Q} / N_t$ (Sirovich 1987, snapshot
      method: see `snapshot_pod`). Exact but $\mathcal{O}(N_t^3)$.
    - ``'randomized'`` (default) — randomized SVD of $\mathbf{Q}$ (Halko, Martinsson
      & Tropp 2011: see `snapshot_pod_randomized`). Returns only the leading
      ``n_modes`` modes; fast and memory-efficient.

    Either way, writing $\mathbf{Q} = \mathbf{X} - \bar{\mathbf{Q}}$ for the zero-mean
    data matrix, ``fit(X)`` stores the orthonormal spatial modes $\boldsymbol{\Psi}$
    and singular values $\boldsymbol{\Sigma}$ of $\mathbf{Q}$, together with the
    temporal coefficients $\boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q}$,
    so that

    $$
    \mathbf{X} \approx \boldsymbol{\Psi}\boldsymbol{\Phi} + \bar{\mathbf{Q}},
    $$

    with equality when all $N_t$ modes are retained (``n_modes >= N_t``, ``method='exact'``).

    Parameters
    ----------
    n_modes : int
        Number of modes retained (the latent-space size). Default 20.
    method : str
        ``'exact'`` or ``'randomized'``. Default ``'randomized'``.
    n_iter : int
        Power-iteration steps for the randomized solver. Default 4.
    random_state : int, optional
        Seed for reproducibility of the randomized solver.
    grid_shape : tuple, optional
        Grid shape ``(Nu, Nx, Ny)`` used to map flat vectors back to the grid.
    domain : list, optional
        Physical domain ``[x0, x1, y0, y1]`` used by the plotting utilities.
    **kwargs
        Pre-set any instance attribute (e.g. pre-computed ``Sigma``, ``Psi``,
        ``Phi``, ``Q_mean``), or pass ``X=...`` to fit directly at construction.

    Attributes
    ----------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_\mathrm{latent},)$.
    Psi : np.ndarray
        Spatial modes with orthonormal columns, shape $(N_x, N_\mathrm{latent})$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_\mathrm{latent}, N_t)$.
    Q_mean : np.ndarray
        Temporal mean $\bar{\mathbf{Q}}$, shape $(N_x, 1)$.

    References
    ----------
    Sirovich (1987). Turbulence and the dynamics of coherent structures.
    *Quart. Appl. Math.*, XLV(3), 561-590.

    Halko, Martinsson & Tropp (2011). Finding structure with randomness.
    *SIAM Review*, 53(2), 217-288.
    """

    # ── class-level defaults ─────────────────────────────────────────────────
    _Sigma:       np.ndarray | None = None
    _Psi:         np.ndarray | None = None
    _Phi:         np.ndarray | None = None
    grid_shape:   tuple | None      = None   # (Nu, Nx, Ny) for grid <-> flat mapping
    domain:       list | None       = None   # [x0, x1, y0, y1]
    field_labels: list                 = ['$u_x$', '$u_y$']
    _TKE:         float | None      = None
    indices_to_original_grid: np.ndarray | None = None
    method:       str           = 'randomized'
    n_iter:       int           = 4
    random_state: int | None = None

    @property
    def Sigma(self) -> np.ndarray:
        if self._Sigma is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Sigma

    @Sigma.setter
    def Sigma(self, v: np.ndarray) -> None:
        self._Sigma = v

    @property
    def Psi(self) -> np.ndarray:
        if self._Psi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Psi

    @Psi.setter
    def Psi(self, v: np.ndarray) -> None:
        self._Psi = v

    @property
    def Phi(self) -> np.ndarray:
        if self._Phi is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Phi

    @Phi.setter
    def Phi(self, v: np.ndarray) -> None:
        self._Phi = v


    def __init__(self,
                 n_modes:      int            = 20,
                 method:       str            = 'randomized',
                 n_iter:       int            = 4,
                 random_state: int | None  = None,
                 grid_shape:   tuple | None = None,
                 domain:       list | None  = None,
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
        if self._Phi is not None and self.N_latent == 20:
            self.N_latent = self._Phi.shape[0]
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
        Run the SVD/eigendecomposition on the zero-mean data matrix Q,
        dispatching to `snapshot_pod` or `snapshot_pod_randomized`
        depending on `method`. Overridden by `SPOD` to filter the
        correlation matrix first.

        Returns
        -------
        tuple of np.ndarray
            ``(Sigma, Psi, Phi)`` with shapes ``(N_latent,)``,
            ``(N_x, N_latent)`` and ``(N_latent, N_t)`` respectively.
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


    def fit(self, X: np.ndarray) -> POD:
        r"""
        Fit the POD to data X.

        Accepts a flat matrix $(N_x, N_t)$ or a raw grid array
        $(N_u, N_t, N_x, N_y)$ / $(N_t, N_x, N_y)$. For raw grid input the
        NaN solid-body mask is detected and stored automatically.

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, flat or raw grid (see above).

        Returns
        -------
        POD
            The fitted instance (``self``).
        """

        Q = self.preprocess_snapshot(X)

        result = self._decompose(Q)
        self.Sigma = result[0]
        self.Psi   = result[1]
        self.Phi   = result[2]
        assert self._Sigma is not None and self._Psi is not None and self._Phi is not None, \
            "Decomposition must return Sigma, Psi, Phi."
        self.N_latent = self.Sigma.shape[0]
        self.fitted = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        r"""
        Project X onto the spatial modes,

        $$
        \mathbf{Z} = \boldsymbol{\Psi}^\mathrm{T}(\mathbf{X} - \bar{\mathbf{Q}}).
        $$

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        np.ndarray
            Latent (POD) coefficients $\mathbf{Z}$, shape $(N_\mathrm{latent}, N_t)$.
        """
        Q = self.preprocess_snapshot(X)
        return self.Psi.T @ Q

    def decode(self, Z: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        r"""
        Reconstruct the state in the original space from latent coefficients,

        $$
        \hat{\mathbf{Q}} = \boldsymbol{\Psi}\mathbf{Z} + \bar{\mathbf{Q}}.
        $$

        Parameters
        ----------
        Z : np.ndarray
            Latent coefficients, shape $(N_\mathrm{latent}, N_t)$.
        idx : np.ndarray, optional
            Indices selecting a subset of rows of `Psi` and `Q_mean`
            (e.g. sensor locations) to reconstruct only those entries.

        Returns
        -------
        np.ndarray
            Reconstructed state, shape $(N_x, N_t)$, or ``(len(idx), N_t)``
            if `idx` is given.
        """

        if idx is not None:
            return self.Psi[idx, :] @ Z + self.Q_mean[idx, :]
        else:
            return self.Psi @ Z + self.Q_mean


    def reconstruct(self, X: np.ndarray | None = None,
                    n_modes: int | None = None,
                    Phi: np.ndarray | None = None) -> np.ndarray:
        r"""
        Full round-trip: encode, decode, then map back to the physical grid
        (when a grid mask is available).

        Parameters
        ----------
        X : np.ndarray, optional
            Raw input, grid or flat. If None, the stored `Phi` is used.
        n_modes : int, optional
            Retain only the first ``n_modes`` modes. Default: all fitted modes.
        Phi : np.ndarray, optional
            Pre-computed latent coefficients, shape $(N_\mathrm{latent}, N_t)$;
            skips the `encode` step.

        Returns
        -------
        np.ndarray
            Reconstructed state, on the physical grid if `grid_shape` and a
            ``to_grid`` mapping are available, otherwise flat.
        """
        if Phi is not None:
            Z = Phi
        elif X is not None:
            Z = self.encode(X)
        else:
            Z = self.Phi

        nm = n_modes if n_modes is not None else self.N_latent
        if nm < self.N_latent:
            X_hat = self.Psi[:, :nm] @ Z[:nm] + self.Q_mean
        else:
            X_hat = self.decode(Z)
        if getattr(self, 'to_grid', None) is not None and self.grid_shape is not None:
            return self._to_physical_grid(X_hat)
        return X_hat


    # ── utilities ─────────────────────────────────────────────────────────────

    def energy_fraction(self):
        r"""
        Relative and cumulative energy per mode, from the eigenvalues
        $\lambda_j = \Sigma_j^2$ of the temporal correlation matrix
        $\mathbf{C}$:

        $$
        \mathrm{rel}_j = \frac{\lambda_j}{\sum_k \lambda_k}, \qquad
        \mathrm{cum}_j = \sum_{k \le j} \mathrm{rel}_k.
        $$

        Returns
        -------
        rel : np.ndarray
            Relative energy per mode, shape $(N_\mathrm{latent},)$.
        cum : np.ndarray
            Cumulative energy fraction captured by the first $j$ modes,
            shape $(N_\mathrm{latent},)$.
        """
        lam = self.Sigma ** 2  # eigenvalues of C = Q^T Q / N_t

        return lam / lam.sum(), np.cumsum(lam) / lam.sum()


    def truncate(self, n_modes: int) -> POD:
        """Truncate to the first n_modes modes in-place."""
        if n_modes >= self.N_latent:
            print(f"Requested n_modes={n_modes} >= N_latent={self.N_latent}. No truncation applied.")
            return self
        self.Psi      = self.Psi[:, :n_modes]
        self.Phi      = self.Phi[:n_modes, :]
        self.Sigma    = self.Sigma[:n_modes]
        self.N_latent = n_modes
        return self


    @property
    def domain_mesh(self):
        """
        Meshgrid for the spatial domain.

        Returns
        -------
        X1 : np.ndarray
            First coordinate array, shape ``(Nx, Ny)``.
        X2 : np.ndarray
            Second coordinate array, shape ``(Nx, Ny)``.
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



# ─────────────────────────────────────────────────────────────────────────────
# SPOD (Sieber)  — inherits from POD, only overrides _decompose
# ─────────────────────────────────────────────────────────────────────────────

class SPOD(POD):
    r"""
    Spectral POD via the filtered correlation matrix (Sieber, Paschereit &
    Oberleithner, 2016).

    Inherits the full `POD` interface (`fit`/`encode`/`decode`/`reconstruct`);
    only `_decompose` is overridden, delegating to `spod_sieber`. The snapshot
    correlation matrix $\mathbf{C} = \mathbf{Q}^\mathrm{T}\mathbf{Q}/N_t$ is
    replaced by a low-pass filtered version before the eigensolve,

    $$
    \tilde{\mathbf{C}} = \mathbf{G}^\mathrm{T} \mathbf{C}\, \mathbf{G},
    $$

    where $\mathbf{G}$ is a banded symmetric Toeplitz filter matrix built from a
    normalised 1-D kernel of half-width ``Nf`` (``filter_kind`` selects
    ``'gaussian'``, ``'box'`` or ``'hann'``). The eigendecomposition of
    $\tilde{\mathbf{C}}$ then follows exactly as in `snapshot_pod`, giving
    orthonormal spatial modes $\boldsymbol{\Psi}$, temporal coefficients
    $\boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q}$ and singular
    values $\boldsymbol{\Sigma}$. Setting ``Nf=0`` skips the filtering step and
    exactly recovers snapshot POD; per Sieber et al. (2016), as ``Nf`` grows
    towards $N_t/2$ the SPOD modes are reported to approach Fourier (DFT) modes.

    Parameters
    ----------
    Nf : int
        Filter half-width (0 recovers POD; $N_t/2$ approaches the DFT limit).
    filter_kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``. Default ``'gaussian'``.
    n_modes : int
        Number of modes to retain. Default 20.
    grid_shape : tuple, optional
        Grid shape ``(Nu, Nx, Ny)`` for grid mapping.
    domain : list, optional
        Physical domain ``[x0, x1, y0, y1]`` for the plotting utilities.
    **kwargs
        Forwarded to `POD.__init__` (e.g. pre-set ``Sigma``, ``Psi``, ``Phi``).

    Attributes
    ----------
    C_tilde : np.ndarray
        Filtered correlation matrix $\tilde{\mathbf{C}}$, shape $(N_t, N_t)$
        (stored after ``fit``).

    References
    ----------
    Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal
    decomposition. *J. Fluid Mech.*, 792, 798–828.
    """

    def __init__(self,
                 Nf:          int            = 0,
                 filter_kind: str            = 'gaussian',
                 n_modes:     int            = 20,
                 grid_shape:  tuple | None = None,
                 domain:      list | None  = None,
                 **kwargs):
        super().__init__(n_modes=n_modes,
                         grid_shape=grid_shape,
                         domain=domain,
                         **kwargs)
        self.Nf                 = Nf
        self.filter_kind        = filter_kind
        self._n_modes_requested = n_modes   # None = keep all after fit

    def _decompose(self, Q: np.ndarray) -> tuple:
        """Filtered-correlation-matrix eigendecomposition, via `spod_sieber`."""
        Sigma, Psi, Phi, C_tilde = spod_sieber(Q, self.Nf, self.filter_kind)
        self.C_tilde = C_tilde
        if self._n_modes_requested is not None:
            nm = min(self._n_modes_requested, Sigma.shape[0])
            return Sigma[:nm], Psi[:, :nm], Phi[:nm]
        self.N_latent = Sigma.shape[0]
        return Sigma, Psi, Phi
