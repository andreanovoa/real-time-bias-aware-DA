"""
romda.models.data_driven.autoencoders
========================

Dimensionality-reduction building blocks for data-driven ROMs.
Only 2D snapshot data is supported for now, but the API is designed to be extensible to 3D and multi-field data in the future.

All projectors share the same sklearn-style API:

    p.fit(X)         -- learn the representation from data  X (N_x, N_t)
    p.encode(X)      -- X (N_x, N_t) --> Z (N_latent, N_t)
    p.decode(Z)      -- Z (N_latent, N_t) --> X_hat (N_x, N_t)
    p.reconstruct(X) -- full round-trip
    p.score(X)       -- mean squared reconstruction error
    p.N_latent       -- size of the latent (bottleneck) space

Class hierarchy
---------------

    Projector (ABC)               shared interface + N_latent + reconstruct/score  [here]
    ├── POD(Projector)            Proper Orthogonal Decomposition (linear)         [pod.py]
    │     N_latent == N_modes retained
    │     Sigma, Psi, Phi         decomposition results
    │     truncate / domain_mesh / energy_fraction / ...  utilities
    │
    └── SPOD(POD)                 Spectral POD (Sieber et al. JFM 2016)            [pod.py]
          inherits all POD helpers; only _decompose is overridden
          to apply the Toeplitz low-pass filter before the eigensolve

Nonlinear projectors (autoencoders) subclass `Projector` with the same interface.

These are pure dimensionality-reduction tools — they have no temporal
forecaster.  Combine with an ESN or LSTM in models/data_driven/ to build
a complete ROM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = ['Projector', 'POD', 'SPOD', 'spod_towne', 'print_spod_towne_summary']


class Projector(ABC):
    r"""
    Abstract base for all dimensionality-reduction building blocks, linear
    (`POD`, `SPOD`) or nonlinear (autoencoders).

    Every projector shares the same sklearn-style interface: `fit` learns the
    representation from data, `encode` maps state space to the latent space,
    and `decode` maps back. `reconstruct` and `score` are provided as
    concrete methods built on top of `encode`/`decode`.

    Attributes
    ----------
    N_latent : int
        Size of the latent (bottleneck) space. For `POD`/`SPOD` this is the
        number of modes retained.
    fitted : bool
        Whether `fit` has been called.
    Q_mean : np.ndarray
        Temporal mean removed from the data during preprocessing, shape
        $(N_x, 1)$. Raises ``AttributeError`` if accessed before `fit`.
    """

    N_latent: int = 20
    fitted:   bool = False
    _Q_mean:  np.ndarray | None = None

    @property
    def Q_mean(self) -> np.ndarray:
        if self._Q_mean is None:
            raise AttributeError("Not fitted — call fit() first.")
        return self._Q_mean

    @Q_mean.setter
    def Q_mean(self, value: np.ndarray) -> None:
        self._Q_mean = value

    @abstractmethod
    def fit(self, X: np.ndarray) -> Projector:
        r"""
        Learn the projection from data.

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        Projector
            The fitted instance (``self``).
        """

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        r"""
        Map snapshot data to the latent representation.

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        np.ndarray
            Latent coefficients $\mathbf{Z}$, shape $(N_\mathrm{latent}, N_t)$.
        """

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        r"""
        Map latent coefficients back to state space.

        Parameters
        ----------
        Z : np.ndarray
            Latent coefficients, shape $(N_\mathrm{latent}, N_t)$.

        Returns
        -------
        np.ndarray
            Reconstructed state $\hat{\mathbf{X}}$, shape $(N_x, N_t)$.
        """

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full round-trip: `encode` then `decode`."""
        return self.decode(self.encode(X))

    def score(self, X: np.ndarray) -> float:
        r"""
        Mean squared reconstruction error in the flat, zero-mean space,

        $$
        \mathrm{MSE} = \frac{1}{N_x N_t}\,
        \lVert \mathbf{Q} - \hat{\mathbf{Q}} \rVert_F^2,
        $$

        where $\mathbf{Q} = \mathrm{preprocess}(X)$ and
        $\hat{\mathbf{Q}} = \mathrm{decode}(\mathrm{encode}(X)) - \bar{\mathbf{Q}}$
        (with $\bar{\mathbf{Q}}$ the stored `Q_mean`).

        Parameters
        ----------
        X : np.ndarray
            Snapshot data, shape $(N_x, N_t)$.

        Returns
        -------
        float
            Mean squared reconstruction error.
        """
        Q = self.preprocess_snapshot(X)
        Q_hat = self.decode(self.encode(X)) - self.Q_mean
        return float(np.mean((Q - Q_hat) ** 2))


    # -------
    # utilities for grid handling
    # -------

    def _to_physical_grid(self, X_hat: np.ndarray) -> np.ndarray:
        """Map flat (N_fluid*Nu, N_t) back to (Nu, N_t, Nx, Ny). Exact inverse of `_to_flat`."""
        Nu, Nx, Ny = self.grid_shape
        if X_hat.ndim == 1:
            X_hat = X_hat[:, np.newaxis]
        Nt = X_hat.shape[1]
        N_fluid = int(self.fluid_mask_flat.sum())

        out = np.full((Nu, Nt, Nx, Ny), np.nan)

        # Invert the flatten: (Nu*N_fluid, Nt) → (Nu, N_fluid, Nt) → (Nu, Nt, N_fluid)
        X_unflatten = X_hat.reshape(Nu, N_fluid, Nt).transpose(0, 2, 1)  # (Nu, Nt, N_fluid)

        for u in range(Nu):
            # X_unflatten[u] is (Nt, N_fluid) — all time steps for field u
            grid_flat = np.full((Nt, Nx*Ny), np.nan)
            grid_flat[:, self.fluid_mask_flat] = X_unflatten[u]  # Place fluid values back

            # Reshape (Nt, Nx*Ny) → (Nt, Nx, Ny) and assign
            out[u] = grid_flat.reshape(Nt, Nx, Ny)

        return out[:, 0] if Nt == 1 else out


    def _to_flat(self, X: np.ndarray) -> np.ndarray:
        """Map raw grid input (Nu, Nt, Nx, Ny) to flat (N_fluid * n_fields, N_t) in
        variable-block ordering: rows [0:N_fluid] are the first field, rows
        [N_fluid:2*N_fluid] the second field, etc."""
        X_masked = X.reshape(X.shape[0], X.shape[1], -1)[:, :, self.fluid_mask_flat] # (Nu, Nt, N_fluid)

        return X_masked.transpose(0, 2, 1).reshape(-1, X.shape[1]) # (n_fields * N_fluid, N_t)


    def grid_index_to_flat_rows(self, grid_idx) -> np.ndarray:
        """
        Map raw-grid indices to rows of the masked flat representation (Psi / Q_mean rows).

        Both the raw grid and the flat representation use variable-block ordering:
        raw index = var * Nx * Ny + g (g = flattened (x, y) position),
        flat row  = var * N_fluid + fluid_pos (position of g among the fluid points).

        Parameters
        ----------
        grid_idx : array-like of int
            Raw-grid indices (e.g., sensor locations). Must correspond to fluid points.

        Returns
        -------
        np.ndarray of int
            Row indices into Psi / Q_mean corresponding to the requested grid points.
        """
        assert self.grid_shape is not None, 'grid_shape must be set to map grid indices.'
        Nu, Nx, Ny = self.grid_shape
        grid_idx = np.asarray(grid_idx).ravel()

        var = grid_idx // (Nx * Ny)
        g = grid_idx % (Nx * Ny)

        if not np.all(self.fluid_mask_flat[g]):
            raise ValueError('Some requested grid points are not fluid points.')

        fluid_idx = np.flatnonzero(self.fluid_mask_flat)
        N_fluid = fluid_idx.size
        fluid_pos = np.searchsorted(fluid_idx, g)
        return var * N_fluid + fluid_pos


    # -----
    # preprocessing for raw grid input.
    # Note: could implement different ones including normalization/standardization.
    # -----

    def preprocess_snapshot(self, X: np.ndarray, subtract_mean=True):
        r"""
        Build the zero-mean data matrix from raw snapshot fields,
        automatically detecting and removing NaN-masked solid-body points.

        Parameters
        ----------
        X : np.ndarray
            Raw snapshot data, either a single field $(N_t, N_x, N_y)$ or a
            stack of fields $(N_u, N_t, N_x, N_y)$.
        subtract_mean : bool
            If True (default), subtract the temporal mean row-wise.

        Returns
        -------
        np.ndarray
            Zero-mean data matrix $\mathbf{Q}$, shape
            $(N_\mathrm{fluid} \cdot n_\mathrm{fields}, N_t)$, ready for decomposition.
        """

        if not self.fitted:
            if X.ndim == 2:
                # Already-flat data matrix (N_x, N_t): no grid/mask handling
                self.fluid_mask_flat = np.ones(X.shape[0], dtype=bool)
                if subtract_mean:
                    self.Q_mean = X.mean(axis=1, keepdims=True)
                else:
                    self.Q_mean = np.zeros_like(X[:, :1])
                Q = X - self.Q_mean
                self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
                return Q

            # if the input is raw grid data, we need to detect the fluid points and flatten the data
            assert X.ndim == 4, f'Expected raw grid input with 4 dimensions, got {X.ndim}.'
            Nu, Nt, Nx, Ny = X.shape
            self.grid_shape = (Nu, Nx, Ny)

            ref = X[0]
            fluid_mask = ~np.isnan(ref[0])
            self.fluid_mask_flat  = fluid_mask.ravel()



            X_masked_flat = self._to_flat(X)                    # (N_fluid * n_fields, N_t)

            if subtract_mean:
                self.Q_mean = X_masked_flat.mean(axis=1, keepdims=True)
            else:
                self.Q_mean = np.zeros_like(X_masked_flat[:, :1])


            Q = X_masked_flat - self.Q_mean #shape (N_fluid * n_fields, N_t)
            # store the total kinetic energy for later use in relative error metrics
            self._TKE = 0.5 * float(np.sum(np.mean(Q**2, axis=1)))
            return Q

        elif X.shape[0] != self.Q_mean.shape[0]:

            # if the decomosition is already fitted, can expect 1 snapshot only
            assert X.ndim in (3, 4), f'Expected flat input with 2, 3 or 4 dimensions, got {X.ndim}.'
            if X.ndim == 3:
                X = X[:, np.newaxis]    # (n_fields, 1, Nx, Ny)

            #check grid
            Nu, _, Nx, Ny = X.shape
            grid_shape = (Nu, Nx, Ny)
            assert grid_shape == self.grid_shape, f'Expected grid shape {self.grid_shape}, got {grid_shape}.'

            X_masked_flat = self._to_flat(X)                    # (N_fluid * n_fields, N_t)
            return X_masked_flat - self.Q_mean #shape (N_fluid * n_fields, N_t)
        else:
            # already flat input, just check dimensions and remove mean
            assert X.ndim == 2, f'Expected flat input with 2 dimensions, got {X.ndim}.'

            return X - self.Q_mean #shape (N_fluid * n_fields, N_t)


# Re-exports — must stay AFTER the Projector definition (the submodules import
# Projector back from this partially-initialised __init__, as in romda.estimators).
from .pod import POD, SPOD  # noqa: E402
from .pod_utils import print_spod_towne_summary, spod_towne  # noqa: E402

