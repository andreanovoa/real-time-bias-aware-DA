"""
rom_base.py
===========
Abstract base classes for Reduced-Order Models (ROMs).

All ROMs share a common sklearn-style API:

    rom.fit(Q)            -- learn the representation from data
    rom.encode(Q)         -- Q (N_x, N_t) --> Z (latent, ...)
    rom.decode(Z)         -- Z (latent, ...) --> Q_hat (N_x, N_t)
    rom.reconstruct(Q)    -- full round-trip: encode then decode
    rom.score(Q)          -- mean squared reconstruction error

This interface allows downstream code (e.g. data assimilation) to swap
decomposition methods without changing any other logic.

Hierarchy
---------
ROM  (abstract)
 ├── LinearROM     -- linear subspace methods (Psi/Phi/Sigma attributes)
 │    ├── POD      -- snapshot POD, exact or randomized
 │    └── SPOD     -- Sieber spectral POD
 └── NonlinearROM  -- neural-network encoder/decoder  [not yet implemented]
      ├── AE       -- Autoencoder
      └── CAE      -- Convolutional Autoencoder
"""

from abc import ABC, abstractmethod
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ROM(ABC):
    """
    Abstract base for all Reduced-Order Models.

    Subclasses must implement ``fit``, ``encode``, and ``decode``.
    ``reconstruct`` and ``score`` are provided as concrete methods built on top.
    """

    @abstractmethod
    def fit(self, Q: np.ndarray) -> 'ROM':
        """
        Learn the ROM from data.

        Parameters
        ----------
        Q : ndarray (N_x, N_t)   Zero-mean data matrix (caller subtracts mean).

        Returns
        -------
        self  (for method chaining)
        """

    @abstractmethod
    def encode(self, Q: np.ndarray) -> np.ndarray:
        """
        Encode Q into the latent representation Z.

        Parameters
        ----------
        Q : ndarray (N_x, N_t)

        Returns
        -------
        Z : ndarray   Latent representation (shape depends on subclass).
        """

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation Z back to the state space.

        Parameters
        ----------
        Z : ndarray   Latent representation (shape depends on subclass).

        Returns
        -------
        Q_hat : ndarray (N_x, N_t)
        """

    def reconstruct(self, Q: np.ndarray) -> np.ndarray:
        """
        Full round-trip: encode Q then decode.

        Parameters
        ----------
        Q : ndarray (N_x, N_t)

        Returns
        -------
        Q_hat : ndarray (N_x, N_t)
        """
        return self.decode(self.encode(Q))

    def score(self, Q: np.ndarray) -> float:
        """
        Mean squared reconstruction error  ||Q - reconstruct(Q)||^2 / N.

        Parameters
        ----------
        Q : ndarray (N_x, N_t)

        Returns
        -------
        mse : float
        """
        return float(np.mean((Q - self.reconstruct(Q)) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Nonlinear ROM stubs  (AE / CAE — to be implemented)
# ─────────────────────────────────────────────────────────────────────────────

class NonlinearROM(ROM):
    """
    Base class for neural-network ROMs.

    The encoder and decoder are learnable functions (e.g. fully-connected or
    convolutional networks).  ``fit`` trains the network end-to-end by
    minimising reconstruction loss.

    Not yet implemented — subclasses must override all abstract methods.
    """


class AE(NonlinearROM):
    """
    Autoencoder ROM  [stub — not yet implemented].

    Architecture: fully-connected encoder/decoder with a bottleneck of size
    ``latent_dim``.  Trained end-to-end with MSE loss.
    """

    def fit(self, Q: np.ndarray) -> 'AE':
        raise NotImplementedError(
            'AE is not yet implemented.  '
            'Planned: PyTorch MLP encoder/decoder trained end-to-end.')

    def encode(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CAE(NonlinearROM):
    """
    Convolutional Autoencoder ROM  [stub — not yet implemented].

    Architecture: convolutional encoder/decoder operating on the 2-D spatial
    grid.  Requires structured (Nx, Ny) input — use the ``grid_shape``
    attribute to restore the 2-D layout before passing to the network.
    """

    def fit(self, Q: np.ndarray) -> 'CAE':
        raise NotImplementedError(
            'CAE is not yet implemented.  '
            'Planned: PyTorch Conv2d encoder/decoder with transposed-conv decoder.')

    def encode(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
