"""
pod_spod.py
=============
Pre-processing, decomposition algorithms, and linear subspace ROM classes.

Functions
---------
snapshot_pod            : Snapshot POD — exact solver (Sirovich 1987)
snapshot_pod_randomized : Randomized snapshot POD (Halko et al. 2011)
spod_sieber             : SPOD via filtered correlation matrix (Sieber et al. 2016)
spod_towne              : SPOD via Welch CSD (Towne et al. 2018)
spod_towne_reconstruct  : inverse SPOD [stub]
print_spod_towne_summary: pretty-print Towne SPOD parameters

Classes
-------
LinearROM   : abstract base for any method with Psi / Phi / Sigma attributes.
POD         : snapshot POD — exact (Sirovich 1987) or randomized (Halko 2011).
SPOD        : Sieber spectral POD (Sieber, Paschereit & Oberleithner, JFM 2016).

Usage
-----
::

    from romda.utils import prepare_data
    from pod_spod import POD, SPOD

    Q, fluid_mask, to_grid = prepare_data([ux_raw, uy_raw])   # (N_fluid*2, N_t)

    pod = POD(n_modes=20).fit(Q)
    Z   = pod.encode(Q)               # (N_modes, N_t) -- POD coefficients
    Q_r = pod.decode(Z)               # (N_fluid*2, N_t) -- reconstruction
    mode1 = to_grid(pod.Psi[:N_fluid, 0])   # (Nx, Ny) with NaN at body

Convention (Mendez 2023)
------------------------
  Psi   (N_x, N_modes)   spatial modes, orthonormal columns
  Phi   (N_modes, N_t)   temporal coefficients  Phi[k, t] = Psi_k . q(t)
  Sigma (N_modes,)       singular values  Sigma_k = sqrt(lambda_k)

References
----------
[Sirovich 1987]  Quart. Appl. Math., XLV(3), 561-590.
[Halko 2011]     SIAM Review, 53(2), 217-288.
[Sieber 2016]    JFM 792, 798-828.
[Towne 2018]     JFM 847, 821-867.
[Mendez 2023]    Data-Driven Fluid Mechanics, Cambridge University Press.
"""

import numpy as np
from scipy.signal import get_window
from scipy.special import gammaincinv



# ── optional project utilities ────────────────────────────────────────────────

try:
    from mpl_toolkits.axes_grid1 import ImageGrid
    _HAS_IMAGEGRID = True
except ImportError:
    _HAS_IMAGEGRID = False

try:
    from pyts.image import RecurrencePlot
    _HAS_PYTS = True
except ImportError:
    _HAS_PYTS = False




def energy_fraction(Sigma):
    """
    Relative energy fraction and cumulative energy.

    Parameters
    ----------
    Sigma : ndarray   Singular values from snapshot_pod or spod_sieber.

    Returns
    -------
    rel : ndarray   Relative energy per mode (sums to 1).
    cum : ndarray   Cumulative relative energy.
    """
    lam = Sigma ** 2
    rel = lam / lam.sum()
    return rel, np.cumsum(rel)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition algorithms
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_pod(Q):
    r"""Snapshot POD — exact solver.

    Solves the eigenvalue problem of the temporal correlation matrix
    $\mathbf{C} = \mathbf{Q}^\mathrm{T} \mathbf{Q} / N_t$.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_t,)$.
    Psi : np.ndarray
        Spatial modes with orthonormal columns, shape $(N_x, N_t)$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_t, N_t)$.
    C : np.ndarray
        Temporal correlation matrix, shape $(N_t, N_t)$.

    References
    ----------
    Sirovich (1987). Turbulence and the dynamics of coherent structures.
    *Quart. Appl. Math.*, XLV(3), 561–590.
    """
    _, N_t = Q.shape
    C      = (Q.T @ Q) / N_t
    lam, A = np.linalg.eigh(C)
    idx      = lam.argsort()[::-1]
    lam, A   = lam[idx], A[:, idx]
    safe_lam = np.where(lam > 0, lam, np.inf)
    Psi      = Q @ A / (np.sqrt(N_t) * np.sqrt(safe_lam))
    Psi[:, lam <= 0] = 0.0
    Phi      = Psi.T @ Q
    Sigma    = np.sqrt(np.where(lam > 0, lam, 0.0))
    return Sigma, Psi, Phi, C


def snapshot_pod_randomized(Q, n_modes=20, n_iter=4, random_state=None):
    r"""Randomized snapshot POD.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    n_modes : int
        Leading modes to compute. Default 20.
    n_iter : int
        Power-iteration steps. Default 4.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_\mathrm{modes},)$.
    Psi : np.ndarray
        Spatial modes (approximately orthonormal), shape $(N_x, N_\mathrm{modes})$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_\mathrm{modes}, N_t)$.

    References
    ----------
    Halko, Martinsson & Tropp (2011). Finding structure with randomness.
    *SIAM Review*, 53(2), 217–288.
    """
    N_x, N_t = Q.shape
    n_modes  = min(n_modes, N_x, N_t)
    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]
        U, s, _ = randomized_svd(Q, n_components=n_modes,
                                 n_iter=n_iter, random_state=random_state)
    except ImportError:
        U_full, s_full, _ = np.linalg.svd(Q, full_matrices=False)
        U, s = U_full[:, :n_modes], s_full[:n_modes]
    Psi   = U
    Sigma = s / np.sqrt(N_t)
    Phi   = Psi.T @ Q
    return Sigma, Psi, Phi


def _filter_kernel(Nf, kind='gaussian'):
    """Symmetric normalised 1-D low-pass filter of half-width Nf."""
    if Nf == 0:
        return np.array([1.0])
    k = np.arange(-Nf, Nf + 1, dtype=float)
    if kind == 'box':
        g = np.ones(2 * Nf + 1)
    elif kind == 'gaussian':
        g = np.exp(-0.5 * (k / (Nf / 2.0)) ** 2)
    elif kind == 'hann':
        g = np.hanning(2 * Nf + 1)
    else:
        raise ValueError(f"Unknown filter kind '{kind}'. "
                         "Choose 'box', 'gaussian' or 'hann'.")
    return g / g.sum()


def _toeplitz_filter_matrix(N_t, Nf, kind='gaussian'):
    """Banded symmetric Toeplitz filter matrix G ∈ ℝ^{N_t × N_t}."""
    g = _filter_kernel(Nf, kind)
    G = np.zeros((N_t, N_t))
    for k in range(-Nf, Nf + 1):
        rows = np.arange(max(0, -k), min(N_t, N_t - k))
        G[rows, rows + k] = g[k + Nf]
    return G


def spod_sieber(Q, Nf, kind='gaussian'):
    r"""Sieber spectral POD.

    Filters the temporal correlation matrix,
    $\tilde{\mathbf{C}} = \mathbf{G}^\mathrm{T} \mathbf{C} \mathbf{G}$, then
    solves the same eigenvalue problem as snapshot POD. ``Nf=0`` recovers standard
    snapshot POD exactly.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    Nf : int
        Filter half-width (0 recovers POD; $N_t/2$ approaches the DFT).
    kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_t,)$.
    Psi : np.ndarray
        Spatial modes with orthonormal columns, shape $(N_x, N_t)$.
    Phi : np.ndarray
        Temporal SPOD coefficients, shape $(N_t, N_t)$.
    C_tilde : np.ndarray
        Filtered correlation matrix, shape $(N_t, N_t)$.

    References
    ----------
    Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal
    decomposition. *J. Fluid Mech.*, 792, 798–828.
    """
    N_x, N_t = Q.shape
    C        = (Q.T @ Q) / N_t
    if Nf == 0:
        C_tilde = C
    else:
        G       = _toeplitz_filter_matrix(N_t, Nf, kind)
        C_tilde = G.T @ C @ G
    lam, A    = np.linalg.eigh(C_tilde)
    idx       = lam.argsort()[::-1]
    lam, A    = lam[idx], A[:, idx]
    safe_lam  = np.where(lam > 0, lam, np.inf)
    Psi       = Q @ A / (np.sqrt(N_t) * np.sqrt(safe_lam))
    Psi[:, lam <= 0] = 0.0
    Phi       = Psi.T @ Q
    Sigma     = np.sqrt(np.where(lam > 0, lam, 0.0))
    return Sigma, Psi, Phi, C_tilde


# ─────────────────────────────────────────────────────────────────────────────
# SPOD — Towne, Schmidt & Colonius (JFM 2018)
# ─────────────────────────────────────────────────────────────────────────────

def spod_towne(Q, dt=1.0, n_fft=None, n_ovlp=None, window='hamming',
               weight=None, conf_level=0.95):
    r"""Spectral POD via Welch-averaged cross-spectral density (Towne et al., 2018).

    Estimates the cross-spectral density matrix at each frequency via Welch's
    method, then solves a per-frequency eigenvalue problem.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    dt : float
        Time step.
    n_fft : int, optional
        Block/FFT length. Default $2^{\lfloor \log_2 (N_t / 10) \rfloor}$.
    n_ovlp : int, optional
        Block overlap. Default ``n_fft // 2``.
    window : str or np.ndarray
        Window name or array of length ``n_fft``.
    weight : np.ndarray, optional
        Spatial integration weights, shape $(N_x,)$.
    conf_level : float
        Confidence level for the chi-squared intervals.

    Returns
    -------
    L : np.ndarray
        Modal energy spectrum, shape ``(n_freq, n_blks)``.
    Psi : np.ndarray
        Complex SPOD spatial modes, shape ``(n_freq, N_x, n_blks)``.
    f : np.ndarray
        Frequency vector, shape ``(n_freq,)``.
    Lc : np.ndarray
        Confidence intervals ``[lower, upper]``, shape ``(n_freq, n_blks, 2)``.
    info : dict
        Effective ``n_fft``, ``n_ovlp``, ``n_blks`` and window used.

    References
    ----------
    Towne, Schmidt & Colonius (2018). Spectral proper orthogonal decomposition and
    its relationship to dynamic mode decomposition and resolvent analysis.
    *J. Fluid Mech.*, 847, 821–867.
    """
    N_x, N_t = Q.shape
    is_real  = np.isrealobj(Q)

    if n_fft is None:
        n_fft = int(2 ** np.floor(np.log2(N_t / 10)))
    if n_ovlp is None:
        n_ovlp = n_fft // 2
    if n_ovlp >= n_fft:
        raise ValueError('n_ovlp must be < n_fft.')

    if isinstance(window, str):
        win = get_window(window, n_fft)
    else:
        win = np.asarray(window, dtype=float)
        if win.size != n_fft:
            raise ValueError(f'window length ({win.size}) must equal n_fft ({n_fft}).')
    win_norm = 1.0 / win.mean()
    win_col  = win[:, None]

    W = np.ones(N_x) if weight is None else np.asarray(weight, dtype=float).ravel()
    if W.size != N_x:
        raise ValueError('weight must have length N_x.')

    n_step = n_fft - n_ovlp
    n_blks = int(np.floor((N_t - n_ovlp) / n_step))
    if n_blks < 2:
        raise ValueError(
            f'Too few blocks ({n_blks}). Reduce n_fft/n_ovlp or use more snapshots.')

    if is_real:
        n_freq = n_fft // 2 + 1
        f      = np.arange(n_freq) / (n_fft * dt)
    else:
        n_freq = n_fft
        f      = np.fft.fftfreq(n_fft, d=dt)

    Q_hat = np.zeros((n_freq, N_x, n_blks), dtype=complex)
    for b in range(n_blks):
        i0     = b * n_step
        Q_blk  = Q[:, i0:i0 + n_fft]
        Q_fft  = np.fft.fft(Q_blk * win_col.T, axis=1) * win_norm / n_fft
        if is_real:
            Q_hat[:, :, b] = Q_fft[:, :n_freq].T
        else:
            Q_hat[:, :, b] = Q_fft.T

    L   = np.zeros((n_freq, n_blks))
    Psi = np.zeros((n_freq, N_x, n_blks), dtype=complex)
    for k in range(n_freq):
        Qf         = Q_hat[k]
        M          = (Qf * W[:, None]).conj().T @ Qf / n_blks
        lam, Theta = np.linalg.eigh(M)
        idx        = lam.argsort()[::-1]
        lam        = np.abs(lam[idx]); Theta = Theta[:, idx]
        Psi[k]     = Qf @ Theta / (np.sqrt(lam) * np.sqrt(n_blks))
        if is_real and 0 < k < n_freq - 1:
            L[k] = 2.0 * lam
        else:
            L[k] = lam

    xi2_up = 2 * gammaincinv(1 - conf_level, n_blks)
    xi2_lo = 2 * gammaincinv(    conf_level, n_blks)
    Lc     = np.stack([L * 2 * n_blks / xi2_lo,
                       L * 2 * n_blks / xi2_up], axis=-1)
    info = dict(n_fft=n_fft, n_ovlp=n_ovlp, n_blks=n_blks,
                window=win, n_freq=n_freq)
    return L, Psi, f, Lc, info


def spod_towne_reconstruct(Psi, A_blk, n_fft, n_ovlp, N_t):  # noqa: ARG001
    """Stub — reconstruction from Towne SPOD not yet implemented."""
    raise NotImplementedError(
        "Full reconstruction (inverse SPOD) is not yet implemented. "
        "See Nekkanti & Schmidt (JFM 2021) for details.")


def print_spod_towne_summary(info):
    """Pretty-print SPOD-Towne parameter summary."""
    print('SPOD (Towne / Welch) parameters')
    print('────────────────────────────────')
    print(f'  Snapshots per block (n_fft)  : {info["n_fft"]}')
    print(f'  Block overlap (n_ovlp)       : {info["n_ovlp"]}')
    print(f'  Number of blocks             : {info["n_blks"]}')
    print(f'  Resolved frequencies         : {info["n_freq"]}')

