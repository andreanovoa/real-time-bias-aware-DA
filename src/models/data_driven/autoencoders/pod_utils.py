"""
pod_utils.py
============
Standalone POD/SPOD decomposition algorithms (functions only). The linear
ROM classes that wrap these algorithms with a sklearn-style fit/encode/decode
interface (`POD`, `SPOD`) live alongside this module in
`romda.models.data_driven.autoencoders`.

Functions
---------
energy_fraction         : relative/cumulative energy from singular values
snapshot_pod            : Snapshot POD — exact solver (Sirovich 1987)
snapshot_pod_randomized : Randomized snapshot POD (Halko et al. 2011)
spod_sieber             : SPOD via filtered correlation matrix (Sieber et al. 2016)
spod_towne              : SPOD via Welch CSD (Towne et al. 2018)
spod_towne_reconstruct  : inverse SPOD [stub]
print_spod_towne_summary: pretty-print Towne SPOD parameters

Usage
-----
::

    from romda.models.data_driven.autoencoders import POD, SPOD

    pod = POD(n_modes=20).fit(Q)      # Q: (N_x, N_t) zero-mean-able data matrix
    Z   = pod.encode(Q)               # (N_modes, N_t) -- POD coefficients
    Q_r = pod.decode(Z)               # (N_x, N_t) -- reconstruction

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
    r"""
    Relative energy fraction and cumulative energy per mode.

    From the eigenvalues $\lambda_j = \Sigma_j^2$ of the (possibly filtered)
    temporal correlation matrix,

    $$
    \mathrm{rel}_j = \frac{\lambda_j}{\sum_k \lambda_k}, \qquad
    \mathrm{cum}_j = \sum_{k \le j} \mathrm{rel}_k.
    $$

    Parameters
    ----------
    Sigma : np.ndarray
        Singular values, as returned by `snapshot_pod`,
        `snapshot_pod_randomized` or `spod_sieber`.

    Returns
    -------
    rel : np.ndarray
        Relative energy per mode (sums to 1).
    cum : np.ndarray
        Cumulative relative energy.
    """
    lam = Sigma ** 2
    rel = lam / lam.sum()
    return rel, np.cumsum(rel)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition algorithms
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_pod(Q):
    r"""Snapshot POD — exact solver.

    Solves the eigenvalue problem of the temporal correlation matrix,

    $$
    \mathbf{C}\mathbf{A} = \mathbf{A}\,\mathrm{diag}(\boldsymbol{\lambda}),
    \qquad \mathbf{C} = \mathbf{Q}^\mathrm{T} \mathbf{Q} / N_t,
    \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge 0,
    $$

    and reconstructs the (large) spatial modes from the eigenvectors
    $\mathbf{A}$ of the (small, $N_t \times N_t$) matrix $\mathbf{C}$ — the
    "method of snapshots" of Sirovich (1987):

    $$
    \boldsymbol{\Psi} = \frac{1}{\sqrt{N_t}}\,
    \mathbf{Q}\mathbf{A}\,\mathrm{diag}(\boldsymbol{\lambda})^{-1/2},
    \qquad
    \boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q},
    \qquad
    \boldsymbol{\Sigma} = \sqrt{\boldsymbol{\lambda}}.
    $$

    $\boldsymbol{\Psi}$ has orthonormal columns and, for the modes with
    $\lambda_j > 0$, $\mathbf{Q} = \boldsymbol{\Psi}\boldsymbol{\Phi}$ exactly.
    Modes with $\lambda_j \le 0$ (numerical noise) are set to zero.

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

    Computes a truncated randomized SVD of $\mathbf{Q}$ (via
    ``sklearn.utils.extmath.randomized_svd``, falling back to a full
    ``numpy.linalg.svd`` if scikit-learn is unavailable),

    $$
    \mathbf{Q} \approx \mathbf{U}\,\mathbf{S}\,\mathbf{V}^\mathrm{T},
    $$

    and returns

    $$
    \boldsymbol{\Psi} = \mathbf{U}, \qquad
    \boldsymbol{\Sigma} = \mathbf{S} / \sqrt{N_t}, \qquad
    \boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q}.
    $$

    The $\boldsymbol{\Sigma} = \mathbf{S}/\sqrt{N_t}$ scaling matches the
    eigenvalue-based normalisation of `snapshot_pod`, since
    $\mathbf{Q}^\mathrm{T}\mathbf{Q}/N_t = \mathbf{V}(\mathbf{S}^2/N_t)\mathbf{V}^\mathrm{T}$.

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
    """
    Symmetric, unit-sum 1-D low-pass filter kernel of half-width Nf
    (length ``2*Nf + 1``), used to build the SPOD-Sieber filter matrix.

    Parameters
    ----------
    Nf : int
        Filter half-width. ``Nf=0`` returns the trivial kernel ``[1.0]``.
    kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``.

    Returns
    -------
    np.ndarray
        Normalised kernel of length ``2*Nf + 1`` (or 1 if ``Nf=0``).
    """
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
    """
    Build the banded symmetric Toeplitz filter matrix G (N_t x N_t) whose
    diagonals hold the kernel from `_filter_kernel`, used to low-pass
    filter the temporal correlation matrix in `spod_sieber`.

    Parameters
    ----------
    N_t : int
        Matrix size (number of snapshots).
    Nf : int
        Filter half-width (bandwidth ``2*Nf + 1``).
    kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``.

    Returns
    -------
    np.ndarray
        Filter matrix G, shape ``(N_t, N_t)``.
    """
    g = _filter_kernel(Nf, kind)
    G = np.zeros((N_t, N_t))
    for k in range(-Nf, Nf + 1):
        rows = np.arange(max(0, -k), min(N_t, N_t - k))
        G[rows, rows + k] = g[k + Nf]
    return G


def spod_sieber(Q, Nf, kind='gaussian'):
    r"""Sieber spectral POD — filtered correlation matrix.

    Low-pass filters the temporal correlation matrix
    $\mathbf{C} = \mathbf{Q}^\mathrm{T}\mathbf{Q}/N_t$ with the banded
    symmetric Toeplitz matrix $\mathbf{G}$ (see `_toeplitz_filter_matrix`,
    built from a normalised kernel of half-width ``Nf``, see
    `_filter_kernel`),

    $$
    \tilde{\mathbf{C}} = \mathbf{G}^\mathrm{T} \mathbf{C} \mathbf{G},
    $$

    then solves the same "method of snapshots" eigenvalue problem as
    `snapshot_pod`, with $\tilde{\mathbf{C}}$ in place of $\mathbf{C}$:

    $$
    \tilde{\mathbf{C}}\mathbf{A} = \mathbf{A}\,\mathrm{diag}(\boldsymbol{\lambda}),
    \qquad
    \boldsymbol{\Psi} = \frac{1}{\sqrt{N_t}}\,
    \mathbf{Q}\mathbf{A}\,\mathrm{diag}(\boldsymbol{\lambda})^{-1/2},
    \qquad
    \boldsymbol{\Phi} = \boldsymbol{\Psi}^\mathrm{T}\mathbf{Q},
    \qquad
    \boldsymbol{\Sigma} = \sqrt{\boldsymbol{\lambda}}.
    $$

    ``Nf=0`` skips the filtering step ($\tilde{\mathbf{C}} = \mathbf{C}$) and
    recovers standard snapshot POD exactly.

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

    Splits $\mathbf{Q}$ into ``n_blks`` overlapping blocks of length ``n_fft``
    (Welch's method, step ``n_fft - n_ovlp``), windows and Fourier-transforms
    each block, then — at every frequency — solves a "method of snapshots"
    eigenvalue problem across blocks (analogous to `snapshot_pod`, but with
    blocks in place of time snapshots) to avoid ever forming the full
    $N_x \times N_x$ cross-spectral density (CSD) matrix.

    For block $b = 0, \dots, n_\mathrm{blk}-1$ starting at snapshot
    $b\,(n_\mathrm{fft}-n_\mathrm{ovlp})$, the windowed block DFT at
    frequency $f_k$ is

    $$
    \hat{\mathbf{q}}^{(b)}_k = \frac{1}{n_\mathrm{fft}\,\bar{w}}
    \sum_{n=0}^{n_\mathrm{fft}-1} w[n]\,\mathbf{q}^{(b)}[n]\,
    e^{-\mathrm{i}2\pi kn/n_\mathrm{fft}}, \qquad \bar{w} = \mathrm{mean}(w),
    $$

    with $w$ the ``window``. Stacking the blocks,
    $\hat{\mathbf{Q}}_k = [\hat{\mathbf{q}}^{(1)}_k, \dots,
    \hat{\mathbf{q}}^{(n_\mathrm{blk})}_k] \in \mathbb{C}^{N_x \times n_\mathrm{blk}}$,
    the (weighted) cross-block Gram matrix and its eigendecomposition give the
    SPOD modes and modal energies at frequency $f_k$:

    $$
    \mathbf{M}_k = \frac{1}{n_\mathrm{blk}}\,
    \hat{\mathbf{Q}}_k^\mathrm{H}\, \mathbf{W}\, \hat{\mathbf{Q}}_k,
    \qquad
    \mathbf{M}_k \boldsymbol{\Theta}_k
    = \boldsymbol{\Theta}_k\, \mathrm{diag}(\boldsymbol{\lambda}_k),
    $$

    $$
    \boldsymbol{\Psi}_k = \frac{1}{\sqrt{n_\mathrm{blk}}}\,
    \hat{\mathbf{Q}}_k\, \boldsymbol{\Theta}_k\,
    \mathrm{diag}(\boldsymbol{\lambda}_k)^{-1/2},
    $$

    with $\mathbf{W} = \mathrm{diag}(\mathrm{weight})$ the spatial weight matrix
    ($\mathbb{I}$ by default). The modal energy spectrum is
    $L_k = \boldsymbol{\lambda}_k$, doubled ($L_k = 2\boldsymbol{\lambda}_k$) at
    interior frequency bins of a real, one-sided spectrum to account for the
    folded negative-frequency energy.

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
        Window name (passed to ``scipy.signal.get_window``) or an array of
        length ``n_fft``.
    weight : np.ndarray, optional
        Spatial integration weights $\mathrm{diag}(\mathbf{W})$, shape
        $(N_x,)$. Uniform weights (no integration) by default.
    conf_level : float
        Target confidence level for the chi-squared-based interval `Lc`
        (Welch's method, nominally $2\,n_\mathrm{blk}$ degrees of freedom).

    Returns
    -------
    L : np.ndarray
        Modal energy spectrum, shape ``(n_freq, n_blks)``.
    Psi : np.ndarray
        Complex SPOD spatial modes, shape ``(n_freq, N_x, n_blks)``.
    f : np.ndarray
        Frequency vector, shape ``(n_freq,)``.
    Lc : np.ndarray
        Nominal confidence bounds ``[lower, upper]`` for `L`, shape
        ``(n_freq, n_blks, 2)``. See Notes.
    info : dict
        Effective ``n_fft``, ``n_ovlp``, ``n_blks``, ``n_freq`` and window used.

    Notes
    -----
    `Lc` is computed from ``scipy.special.gammaincinv(1 - conf_level, n_blks)``
    and ``scipy.special.gammaincinv(conf_level, n_blks)``, intended as a
    chi-squared confidence interval in the spirit of Welch's method. However,
    `scipy.special.gammaincinv(a, y)` requires its second argument $y$
    (a probability) in $[0, 1]$, whereas here it is called with $y =$
    ``n_blks`` (an integer $\ge 2$); numerically this returns ``nan`` for the
    ``n_blks`` values produced by this function. Treat `Lc` as unverified
    until this is checked against the intended formula.

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
    """
    Reconstruct snapshots from Towne SPOD modes  [stub — not yet implemented].

    Parameters
    ----------
    Psi : np.ndarray
        SPOD spatial modes, as returned by `spod_towne`.
    A_blk : np.ndarray
        Block expansion coefficients.
    n_fft : int
        Block/FFT length used by `spod_towne`.
    n_ovlp : int
        Block overlap used by `spod_towne`.
    N_t : int
        Number of snapshots in the reconstructed series.

    Raises
    ------
    NotImplementedError
        Always — inverse SPOD is not yet implemented.

    References
    ----------
    Nekkanti & Schmidt (2021). Frequency-time analysis, low-rank reconstruction
    and denoising of turbulent flows using SPOD. *J. Fluid Mech.*, 926, A26.
    """
    raise NotImplementedError(
        "Full reconstruction (inverse SPOD) is not yet implemented. "
        "See Nekkanti & Schmidt (JFM 2021) for details.")


def print_spod_towne_summary(info):
    """
    Pretty-print the block/frequency parameters of a `spod_towne` run.

    Parameters
    ----------
    info : dict
        The ``info`` dictionary returned by `spod_towne` (must contain
        ``n_fft``, ``n_ovlp``, ``n_blks`` and ``n_freq``).
    """
    print('SPOD (Towne / Welch) parameters')
    print('────────────────────────────────')
    print(f'  Snapshots per block (n_fft)  : {info["n_fft"]}')
    print(f'  Block overlap (n_ovlp)       : {info["n_ovlp"]}')
    print(f'  Number of blocks             : {info["n_blks"]}')
    print(f'  Resolved frequencies         : {info["n_freq"]}')

