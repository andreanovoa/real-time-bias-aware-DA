"""
pod_spod.py
===========
POD and SPOD variants for fluid-dynamics data.

Algorithms
----------
snapshot_pod            : Snapshot POD (Sirovich 1987) — exact solver
snapshot_pod_randomized : Randomized snapshot POD (Halko et al. 2011) — fast, truncated
spod_sieber             : SPOD via filtered correlation matrix (Sieber et al., JFM 2016)
spod_towne              : SPOD via Welch cross-spectral density (Towne et al., JFM 2018)

Utilities
---------
prepare_data  : Build zero-mean Q from raw (N_t, Nx, Ny) fields; handles NaN masking
                and returns the matching to_grid() callable for visualization.

Convention following [Mendez 2023]
----------
All functions expect a data matrix  Q : (N_x, N_t)
  - rows    --> spatial degrees of freedom  (flattened, NaN-masked, etc.)
  - columns --> time snapshots
Q must be zero-mean (subtract temporal mean before passing).

Decomposition:   D ≈  sum_k  Sigma_k  Psi_k(x)  Phi_k(t)

  Psi   [N_x, N_modes]   spatial modes  (orthonormal columns)
  Phi   [N_t, N_modes]   temporal coefficients  (Phi[:, k] = Psi_k · Q)
  Sigma [N_modes]        singular values  (Sigma_k = sqrt(lambda_k))

References
----------
[Sirovich 1987]   Sirovich, L., Quart. Appl. Math., XLV(3), 561-590.
[Sieber 2016]     Sieber, M., Paschereit, C. O. & Oberleithner, K., JFM 792, 798-828.
[Towne 2018]      Towne, A., Schmidt, O. T. & Colonius, T., JFM 847, 821-867.
[Mendez 2023]     Mendez, M. A., Generalised and Multiscale Modal Analysis, in
                  Data-Driven Fluid Mechanics: Combining First Principles and
                  Machine Learning, Cambridge University Press, pp. 153-181.
"""

import numpy as np
from scipy.signal import get_window


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Snapshot POD  (Sirovich 1987)
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_pod(Q):
    """
    Snapshot POD (Sirovich 1987).

    Solves the *temporal* eigenvalue problem (cheaper when N_t << N_x):
        C a_k = lambda_k a_k,   C = Q^T Q / N_t
    then recovers spatial modes via the snapshot formula:
        Psi_k = Q a_k / (sqrt(N_t) * sqrt(lambda_k))

    Naming follows the convention in [Mendez 2023]:
        D ≈  sum_k  Sigma_k  Psi_k(x)  Phi_k(t)

    Parameters
    ----------
    Q : ndarray  (N_x, N_t)   Zero-mean data matrix.

    Returns
    -------
    Sigma : ndarray (N_t,)      Singular values, descending  (Sigma_k = sqrt(lambda_k)).
    Psi   : ndarray (N_x, N_t)  Spatial modes (columns), orthonormal.
    Phi   : ndarray (N_t, N_t)  Temporal coefficients  Phi[k, t] = Psi_k * q(t).
    C     : ndarray (N_t, N_t)  Temporal correlation matrix.
    """
    N_x, N_t = Q.shape
    C        = (Q.T @ Q) / N_t
    lam, A   = np.linalg.eigh(C)
    idx      = lam.argsort()[::-1]
    lam, A   = lam[idx], A[:, idx]
    safe_lam = np.where(lam > 0, lam, np.inf)   # avoid div-by-zero for zero/neg eigenvalues
    Psi      = Q @ A / (np.sqrt(N_t) * np.sqrt(safe_lam))
    Psi[:, lam <= 0] = 0.0
    Phi      = Psi.T @ Q
    Sigma    = np.sqrt(np.where(lam > 0, lam, 0.0))
    return Sigma, Psi, Phi, C


# ─────────────────────────────────────────────────────────────────────────────
# 1b.  Randomized Snapshot POD  (Halko, Martinsson & Tropp 2011)
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_pod_randomized(Q, n_modes=20, n_iter=4, random_state=None):
    """
    Randomized snapshot POD (Halko, Martinsson & Tropp 2011).

    Applies a randomized SVD to Q directly instead of forming the full
    temporal correlation matrix.  Returns only the leading n_modes — much
    faster than the exact solver when n_modes << N_t and N_x is large.

    This is the algorithm used by Mendez 2023 (modulo_vki) with
    eig_solver='svd_sklearn_randomized'.  Results are approximate but
    statistically close to the exact decomposition for dominant modes.

    Algorithm
    ---------
    Q  ≈  U  S  V^T          (truncated randomized SVD)
    Psi   = U                 (N_x, n_modes)  left singular vectors
    Sigma = S / sqrt(N_t)     singular values in snapshot-POD sense
    Phi   = Psi^T Q           (n_modes, N_t)  temporal coefficients

    Relationship to exact snapshot_pod
    -----------------------------------
    lam_k  = Sigma_k**2  =  s_k**2 / N_t   (eigenvalue of C = Q^T Q / N_t)
    Psi_k  = U_k                             (same up to numerical noise & sign)
    Phi_k  = Psi_k^T Q                       (same formula)

    Parameters
    ----------
    Q            : ndarray (N_x, N_t)   Zero-mean data matrix.
    n_modes      : int                  Number of leading modes to compute.
                                        Default: 20.
    n_iter       : int                  Power-iteration steps for accuracy.
                                        Higher --> more accurate, slower.
                                        Default: 4.
    random_state : int | None           Seed for reproducibility.

    Returns
    -------
    Sigma : ndarray (n_modes,)      Singular values, descending.
    Psi   : ndarray (N_x, n_modes)  Spatial modes (columns), approximately
                                    orthonormal.
    Phi   : ndarray (n_modes, N_t)  Temporal coefficients.

    Notes
    -----
    * C is not returned (never explicitly formed).
    * Sign of each mode is arbitrary (same ambiguity as exact POD).
    * Requires scikit-learn; falls back to truncated numpy SVD if unavailable.

    References
    ----------
    Halko, N., Martinsson, P. G. & Tropp, J. A. (2011).
        Finding structure with randomness: probabilistic algorithms for
        constructing approximate matrix decompositions.
        SIAM Review, 53(2), 217-288.
    """
    N_x, N_t = Q.shape
    n_modes  = min(n_modes, N_x, N_t)

    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]
        U, s, _ = randomized_svd(Q, n_components=n_modes,
                                 n_iter=n_iter,
                                 random_state=random_state)
    except ImportError:
        # fallback: truncated deterministic SVD (exact but O(min(N_x,N_t)^2 * max))
        U_full, s_full, _ = np.linalg.svd(Q, full_matrices=False)
        U, s = U_full[:, :n_modes], s_full[:n_modes]

    Psi   = U                                   # (N_x, n_modes)
    Sigma = s / np.sqrt(N_t)                    # snapshot-POD singular values
    Phi   = Psi.T @ Q                           # (n_modes, N_t)
    return Sigma, Psi, Phi


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPOD — Sieber, Paschereit & Oberleithner (JFM 2016)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_kernel(Nf, kind='gaussian'):
    """
    Symmetric normalised 1-D low-pass filter of half-width Nf.

    Parameters
    ----------
    Nf   : int   Half-width (full length = 2*Nf + 1).  Nf=0 --> identity.
    kind : str   'box' | 'gaussian' | 'hann'

    Returns
    -------
    g : ndarray (2*Nf+1,)
    """
    if Nf == 0:
        return np.array([1.0])
    k = np.arange(-Nf, Nf + 1, dtype=float)
    if kind == 'box':
        g = np.ones(2 * Nf + 1)
    elif kind == 'gaussian':
        sigma = Nf / 2.0
        g = np.exp(-0.5 * (k / sigma) ** 2)
    elif kind == 'hann':
        g = np.hanning(2 * Nf + 1)
    else:
        raise ValueError(f"Unknown filter kind '{kind}'. Choose 'box', 'gaussian' or 'hann'.")
    return g / g.sum()


def _toeplitz_filter_matrix(N_t, Nf, kind='gaussian'):
    """
    Banded symmetric Toeplitz filter matrix G ∈ ℝ^{N_t × N_t}.

    G[i, j] = g[i - j]  for |i - j| ≤ Nf, else 0.
    """
    g = _filter_kernel(Nf, kind)
    G = np.zeros((N_t, N_t))
    for k in range(-Nf, Nf + 1):
        gk   = g[k + Nf]
        rows = np.arange(max(0, -k), min(N_t, N_t - k))
        cols = rows + k
        G[rows, cols] = gk
    return G


def spod_sieber(Q, Nf, kind='gaussian'):
    """
    Spectral POD — Sieber, Paschereit & Oberleithner (JFM 2016).

    Replaces the snapshot correlation matrix C with a filtered version
        C̃ = G^T C G
    where G is a banded symmetric Toeplitz matrix built from a low-pass
    filter kernel {g_k}.  All subsequent steps are identical to snapshot POD.

    Setting Nf=0 (G = I) exactly recovers standard snapshot POD.

    Parameters
    ----------
    Q    : ndarray (N_x, N_t)  Zero-mean data matrix.
    Nf   : int                 Filter half-width (0 = snapshot POD).
    kind : str                 'box' | 'gaussian' | 'hann'

    Returns
    -------
    Sigma   : ndarray (N_t,)      SPOD singular values, descending  (Sigma_k = sqrt(lambda_k)).
    Psi     : ndarray (N_x, N_t)  SPOD spatial modes (columns), orthonormal.
    Phi     : ndarray (N_t, N_t)  Temporal SPOD coefficients  Phi[k, t].
    C_tilde : ndarray (N_t, N_t)  Filtered correlation matrix.
    """
    N_x, N_t  = Q.shape
    C         = (Q.T @ Q) / N_t
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
# 3.  SPOD — Towne, Schmidt & Colonius (JFM 2018)   [Welch / CSD method]
# ─────────────────────────────────────────────────────────────────────────────

def spod_towne(Q, dt=1.0, n_fft=None, n_ovlp=None, window='hamming', weight=None,
               conf_level=0.95):
    """
    Spectral POD — Towne, Schmidt & Colonius (JFM 2018).

    Estimates the cross-spectral density (CSD) matrix at each resolved
    frequency via Welch's method (overlapping windowed blocks + FFT), then
    solves a per-frequency eigenvalue problem.  SPOD modes at frequency f_k
    are the leading eigenvectors of the CSD matrix at that frequency.

    This is a faithful Python translation of Schmidt & Towne's spod.m
    (https://github.com/SpectralPOD/spod_matlab).

    Algorithm
    ---------
    For each block b (b = 1 … n_blks):
        1. Extract block  Q_b = Q[:, idx_b]                (N_x × n_fft)
        2. Apply window   Q_b ← Q_b * w                    (broadcast)
        3. FFT            Q̂_b = FFT(Q_b, axis=1) / n_fft

    For each frequency bin k:
        4. Collect FFT realisations  Q̂_f  ∈  ℂ^{N_x × n_blks}
        5. CSD matrix     M = (Q̂_f)^H  (W ⊙ Q̂_f) / n_blks    (n_blks × n_blks)
        6. Eigendecomp    M Theta = Theta Lambda
        7. SPOD modes     Psi = Qhat_f Theta / (sqrt(Lambda) sqrt(n_blks))   (N_x x n_blks)

    Parameters
    ----------
    Q          : ndarray (N_x, N_t)  Zero-mean data matrix (real or complex).
    dt         : float               Time step (sets physical frequency axis).
    n_fft      : int | None          Block/FFT length.  Default: 2^floor(log2(N_t/10)).
    n_ovlp     : int | None          Block overlap.  Default: n_fft // 2.
    window     : str | ndarray       Window name (scipy.signal.get_window) or
                                     array of length n_fft.  Default: 'hamming'.
    weight     : ndarray (N_x,) | None  Spatial integration weights for the
                                     inner product.  Default: uniform.
    conf_level : float               Confidence level for chi-squared intervals
                                     (0.95 --> 95%).

    Returns
    -------
    L    : ndarray (n_freq, n_blks)          Modal energy spectrum  (= Sigma_k(f)^2, real).
    Psi  : ndarray (n_freq, N_x, n_blks)     Complex SPOD spatial modes.
                                              Visualise as  real(Psi[k] * exp(2*pi*i f[k] t)).
    f    : ndarray (n_freq,)                 Frequency vector.
    Lc   : ndarray (n_freq, n_blks, 2)       Confidence interval [lower, upper].
    info : dict                              n_fft, n_ovlp, n_blks, window used.

    Notes
    -----
    * For real-valued Q a one-sided (positive-frequency) spectrum is returned
      and energies at interior frequencies are doubled (consistent with Welch).
    * The CSD matrix is formed in the *small* space (n_blks × n_blks), which
      is efficient when n_blks << N_x.
    * SPOD modes at each frequency are mutually orthogonal with respect to the
      weight W: ‖Ψ_j^H W Ψ_k‖ = δ_{jk}.
    """
    from scipy.special import gammaincinv

    N_x, N_t = Q.shape
    is_real  = np.isrealobj(Q)

    # ── default parameters ────────────────────────────────────────────────
    if n_fft is None:
        n_fft = int(2 ** np.floor(np.log2(N_t / 10)))
    if n_ovlp is None:
        n_ovlp = n_fft // 2
    if n_ovlp >= n_fft:
        raise ValueError('n_ovlp must be < n_fft.')

    # ── window ────────────────────────────────────────────────────────────
    if isinstance(window, str):
        win = get_window(window, n_fft)
    else:
        win = np.asarray(window, dtype=float)
        if win.size != n_fft:
            raise ValueError(f'window length ({win.size}) must equal n_fft ({n_fft}).')
    win_norm  = 1.0 / win.mean()        # amplitude normalisation (matches MATLAB)
    win_col   = win[:, None]            # (n_fft, 1) for broadcasting

    # ── spatial weights ───────────────────────────────────────────────────
    if weight is None:
        W = np.ones(N_x)
    else:
        W = np.asarray(weight, dtype=float).ravel()
        if W.size != N_x:
            raise ValueError('weight must have length N_x.')

    # ── block layout ──────────────────────────────────────────────────────
    n_step  = n_fft - n_ovlp
    n_blks  = int(np.floor((N_t - n_ovlp) / n_step))
    if n_blks < 2:
        raise ValueError(
            f'Too few blocks ({n_blks}). Reduce n_fft or n_ovlp, or use more snapshots.')

    # ── frequency axis ────────────────────────────────────────────────────
    if is_real:
        n_freq = n_fft // 2 + 1
        f      = np.arange(n_freq) / (n_fft * dt)
    else:
        n_freq = n_fft
        f      = np.fft.fftfreq(n_fft, d=dt)

    # ── Step 1–3: compute all FFT blocks ─────────────────────────────────
    # Q_hat : (n_freq, N_x, n_blks)  complex
    Q_hat = np.zeros((n_freq, N_x, n_blks), dtype=complex)

    for b in range(n_blks):
        i0      = b * n_step
        i1      = i0 + n_fft
        Q_blk   = Q[:, i0:i1]                         # (N_x, n_fft)
        Q_win   = Q_blk * win_col.T                    # (N_x, n_fft) windowed
        Q_fft   = np.fft.fft(Q_win, axis=1)           # (N_x, n_fft)
        Q_fft  *= win_norm / n_fft
        if is_real:
            Q_hat[:, :, b] = Q_fft[:, :n_freq].T      # (n_freq, N_x)
        else:
            Q_hat[:, :, b] = Q_fft.T                  # (n_freq, N_x)

    # ── Step 4–7: per-frequency eigenvalue problem ────────────────────────
    L   = np.zeros((n_freq, n_blks))
    Psi = np.zeros((n_freq, N_x, n_blks), dtype=complex)

    for k in range(n_freq):
        Qf   = Q_hat[k]                                # (N_x, n_blks)
        # CSD in small (block) space
        M    = (Qf * W[:, None]).conj().T @ Qf / n_blks   # (n_blks, n_blks)
        lam, Theta = np.linalg.eigh(M)
        idx  = lam.argsort()[::-1]
        lam  = lam[idx]; Theta = Theta[:, idx]
        lam  = np.abs(lam)                              # numerical safety
        # SPOD spatial modes (N_x × n_blks)
        psi_k   = Qf @ Theta / (np.sqrt(lam) * np.sqrt(n_blks))
        Psi[k]  = psi_k
        # one-sided energy doubling for real data (matches Welch convention)
        if is_real and 0 < k < n_freq - 1:
            L[k] = 2.0 * lam
        else:
            L[k] = lam

    # ── Confidence intervals (chi-squared, same as MATLAB) ───────────────
    xi2_up  = 2 * gammaincinv(1 - conf_level, n_blks)
    xi2_lo  = 2 * gammaincinv(    conf_level, n_blks)
    Lc      = np.stack([L * 2 * n_blks / xi2_lo,
                        L * 2 * n_blks / xi2_up], axis=-1)  # (n_freq, n_blks, 2)

    info = dict(n_fft=n_fft, n_ovlp=n_ovlp, n_blks=n_blks,
                window=win, n_freq=n_freq)
    return L, Psi, f, Lc, info


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(fields, subtract_mean=True):
    """
    Build the zero-mean data matrix Q from raw snapshot fields,
    automatically detecting and removing NaN-masked solid-body points.

    This is the standard pre-processing step before any POD or SPOD call.
    NaN values in the raw data are assumed to mark a solid body interior
    (e.g. a cylinder); they are excluded from Q so that no NaN propagates
    through the decomposition.

    Parameters
    ----------
    fields : ndarray (N_t, Nx, Ny)  or  list / tuple of such arrays
        Raw snapshot data on a structured 2-D grid.
        NaN values mark solid-body (or otherwise invalid) grid points.
        The NaN pattern must be the same for every field and every snapshot.
        Pass a list to stack multiple fields (e.g. [ux, uy]) into a single
        Q with  N_fluid * n_fields  rows.
    subtract_mean : bool
        If True (default), subtract the temporal mean row-wise so that Q
        is zero-mean in time — required by all POD/SPOD algorithms here.

    Returns
    -------
    Q          : ndarray (N_fluid * n_fields, N_t)
                 Zero-mean data matrix ready to pass to snapshot_pod,
                 snapshot_pod_randomized, spod_sieber, or spod_towne.
    fluid_mask : ndarray (Nx, Ny)  bool
                 True at fluid (valid) grid points, False at solid body.
    to_grid    : callable  (N_fluid,) --> (Nx, Ny)
                 Re-embeds a mode or field vector (N_fluid real or complex
                 values) back onto the full 2-D mesh, placing NaN at solid
                 body points.  Use this for every mode you want to plot.

    Examples
    --------
    Single field::

        Q, mask, to_grid = prepare_data(ux_raw)
        Sigma, Psi, Phi, C = snapshot_pod(Q)
        mode1 = to_grid(Psi[:, 0])   # (Nx, Ny) with NaN at body

    Multiple stacked fields::

        Q, mask, to_grid = prepare_data([ux_raw, uy_raw])
        # Q has 2*N_fluid rows; to_grid works on each N_fluid-sized slice
        N_fluid = mask.sum()
        ux_mode1 = to_grid(Psi[:N_fluid,      0])
        uy_mode1 = to_grid(Psi[N_fluid:2*N_fluid, 0])
    """
    # normalise input to a list
    if isinstance(fields, np.ndarray):
        fields = [fields]

    ref = fields[0]
    if ref.ndim != 3:
        raise ValueError(
            f'Each field must be 3-D (N_t, Nx, Ny); got shape {ref.shape}.')
    N_t, Nx, Ny = ref.shape

    # build mask from first snapshot of first field
    fluid_mask = ~np.isnan(ref[0])          # (Nx, Ny)  True = fluid
    flat_mask  = fluid_mask.ravel()         # (Nx*Ny,)

    # flatten, mask, and stack each field
    rows = []
    for fld in fields:
        flat = fld.reshape(N_t, -1)[:, flat_mask].T    # (N_fluid, N_t)
        rows.append(flat)
    Q = np.vstack(rows)                                 # (N_fluid*n_fields, N_t)

    if subtract_mean:
        Q -= Q.mean(axis=1, keepdims=True)

    def to_grid(vec):
        """(N_fluid,) --> (Nx, Ny) with NaN at solid-body points."""
        g = np.full(Nx * Ny, np.nan)
        g[flat_mask] = np.real(vec)
        return g.reshape(Nx, Ny)

    return Q, fluid_mask, to_grid


def energy_fraction(Sigma):
    """
    Relative energy fraction and cumulative energy for a POD/SPOD result.

    Parameters
    ----------
    Sigma : ndarray   Singular values returned by snapshot_pod or spod_sieber.
                      Energies are  Sigma**2  (eigenvalues); the function squares
                      internally so you pass Sigma directly.

    Returns
    -------
    rel : ndarray   Relative energy per mode  (sums to 1).
    cum : ndarray   Cumulative relative energy.
    """
    lam   = Sigma ** 2
    total = lam.sum()
    rel   = lam / total
    cum   = np.cumsum(rel)
    return rel, cum


def spod_towne_reconstruct(Psi, A_blk, n_fft, n_ovlp, N_t):
    """
    Reconstruct the time series from Towne SPOD modes and block coefficients.

    Parameters
    ----------
    Psi    : (n_freq, N_x, n_modes)  Complex SPOD spatial modes.
    A_blk  : (n_freq, n_modes, n_blks) Block expansion coefficients.
    n_fft  : int
    n_ovlp : int
    N_t    : int  Length of original time series.

    Returns
    -------
    Q_rec : (N_x, N_t)  Reconstructed (real) field.
    """
    raise NotImplementedError(
        "Full reconstruction (inverse SPOD) is not yet implemented. "
        "See Nekkanti & Schmidt (JFM 2021) for details."
    )


def print_spod_towne_summary(info):
    """Pretty-print SPOD-Towne parameter summary (mirrors MATLAB console output)."""
    print('SPOD (Towne / Welch) parameters')
    print('────────────────────────────────')
    print(f'  Snapshots per block (n_fft)  : {info["n_fft"]}')
    print(f'  Block overlap (n_ovlp)       : {info["n_ovlp"]}')
    print(f'  Number of blocks             : {info["n_blks"]}')
    print(f'  Resolved frequencies         : {info["n_freq"]}')