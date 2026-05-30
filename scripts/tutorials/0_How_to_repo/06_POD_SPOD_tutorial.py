"""
06_POD_SPOD_tutorial.py
=======================

Tutorial: POD and SPOD classes
-------------------------------
Demonstrates the ``POD`` and ``SPOD`` classes from ``tools``,
applied to DNS data of the wake past a circular cylinder at Re = 100.

Contents
--------
  1. Setup and data loading
  2. Fit POD (exact solver)
  3. Energy spectrum
  4. Spatial modes
  5. Temporal coefficients
  6. Encode / decode / reconstruct
  7. SPOD — Sieber 2016 (same API, one extra parameter)
  8. POD vs SPOD comparison
  9. Towne SPOD (Welch / per-frequency eigenproblem)

Run from the tutorials directory::

    python 06_POD_SPOD_tutorial.py

or open in VS Code / Spyder as a "percent-cell" script (cells separated by ``# %%``).
"""

# %% ── 1. Setup ───────────────────────────────────────────────────────────────
import sys, os

# make src/ importable when running from tutorials/
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if os.path.join(_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat

from utils import set_working_directories

# ── POD / SPOD classes ────────────────────────────────────────────────────────
from tools import POD, SPOD
# ── utility functions ─────────────────────────────────────────────────────────
from tools import prepare_data, energy_fraction, spod_towne, print_spod_towne_summary


# %% ── 2. Load data ───────────────────────────────────────────────────────────
data_folder = set_working_directories('wakes')[0]

mat    = loadmat(os.path.join(data_folder, 'circle_re_100.mat'))
ux_raw = mat['ux']   # (N_t, Nx, Ny) — NaN marks cylinder body interior
uy_raw = mat['uy']
N_t, Nx, Ny = ux_raw.shape
print(f'Snapshots  : N_t = {N_t}')
print(f'Grid       : {Nx} × {Ny}')

# prepare_data:
#   • detects the NaN mask (cylinder interior) from the first snapshot
#   • flattens fluid points to (N_fluid, N_t) and subtracts the temporal mean
#   • returns to_grid() for re-embedding flat vectors onto the 2-D mesh
#
# Stacking [ux, uy] gives modes that capture both velocity components at once.
Q, fluid_mask, to_grid = prepare_data([ux_raw, uy_raw], subtract_mean=True)
N_fluid = fluid_mask.sum()

print(f'\nData matrix Q : {Q.shape}   (2 × {N_fluid} fluid pts, {N_t} snapshots)')
print(f'Cylinder body : {(~fluid_mask).sum()} NaN pts excluded')

# %% ── quick snapshot ─────────────────────────────────────────────────────────
snap = np.where(~fluid_mask, np.nan,
                ux_raw[N_t // 2] - np.nanmean(ux_raw, axis=0))

fig, ax = plt.subplots(figsize=(11, 3.5))
vmax = np.nanpercentile(np.abs(snap), 99)
im   = ax.pcolormesh(snap.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label="$u'_x$")
ax.set_aspect('equal')
ax.set_title(f"Streamwise velocity fluctuation — snapshot {N_t//2}/{N_t}")
plt.tight_layout(); plt.show()


# %% ── 3. Fit POD ─────────────────────────────────────────────────────────────
#
# POD  —  sklearn-style interface
#
#   pod = POD(n_modes, method)   # method: 'exact' | 'randomized' (default)
#   pod.fit(Q)                   # learn modes from zero-mean Q (N_x, N_t)
#
# After fit():
#   pod.Psi    (N_x, N_modes)   spatial modes, orthonormal columns
#   pod.Phi    (N_modes, N_t)   temporal coefficients
#   pod.Sigma  (N_modes,)       singular values  Σ_k = √λ_k
#   pod.Q_mean (N_x, 1)         temporal mean
#
N_modes = 20

pod = POD(n_modes=N_modes, method='exact').fit(Q)

print(f'Fitted POD : {pod.N_modes} modes')
print(f'Psi   : {pod.Psi.shape}')
print(f'Phi   : {pod.Phi.shape}')
print(f'Sigma : {pod.Sigma.shape}')
print(f'Leading singular values: {pod.Sigma[:6].round(4)}')


# %% ── 4. Energy spectrum ─────────────────────────────────────────────────────
rel, cum = pod.energy_fraction()
print(f'Mode 1     : {rel[0]*100:.2f}% energy')
print(f'Modes 1–2  : {cum[1]*100:.2f}%')
print(f'Modes 1–4  : {cum[3]*100:.2f}%')

POD.plot_spectrum(pod, max_mode=N_modes)
plt.suptitle('POD energy spectrum — cylinder wake Re = 100', fontsize=12, y=1.01)
plt.show()

# Modes 1 & 2 form a conjugate pair (equal energy) = Kármán vortex shedding.
# Modes 3 & 4 capture the first harmonic.


# %% ── 5. Spatial modes ───────────────────────────────────────────────────────
# Psi rows: first N_fluid = ux component, next N_fluid = uy component
Psi_ux = pod.Psi[:N_fluid, :]    # (N_fluid, N_modes)
Psi_uy = pod.Psi[N_fluid:, :]

n_show = 4
fig = plt.figure(figsize=(15, 6))
gs  = gridspec.GridSpec(2, n_show, hspace=0.45, wspace=0.25)

for k in range(n_show):
    for row, Psi_c, label in [(0, Psi_ux, '$u_x$'), (1, Psi_uy, '$u_y$')]:
        mode_2d = to_grid(Psi_c[:, k])
        vmax    = np.nanpercentile(np.abs(mode_2d), 98)
        ax      = fig.add_subplot(gs[row, k])
        im      = ax.pcolormesh(mode_2d.T, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(f'Mode {k+1}  ({rel[k]*100:.1f}%)  {label}', fontsize=9)
        ax.set_aspect('equal')

fig.suptitle('POD spatial modes — cylinder wake Re = 100', fontsize=12)
plt.show()


# %% ── 6. Temporal coefficients ───────────────────────────────────────────────
POD.plot_time_coefficients(pod, num_modes=10)
plt.suptitle('Temporal coefficient matrix Φ (first 10 modes)', fontsize=11, y=1.01)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(pod.Phi[0], lw=0.8, label='$\\phi_1$')
axes[0].plot(pod.Phi[1], lw=0.8, ls='--', label='$\\phi_2$')
axes[0].set(xlabel='Snapshot', ylabel='Amplitude',
            title='Temporal coefficients — modes 1 & 2')
axes[0].legend()

axes[1].plot(pod.Phi[0], pod.Phi[1], lw=0.5, color='steelblue')
axes[1].set(xlabel='$\\phi_1$', ylabel='$\\phi_2$',
            title='Phase portrait $(\\phi_1, \\phi_2)$ — limit cycle',
            aspect='equal')

freqs = np.fft.rfftfreq(N_t)
for k in range(4):
    axes[2].semilogy(freqs, np.abs(np.fft.rfft(pod.Phi[k])),
                     lw=1.2, alpha=0.85, label=f'mode {k+1}')
axes[2].set_xlim(0, 0.35)
axes[2].set(xlabel='Normalised frequency', ylabel='|FFT|',
            title='Spectral content of $\\phi_k(t)$')
axes[2].legend(fontsize=8)

plt.suptitle('POD temporal coefficients — cylinder wake Re = 100', fontsize=11)
plt.tight_layout(); plt.show()


# %% ── 7. Encode / decode / reconstruct ──────────────────────────────────────
#
#   Z     = pod.encode(Q)             (N_modes, N_t)   latent representation
#   Q_hat = pod.decode(Z)             (N_x, N_t)       state-space reconstruction
#   Q_hat = pod.reconstruct(Q)        round-trip  encode → decode
#   Q_hat = pod.reconstruct(Q, n_modes=r)  truncated reconstruction with r modes
#   mse   = pod.score(Q)              mean squared reconstruction error
#
Z     = pod.encode(Q)
Q_hat = pod.decode(Z)

print(f'Latent Z    : {Z.shape}')
print(f'Q_hat       : {Q_hat.shape}')
print(f'encode(Q) == Phi: {np.allclose(Z, pod.Phi, atol=1e-10)}')
print(f'MSE ({N_modes} modes): {pod.score(Q):.2e}')

# Reconstruction error vs number of modes retained
mode_range = [1, 2, 4, 6, 8, 10, 15, 20]
mse_list   = [pod.score(pod.reconstruct(Q, n_modes=r)) for r in mode_range]

snap_idx = N_t // 2
Q_r2     = pod.reconstruct(Q, n_modes=2)
vmax     = np.nanpercentile(np.abs(to_grid(Q[:N_fluid, snap_idx])), 98)

fig = plt.figure(figsize=(16, 4))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.2], wspace=0.35)

# left: MSE vs modes
ax_mse = fig.add_subplot(gs[0])
ax_mse.semilogy(mode_range, mse_list, 'o-', color='steelblue')
ax_mse.set(xlabel='Modes retained $r$', ylabel='Reconstruction MSE',
           title='Reconstruction error vs modes')

# right: three flow-field panels
gs_right = gs[1].subgridspec(1, 3, wspace=0.5)
for col, (data, title, cmap) in enumerate([
        (to_grid(Q[:N_fluid, snap_idx]),                                    'Original $u_x$', 'RdBu_r'),
        (to_grid(Q_r2[:N_fluid, snap_idx]),                                 '2-mode recon.',  'RdBu_r'),
        (to_grid(np.abs(Q[:N_fluid, snap_idx] - Q_r2[:N_fluid, snap_idx])), 'Absolute error', 'Reds'),
    ]):
    ax = fig.add_subplot(gs_right[col])
    kw = dict(vmin=-vmax, vmax=vmax) if cmap == 'RdBu_r' else dict(vmin=0, vmax=vmax)
    im = ax.pcolormesh(data.T, cmap=cmap, shading='auto', **kw)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')

fig.suptitle(f'Reconstruction — snapshot {snap_idx}', fontsize=12)
plt.show()


# %% ── 8. SPOD (Sieber 2016) ──────────────────────────────────────────────────
#
# SPOD has the same interface as POD — only `Nf` (filter half-width) is added.
#
#   spod = SPOD(Nf=Nf, filter_kind='gaussian').fit(Q)
#
# Internally, the correlation matrix C is replaced by:
#
#   C_tilde = G^T C G
#
# where G is a banded symmetric Toeplitz filter matrix.
# Setting Nf=0 recovers standard snapshot POD exactly.
# Setting Nf → N_t/2 recovers the DFT.
#
# A natural choice for Nf is half the dominant temporal scale (here, the
# Kármán shedding period).

fft_phi1 = np.abs(np.fft.rfft(pod.Phi[0]))
f_peak   = np.fft.rfftfreq(N_t)[np.argmax(fft_phi1)]
T_snaps  = int(round(1.0 / f_peak))
Nf       = T_snaps // 2

print(f'Dominant frequency : f = {f_peak:.4f}  (normalised)')
print(f'Shedding period    : ~{T_snaps} snapshots')
print(f'Filter half-width  : Nf = {Nf}')

spod = SPOD(Nf=Nf, filter_kind='gaussian', n_modes=N_modes).fit(Q)
print(f'\nSPOD fitted: {spod.N_modes} modes')

# SPOD reconstruction API is identical to POD
mse_spod = spod.score(Q)
print(f'SPOD MSE ({N_modes} modes): {mse_spod:.2e}')


# %% ── 9. POD vs SPOD comparison ──────────────────────────────────────────────
n_comp = 4
fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(2, n_comp, hspace=0.50, wspace=0.25)

for k in range(n_comp):
    for row, psi_src, label in [
            (0, pod.Psi[:N_fluid, :],  'POD'),
            (1, spod.Psi[:N_fluid, :], f'SPOD  $N_f$={Nf}'),
        ]:
        mode_2d = to_grid(psi_src[:, k])
        vmax    = np.nanpercentile(np.abs(mode_2d), 98)
        ax      = fig.add_subplot(gs[row, k])
        im      = ax.pcolormesh(mode_2d.T, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(f'{label}  mode {k+1}', fontsize=9)
        ax.set_aspect('equal')

fig.suptitle('POD vs SPOD — spatial modes ($u_x$) — cylinder wake Re = 100',
             fontsize=12, y=1.01)
plt.show()

# Quantitative: cosine similarity between POD and SPOD modes
cos_sim = np.array([
    abs(pod.Psi[:, k] @ spod.Psi[:, k]) /
    (np.linalg.norm(pod.Psi[:, k]) * np.linalg.norm(spod.Psi[:, k]))
    for k in range(N_modes)
])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(1, N_modes + 1), cos_sim, color='steelblue', edgecolor='white')
axes[0].axhline(1.0, lw=1, ls='--', color='gray')
axes[0].set(ylim=[0, 1.05], xlabel='Mode index', ylabel='|cos similarity|',
            title=f'Spatial agreement: POD vs SPOD ($N_f$={Nf})')

axes[1].semilogy(freqs, np.abs(np.fft.rfft(pod.Phi[0])),
                 lw=1.5, alpha=0.8, label='POD mode 1')
axes[1].semilogy(freqs, np.abs(np.fft.rfft(spod.Phi[0])),
                 lw=1.5, alpha=0.8, ls='--', label=f'SPOD mode 1 ($N_f$={Nf})')
axes[1].axvline(f_peak, color='tomato', lw=1.2, ls=':', label=f'$f_s$={f_peak:.3f}')
axes[1].set_xlim(0, 4 * f_peak)
axes[1].set(xlabel='Normalised frequency', ylabel='|FFT|',
            title='Spectral content of leading temporal coefficient')
axes[1].legend(fontsize=9)

plt.suptitle('POD vs SPOD — single-frequency wake', fontsize=12)
plt.tight_layout(); plt.show()

print('\nFor this single-frequency flow, SPOD reproduces POD modes with near-unity similarity.')
print('SPOD adds the most value for multi-frequency flows (harmonics, competing instabilities).')


# %% ── 10. Towne SPOD (Welch / per-frequency) ─────────────────────────────────
#
# spod_towne() performs a Welch block-average of the CSD matrix and solves
# a per-frequency eigenvalue problem (Towne, Schmidt & Colonius, JFM 2018).
# It is a standalone function — no class wrapper yet.
#
#   L, Psi_f, f, Lc, info = spod_towne(Q, dt=1.0)
#
#   L     (n_freq, n_blks)         modal energy spectrum
#   Psi_f (n_freq, N_x, n_blks)   complex SPOD spatial modes
#   f     (n_freq,)                frequency vector
#   Lc    (n_freq, n_blks, 2)      confidence intervals (lower, upper)
#   info  dict                     n_fft, n_ovlp, n_blks, n_freq

L, Psi_f, f_towne, Lc, info = spod_towne(Q, dt=1.0)
print_spod_towne_summary(info)

fig, ax = plt.subplots(figsize=(10, 4))
for m in range(info['n_blks']):
    ax.semilogy(f_towne, L[:, m], color='steelblue', lw=0.8, alpha=0.5)
ax.semilogy(f_towne, L[:, 0], color='steelblue', lw=1.5, label='Leading mode')
ax.axvline(f_peak, color='tomato', ls='--', lw=1.2, label=f'$f_s$={f_peak:.3f}')
ax.set(xlabel='Normalised frequency', ylabel='Modal energy',
       title='Towne SPOD energy spectrum — cylinder wake Re = 100',
       xlim=[0, 4 * f_peak])
ax.legend(); plt.tight_layout(); plt.show()

# Leading spatial mode at the shedding frequency
idx_fs       = np.argmin(np.abs(f_towne - f_peak))
leading_mode = Psi_f[idx_fs, :N_fluid, 0]   # complex mode, ux component

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, data, title in [
        (axes[0], to_grid(leading_mode.real), 'Re(mode 1)'),
        (axes[1], to_grid(leading_mode.imag), 'Im(mode 1)'),
    ]:
    vmax = np.nanpercentile(np.abs(data[~np.isnan(data)]), 98)
    im   = ax.pcolormesh(data.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Towne SPOD  {title}  at $f_s$={f_peak:.3f}', fontsize=10)
    ax.set_aspect('equal')

plt.suptitle('Towne SPOD leading mode — $u_x$ component', fontsize=12)
plt.tight_layout(); plt.show()


# %% ── Summary ────────────────────────────────────────────────────────────────
#
# POD / SPOD API
# ──────────────
#   from tools import POD, SPOD, prepare_data
#
#   Q, mask, to_grid = prepare_data([ux_raw, uy_raw])   # NaN mask + zero-mean
#
#   pod  = POD(n_modes=20, method='exact').fit(Q)       # or method='randomized'
#   spod = SPOD(Nf=Nf, filter_kind='gaussian').fit(Q)   # same API, one extra param
#
#   Z     = pod.encode(Q)            # (N_modes, N_t)
#   Q_hat = pod.decode(Z)            # (N_x, N_t)
#   Q_hat = pod.reconstruct(Q, n_modes=r)
#   mse   = pod.score(Q)
#
#   POD.plot_spectrum(pod)
#   POD.plot_time_coefficients(pod, num_modes=10)
#   mode_2d = to_grid(pod.Psi[:N_fluid, k])
#
# Towne SPOD (standalone)
# ───────────────────────
#   from tools import spod_towne, print_spod_towne_summary
#   L, Psi_f, f, Lc, info = spod_towne(Q, dt=1.0)
