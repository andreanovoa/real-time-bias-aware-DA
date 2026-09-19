"""Visualize the assimilated predictions themselves (not just error metrics).

Runs the default experiments and plots the raw predictions in the repo's
standard colors (gray truth, teal model estimate, red observations):

- Lorenz63: the three state variables vs time — ensemble mean +/- 1 std band,
  with the assimilated observations overlaid.
- Lorenz96: spatiotemporal (time x variable) fields of truth, ensemble mean
  and spread, plus timeseries at a few observed/unobserved variables.
- Cylinder: assimilated sensor measurements, plus global quantities the
  POD-ESN should reconstruct (leading POD coefficients and modal TKE).

One PDF page per figure, saved to ``figs/predictions.pdf``.

Usage
-----
    python plot_predictions.py
"""

import inspect
import os

import main_cylinder as cyl
import main_lorenz63 as l63
import main_lorenz96 as l96
import matplotlib.pyplot as plt
import numpy as np
from common import FIGS_FOLDER
from matplotlib.backends.backend_pdf import PdfPages
from romda.data_assimilation import run_da_loop

# Repo palette (src/plotting/utils_plotting.Palette)
TRUTH = '#808080'
ENS = '#20B2AA'
OBS = '#D62728'


def _default_std_obs(case):
    return inspect.signature(case.run_experiment).parameters['std_obs'].default


def _state_hist(filter_ens):
    """(t, states (Nt, Nphi, m)) of the filtered ensemble."""
    model = filter_ens.model
    return model.hist_t, model.hist[:, :model.Nphi, :]


def _truth_state(truth_full):
    """(Nt, Nx) truth trajectory, collapsing a trailing realization axis."""
    y = truth_full.y_true
    return np.mean(y, axis=-1) if y.ndim == 3 else y


def _window(ax, truth, t_ref):
    for t_edge in (truth.t_obs[0], truth.t_obs[-1]):
        ax.axvline(t_edge / t_ref, color='0.5', lw=0.8, ls=':')


def _band(ax, t, m, s, first):
    ax.fill_between(t, m - s, m + s, color=ENS, alpha=0.35, lw=0,
                    label='ensemble spread ($\\pm 1\\sigma$)' if first else None)
    ax.plot(t, m, color=ENS, lw=1.2, ls='--', dashes=(2, .5),
            label='model estimate' if first else None)


def _obs_overlay(ax, truth, col, t_ref, first):
    """Raw (noisy) truth line and the assimilated observations of one column."""
    ax.plot(truth.t_true / t_ref, np.squeeze(truth.y_raw)[:, col], color=TRUTH,
            lw=0.6, alpha=0.45, label='raw truth' if first else None)
    ax.plot(truth.t_obs / t_ref, np.squeeze(truth.y_obs)[:, col], '.', color=OBS,
            ms=5, ls='none', label='data' if first else None)


def _fig_legend(fig):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=len(labels), frameon=False,
               loc='outside lower center', fontsize=9)


def page_lorenz63(pdf):
    filter_ens, truth, _ = l63.run_experiment()
    t, phi = _state_hist(filter_ens)
    mean, std = np.mean(phi, axis=-1), np.std(phi, axis=-1)

    # Deterministic twin: re-integrate the truth observing every state
    truth_full = l63.build_truth(observe_dims=[0, 1, 2], add_noise=False)
    y_full = _truth_state(truth_full)

    obs_dims = l63.truth_params['observe_dims']
    t_ref = l63.t_lyap
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout='constrained')
    for k, (ax, name) in enumerate(zip(axs, ['x', 'y', 'z'])):
        first = k == 0
        ax.plot(truth_full.t_true / t_ref, y_full[:, k], color=TRUTH, lw=1.5,
                alpha=0.8, label='truth' if first else None)
        _band(ax, t / t_ref, mean[:, k], std[:, k], first)
        if k in obs_dims:
            _obs_overlay(ax, truth, obs_dims.index(k), t_ref, first)
        ax.set_ylabel(f'${name}$' + ('' if k in obs_dims else ' (unobserved)'))
        _window(ax, truth, t_ref)
    axs[-1].set_xlim(t[0] / t_ref, truth_full.t_true[-1] / t_ref)
    axs[-1].set_xlabel('$t / t_\\mathrm{lyap}$')
    _fig_legend(fig)
    fig.suptitle(f"Lorenz63 twin — EnSRKF, m={l63.ensemble_params['m']}, "
                 f"Nt_obs={l63.truth_params['Nt_obs']}, "
                 f"noise_level={l63.truth_params['noise_level']}, "
                 f"std_obs={_default_std_obs(l63)}, "
                 f"inflation={l63.ensemble_params['inflation_factor']}", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def page_lorenz96(pdf):
    filter_ens, truth, _ = l96.run_experiment()
    t, phi = _state_hist(filter_ens)
    mean, std = np.mean(phi, axis=-1), np.std(phi, axis=-1)

    truth_full = l96.build_truth(observed_idx=list(range(l96.Nx)), add_noise=False)
    t_ref = l96.t_lyap
    y_ref = _truth_state(truth_full)    # (Nt_true, Nx)

    observed_idx = l96.truth_params['observed_idx']
    config = (f"EnSRKF, m={l96.ensemble_params['m']}, "
              f"Nt_obs={l96.truth_params['Nt_obs']}, "
              f"Nq={len(observed_idx)}/{l96.Nx}, "
              f"noise_level={l96.truth_params['noise_level']}, "
              f"std_obs={_default_std_obs(l96)}, "
              f"inflation={l96.ensemble_params['inflation_factor']}")

    # --- Page: spatiotemporal fields --- #
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True, sharey=True,
                            layout='constrained')
    vmax = np.max(np.abs(y_ref))
    fields = [(y_ref.T, truth_full.t_true, 'truth', dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax)),
              (mean.T, t, 'ensemble mean', dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax)),
              (std.T, t, 'ensemble spread (std)', dict(cmap='Blues'))]
    for ax, (field, tt, title, kw) in zip(axs, fields):
        im = ax.pcolormesh(tt / t_ref, np.arange(l96.Nx), field,
                           shading='auto', rasterized=True, **kw)
        ax.set(ylabel='variable index', title=title)
        _window(ax, truth, t_ref)
        fig.colorbar(im, ax=ax, pad=0.01)
    axs[-1].set_xlim(t[0] / t_ref, truth_full.t_true[-1] / t_ref)
    axs[-1].set_xlabel('$t / t_\\mathrm{lyap}$')
    fig.suptitle(f'Lorenz96 twin — {config}', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page: timeseries at a few observation locations --- #
    show_vars = [0, 1, 20, 21]
    fig, axs = plt.subplots(len(show_vars), 1, figsize=(9, 2.2 * len(show_vars)),
                            sharex=True, layout='constrained')
    for i, (ax, k) in enumerate(zip(axs, show_vars)):
        first = i == 0
        observed = k in observed_idx
        ax.plot(truth_full.t_true / t_ref, y_ref[:, k], color=TRUTH, lw=1.5,
                alpha=0.8, label='truth' if first else None)
        _band(ax, t / t_ref, mean[:, k], std[:, k], first)
        if observed:
            _obs_overlay(ax, truth, observed_idx.index(k), t_ref, first)
        ax.set_ylabel(f'$x_{{{k}}}$' + ('' if observed else '\n(unobserved)'))
        _window(ax, truth, t_ref)
    axs[-1].set_xlim(t[0] / t_ref, truth_full.t_true[-1] / t_ref)
    axs[-1].set_xlabel('$t / t_\\mathrm{lyap}$')
    _fig_legend(fig)
    fig.suptitle(f'Lorenz96 twin — {config}', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def page_cylinder(pdf, std_obs=0.1, std_phi=1.5, Nt_obs=10, inflation_factor=1.1):
    (X_train, _), (X_filter, X_filter_true), simulation_dir = cyl.load_data()
    rom = cyl.build_rom(X_train, simulation_dir)
    truth = cyl.build_truth(rom, X_filter, X_filter_true, Nt_obs=Nt_obs)
    ensemble = cyl.build_ensemble(rom, m=cyl.ensemble_params['m'], std_phi=std_phi,
                                  inflation_factor=inflation_factor)
    filter_ens = run_da_loop(ensemble, truth, std_obs=std_obs, t_extra=2. * rom.t_CR)

    model = filter_ens.model
    t, y = model.hist_t, model.get_observable_hist()      # (Nt, Nq, m)
    y_mean, y_std = np.mean(y, axis=-1), np.std(y, axis=-1)
    t_ref = model.t_CR                                    # convective time unit
    t_end = min(t[-1], truth.t_true[-1]) / t_ref

    config = (f"EnSRKF, m={cyl.ensemble_params['m']}, "
              f"Nt_obs={Nt_obs}, "
              f"N_modes={cyl.rom_params['N_modes']}, "
              f"std_phi={std_phi}, std_obs={std_obs}, "
              f"inflation={inflation_factor}")

    # --- Page: assimilated sensor measurements --- #
    Nq = min(y_mean.shape[1], 4)
    fig, axs = plt.subplots(Nq, 1, figsize=(9, 2.2 * Nq), sharex=True,
                            layout='constrained')
    for k, ax in enumerate(np.atleast_1d(axs)):
        first = k == 0
        ax.plot(truth.t_true / t_ref, truth.y_true[:, k], color=TRUTH, lw=1.5,
                alpha=0.8, label='truth' if first else None)
        _band(ax, t / t_ref, y_mean[:, k], y_std[:, k], first)
        _obs_overlay(ax, truth, k, t_ref, first)
        ax.set_ylabel(f'sensor {k}')
        _window(ax, truth, t_ref)
    np.atleast_1d(axs)[-1].set_xlim(t[0] / t_ref, t_end)
    np.atleast_1d(axs)[-1].set_xlabel('$t / t_\\mathrm{CR}$')
    _fig_legend(fig)
    fig.suptitle(f'POD-ESN cylinder, sensor measurements — {config}', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page: global quantities (POD coefficients and modal TKE) --- #
    N_modes = cyl.rom_params['N_modes']
    a_true = np.asarray(rom.encode(X_filter_true))        # (N_modes, Nt_true)
    assert a_true.shape[0] == N_modes, a_true.shape
    t_true = np.arange(a_true.shape[1]) * rom.dt

    a_ens = model.hist[:, :N_modes, :]                    # (Nt, N_modes, m)
    tke_ens = 0.5 * np.sum(a_ens ** 2, axis=1)            # (Nt, m)
    tke_true = 0.5 * np.sum(a_true ** 2, axis=0)

    panels = [(f'$a_{{{k + 1}}}$', a_true[k], np.mean(a_ens[:, k], -1),
               np.std(a_ens[:, k], -1)) for k in range(2)]
    panels.append(('modal TKE\n$\\frac{1}{2}\\sum_r a_r^2$', tke_true,
                   np.mean(tke_ens, -1), np.std(tke_ens, -1)))

    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 2.4 * len(panels)),
                            sharex=True, layout='constrained')
    for i, (ax, (lbl, ref, m_, s_)) in enumerate(zip(axs, panels)):
        first = i == 0
        ax.plot(t_true / t_ref, ref, color=TRUTH, lw=1.5, alpha=0.8,
                label='truth (projected)' if first else None)
        _band(ax, t / t_ref, m_, s_, first)
        ax.set_ylabel(lbl)
        _window(ax, truth, t_ref)
    axs[-1].set_xlim(t[0] / t_ref, t_end)
    axs[-1].set_xlabel('$t / t_\\mathrm{CR}$')
    _fig_legend(fig)
    fig.suptitle(f'POD-ESN cylinder, global quantities — {config}', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIGS_FOLDER, exist_ok=True)
    filename = os.path.join(FIGS_FOLDER, 'predictions.pdf')
    with PdfPages(filename) as pdf:
        page_lorenz63(pdf)
        page_lorenz96(pdf)
        page_cylinder(pdf)
    print(f'Saved {filename}')
