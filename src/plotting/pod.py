


# ── plots ─────────────────────────────────────────────────────────────────

from tools.autoencoders import POD
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
from typing import Optional, Union

from utils import (save_figs_to_pdf,
                    get_figsize_based_on_domain) 

from pyts.image import RecurrencePlot
from mpl_toolkits.axes_grid1 import ImageGrid



def plot_modes(case: POD, 
                num_modes: int = 2, save: bool = False,
                cmap: str = 'viridis', dim=None,
                n_col=4):
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
    
    assert case.fitted, "POD must be fitted before plotting modes."

    Psi = case.Psi.copy()
    if Psi.ndim == 2:
        Psi = case._to_physical_grid(Psi)
    dim = np.arange(Psi.shape[0]) if dim is None else (
            [dim] if isinstance(dim, int) else dim)
    X1, X2 = case.domain_mesh

    num_modes = min(num_modes, Psi.shape[1])
    n_col = min(n_col, num_modes)
    n_row = int(np.ceil(num_modes / n_col))
    
    for jj, d in enumerate(dim):
        data = Psi[d]
        fig  = plt.figure(figsize=(n_col * 2, n_row), layout='constrained')
        axs  = fig.subplots(nrows=n_row, ncols=n_col,
                            sharex=True, sharey=True)
        axs  = [axs] if case.N_latent == 1 else axs.ravel()
        norm = colors.Normalize(vmin=data[0].min(),
                                vmax=data[0].max())
        
        for kk, ax in zip(range(num_modes), axs):
            im = ax.pcolormesh(X1, X2, data[kk],
                                cmap=mpl.colormaps[cmap], norm=norm,
                                rasterized=True)
            ax.set_title(f'mode {kk}', fontsize='xx-small')
            ax.set_aspect('equal')
            if kk >= num_modes - n_col:
                ax.set_xlabel('$y$')
            if kk % n_col == 0:
                ax.set_ylabel('$x$')
        fig.colorbar(im, ax=axs, shrink=0.25, aspect=20) #type: ignore
        if save: 
            plt.savefig(f'modes_dim{jj}.png', dpi=300)


def plot_time_coefficients(case: POD, 
                           Phi: np.ndarray = None,
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
def plot_spectrum(case: POD, max_mode: Optional[int] = None):
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
        energy_frac = Lambda / 2 / (case._Phi.shape[0] if case._Phi is not None else 1)
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



def plot_flows_rms(case: POD,
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
