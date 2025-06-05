
import numpy as np
import os as os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from matplotlib import rc
from essentials.Util import colour_noise

rc('animation', html='jshtml')

from matplotlib.animation import FuncAnimation, PillowWriter

from essentials.models_physical import Lorenz63
from essentials.Util import load_from_mat_file, save_to_mat_file, set_working_directories


def create_Lorenz63_dataset(noise_level=0.02, num_lyap_times=300, seed=0, **kwargs):
    # Load or create training data from the Lorenz 63 model
    data_folder = set_working_directories('Lorenz/')[0]

    model = Lorenz63(**kwargs)

    # Default filename
    filename = f"{''.join([f'{key}{val:.2f}_' for key, val in model.get_default_params.items()])}Nlyap{num_lyap_times}_noise{noise_level}_seed{seed}"

    t_lyap = model.t_lyap
    dt = model.dt
    N_lyap = int(t_lyap / dt)

    # if os.path.isfile(data_folder + filename):
    #     dataset = load_from_mat_file(data_folder + filename)
    #     print('Loaded case')
    # else:

    try:
        dataset = load_from_mat_file(data_folder + filename)
        print('Loaded case')
    except FileNotFoundError:

        # Create a time series from the model
        model.create_long_timeseries(Nt=num_lyap_times * N_lyap)

        # Get the model observables and time
        t = model.hist_t
        all_data = model.get_observable_hist()[..., 0].copy()
        all_data_clean = all_data.copy()

        # Add noise to the data
        all_data = add_noise_to_data(all_data, noise_level, seed)

        # Save data for future use
        dataset = dict(clean_data=all_data_clean,
                       noisy_data=all_data,
                       t=t,
                       N_lyap=N_lyap)
        save_to_mat_file(data_folder + filename, dataset)

    return dataset


def add_noise_to_data(data, noise_level, seed=0):
    """
    Add Normal noise of strength noise_level to the provided data.

    Args:
        data: shape [N_time x N_dim]
        noise_level:
        seed:

    Returns:

    """
    rng_noise = np.random.default_rng(seed)
    U_std = np.std(data, axis=0)
    for dd in range(data.shape[1]):
        data[:, dd] += rng_noise.normal(loc=0, scale=noise_level * U_std[dd], size=data.shape[0])
    return data


def add_noise_to_flow(U, V, noise_level=0.05, noise_type="gauss", spatial_smooth=0):
    """
    Adds noise to a 3D velocity field (Nt x Nx x Ny).

    Args
        U, V : numpy.ndarray
            3D arrays representing the velocity components (Nt x Nx x Ny).
        
        noise_level : float, optional, default=0.05
            The standard deviation of the noise as a fraction of the maximum absolute velocity.
        
        noise_type : str, optional, default="gauss"
            The type of noise to apply:
            - 'gauss': Gaussian (white) noise.
            - 'pink', 'brown', 'blue', 'violet': Colored noise.

        spatial_smooth : float, optional, default=0
            Standard deviation for Gaussian smoothing (0 means no smoothing).

    Returns
        U_noisy, V_noisy : numpy.ndarray
            Noisy velocity fields with the same shape as U and V.
    """
    rng = np.random.default_rng()  # Random generator

    U = U.copy()
    V = V.copy()
    
    U[np.isnan(U) | np.isinf(U)] = 0  # Replace NaN/Inf values
    V[np.isnan(V) | np.isinf(V)] = 0  # Replace NaN/Inf values

    # Compute noise amplitude
    max_vel = max(np.max(np.abs(U)), np.max(np.abs(V)))
    noise_amp = noise_level * max_vel

    def generate_noise(shape, noise_type):
        """Generates 3D noise (Nt x Nx x Ny)."""
        Nt, Nx, Ny = shape
        if noise_type == "gauss":
            return rng.normal(scale=noise_amp, size=shape)
        else:
            noise_white = np.fft.rfftn(rng.standard_normal(shape) * noise_amp)
            # Frequency domain shapes
            Nt_f, Nx_f, Ny_f = noise_white.shape

            S_time = colour_noise(Nt_f, noise_colour=noise_type)[:, None, None]
            S_x = colour_noise(Nx_f, noise_colour=noise_type)[None, :, None]
            S_y = colour_noise(Ny_f, noise_colour=noise_type)[None, None, :]

            S = S_time * S_x * S_y
            S[np.isnan(S) | np.isinf(S)] = 0

            power = np.mean(S ** 2)
            if power > 0:
                S /= np.sqrt(power)

    # Generate noise
    

    noise_U = generate_noise(U.shape, noise_type)
    noise_V = generate_noise(V.shape, noise_type)

    # Apply optional spatial smoothing
    if spatial_smooth > 0:
        noise_U = ndimage.gaussian_filter(noise_U, sigma=(0, spatial_smooth, spatial_smooth))
        noise_V = ndimage.gaussian_filter(noise_V, sigma=(0, spatial_smooth, spatial_smooth))

    # Add noise to velocity field
    
    U_noisy = U + noise_U
    V_noisy = V + noise_V

    return U_noisy, V_noisy


def plot_timeseries(*plot_cases, zoom_window=None, add_pdf=False, t_factor=1, dims='all'):
    """
    Plot the time evolution of the observables in a object of class model
    """

    if not isinstance(plot_cases, (list, tuple)):
        plot_cases = [plot_cases]


    # Ensure dims is a list of integers
    if dims == 'all':
        dims = list(range(plot_cases[0].Nq))
    elif isinstance(dims, int):
        dims = [dims]
    elif not isinstance(dims, (list, tuple, np.ndarray)):
        raise ValueError(f"`dims` must be 'all', int, or list of ints. Got: {dims}")


    fig = plt.figure(figsize=(10, 2.5*len(dims)), layout="constrained")
    axs = fig.subplots(len(dims), 2 + add_pdf, sharey='row', sharex='col')
    if len(dims) == 1:
        axs = [axs]

    xlabel = '$t$'
    if t_factor != 1:
        xlabel += '$/T$'

    for plot_case in plot_cases:
        y = plot_case.get_observable_hist()  # history of the model observables
        lbl = plot_case.obs_labels

        t_h = plot_case.hist_t / t_factor
        if zoom_window is None:
            zoom_window = [t_h[-1] - plot_case.t_CR, t_h[-1]]

        for ii, ax in zip(dims, axs):
            [ax[jj].plot(t_h, y[:, ii]) for jj in range(2)]
            ax[0].set(ylabel=lbl[ii])
            if add_pdf:
                ax[2].hist(y[:, ii], alpha=0.5, histtype='stepfilled', bins=20, density=True, orientation='horizontal',
                           stacked=False)

        axs[-1][0].set(xlabel=xlabel, xlim=[t_h[0], t_h[-1]])
        axs[-1][1].set(xlabel=xlabel, xlim=zoom_window)
        if plot_case.ensemble:
            plt.gcf().legend([f'$mi={mi}$' for mi in range(plot_case.m)], loc='center left', bbox_to_anchor=(1.0, .75),
                             ncol=1, frameon=False)


def plot_parameters(plot_case):
    """
    Plot the time evolution of the parameters in a object of class model
    """

    colors_alpha = ['g', 'sandybrown', [0.7, 0.7, 0.87], 'b', 'r', 'gold', 'deepskyblue']

    t_h = plot_case.hist_t
    t_zoom = int(plot_case.t_CR / plot_case.dt)

    hist_alpha = plot_case.hist[:, -plot_case.Na:]
    mean_alpha = np.mean(hist_alpha, axis=-1)
    std_alpha = np.std(hist_alpha, axis=-1)

    fig = plt.figure(figsize=(5, 1.5*plot_case.Na), layout="constrained")
    axs = fig.subplots(plot_case.Na, 1, sharex='col')
    if isinstance(axs, plt.Axes):
        axs = [axs]

    axs[-1].set(xlabel='$t$', xlim=[plot_case.hist_t[0], plot_case.hist_t[-1]]);
    for ii, ax, p, c in zip(range(len(axs)), axs, plot_case.est_a, colors_alpha):
        avg, s, all_h = [xx[:, ii] for xx in [mean_alpha, std_alpha, hist_alpha]]
        ax.fill_between(t_h, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=c, label='2 std')
        ax.plot(t_h, avg, color=c, label='mean', lw=2)
        ax.plot(t_h, all_h, color=c, lw=1., alpha=0.3)
        # ax.legend(loc='upper right', fontsize='small', ncol=2)
        # print(min(avg) + 3 * max(s), max(avg) + 3 * max(s))
        ax.set(ylabel=plot_case.alpha_labels[p], ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])



def plot_train_dataset(clean_data, noisy_data, t, *split_times):
    # Visualize the training dataset
    fig = plt.figure(figsize=(12.5, 5), layout='tight')
    sfs = fig.subfigures(1, 2, width_ratios=[1.2, 1])

    axs = sfs[0].subplots(nrows=clean_data.shape[1], ncols=1, sharex='col', sharey='row')

    for axi, ax in enumerate(axs):
        ax.plot(t, clean_data[:, axi], c='k', lw=1.2, label='Truth')
        ax.plot(t, noisy_data[:, axi], c='r', lw=0.4, label='Noisy data')
        for kk, c, lbl in zip(range(1, len(split_times) + 2), ['C2', 'C1', 'C4', 'lightgray'],
                              ['Train', 'Validation', 'Test', 'Not used']):
            ax.axvspan(sum(split_times[:kk]), sum(split_times[:kk + 1]), facecolor=c, alpha=0.3,
                       label=lbl + ' data')
            ax.axvline(x=sum(split_times[:kk]), color='k', lw=0.5, dashes=[10, 5])
        if axi == 0:
            ax.legend(loc='lower center', ncol=10, bbox_to_anchor=(0.5, 1.0))
    axs[-1].set(xlabel='$t/T$', xlim=[0, t[-1]]);

    ax = sfs[1].add_subplot(111, projection='3d')
    _nt = 1000
    ax.plot(clean_data[-_nt:, 0], clean_data[-_nt:, 1], clean_data[-_nt:, 2], '.-',
            c='k', ms=2, lw=.5, label='Truth')
    ax.plot(noisy_data[-_nt:, 0], noisy_data[-_nt:, 1], noisy_data[-_nt:, 2], '.',
            c='r', ms=1, label='Noisy data')
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$');



def animate_flowfields(datsets, n_frames=40, cmaps=None, titles=None, rms_cmap='Reds'):

    if cmaps is None:
        cmaps = ['viridis'] * len(datsets)
    if titles is None:
        titles = [f'Dataset {i+1}' for i in range(len(datsets))]

    fig, axs = plt.subplots(1, len(datsets), sharex=True, sharey=True, figsize=(1.5*len(datsets), 4), layout='constrained')

    ims = []
    for ax, D, ttl, cmap in zip(axs, datsets, titles, cmaps):
        if 'RMS' in ttl:
            ims.append(ax.pcolormesh(D[0], rasterized=True, cmap=plt.get_cmap(rms_cmap), vmin=0, vmax=1))
        else:
            ims.append(ax.pcolormesh(D[0], rasterized=True, cmap=plt.get_cmap(cmap)))
        ax.set(title=ttl, xticks=[], yticks=[])
        fig.colorbar(ims[-1], ax=ax, orientation='horizontal')

    def animate(ti):
        [im.set_array(D[ti]) for im, D in zip(ims, datsets)]
        print(f'Frame {ti + 1}/{n_frames}', flush=True, end='\r')
        return ims

    plt.close(fig)

    return FuncAnimation(fig, animate, frames=n_frames)

# Add test funciton
