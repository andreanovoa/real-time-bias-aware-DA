import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    return plt.loglog(f, np.abs(np.fft.rfft(s)))[0]


def noise_psd(N, psd=lambda f: 1):
    N = int(N)
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N: noise_psd(N, f)


@PSDGenerator
def white_noise(f):
    return 1


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)


@PSDGenerator
def violet_noise(f):
    return f


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float('inf'), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))


SAMPLE_FREQ_HZ = 125
SAMPLE_INTV_SEC = 1 / SAMPLE_FREQ_HZ


def plot_test_points(sample_count: int = 2500):
    n = sample_count if sample_count else 1000
    fig, ax_list = plt.subplots(5, 1, figsize=(12, 8), tight_layout=True)
    i = 0
    for ax, G, c, l in zip(ax_list,
            [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise],
            ['brown', 'hotpink', 'white', 'blue', 'violet'],
            ['brown', 'pink', 'white', 'blue', 'violet']):
        t = [x * SAMPLE_INTV_SEC for x in range(0, n)]
        ax.plot(t, G(n), color=c, linewidth=1.5, label=l)
        ax.legend(loc='lower left')
        ax.set_xlabel("Time [sec]")
        i += 1
    plt.suptitle(f"Colored Noise (n={n} points; sampling rate = {SAMPLE_FREQ_HZ}Hz)")


if __name__ == '__main__':

    Nt = 10000
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8), tight_layout=True)
    for func, color in zip(
            [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise],
            ['brown', 'hotpink', 'white', 'blue', 'violet']):
        plot_spectrum(func(Nt)).set(color=color, linewidth=3)
    plt.legend(['brownian', 'pink', 'white', 'blue', 'violet'])
    plt.suptitle("Colored Noise")
    plt.ylim([1e-3, None])

    plot_test_points()


    def colour_noise(ff, noise_type='pink', beta=1):
        if noise_type.lower() == 'white':
            return np.ones(ff.shape)
        elif noise_type.lower() == 'blue':
            return np.sqrt(ff)
        elif noise_type.lower() == 'violet':
            return ff
        elif 'pink' in noise_type.lower():
            return 1 / np.where(ff == 0, float('inf'), ff**(1/beta))
        elif 'brown' in noise_type.lower():
            return 1 / np.where(ff == 0, float('inf'), ff)
        else:
            raise ValueError('{} noise type not defined'.format(noise_type))

    # X_time_domain = np.random.randn(N)
    # X_freq_domain = np.fft.rfft(X_time_domain)
    psd = np.fft.rfftfreq(Nt)

    nosie_white = np.fft.rfft(np.random.randn(Nt))
    freq = np.fft.rfftfreq(len(nosie_white))

    plt.figure()
    for beta_val, c in zip(np.arange(5), ['white', 'hotpink', 'brown', 'blue', 'violet']):
        # Generate the noise signal
        noise = colour_noise(psd, beta=beta_val, noise_type=c)

        # Normalize S
        noise = nosie_white * noise / np.sqrt(np.mean(noise ** 2))

        # Plot prequency spectrum
        PSD = np.fft.irfft(noise)
        freq = np.fft.rfftfreq(len(PSD))
        plt.loglog(freq, np.abs(np.fft.rfft(PSD)), color=c)

    plt.legend(['white', 'pink', 'red', 'blue', 'violet'])
    plt.ylim([1e-3, None])

    plt.show()

