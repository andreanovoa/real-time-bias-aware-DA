
import contextlib
import glob
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image

# Figures colors
color_true = 'gray'
color_unbias = '#000080ff'
color_bias = '#20b2aae5'
color_obs = 'r'
color_b = 'indigo'
colors_alpha = ['green', 'sandybrown', [0.7, 0.7, 0.87], 'blue', 'red', 'gold', 'deepskyblue']


@dataclass(frozen=True)
class Palette:
    TRUE = "#808080"
    UNBIASED = "#000080"
    BIASED = "#20B2AA"

    OBS = "#D62728"

    BIAS_STATE = "#4B0082"
    BIAS_OBS = "#BA55D3"
    BIAS_OBS_NOISY = "#DD9700"

    PARAMS = ["#2CA02C", "#F4A460", "#B3B3DE", "#1F77B4", "#D62728", "#FFD700", "#00BFFF"]

    @staticmethod
    def get_color(name: str, alpha: float = 1.0):
        val = getattr(Palette, name.upper())
        if isinstance(val, list):
            return [colors.to_rgba(c, alpha) for c in val]
        return colors.to_rgba(val, alpha)

    @staticmethod
    def get_color_params(n=-1, alpha: float = 1.0):
        return [colors.to_rgba(c, alpha) for c in Palette.PARAMS[:n]]

    @property
    def y_unbias_props(self):

        return dict(marker='none', linestyle='--', dashes=(10, 1), lw=.8, color=self.get_color('UNBIASED'))

    @property
    def y_biased_props(self):
        return dict(marker='none', linestyle='-', lw=.8, color=self.get_color('BIASED', .7))

    @property
    def y_biased_mean_props(self):
        return dict(marker='none', linestyle='--', dashes=(2, .5), lw=1, color=self.get_color('BIASED', 1.0))

    @property
    def true_noisy_props(self):
        return dict(marker='none', linestyle='-', lw=1.2, color=self.get_color('TRUE', .7))
    @property
    def true_props(self):
        return dict(marker='none', linestyle='-', lw=2, color=self.get_color('TRUE', .7))
    @property
    def obs_props(self):
        return dict(marker='.', linestyle='none', markersize=7, markeredgecolor='none', color=self.get_color('OBS'))
    @property
    def washout_props(self):
        return dict(marker='x', linestyle='none', markersize=3, markeredgecolor=self.get_color('OBS'))
    @property
    def bias_props(self):
        return dict(marker='none', linestyle='--', dashes=(10, 1), lw=.5, color=self.get_color('BIAS_STATE'))
    @property
    def bias_obs_props(self):
        return dict(lw=1.5, color=self.get_color('BIAS_OBS', alpha=0.7))
    @property
    def bias_obs_noisy_props(self):
        return dict(lw=1.5, color=self.get_color('BIAS_OBS_NOISY', 0.2))


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Create a categorical colormap with a specified number of categories and subcategories.

    Parameters
    ----------
        nc : int
            Number of categories.
        nsc : int
            Number of subcategories.
        cmap : str, optional
            Colormap to use. Default is "tab10".
        continuous : bool, optional
            Whether to use a continuous colormap. Default is False.
    Returns
    -------
        np.ndarray with shape (nc * nsc, 3)
            Array of colors. Category 1 will be in rows 0 to nsc-1, category 2 in rows nsc to 2*nsc-1, and so on.
    """

    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * nsc:(i + 1) * nsc, :] = rgb
    return cols



def folder_to_gif(folder, img_type='.png', gif_name='movie.gif'):
    """
    Convert all the images inside a folder into a gif.

    """
    if img_type[0] != '.':
        img_type = f'.{img_type}'

    fp_in = folder + f'*{img_type}'
    fp_out = folder + gif_name

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out,
                 format='GIF',
                 append_images=imgs,
                 save_all=True,
                 duration=200,
                 loop=0)
