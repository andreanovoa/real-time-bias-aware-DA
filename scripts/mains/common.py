"""Shared helpers for the main scripts and ablation studies.

These utilities wrap the sequential data assimilation loop
(`forecast_step` / `analysis_step`) and compute scalar error metrics so that
the same driver can be reused by every test case (Lorenz63/96 twin
experiments, annular thermoacoustics, POD-ESN cylinder wake).
"""

import argparse
import multiprocessing
import os

try:  # Python >= 3.14 defaults to forkserver on Linux; IVPIntegrator pools break there
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import yaml
from romda.estimators import EnKF, EnSRKF, rBA_EnKF
from romda.metrics import ensemble_metrics
from romda.observations import Observations

#: Filters `resolve_da_method` accepts by name, so the mains can sweep a plain string axis.
DA_METHODS = dict(EnKF=EnKF, EnSRKF=EnSRKF, rBA_EnKF=rBA_EnKF)


def resolve_da_method(name):
    """DA-method registry: 'EnKF' / 'EnSRKF' / 'rBA_EnKF' in, the estimator class out."""
    if isinstance(name, type):
        return name
    if name not in DA_METHODS:
        raise ValueError(f"Unknown DA method '{name}'. Available: {sorted(DA_METHODS)}")
    return DA_METHODS[name]

#: repo-root configs/ -- one YAML of defaults per main (mains/<case>.yml, tai_da/<case>.yml)
CONFIGS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'configs')


def load_main_config(name: str) -> dict:
    """The default blocks of a main from ``configs/<name>.yml`` (e.g. 'tai_da/annular')."""
    with open(os.path.join(CONFIGS_FOLDER, f'{name}.yml')) as f:
        return yaml.safe_load(f)


def observation_covariance(truth: Observations, std_obs: float) -> np.ndarray:
    """Diagonal observation-error covariance matrix.

    The variance of each observed dimension is ``(std_obs * max|y_obs|)^2``,
    i.e., ``std_obs`` is interpreted as a fraction of the observation amplitude.
    """
    Nq = truth.y_obs.shape[1]
    return np.diag(std_obs * np.ones(Nq)) * np.max(abs(truth.y_obs), axis=0) ** 2


#: promoted to the repo proper (`romda.metrics`); the alias keeps the mains working
compute_metrics = ensemble_metrics


def print_metrics(metrics: dict, header: str = 'Metrics') -> None:
    print(f'\n ------------------ {header} ------------------ ')
    for key, val in metrics.items():
        print(f'\t {key} = {val:.4f}' if isinstance(val, float) else f'\t {key} = {val}')


# Figures saved by the mains with --save-figs are versioned in git
FIGS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


def parse_main_args(description: str = None) -> argparse.Namespace:
    """Common command-line interface of the main scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--save-figs', action='store_true',
                        help=f'save figures to {FIGS_FOLDER} instead of showing them')
    return parser.parse_args()


def show_or_save_figs(save: bool, prefix: str) -> None:
    """Either open the interactive figures or save them as PNGs in ``figs/``."""
    if not save:
        plt.show()
        return

    os.makedirs(FIGS_FOLDER, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = os.path.join(FIGS_FOLDER, f'{prefix}_{num:02d}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
