"""Ablation studies for the Lorenz63 twin experiment.

Usage
-----
    python run_lorenz63.py --study ensemble_size [--seeds 5] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m`` for EnKF vs EnSRKF.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``inflation`` : multiplicative inflation factor.
- ``observation_noise`` : truth noise level vs assumed observation error.
- ``regularization`` : bias-aware r-EnKF regularization factor (biased truth).

Results are appended to ``results/ablations/lorenz63_<study>.jsonl`` and can be
re-plotted at any time with ``--plot`` (no new runs if the sweep is complete).
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mains'))

import main_lorenz63 as case
from romda.estimators import EnKF, EnSRKF, rBA_EnKF
from dev.experiments.sweep import load_results, plot_ablation, run_sweep
from romda.utils import set_working_directories

DA_METHODS = dict(EnKF=EnKF, EnSRKF=EnSRKF, rBA_EnKF=rBA_EnKF)

results_folder = set_working_directories('ablations')[1]

# Ablation figures are versioned in git; the .jsonl records stay in results/
figs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               da_method='EnSRKF',
               m=50,
               Nt_obs=20,
               inflation_factor=1.02,
               noise_level=0.05,
               std_obs=0.05,
               regularization_factor=None,
               manual_bias=None):
    """Map a flat grid point onto ``main_lorenz63.run_experiment`` kwargs."""
    truth_kwargs = dict(Nt_obs=Nt_obs, noise_level=noise_level, manual_bias=manual_bias)
    ensemble_kwargs = dict(m=m, inflation_factor=inflation_factor, seed=seed)
    if regularization_factor is not None:
        ensemble_kwargs['regularization_factor'] = regularization_factor

    _, _, metrics = case.run_experiment(da_method=DA_METHODS[da_method],
                                        std_obs=std_obs,
                                        truth_kwargs=truth_kwargs,
                                        ensemble_kwargs=ensemble_kwargs)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[5, 10, 20, 50, 100],
                  da_method=['EnKF', 'EnSRKF']),
        x_key='m', group_key='da_method', logx=True,
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[5, 10, 20, 40, 80]),
        x_key='Nt_obs', logx=True,
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.01, 1.02, 1.05, 1.1]),
        x_key='inflation_factor',
    ),
    observation_noise=dict(
        axes=dict(noise_level=[0.01, 0.05, 0.1, 0.2],
                  std_obs=[0.01, 0.05, 0.1]),
        x_key='noise_level', group_key='std_obs',
    ),
    regularization=dict(
        axes=dict(regularization_factor=list(np.linspace(0., 5., 6)),
                  da_method=['rBA_EnKF'],
                  manual_bias=['linear']),
        x_key='regularization_factor',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES.keys(), required=True)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--quick', action='store_true', help='2 values per axis, 2 seeds')
    parser.add_argument('--plot', action='store_true', help='only plot existing results')
    args = parser.parse_args()

    study = STUDIES[args.study]
    axes = study['axes']
    n_seeds = args.seeds

    if args.quick:
        axes = {k: v[:2] for k, v in axes.items()}
        n_seeds = min(n_seeds, 2)

    results_file = f'{results_folder}lorenz63_{args.study}.jsonl'

    if args.plot:
        records = load_results(results_file)
    else:
        records = run_sweep(experiment, axes, results_file, n_seeds=n_seeds)

    plot_ablation(records,
                  x_key=study['x_key'],
                  group_key=study.get('group_key'),
                  logx=study.get('logx', False),
                  title=f'Lorenz63 — {args.study}',
                  filename=os.path.join(figs_folder, f'lorenz63_{args.study}.png'))
    plt.show()
