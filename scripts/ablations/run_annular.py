"""Ablation studies for bias-aware assimilation of the annular experiment data.

Requires the azimuthal dataset (downloaded automatically from
https://zenodo.org/records/15609832 on first run).

Usage
-----
    python run_annular.py --study regularization [--seeds 3] [--quick] [--plot]

Studies
-------
- ``regularization`` : r-EnKF bias regularization factor vs the bias-blind EnKF
  baseline (``da_method='EnKF'`` records).
- ``ensemble_size`` : number of members ``m``.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``equivalence_ratio`` : robustness across the four available operating
  conditions (ER = 0.4875 ... 0.5625).

Results are appended to ``results/ablations/annular_<study>.jsonl`` and can be
re-plotted at any time with ``--plot``.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mains'))

import main_annular as case
from romda.estimators import EnKF, rBA_EnKF
from dev.experiments.sweep import load_results, plot_ablation, run_sweep
from romda.utils import set_working_directories

DA_METHODS = dict(EnKF=EnKF, rBA_EnKF=rBA_EnKF)

results_folder = set_working_directories('ablations')[1]

# Ablation figures are versioned in git; the .jsonl records stay in results/
figs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')

ERs = 0.4875 + np.arange(0, 4) * 0.025    # available equivalence ratios (the data grid)


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               da_method='rBA_EnKF',
               ER=ERs[1],
               regularization_factor=5.,
               m=20,
               Nt_obs=35,
               inflation_factor=1.0,
               std_obs=0.05):
    """Map a flat grid point onto ``main_annular.run_experiment`` kwargs.

    The ESN bias estimator is trained once per (ER, Nt_obs) and cached on disk
    by ``build_bias_estimator``, so sweeping m/regularization reuses it.
    """
    truth_kwargs = dict(Nt_obs=Nt_obs)
    ensemble_kwargs = dict(m=m, inflation_factor=inflation_factor, seed=seed)

    _, _, metrics = case.run_experiment(ER=ER,
                                        regularization_factor=regularization_factor,
                                        da_method=DA_METHODS[da_method],
                                        std_obs=std_obs,
                                        truth_kwargs=truth_kwargs,
                                        ensemble_kwargs=ensemble_kwargs)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    regularization=dict(
        axes=dict(regularization_factor=list(np.linspace(0., 10., 11)),
                  da_method=['rBA_EnKF']),
        x_key='regularization_factor',
    ),
    ensemble_size=dict(
        axes=dict(m=[10, 20, 40, 60, 80],
                  da_method=['rBA_EnKF', 'EnKF']),
        x_key='m', group_key='da_method', logx=True,
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[15, 25, 35, 50, 70]),
        x_key='Nt_obs', logx=True,
    ),
    equivalence_ratio=dict(
        axes=dict(ER=list(ERs),
                  da_method=['rBA_EnKF', 'EnKF']),
        x_key='ER', group_key='da_method',
    ),
)


# ------------------------------- Main -------------------------------- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES.keys(), required=True)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--quick', action='store_true', help='2 values per axis, 1 seed')
    parser.add_argument('--plot', action='store_true', help='only plot existing results')
    args = parser.parse_args()

    study = STUDIES[args.study]
    axes = study['axes']
    n_seeds = args.seeds

    if args.quick:
        axes = {k: v[:2] for k, v in axes.items()}
        n_seeds = 1

    results_file = f'{results_folder}annular_{args.study}.jsonl'

    if args.plot:
        records = load_results(results_file)
    else:
        records = run_sweep(experiment, axes, results_file, n_seeds=n_seeds)

    plot_ablation(records,
                  x_key=study['x_key'],
                  group_key=study.get('group_key'),
                  logx=study.get('logx', False),
                  title=f'Annular — {args.study}',
                  filename=os.path.join(figs_folder, f'annular_{args.study}.png'))
    plt.show()
