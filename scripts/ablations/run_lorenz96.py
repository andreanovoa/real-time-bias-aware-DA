"""Ablation studies for the Lorenz96 twin experiment.

Usage
-----
    python run_lorenz96.py --study ensemble_size [--seeds 5] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m`` for EnKF vs EnSRKF.
- ``observation_sparsity`` : time steps between observations ``Nt_obs`` x
  observing every k-th of the ``Nx = 40`` variables (one combined figure).
- ``inflation`` : multiplicative inflation factor (undersampled regime, small m).

Results are appended to ``results/ablations/lorenz96_<study>.jsonl`` and can be
re-plotted at any time with ``--plot``.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mains'))

import main_lorenz96 as case
from romda.estimators import EnKF, EnSRKF
from dev.experiments.sweep import load_results, plot_ablation, run_sweep
from romda.utils import set_working_directories

DA_METHODS = dict(EnKF=EnKF, EnSRKF=EnSRKF)

results_folder = set_working_directories('ablations')[1]

# Ablation figures are versioned in git; the .jsonl records stay in results/
figs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')

Nx = case.CFG['truth']['params']['Nx']


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               da_method='EnSRKF',
               m=50,
               Nt_obs=10,
               obs_every=2,
               inflation_factor=1.05,
               noise_level=0.05,
               std_obs=0.05):
    """Map a flat grid point onto ``main_lorenz96.run_experiment`` kwargs."""
    observed_idx = list(range(0, Nx, obs_every))
    truth_kwargs = dict(Nt_obs=Nt_obs, noise_level=noise_level, observe_every=obs_every)
    ensemble_kwargs = dict(m=m, inflation_factor=inflation_factor, seed=seed)

    _, _, metrics = case.run_experiment(da_method=DA_METHODS[da_method],
                                        std_obs=std_obs,
                                        truth_kwargs=truth_kwargs,
                                        ensemble_kwargs=ensemble_kwargs)
    metrics['Nq'] = len(observed_idx)
    return metrics


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[10, 20, 50, 100],
                  da_method=['EnKF', 'EnSRKF']),
        x_key='m', group_key='da_method', logx=True,
    ),
    observation_sparsity=dict(
        # Temporal (Nt_obs) x spatial (obs_every) observation sparsity in one figure
        axes=dict(Nt_obs=[5, 10, 20, 40],
                  obs_every=[1, 2, 4, 8]),
        x_key='Nt_obs', group_key='obs_every', logx=True,
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.02, 1.05, 1.1],
                  m=[20]),      # inflation matters most for small ensembles
        x_key='inflation_factor',
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

    results_file = f'{results_folder}lorenz96_{args.study}.jsonl'

    if args.plot:
        records = load_results(results_file)
    else:
        records = run_sweep(experiment, axes, results_file, n_seeds=n_seeds)

    plot_ablation(records,
                  x_key=study['x_key'],
                  group_key=study.get('group_key'),
                  logx=study.get('logx', False),
                  title=f'Lorenz96 — {args.study}',
                  filename=os.path.join(figs_folder, f'lorenz96_{args.study}.png'))
    plt.show()
