"""Ablation studies for the POD-ESN cylinder wake assimilation.

Requires the cylinder wake dataset (downloaded automatically on first run).
The POD-ESN reduced-order model is trained once per configuration and cached
on disk, so repeated sweeps reuse it.

Usage
-----
    python run_cylinder.py --study ensemble_size [--seeds 3] [--quick] [--plot]

Studies
-------
- ``ensemble_size`` : number of members ``m``.
- ``assimilation_frequency`` : time steps between observations ``Nt_obs``.
- ``sensors`` : number of point sensors (QR-pivot placement on the flow field).
- ``wout_estimation`` : state-only vs joint state + ``Wout`` estimation, over
  the parameter uncertainty ``std_alpha`` and initial state uncertainty
  ``std_phi``.
- ``inflation`` : multiplicative inflation factor (the periodic wake keeps the
  seed spread tiny; inflation counteracts the spread collapse).

Results are appended to ``results/ablations/cylinder_<study>.jsonl`` and can be
re-plotted at any time with ``--plot``.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mains'))

import main_pod_esn_cylinder as case
from dev.experiments.sweep import load_results, plot_ablation, run_sweep
from romda.utils import set_working_directories

results_folder = set_working_directories('ablations')[1]

# Ablation figures are versioned in git; the .jsonl records stay in results/
figs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')


# ------------------------- Experiment adapter ------------------------- #

def experiment(seed=0,
               m=50,
               Nt_obs=25,
               N_sensors=None,
               std_phi=1.5,
               est_Wout=False,
               std_alpha=0.02,
               inflation_factor=None,
               std_obs=0.1):
    """Map a flat grid point onto ``main_cylinder.run_experiment`` kwargs.

    ``N_sensors=None`` keeps the default (down-sampled QR) sensor placement of
    the trained ROM; an integer re-selects that many QR-pivot sensors.
    """
    rom_kwargs = {}
    truth_kwargs = dict(Nt_obs=Nt_obs)
    ensemble_kwargs = dict(m=m, std_phi=std_phi, seed=seed)
    if inflation_factor is not None:
        ensemble_kwargs['inflation_factor'] = inflation_factor
    if est_Wout:
        ensemble_kwargs.update(est_alpha=['Wout'], std_alpha=std_alpha)

    (X_train, _), (X_filter, X_filter_true), simulation_dir = case.load_data()
    rom = case.build_rom(X_train, simulation_dir, **rom_kwargs)

    if N_sensors is not None:
        rom = rom.copy()
        rom.select_sensors(N_sensors=N_sensors,
                           domain_of_measurement=[4, 7, -1, 1],   # x_min, x_max, y_min, y_max (wake)
                           down_sample_measurement=(10, 10),
                           qr_selection=True)

    truth = case.build_truth(rom, X_filter, X_filter_true, **truth_kwargs)
    ensemble = case.build_ensemble(rom, **ensemble_kwargs)

    from common import compute_metrics
    from romda.data_assimilation import run_da_loop
    filter_ens = run_da_loop(ensemble, truth, std_obs=std_obs, t_extra=2. * rom.t_CR)
    return compute_metrics(filter_ens, truth)


# ------------------------- Study definitions ------------------------- #

STUDIES = dict(
    ensemble_size=dict(
        axes=dict(m=[10, 50, 100, 200]),
        x_key='m', logx=True,
    ),
    assimilation_frequency=dict(
        axes=dict(Nt_obs=[10, 20, 30, 50]),
        x_key='Nt_obs', logx=True,
    ),
    sensors=dict(
        axes=dict(N_sensors=[1, 2, 3, 4, 8]),
        x_key='N_sensors',
    ),
    inflation=dict(
        axes=dict(inflation_factor=[1.0, 1.01, 1.02, 1.05, 1.1, 1.2]),
        x_key='inflation_factor',
    ),
    wout_estimation=dict(
        axes=dict(est_Wout=[False, True],
                  std_phi=[0.1, 0.5, 1.1],
                  std_alpha=[0.01, 0.05, 0.1, 0.2]),
        x_key='std_alpha', group_key='est_Wout',
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

    results_file = f'{results_folder}cylinder_{args.study}.jsonl'

    if args.plot:
        records = load_results(results_file)
    else:
        records = run_sweep(experiment, axes, results_file, n_seeds=n_seeds)

    plot_ablation(records,
                  x_key=study['x_key'],
                  group_key=study.get('group_key'),
                  logx=study.get('logx', False),
                  title=f'POD-ESN cylinder — {args.study}',
                  filename=os.path.join(figs_folder, f'cylinder_{args.study}.png'))
    plt.show()
