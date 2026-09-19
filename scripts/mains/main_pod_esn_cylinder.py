"""Real-time data assimilation of the cylinder wake with a POD-ESN model.

A POD-ESN reduced-order model is trained on noisy snapshots of the cylinder
wake (Re = 100), sparse sensors are placed on the flow field (QR pivoting) or
directly on the POD coefficients, and the remaining snapshots provide the
observations for sequential filtering with the EnSRKF. Optionally the ESN
output weights (``Wout`` singular values) are appended to the augmented state
and estimated online. Defaults live in ``configs/rom_da/pod_esn_cylinder.yml`` — the ROM-DA
campaign: data-driven forecast models, no true model required.

Run as a script for a single default experiment, or import the builder
functions from the ablation studies in ``scripts/ablations``.
"""

import os

import numpy as np
from common import (
    compute_metrics,
    load_main_config,
    parse_main_args,
    print_metrics,
    resolve_da_method,
    show_or_save_figs,
)
from romda.data_assimilation import run_da_loop
from romda.estimators import EnsembleEstimator
from romda.models.data_driven import POD_ESN
from romda.observations import Observations
from romda.utils import load_cylinder_dataset, load_from_pickle_file, save_to_pickle_file, set_working_directories

data_folder, results_folder, figs_folder = set_working_directories('wakes')

CFG = load_main_config('rom_da/pod_esn_cylinder')

# Number of training/validation snapshots; the rest are used for assimilation
N_train, N_val = CFG['data']['N_train'], CFG['data']['N_val']


# ------------------------------------- Builders -------------------------------------- #

def load_data(**overrides):
    """Load the (noisy, true) cylinder snapshots and split train/filter sets."""
    params = {**CFG['data'], **overrides}
    params.pop('N_train'), params.pop('N_val')
    X_true, X_noisy, simulation_dir = load_cylinder_dataset(**params)

    X_train, X_train_true = [yy[:N_train + N_val].transpose(-1, 0, 1, 2) for yy in [X_noisy, X_true]]
    X_filter, X_filter_true = [yy[N_train + N_val:].transpose(-1, 0, 1, 2) for yy in [X_noisy, X_true]]

    return (X_train, X_train_true), (X_filter, X_filter_true), simulation_dir


def build_rom(X_train, simulation_dir, use_cache=True, **overrides) -> POD_ESN:
    """Train (or load from cache) the POD-ESN reduced-order model."""
    params = {**CFG['rom'], **overrides}
    params.setdefault('t_train', N_train * params['dt'])
    params.setdefault('t_val', N_val * params['dt'])
    case_filename = (f"{simulation_dir}POD{params['N_modes']}_ESN{params['N_units']}"
                     f"_Ntrain{X_train.shape[1]}")

    if use_cache and os.path.isfile(case_filename):
        rom = load_from_pickle_file(case_filename)
        print(f'Loaded POD-ESN case from {case_filename}')
    else:
        rom = POD_ESN(data=X_train, figs_folder=simulation_dir, **params)
        rom.name = case_filename
        if use_cache:
            save_to_pickle_file(case_filename, rom)

    return rom


def build_truth(rom: POD_ESN, X_filter, X_filter_true, **overrides) -> Observations:
    """Sample sensor measurements from the held-out snapshots."""
    params = {**CFG['truth'], **overrides}

    N_test = X_filter.shape[1]
    X_flat = X_filter.transpose(1, 0, 2, 3).reshape(N_test, -1)            # Nt x Ndof
    X_flat_true = X_filter_true.transpose(1, 0, 2, 3).reshape(N_test, -1)  # Nt x Ndof

    t_true = np.arange(N_test) * rom.dt
    params['t_stop'] = min(params['t_stop'], t_true[-10])

    return Observations(y_raw=X_flat[:, rom.sensor_locations],
                        y_true=X_flat_true[:, rom.sensor_locations],
                        t_true=t_true,
                        **params)


def build_ensemble(rom: POD_ESN, da_method=None, **overrides) -> EnsembleEstimator:
    params = {**CFG['da'], **overrides}
    params.pop('std_obs', None)
    da_method = resolve_da_method(da_method or params.pop('method'))
    params.pop('method', None)
    return da_method(parent_model=rom, **params)


def run_experiment(da_method=None,
                   std_obs=None,
                   data_kwargs=None,
                   rom_kwargs=None,
                   truth_kwargs=None,
                   ensemble_kwargs=None,
                   t_extra=None):
    """Train/load the POD-ESN, build observations from held-out snapshots, filter.

    Returns
    -------
    (EnsembleEstimator, Observations, dict)
        Filtered ensemble, truth, and the error metrics.
    """
    (X_train, _), (X_filter, X_filter_true), simulation_dir = load_data(**(data_kwargs or {}))

    rom = build_rom(X_train, simulation_dir, **(rom_kwargs or {}))
    truth = build_truth(rom, X_filter, X_filter_true, **(truth_kwargs or {}))
    ensemble = build_ensemble(rom, da_method=da_method, **(ensemble_kwargs or {}))

    if t_extra is None:
        t_extra = 2. * rom.t_CR

    filter_ens = run_da_loop(ensemble, truth,
                             std_obs=CFG['da']['std_obs'] if std_obs is None else std_obs,
                             t_extra=t_extra)
    return filter_ens, truth, compute_metrics(filter_ens, truth)


# --------------------------------------- Main ---------------------------------------- #

if __name__ == '__main__':

    args = parse_main_args(description=__doc__)

    # State estimation only
    filter_ens, truth, metrics = run_experiment()
    print_metrics(metrics, header='POD-ESN cylinder wake (state estimation)')
    filter_ens.visualize_history(truth=truth, plot_members=True)

    # State estimation + online update of the Wout singular values
    filter_ens_W, truth_W, metrics_W = run_experiment(
        ensemble_kwargs=dict(std_phi=0.1, est_alpha=['Wout'], std_alpha=0.02))
    print_metrics(metrics_W, header='POD-ESN cylinder wake (state + Wout estimation)')
    filter_ens_W.visualize_history(truth=truth_W, plot_members=True)

    os.makedirs(results_folder, exist_ok=True)
    save_to_pickle_file(f'{results_folder}cylinder_main_results',
                        truth, filter_ens, filter_ens_W)

    show_or_save_figs(args.save_figs, prefix='cylinder')
