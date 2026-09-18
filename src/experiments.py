"""Physical-model and thermoacoustic data-assimilation experiments.

The public experiment entry point is :func:`run_experiment`. YAML files in
``configs/twin_da`` and ``configs/tai_da`` describe physical twin experiments
and the annular experimental-data case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from romda.bias_estimators import ESN_bias
from romda.data_assimilation import run_da_loop
from romda.estimators import EnKF, EnSRKF, rBA_EnKF
from romda.metrics import ensemble_metrics
from romda.models import physical
from romda.observations import Observations
from romda.utils import set_working_directories

DA_METHODS = {'EnKF': EnKF, 'EnSRKF': EnSRKF, 'rBA_EnKF': rBA_EnKF}
_DA_ONLY = ('measured', 'n_measured', 'ring', 'bayesian_update', 'case')


def _resolve_model(name):
    model_class = getattr(physical, name, None)
    if not isinstance(model_class, type):
        raise ValueError(f"Unknown physical model '{name}'")
    return model_class


def _resolve_da_method(name):
    try:
        return DA_METHODS[name] if isinstance(name, str) else name
    except KeyError as exc:
        raise ValueError(f"Unknown data-assimilation method '{name}'") from exc


def _build_truth(model_class, cfg, overrides=None):
    params = {**cfg, **(overrides or {})}
    params.pop('model', None)
    constructor = params.pop('params', None) or {}

    if 'ER' in params:
        from romda.utils import get_annular_data

        ER = params.pop('ER')
        data_folder = set_working_directories(model_class.__name__.lower())[0]
        get_annular_data(data_folder)
        params.setdefault('t_start', model_class.t_transient)
        params['t_stop'] = params['t_start'] + model_class.t_CR * params.pop('t_stop_cr')
        params.setdefault('t_max', params['t_stop'] + model_class.t_transient)
        return Observations(model=os.path.join(data_folder, f'ER_{ER}'), **params)

    if 'observe_every' in params:
        params['observed_idx'] = list(range(0, constructor['Nx'], params.pop('observe_every')))
    window = params.pop('window', None)
    if window is None:
        window = [value * model_class.t_lyap for value in params.pop('window_lyap')]
    t_start, t_stop, t_max = window
    return Observations(model=model_class, t_start=t_start, t_stop=t_stop, t_max=t_max,
                        **constructor, **params)


def _build_ensemble(model_class, cfg, truth, truth_cfg, method=None, overrides=None):
    params = {k: v for k, v in {**cfg, **(overrides or {})}.items()
              if k not in _DA_ONLY and k not in ('method', 'std_obs')}
    da_method = _resolve_da_method(method or cfg.get('method', cfg.get('da_method')))
    if da_method is not rBA_EnKF:
        params.pop('regularization_factor', None)

    constructor = dict(truth_cfg.get('params') or {})
    layout = {k: truth_cfg[k] for k in ('observe_dims', 'observed_idx') if k in truth_cfg}
    estimated = set(params.get('std_alpha') or ())
    shared = {k: v for k, v in constructor.items()
              if k not in estimated and k != 'dt'}
    return da_method(parent_model=model_class, dt=truth.dt, **shared, **layout, **params)


def _build_bias(cfg, ensemble, truth, model_class, experimental=False, overrides=None):
    params = {**(cfg or {}), **(overrides or {})}
    for key in ('t_train', 't_val', 't_test'):
        if f'{key}_lyap' in params:
            params[key] = params.pop(f'{key}_lyap') * model_class.t_lyap
    name = model_class.__name__.lower()
    if experimental:
        params.setdefault('t_train', model_class.t_transient / 3.)
        params.setdefault('t_test', model_class.t_CR * 2)
        params.setdefault('t_val', model_class.t_CR * 2)
        tag = f'ESN_train_data_{name}_raw'
    else:
        tag = f'ESN_train_data_{name}_{truth.name_bias}'
    results_folder = set_working_directories(name)[1]
    os.makedirs(results_folder, exist_ok=True)
    return ESN_bias(rom=ensemble.model, reference_data=truth,
                    training_data_filename=f'{results_folder}{tag}',
                    N_ens=ensemble.m, **params)


def run_experiment(config, seed=0, da_method=None, std_obs=None,
                   truth_kwargs=None, ensemble_kwargs=None, bias_kwargs=None,
                   t_extra=None):
    """Run a physical twin or thermoacoustic experiment.

    Parameters
    ----------
    config : dict or path-like
        Resolved experiment mapping or a YAML configuration path.
    seed : int, default=0
        Seed used to redraw the physical-model ensemble.

    Returns
    -------
    tuple
        Filtered ensemble, truth observations, and scalar metrics.
    """
    if isinstance(config, (str, Path)):
        with open(config) as stream:
            config = yaml.safe_load(stream)
    truth_cfg = {**config['truth'], **(truth_kwargs or {})}
    model_class = _resolve_model(truth_cfg['model'])
    experimental = 'ER' in truth_cfg
    truth = _build_truth(model_class, config['truth'], truth_kwargs)
    ensemble_options = dict(ensemble_kwargs or {})
    if seed:
        ensemble_options.setdefault('seed', seed)
    ensemble = _build_ensemble(model_class, config['da'], truth, truth_cfg,
                               method=da_method, overrides=ensemble_options)
    if isinstance(ensemble, rBA_EnKF):
        ensemble.bias = _build_bias(config.get('bias'), ensemble, truth, model_class,
                                    experimental=experimental, overrides=bias_kwargs)
    if t_extra is None and experimental:
        t_extra = 5. * model_class.t_CR
    filtered = run_da_loop(ensemble, truth,
                           std_obs=config['da']['std_obs'] if std_obs is None else std_obs,
                           t_extra=t_extra)
    return filtered, truth, ensemble_metrics(filtered, truth)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='path to a physical experiment YAML file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args(argv)
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    for item in args.overrides:
        key, _, raw = item.partition('=')
        if not key or not _:
            parser.error(f'--set expects KEY=VALUE, got {item!r}')
        block = config
        parts = key.split('.')
        for part in parts[:-1]:
            block = block.setdefault(part, {})
        try:
            block[parts[-1]] = json.loads(raw)
        except json.JSONDecodeError:
            block[parts[-1]] = raw
    _, _, metrics = run_experiment(config, seed=args.seed)
    print(json.dumps(metrics))


if __name__ == '__main__':
    _main()