"""Twin experiment on the Rijke tube (Galerkin low-order model).

Acoustic pressure/velocity Galerkin modes coupled to a time-delayed
heat-release law, observed through pressure 'microphones' along the tube,
estimating the heat-release gain ``beta`` and flame time delay ``tau`` online.
The whole experiment is described by ``configs/tai_da/rijke.yml`` and runs
through `romda.experiments.physical` — the same flow as

    python -m romda.experiments configs/tai_da/rijke.yml [--seed N] [--set da.m=32]

Run as a script for the default experiment plus figures, or import
`run_experiment` from the ablation studies in ``scripts/ablations``.
"""

from common import load_main_config, parse_main_args, print_metrics, show_or_save_figs
from romda import experiments as physical
from romda.models.physical import Rijke

CFG = load_main_config('tai_da/rijke')


def run_experiment(**kwargs):
    """`romda.experiments.physical.run_experiment` on the rijke config; accepts the
    same overrides (`da_method`, `std_obs`, `truth_kwargs`, `ensemble_kwargs`,
    `t_extra`, `seed`).

    Returns
    -------
    (EnsembleEstimator, Observations, dict)
        Filtered ensemble, truth, and the error metrics.
    """
    return physical.run_experiment(CFG, **kwargs)


# --------------------------------------- Main ---------------------------------------- #

if __name__ == '__main__':

    args = parse_main_args(description=__doc__)

    filter_ens, truth, metrics = run_experiment()
    print_metrics(metrics, header='Rijke tube twin experiment')

    filter_ens.visualize_history(truth=truth,
                                 plot_members=False,
                                 reference_t=Rijke.t_CR,
                                 reference_a=truth.true_parameters)
    show_or_save_figs(args.save_figs, prefix='rijke')
