"""Twin experiment on the Lorenz (1996) system.

A higher-dimensional chaotic benchmark (``Nx`` cyclic state variables driven
by a constant forcing ``F``) observed at a subset of grid points, with the
forcing estimated online. The whole experiment is described by
``configs/twin_da/lorenz96.yml`` and runs through `romda.experiments.physical` —
the same flow as

    python -m romda.experiments configs/twin_da/lorenz96.yml [--seed N] [--set da.m=32]

Run as a script for the default experiment plus figures, or import
`run_experiment` from the ablation studies in ``scripts/ablations``.
"""

from common import load_main_config, parse_main_args, print_metrics, show_or_save_figs
from romda import experiments as physical
from romda.models.physical import Lorenz96

CFG = load_main_config('twin_da/lorenz96')


def run_experiment(**kwargs):
    """`romda.experiments.physical.run_experiment` on the lorenz96 config; accepts
    the same overrides (`da_method`, `std_obs`, `truth_kwargs`, `ensemble_kwargs`,
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
    print_metrics(metrics, header='Lorenz96 twin experiment')

    filter_ens.visualize_history(truth=truth,
                                 plot_members=False,
                                 reference_t=Lorenz96.t_lyap,
                                 reference_a=truth.true_parameters)
    show_or_save_figs(args.save_figs, prefix='lorenz96')
