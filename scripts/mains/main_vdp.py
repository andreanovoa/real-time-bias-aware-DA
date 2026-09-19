"""Twin experiment on the Van der Pol thermoacoustic oscillator.

The lowest-order thermoacoustic test case: a self-excited oscillator with a
saturated heat-release law, estimating the growth rate ``beta`` and damping
``zeta`` online with a bias-blind (EnKF/EnSRKF) or bias-aware (rBA-EnKF + ESN)
method. The whole experiment is described by ``configs/tai_da/vdp.yml`` and
runs through `romda.experiments.physical` — the same flow as

    python -m romda.experiments configs/tai_da/vdp.yml [--seed N] [--set da.m=32]

Run as a script for the default experiment plus figures, or import
`run_experiment` from the ablation studies in ``scripts/ablations``.
"""

from common import load_main_config, parse_main_args, print_metrics, show_or_save_figs
from romda import experiments as physical
from romda.models.physical import VdP

CFG = load_main_config('tai_da/vdp')


def run_experiment(**kwargs):
    """`romda.experiments.physical.run_experiment` on the vdp config; accepts the
    same overrides (`da_method`, `std_obs`, `truth_kwargs`, `ensemble_kwargs`,
    `bias_kwargs`, `t_extra`, `seed`).

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
    print_metrics(metrics, header='VdP twin experiment')

    filter_ens.visualize_history(truth=truth,
                                 plot_members=True,
                                 reference_t=VdP.t_CR,
                                 reference_a=truth.true_parameters)
    show_or_save_figs(args.save_figs, prefix='vdp')
