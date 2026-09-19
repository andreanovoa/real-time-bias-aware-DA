"""Generic harness for the ablation studies.

An ablation study is defined by a set of *axes* (parameter name -> list of
values). The harness expands the Cartesian product of the axes, runs a
user-supplied experiment function for each grid point and random seed, and
stores one record per run in a JSON-lines file so that interrupted sweeps can
be resumed. Aggregation and plotting helpers summarize the results as
mean +/- std over the repeated seeds.

The experiment function receives the grid point as keyword arguments plus a
``seed`` and must return a dict of scalar metrics (see
``romda.metrics.state_metrics`` / ``scripts/mains/common.compute_metrics``).

Ported unchanged from ``scripts/ablations/ablation_tools.py`` (which is now a
thin re-export shim over this module).
"""

import itertools
import json
import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

__all__ = [
    'expand_grid',
    'load_results',
    'filter_context',
    'run_sweep',
    'summarize',
    'collect',
    'METRIC_LABELS',
    'AXIS_HUES',
    'plot_ablation',
    'dedupe',
    'add_lyapunov_axis',
]


def expand_grid(axes: dict) -> list:
    """Cartesian product of the axes: dict of lists -> list of dicts."""
    keys = list(axes.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def _record_id(point: dict, seed: int) -> str:
    return json.dumps({**point, 'seed': seed}, sort_keys=True, default=str)


def load_results(results_file: str) -> list:
    """Load all completed records from a JSON-lines results file."""
    records = []
    if os.path.isfile(results_file):
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _context_str(context, implicit=None) -> str:
    # `implicit` is filled in on both sides of the comparison: a record stamped without
    # a key matches for as long as the current value is the implicit one
    if context is not None:
        for block, values in (implicit or {}).items():
            context = {**context, block: {**values, **context.get(block, {})}}
    return json.dumps(context, sort_keys=True, default=str)


def filter_context(records: list, context, implicit=None) -> list:
    """Only the records produced under these non-swept parameters (see `run_sweep`).

    Records written before context stamping existed (no ``context`` field) are
    dropped too — there is no way to know which defaults they ran under.
    `implicit`, ``{block: {key: value}}``, is what a key meant before the defaults
    spelt it out: a new default key whose value is the implicit one keeps the
    records that were stamped without it.
    """
    ctx = _context_str(context, implicit)
    return [r for r in records if _context_str(r.get('context'), implicit) == ctx]


def run_sweep(experiment,
              axes: dict,
              results_file: str,
              n_seeds: int = 5,
              resume: bool = True,
              context=None,
              implicit=None) -> list:
    """Run ``experiment(seed=..., **point)`` over the grid and repeated seeds.

    Parameters
    ----------
    experiment : callable
        Maps a grid point + seed to a dict of scalar metrics.
    axes : dict
        Parameter name -> list of values to sweep (Cartesian product).
    results_file : str
        JSON-lines output; one record per (point, seed), written incrementally.
    n_seeds : int
        Number of repeated runs per grid point (seeds 0..n_seeds-1).
    resume : bool
        Skip (point, seed) combinations already present in ``results_file``.
    context, implicit
        The non-swept defaults stamped into each record, and what their absent keys
        mean (see `filter_context`).

    Returns
    -------
    list
        All records (previous + new). Failed runs are recorded with
        ``status='failed'`` and skipped by the aggregation helpers.
    """
    os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)

    records = load_results(results_file) if resume else []
    # Only 'ok' records count as done: failed runs (e.g. from a transient
    # environment problem) are retried on the next invocation. With `context` given
    # (the experiment's non-swept defaults), a record additionally has to match it —
    # changing a default re-runs the grid instead of silently reusing results
    # computed under the old values.
    ctx = _context_str(context, implicit) if context is not None else None
    done = {_record_id({k: r[k] for k in axes}, r['seed']) for r in records
            if all(k in r for k in axes) and r.get('status') == 'ok'
            and (ctx is None or _context_str(r.get('context'), implicit) == ctx)}

    grid = expand_grid(axes)
    total = len(grid) * n_seeds
    print(f'Ablation sweep: {len(grid)} grid points x {n_seeds} seeds = {total} runs '
          f'({len(done)} already done)')

    with open(results_file, 'a') as f:
        for point in grid:
            for seed in range(n_seeds):
                if _record_id(point, seed) in done:
                    continue

                label = ', '.join(f'{k}={v}' for k, v in point.items())
                print(f'\n>>> Running [{label}] seed={seed}')
                t0 = time.time()
                record = {**point, 'seed': seed}
                if context is not None:
                    record['context'] = context
                try:
                    metrics = experiment(seed=seed, **point)
                    record.update(metrics)
                    record['status'] = 'ok'
                except Exception:
                    traceback.print_exc()
                    record['status'] = 'failed'
                record['walltime'] = time.time() - t0

                f.write(json.dumps(record, default=str) + '\n')
                f.flush()
                records.append(record)

    return records


def summarize(records: list, x_key: str, y_key: str, group: dict = None):
    """Aggregate a metric over seeds: mean and std of ``y_key`` vs ``x_key``.

    Parameters
    ----------
    records : list
        Records from `load_results` / `run_sweep`.
    x_key : str
        Ablation axis for the x-values.
    y_key : str
        Metric to aggregate.
    group : dict, optional
        Restrict to records matching these key-value pairs exactly.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Sorted unique x-values, mean and std of the metric at each x.
    """
    rows = [r for r in records
            if r.get('status') == 'ok' and y_key in r
            and all(r.get(k) == v for k, v in (group or {}).items())]

    xs = sorted({r[x_key] for r in rows})
    means, stds = [], []
    for x in xs:
        vals = np.array([r[y_key] for r in rows if r[x_key] == x], dtype=float)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return np.array(xs, dtype=float), np.array(means), np.array(stds)


def collect(records: list, x_key: str, y_key: str, group: dict = None):
    """Raw per-seed metric values grouped by the ablation axis.

    Returns
    -------
    (list, list of np.ndarray)
        Sorted unique x-values and, for each, the finite metric values of the
        repeated runs.
    """
    rows = [r for r in records
            if r.get('status') == 'ok' and y_key in r
            and all(r.get(k) == v for k, v in (group or {}).items())]

    xs = sorted({r[x_key] for r in rows})
    values = []
    for x in xs:
        vals = np.array([r[y_key] for r in rows if r[x_key] == x], dtype=float)
        values.append(vals[np.isfinite(vals)])
    return xs, values


# All RMSE/spread metrics are normalized by the RMS amplitude of the unbiased
# truth over the assimilation window (see scripts/mains/common.compute_metrics)
METRIC_LABELS = dict(rmse_da='normalized RMSE (assimilation)',
                     rmse_forecast='normalized RMSE (forecast)',
                     spread_da='normalized spread (assimilation)',
                     spread_forecast='normalized spread (forecast)',
                     valid_time_forecast='valid time / $T_\\lambda$ (forecast)',
                     spread_skill='spread-skill ratio (after synchronization)',
                     coverage_2sigma='coverage of the $2\\sigma$ band',
                     rank_outliers='truth outside the ensemble range',
                     rho_mean='mean inflation factor',
                     )

# One hue per subplot (left to right); each entry is a (light, dark) anchor pair
# validated for colorblind-safe separation and lightness/chroma bands.
AXIS_HUES = [('#5598e7', '#184f95'),    # blues
             ('#57b657', '#1a6b1a'),    # greens
             ('#9a9a94', '#3f3f3c')]    # neutral fallback for extra panels


def _shades(anchors, n):
    """n shades of one hue: n=1 -> midpoint, n=2 -> the validated (light, dark)
    anchors, n>2 -> interpolated (rely on the legend to separate close shades)."""
    light, dark = (np.array(to_rgb(c)) for c in anchors)
    if n == 1:
        return [tuple((light + dark) / 2)]
    return [tuple(light + (dark - light) * i / (n - 1)) for i in range(n)]


def _fmt(x):
    return f'{x:g}' if isinstance(x, float) else str(x)


def _cap_offscale(ax, overlays):
    """Clip the y-axis to the box-plot scale and flag off-scale runs.

    ``overlays`` is the plotted data as (position, values, color). The cap
    keeps every whisker (data within Q3 + 1.5 IQR) visible; it only kicks in
    when some run lies far above that (a diverged filter can sit orders of
    magnitude above the boxes and squash them flat). Off-scale runs become an
    upward arrow at the top edge annotated with their actual value.
    """
    whisker_tops, vmax, vmin = [], -np.inf, np.inf
    for _, vals, _ in overlays:
        if not len(vals):
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        inliers = vals[vals <= q3 + 1.5 * (q3 - q1)]
        whisker_tops.append(inliers.max() if len(inliers) else vals.max())
        vmax = max(vmax, vals.max())
        vmin = min(vmin, vals.min())
    if not whisker_tops:
        return
    # Metrics are normalized: > 1 means worse than no assimilation, off the chart
    cap = min(1.15 * max(whisker_tops), 1.0)
    if vmax <= cap:
        return
    ax.set_ylim(bottom=vmin - 0.05 * (cap - vmin), top=cap)

    # One arrow + stacked labels per x-slot (labels at one slot would overlap)
    by_pos = {}
    for pos, vals, color in overlays:
        for v in vals[vals > cap]:
            by_pos.setdefault(pos, []).append((v, color))
    for pos, offs in by_pos.items():
        for i, (v, color) in enumerate(sorted(offs, reverse=True)):
            ax.annotate(f'{v:.3g}', xy=(pos, cap), xytext=(0, -13 - 11 * i),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=color,
                        arrowprops=dict(arrowstyle='->', color=color) if i == 0 else None)


def plot_ablation(records: list,
                  x_key: str,
                  y_keys=('rmse_da', 'rmse_forecast'),
                  group_key: str = None,
                  logx: bool = False,
                  kind: str = 'box',
                  title: str = None,
                  filename: str = None):
    """Plot the seed distribution of metric(s) vs an ablation axis.

    Box plots (default) or violins (``kind='violin'``) over the repeated seeds,
    with the individual runs overlaid as dots so that diverged cases are
    visible rather than hidden in a summary statistic. The axis positions are
    categorical (one slot per swept value), so ``logx`` is accepted for
    backward compatibility but has no effect. If ``group_key`` is given, the
    groups share each x-slot and are told apart by light-to-dark shades of the
    subplot's hue (blues on the first panel, greens on the second).
    """
    fig, axs = plt.subplots(1, len(y_keys), figsize=(5 * len(y_keys), 4),
                            sharex=True, layout='constrained')
    axs = np.atleast_1d(axs)

    group_vals = sorted({r[group_key] for r in records if group_key in r},
                        key=str) if group_key else [None]
    groups = [({group_key: gv} if group_key else None) for gv in group_vals]

    # Common categorical positions across groups and subplots
    all_xs = sorted({r[x_key] for r in records if r.get('status') == 'ok'})
    slots = {x: i for i, x in enumerate(all_xs)}

    # Overlaid groups get narrower with each shade so nested boxes stay visible
    box_widths = np.linspace(0.7, 0.35, len(group_vals)) if len(group_vals) > 1 else [0.55]

    for ai, (ax, y_key) in enumerate(zip(axs, y_keys)):
        shades = _shades(AXIS_HUES[ai % len(AXIS_HUES)], len(group_vals))
        overlays = []
        for gi, (gv, group) in enumerate(zip(group_vals, groups)):
            xs, values = collect(records, x_key, y_key, group=group)
            positions = [slots[x] for x, vals in zip(xs, values) if len(vals)]
            values = [vals for vals in values if len(vals)]
            if not values:
                continue
            color = shades[gi]

            if kind == 'violin':
                parts = ax.violinplot(values, positions=positions, widths=box_widths[gi],
                                      showmedians=True, showextrema=False)
                for body in parts['bodies']:
                    body.set_facecolor(color)
                    body.set_alpha(0.3)
                parts['cmedians'].set_color(color)
            else:
                bp = ax.boxplot(values, positions=positions, widths=box_widths[gi],
                                patch_artist=True, zorder=2 + gi,
                                boxprops=dict(edgecolor=color, linewidth=1.5),
                                whiskerprops=dict(color=color, linewidth=1.2),
                                capprops=dict(color=color, linewidth=1.2),
                                medianprops=dict(color=color, linewidth=1.5),
                                flierprops=dict(marker='+', markeredgecolor=color))
                for box in bp['boxes']:
                    box.set(facecolor=(*color, 0.25))

            # Overlay the individual seed runs
            for pos, vals in zip(positions, values):
                ax.plot(np.full(vals.size, pos), vals, '.', color=color,
                        ms=4, zorder=4 + gi)
                overlays.append((pos, vals, color))

        # the cap assumes a normalized error (low is good, 1.0 = no skill); the valid time,
        # the calibration scores and the inflation factor are not on that scale
        if y_key.startswith('rmse'):
            _cap_offscale(ax, overlays)
        ax.set_xticks(range(len(all_xs)), [_fmt(x) for x in all_xs])
        ax.set(xlabel=x_key, ylabel=METRIC_LABELS.get(y_key, y_key))
        if group_key:
            handles = [Patch(facecolor=(*shades[gi], 0.4), edgecolor=shades[gi],
                             label=f'{group_key}={_fmt(gv)}')
                       for gi, gv in enumerate(group_vals)]
            ax.legend(handles=handles, frameon=False)

    if title:
        fig.suptitle(title)
    if filename:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        fig.savefig(filename, dpi=200)
        print(f'Saved figure to {filename}')
    return fig


def dedupe(records: list, axes: dict) -> list:
    """Keep the last record per (grid point, seed).

    `run_sweep` appends, and only skips a point it can already see in the file when it
    starts. Two sweeps of the same study running at once — or one started before an
    earlier one had finished writing — therefore leave repeats behind, which would be
    counted twice in every box.
    """
    seen = {}
    for r in records:
        key = json.dumps({**{k: r.get(k) for k in axes}, 'seed': r.get('seed')},
                         sort_keys=True, default=str)
        seen[key] = r
    return list(seen.values())


def add_lyapunov_axis(fig, records, x_key, model):
    """Second x-axis in Lyapunov times, for the studies swept over `Nt_obs`.

    `Nt_obs` is in model steps, and what actually controls a chaotic filtering problem
    is the observation interval measured against the system's own predictability time —
    e.g. 55 steps for Lorenz-63 at `dt = 0.02`. `plot_ablation` puts the swept values in
    categorical slots, so this is a relabelling of the same ticks rather than a
    rescaling. `model` is the physical model class (anything with `dt` and `t_lyap`).
    """
    if x_key != 'Nt_obs':
        return

    m = model()
    xs = sorted({r[x_key] for r in records if r.get('status') == 'ok'})
    for ax in list(fig.axes):                     # snapshot: each call appends an axes
        sec = ax.secondary_xaxis('top')
        sec.set_xticks(range(len(xs)), [f'{x * m.dt / m.t_lyap:.2f}' for x in xs])
        sec.set_xlabel(f'observation interval / $T_\\lambda$'
                       f'   ($T_\\lambda$ = {m.t_lyap / m.dt:.0f} steps)', fontsize=9)
