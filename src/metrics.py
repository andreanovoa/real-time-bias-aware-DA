"""Scalar and time-resolved error metrics for deployed forecast models.

Full-state scoring of a deployed forecast result against a known truth
record: `valid_time` (time-to-threshold of one error series), `state_metrics`
(scalar summary comparable across deployments) and `state_error` (the error
curve behind it). Ported from `dev/ESN-DA/esnda_tools.py` with metric keys
unchanged, so existing jsonl ablation records stay comparable.
"""

import numpy as np
from romda.utils import interpolate


def valid_time(pred, target, norm, threshold=0.3, dt=1.0):
    """Time until the normalized error first exceeds `threshold`, and the error series."""
    err = np.sqrt((((pred - target) / norm) ** 2).mean(axis=1))
    k = int(np.argmax(err > threshold)) if (err > threshold).any() else len(err)
    return k * dt, err


def _check_truth_covers(t, t_true):
    """Refuse to score a model run that extends past the end of the truth record —
    the interpolation would silently extrapolate garbage there."""
    if t[-1] > t_true[-1] + 1e-9:
        raise ValueError(
            f'The model history runs to t = {t[-1]:.2f} but the truth record ends at '
            f't = {t_true[-1]:.2f}: more assimilation cycles (N_da) and/or t_extra than '
            'the available truth covers. Run the truth for longer (hold out more test '
            'data), or reduce N_da / Nt_obs / t_extra.')


def state_metrics(result, truth, t_true, y_true, t_lyap=None, vt_threshold=0.5, t_sync=None):
    """Full-state metrics of a `deploy` result, comparable across both deployments.

    Scored on the full clean state, normalized by the truth's RMS amplitude over the
    assimilation window (as `scripts/mains.compute_metrics`). `rmse_*`/`std_*` are
    the mean/std over each member's own RMSE; `rmse_mean_*` scores the ensemble-mean
    trajectory and `spread_*` the RMS spread; estimated parameters add
    `alpha_<name>` (+std). `valid_time_*` is the free-forecast valid time (error
    below `vt_threshold` after the last observation, censored at the end of the
    run; in Lyapunov times when `t_lyap` is given).

    The ensemble calibration is scored on the post-synchronization window
    `t_obs[0] + t_sync <= t <= t_obs[-1]` (`t_sync` defaults to `2 * t_lyap`, or to 0
    without `t_lyap`), because the members that leave the attractor before the
    ensemble locks dominate the `*_da` scores. `rmse_mean_sync`/`spread_sync` are
    `rmse_mean_da`/`spread_da` on that window. `spread_skill` is their ratio, spread
    over error, times the finite-ensemble factor sqrt((m+1)/(m-1)) (Fortin et al.
    2014), so that a calibrated ensemble (the truth is statistically one more member)
    gives 1 for every `m`. `coverage_2sigma` is the fraction of (time, component)
    entries with |ensemble mean - truth| <= 2 member std. This band has no
    finite-ensemble factor: a calibrated ensemble gives 0.88 for m = 8, 0.93 for
    m = 20, 0.95 for m = 64, and 0.954 as m -> inf. `rank_outliers` is the fraction of
    entries in which the truth falls outside the ensemble range, i.e. in the two
    extreme bins of the rank histogram (2/(m+1) for a calibrated ensemble).
    `alpha_cover_<name>` is 1.0 when the true parameter is within 2 std of the final
    ensemble mean, and 0.0 otherwise."""
    t, y = result['t'], result['y']
    _check_truth_covers(t, t_true)
    t_obs = truth.t_obs
    y_ref = interpolate(t_true, y_true, t)
    win_da = (t >= t_obs[0]) & (t <= t_obs[-1])
    win_fc = t > t_obs[-1]
    if t_sync is None:
        t_sync = 2 * t_lyap if t_lyap else 0.
    win_sync = win_da & (t >= t_obs[0] + t_sync)
    scale = np.sqrt(np.mean(y_ref[win_da] ** 2))

    def nrmse(w):
        return float(np.sqrt(np.mean((y[w] - y_ref[w]) ** 2)) / scale) if w.any() else np.nan

    metrics = dict(rmse_mean_da=nrmse(win_da), rmse_mean_forecast=nrmse(win_fc),
                   rmse_mean_sync=nrmse(win_sync))

    def first_exceed(err_fc):
        """Valid time of one error series over `win_fc`, clock at the last observation."""
        if not win_fc.any():
            return np.nan
        t_fc = t[win_fc]
        over = err_fc > vt_threshold
        t_v = t_fc[int(np.argmax(over))] if over.any() else t_fc[-1]
        return float((t_v - t_obs[-1]) / (t_lyap or 1.0))

    err_fc_mean = (np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale)[win_fc]
    metrics['valid_time_mean_forecast'] = first_exceed(err_fc_mean)

    ensemble = result.get('ensemble')
    members = result.get('y_members')                          # open-loop ensemble
    if ensemble is not None:
        members = ensemble.model.hist[:, :ensemble.model.N_dim, :]
    if members is None:
        metrics.update(rmse_da=metrics['rmse_mean_da'],
                       rmse_forecast=metrics['rmse_mean_forecast'],
                       valid_time_forecast=metrics['valid_time_mean_forecast'],
                       std_da=np.nan, std_forecast=np.nan, valid_time_std=np.nan,
                       spread_da=np.nan, spread_forecast=np.nan,
                       spread_sync=np.nan, spread_skill=np.nan,
                       coverage_2sigma=np.nan, rank_outliers=np.nan)
        return metrics

    e = members - y_ref[:, :, None]                            # (Nt, N_dim, m)
    spread = members.std(axis=-1)

    def per_member(w):
        if not w.any():
            return np.nan, np.nan
        e_m = np.sqrt(np.mean(e[w] ** 2, axis=(0, 1))) / scale   # (m,) member RMSEs
        return float(e_m.mean()), float(e_m.std())

    def nspread(w):
        return float(np.sqrt(np.mean(spread[w] ** 2)) / scale) if w.any() else np.nan

    metrics['rmse_da'], metrics['std_da'] = per_member(win_da)
    metrics['rmse_forecast'], metrics['std_forecast'] = per_member(win_fc)
    metrics.update(spread_da=nspread(win_da), spread_forecast=nspread(win_fc))

    # calibration after the ensemble locks: spread against error, 2-sigma band, rank extremes
    def fraction(flag):
        return float(flag[win_sync].mean()) if win_sync.any() else np.nan

    rmse_sync, m = metrics['rmse_mean_sync'], members.shape[-1]
    metrics.update(spread_sync=nspread(win_sync),
                   coverage_2sigma=fraction(np.abs(y - y_ref) <= 2 * spread),
                   rank_outliers=fraction((e.min(axis=-1) > 0) | (e.max(axis=-1) < 0)))
    # E[error^2] = (1 + 1/m) sigma^2 and E[spread^2] = (1 - 1/m) sigma^2 when calibrated
    metrics['spread_skill'] = (float(np.sqrt((m + 1) / (m - 1)) * metrics['spread_sync'] / rmse_sync)
                               if rmse_sync > 0 else np.nan)

    err_fc_members = (np.sqrt((e ** 2).mean(axis=1)) / scale)[win_fc]     # (Nt_fc, m)
    vts = [first_exceed(err_fc_members[:, j]) for j in range(members.shape[-1])]
    metrics['valid_time_forecast'] = float(np.mean(vts))
    metrics['valid_time_std'] = float(np.std(vts))

    if ensemble is not None and ensemble.Na > 0:
        model = ensemble.model
        alpha_hist = model.hist[-1, model.Nphi:model.Nphi + ensemble.Na, :]
        true_params = getattr(truth, 'true_parameters', None) or {}
        for key, val, s in zip(ensemble.est_alpha,
                               alpha_hist.mean(axis=-1), alpha_hist.std(axis=-1)):
            metrics[f'alpha_{key}'] = float(val)
            metrics[f'alpha_std_{key}'] = float(s)
            if key in true_params:
                a_true = true_params[key]
                metrics[f'alpha_error_{key}'] = float(abs(val - a_true) / max(abs(a_true), 1e-12))
                metrics[f'alpha_cover_{key}'] = float(abs(val - a_true) <= 2 * s)

    return metrics


def state_error(result, t_true, y_true, per_member=True):
    """Normalized full-state error of a `deploy` result over time, as `(t, err, spread)`.

    `per_member=True` scores each member's own curve (mean/std across members);
    otherwise `err` is the ensemble-mean trajectory's error and `spread` the RMS
    spread. `spread` is all-NaN for the single-network open loop."""
    t, y = result['t'], result['y']
    _check_truth_covers(t, t_true)
    y_ref = interpolate(t_true, y_true, t)
    scale = np.sqrt(np.mean(y_ref ** 2))

    ensemble = result.get('ensemble')
    hist = result.get('y_members')                            # open-loop ensemble
    if ensemble is not None:
        hist = ensemble.model.hist[:, :ensemble.model.N_dim, :]
    if hist is None:
        err = np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale
        return t, err, np.full_like(err, np.nan)
    if per_member:
        e = np.sqrt(((hist - y_ref[:, :, None]) ** 2).mean(axis=1)) / scale   # (Nt, m)
        return t, e.mean(axis=-1), e.std(axis=-1)

    err = np.sqrt(((y - y_ref) ** 2).mean(axis=1)) / scale
    spread = np.sqrt((hist.std(axis=-1) ** 2).mean(axis=1)) / scale
    return t, err, spread


def ensemble_metrics(filter_ens, truth) -> dict:
    """Scalar error metrics of a filtered ensemble against the unbiased truth.

    The physical-twin counterpart of `state_metrics` (which scores `deploy`
    results): works off the estimator's model histories. All RMS errors are
    computed on the ensemble-mean observables and normalized by the RMS
    amplitude of the truth over the assimilation window.

    Returns
    -------
    dict
        ``rmse_da`` / ``rmse_forecast`` : normalized RMSE over the assimilation
        window and the post-assimilation forecast; ``spread_da`` /
        ``spread_forecast`` : normalized mean ensemble spread; ``alpha_<name>``
        (+ ``alpha_error_<name>`` when the truth stores ``true_parameters``) :
        final ensemble-mean parameter estimates.
    """
    model = filter_ens.model
    t = model.hist_t
    y = model.get_observable_hist()          # (Nt, Nq, m)
    y_mean = np.mean(y, axis=-1)
    spread = np.std(y, axis=-1)

    y_true = truth.y_true
    if y_true.ndim == 3:
        y_true = np.mean(y_true, axis=-1)
    y_ref = interpolate(truth.t_true, y_true, t)

    t_first, t_last = truth.t_obs[0], truth.t_obs[-1]
    win_da = (t >= t_first) & (t <= t_last)
    win_fc = t > t_last

    scale = np.sqrt(np.mean(y_ref[win_da] ** 2))

    def _nrmse(window):
        return float(np.sqrt(np.mean((y_mean[window] - y_ref[window]) ** 2)) / scale)

    def _nspread(window):
        return float(np.mean(spread[window]) / scale)

    metrics = dict(rmse_da=_nrmse(win_da),
                   rmse_forecast=_nrmse(win_fc) if win_fc.any() else np.nan,
                   spread_da=_nspread(win_da),
                   spread_forecast=_nspread(win_fc) if win_fc.any() else np.nan,
                   )

    # Parameter estimates (rows Nphi..Nphi+Na of the state augmentation)
    if filter_ens.Na > 0:
        alpha_hist = model.hist[:, model.Nphi:model.Nphi + filter_ens.Na, :]
        alpha_final = np.mean(alpha_hist[-1], axis=-1)
        for key, val in zip(filter_ens.est_alpha, alpha_final):
            metrics[f'alpha_{key}'] = float(val)
            if truth.true_parameters and key in truth.true_parameters:
                a_true = truth.true_parameters[key]
                metrics[f'alpha_error_{key}'] = float(abs(val - a_true) / max(abs(a_true), 1e-12))

    return metrics
