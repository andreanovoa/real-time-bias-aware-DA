
from typing import Optional

import numpy as np
from numpy.linalg import inv
from romda.estimators.base import Estimator

# ══════════════════════════════════════════════════════════════════════════════
# Deterministic estimators  (own mean + covariance, self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicEstimator(Estimator):
    r"""Base for non-ensemble estimators that own an explicit mean and covariance (e.g., KF, UKF).

    Unlike `EnsembleEstimator`, the mean $\boldsymbol{\psi}$ and covariance
    $\mathbf{C}_{\psi\psi}$ live on the estimator itself; the (optional) model is
    only used to advance the mean in time. With `model=None` the mean is advanced
    by the linear map `F` instead, one application per step, and `current_time`
    counts steps.

    Parameters
    ----------
    N : int
        State dimension.
    Nq : int
        Observable dimension.
    Cdd : ndarray, shape (Nq, Nq)
        Default observation-noise covariance.
    psi0 : ndarray, shape (N,)
        Initial state mean.
    Cpp0 : ndarray, shape (N, N)
        Initial state covariance $\mathbf{C}_{\psi\psi,\,0|0}$.
    M : ndarray, shape (Nq, N), or callable, optional
        Measurement operator. Defaults to `model.M`.
    model : Model, optional
        Nonlinear model used for the mean forecast. Either `model` or `F` must be
        provided.
    F : ndarray, shape (N, N), optional
        Linear transition matrix, used instead of `model` for the mean forecast.
    Q : ndarray, shape (N, N), optional
        Process-noise covariance added at every covariance propagation
        (defaults to zeros).
    """

    def __init__(
        self,
        N:    int,
        Nq:   int,
        Cdd:  np.ndarray,
        psi0: np.ndarray,
        Cpp0: np.ndarray,
        M=None,
        model=None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        **kwargs,
    ):
        #  Apply general DA attributes from Estimator base (same order as
        #  EnsembleEstimator: config first, then the estimator's own state).
        super().__init__(**kwargs)

        if model is None and F is None:
            raise ValueError(
                f"{self.__class__.__name__}: provide either a Model instance "
                "(model=...) or a linear transition matrix (F=...)."
            )

        self.N    = N
        self._Nq  = Nq
        self.Cdd  = np.atleast_2d(Cdd)
        self.Q    = np.atleast_2d(Q) if Q is not None else np.zeros((N, N))

        self._model = model
        self._F     = np.atleast_2d(F) if F is not None else None

        # Resolve measurement operator M
        if M is not None:
            self._M = M
        elif model is not None:
            self._M = model.M
        else:
            raise ValueError("M must be supplied when no model is provided.")

        self._psi = np.array(psi0, dtype=float).ravel()
        self._Cpp = np.array(Cpp0, dtype=float)          # posterior covariance P_{k|k}
        self._t: float = float(model.current_time) if model is not None else 0.0

    # ── Overrides of base properties that delegate to the (possibly absent) model ──

    @property
    def Nq(self) -> int:
        return self._Nq

    @property
    def M(self):
        return self._M

    @property
    def current_state(self) -> np.ndarray:
        """Current mean estimate (owned by the estimator, not the model)."""
        return self._psi

    @property
    def current_time(self) -> float:
        return self._t

    @property
    def Cpp(self) -> np.ndarray:
        """Current state covariance (forecast after `forecast_step`, posterior after `analysis_step`)."""
        return self._Cpp

    # ── Forecast (mean; covariance is propagated by subclasses) ──────────────

    def forecast_step(self, t_end=None, **kwargs) -> None:
        """Advance the state mean: via the model if present, else via ``F``.

        With a model, this delegates to `Estimator.forecast_step` (which
        advances the model history) and then refreshes the mean from the
        model's current state. Without a model, ``F`` is applied ``Nt`` times
        (default 1) and `current_time` counts steps; `t_end` is not supported.
        """
        if self._model is not None:
            super().forecast_step(t_end=t_end, **kwargs)
            psi = np.asarray(self.model.current_state, dtype=float)
            self._psi = psi.mean(axis=-1) if psi.ndim > 1 else psi
            self._t = float(self.model.current_time)
            return

        if t_end is not None:
            raise ValueError(
                "t_end requires a model; F-only mode advances Nt steps (default 1)."
            )
        Nt = kwargs.get("Nt", 1)
        for _ in range(Nt):
            self._psi = self._F @ self._psi
        self._t += Nt

    # ── internal forecast helpers ─────────────────────────────────────────────

    def _propagate_Cpp(self, J: np.ndarray) -> np.ndarray:
        r"""Forecast covariance: $\mathbf{C}^\mathrm{f}_{\psi\psi} = \mathbf{J}\mathbf{C}_{\psi\psi}\mathbf{J}^\mathrm{T} + \mathbf{Q}$.

        Parameters
        ----------
        J : ndarray, shape (N, N)
            Jacobian of the forecast map (equal to `F` for linear models).
        """
        return J @ self._Cpp @ J.T + self.Q


# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilter(DeterministicEstimator):
    r"""Exact linear (or linearized) Kalman filter.

    Implements the standard two-step KF recursion. All `DeterministicEstimator`
    parameters apply, plus:

    Parameters
    ----------
    F_jac : ndarray, shape (N, N), optional
        Jacobian of the forecast map over ONE model step, used to propagate the
        covariance (applied once per step spanned by each `forecast_step`).
        Defaults to `F` when `model=None`, to `model.F` when the model exposes a
        linear transition matrix, and to the identity otherwise (covariance then
        grows only by $\mathbf{Q}$ each step).

    Notes
    -----
    Predict (mean via `F` or `model.time_integrate`, covariance via `F_jac`):

    $$
    \boldsymbol{\psi}^\mathrm{f} = \mathbf{F}\,\boldsymbol{\psi}_{k-1|k-1}, \qquad
    \mathbf{C}^\mathrm{f}_{\psi\psi} = \mathbf{F}_\mathrm{jac}\,\mathbf{C}_{\psi\psi,\,k-1|k-1}\,\mathbf{F}_\mathrm{jac}^\mathrm{T} + \mathbf{Q}.
    $$

    Update:

    $$
    \mathbf{S} = \mathbf{M}\mathbf{C}^\mathrm{f}_{\psi\psi}\mathbf{M}^\mathrm{T} + \mathbf{C}_{dd}, \qquad
    \mathbf{K} = \mathbf{C}^\mathrm{f}_{\psi\psi}\mathbf{M}^\mathrm{T}\mathbf{S}^{-1},
    $$

    $$
    \boldsymbol{\psi}^\mathrm{a} = \boldsymbol{\psi}^\mathrm{f}
    + \mathbf{K}\left(\mathbf{d} - \mathbf{M}\boldsymbol{\psi}^\mathrm{f}\right), \qquad
    \mathbf{C}^\mathrm{a}_{\psi\psi} = (\mathbb{I} - \mathbf{K}\mathbf{M})\,\mathbf{C}^\mathrm{f}_{\psi\psi}.
    $$
    """

    def __init__(self, *, F_jac: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        # Jacobian for covariance propagation (over one model step)
        if F_jac is not None:
            self._F_jac = np.atleast_2d(F_jac)
        elif self._F is not None:
            self._F_jac = self._F             # linear model: the Jacobian is F itself
        elif self._model is not None and hasattr(self._model, 'F'):
            self._F_jac = np.atleast_2d(np.asarray(self._model.F, dtype=float))
        else:
            self._F_jac = np.eye(self.N)      # identity: Cpp grows only by Q

    # ── KF predict ───────────────────────────────────────────────────────────

    def forecast_step(self, t_end=None, **kwargs) -> None:
        r"""KF prediction step.

        Advances the state mean via `model.time_integrate` or $\mathbf{F}\boldsymbol{\psi}$,
        then propagates the covariance once per model step spanned:
        $\mathbf{C}^\mathrm{f}_{\psi\psi} = \mathbf{F}_\mathrm{jac}\,\mathbf{C}_{\psi\psi,\,k-1|k-1}\,\mathbf{F}_\mathrm{jac}^\mathrm{T} + \mathbf{Q}$.
        """
        t_prev = self._t
        super().forecast_step(t_end=t_end, **kwargs)

        # Propagate covariance per step: Cpp <- J Cpp J^T + Q, once per model step,
        # so F_jac and Q keep their single-step meaning whatever interval was spanned.
        if self._model is not None:
            n_steps = max(1, int(round((self._t - t_prev) / self._model.dt)))
        else:
            n_steps = kwargs.get("Nt", 1)
        for _ in range(n_steps):
            self._Cpp = self._propagate_Cpp(self._F_jac)

    # ── KF update ────────────────────────────────────────────────────────────
    @property
    def M_mat(self):
        if not hasattr(self, '_M_mat'):
            if isinstance(self.M, np.ndarray):
                M_mat = self.M
                if M_mat.shape[1] != self.N:
                    # model.M acts on the augmented state [phi; alpha; y]; apply
                    # the same column-truncation rule as Estimator._MA.
                    Nq = M_mat.shape[0]
                    M_mat = np.hstack((M_mat[:, :self.N - Nq], M_mat[:, -Nq:]))
            else:
                M_mat = np.column_stack(
                    [self._MA(np.eye(self.N)[:, i]) for i in range(self.N)]
                )
            self._M_mat = M_mat

        return self._M_mat

    def analysis_step(self,
                      d: np.ndarray,
                      Cdd: np.ndarray,
                      return_analysis: bool = False) -> Optional[np.ndarray]:
        r"""Kalman update step.

        $\mathbf{S} = \mathbf{M}\mathbf{C}^\mathrm{f}_{\psi\psi}\mathbf{M}^\mathrm{T} + \mathbf{C}_{dd}$,
        $\mathbf{K} = \mathbf{C}^\mathrm{f}_{\psi\psi}\mathbf{M}^\mathrm{T}\mathbf{S}^{-1}$,
        $\boldsymbol{\psi}^\mathrm{a} = \boldsymbol{\psi}^\mathrm{f} + \mathbf{K}(\mathbf{d} - \mathbf{M}\boldsymbol{\psi}^\mathrm{f})$,
        $\mathbf{C}^\mathrm{a}_{\psi\psi} = (\mathbb{I} - \mathbf{K}\mathbf{M})\,\mathbf{C}^\mathrm{f}_{\psi\psi}$.

        When a model is present, the analysed mean is written back into the
        model history so the next `forecast_step` starts from the analysis.

        Parameters
        ----------
        d : ndarray, shape (Nq,)
            Observation vector.
        Cdd : ndarray, shape (Nq, Nq)
            Observation-noise covariance.

        Returns
        -------
        ndarray, shape (N,)
            Posterior mean $\boldsymbol{\psi}_{k|k}$, only if `return_analysis=True`.
        """
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)

        Cpp_f = self._Cpp                                        # forecast covariance P_{k|k-1}
        S     = self.M_mat @ Cpp_f @ self.M_mat.T + Cdd          # innovation covariance (Nq, Nq)
        K     = Cpp_f @ self.M_mat.T @ inv(S)                    # Kalman gain           (N,  Nq)

        self._psi = self._psi + K @ (d - self.M_mat @ self._psi)
        self._Cpp = (np.eye(self.N) - K @ self.M_mat) @ Cpp_f    # posterior covariance P_{k|k}

        if self._model is not None:
            self.update_history(self._psi[:, np.newaxis], self._t, modify_saved_states=True)
        self.assimilated_data = (d, self._t)

        if return_analysis:
            return self._psi.copy()


# ─────────────────────────────────────────────────────────────────────────────


