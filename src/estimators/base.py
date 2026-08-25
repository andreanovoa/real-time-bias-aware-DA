"""Abstract base class shared by all state/parameter estimators.

See the package docstring (`romda.estimators`) for the class hierarchy, the
forecast strategy, and the notation table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy

import numpy as np
from romda.bias_estimators import Bias
from romda.models import Model

# ══════════════════════════════════════════════════════════════════════════════
# Abstract base — Estimator
# ══════════════════════════════════════════════════════════════════════════════

class Estimator(ABC):
    r"""Abstract base class shared by all state/parameter estimators.

    Every concrete estimator (`EnKF`, `EnSRKF`, `rBA_EnKF`, `KalmanFilter`, ...)
    owns a `Model` instance for the forecast step and, optionally, a
    `Bias` instance for bias-aware assimilation. The observation operator maps state
    to observation space, $\mathbf{y} = \mathbf{M}\boldsymbol{\psi}$ if $\mathbf{M}$
    is a matrix, or $\mathbf{y} = \mathbf{M}(\boldsymbol{\psi})$ if $\mathbf{M}$ is
    callable; if not supplied explicitly it is read from `model.M`.

    Subclasses must implement `analysis_step(d, Cdd, **kwargs)`: the Bayesian update
    given observation $\mathbf{d}$ and observation-noise covariance $\mathbf{C}_{dd}$.
    Implementations build the forecast state internally, run the filter update,
    validate the result, and update the model history in-place.

    Notation
    --------
    | Symbol | Meaning | Shape |
    | --- | --- | --- |
    | $N$ | augmented state dimension ($N_\phi{+}N_\alpha$, or $+N_q$ once observables are appended) | |
    | $N_\phi$ | model state dimension | |
    | $N_\alpha$ | number of estimated parameters | |
    | $N_q$ | number of observables | |
    | $m$ | ensemble size | |
    | $\boldsymbol{\psi}$ | state vector / ensemble | $(N,)$ or $(N, m)$ |
    | $\mathbf{M}$ | measurement operator (matrix or callable) | $(N_q, N)$ |
    | $\mathbf{d}$ | observation vector | $(N_q,)$ |
    | $\mathbf{C}_{dd}$ | observation-noise covariance | $(N_q, N_q)$ |
    | $\mathbf{C}_{\psi\psi}$ | forecast (prior) covariance | $(N, N)$ |
    | $\mathbf{K}$ | Kalman gain | $(N, N_q)$ |

    Attributes
    ----------
    est_phi : bool
        Whether to estimate the model state (default True).
    est_alpha : list of str
        Names of model parameters to estimate (default ``[]``).
    est_bias : bool
        Whether to estimate an observation bias (default False).
    start_param : int
        Analysis step at which parameter estimation starts; parameters are
        frozen in earlier analyses (0 = active from the first analysis).
    start_bias : int
        Analysis step at which a bias-aware filter starts its bias-aware
        update; a plain EnKF update is applied in earlier analyses
        (0 = active from the first analysis).
    inflation_factor : float
        Covariance/ensemble inflation factor (default 1.0).
    inflation_factor_rejection : float
        Inflation applied after a rejected analysis (default 1.002).
    results_folder : str or None
        Optional path for saving results.

    References
    ----------
    Kalman (1960). A new approach to linear filtering and prediction problems.
    *J. Basic Eng.*, 82(1), 35-45.

    Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer.

    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    """

    # ── General DA class-level defaults ──────────────────────────────────────

    est_phi: bool = True
    est_bias: bool = False

    # Warm-up windows, counted in analysis steps (0 = active from the start).
    # The Bayesian *bias* update is configured on the Bias estimator itself
    # (Bias.bayesian_update), not here.
    start_param: int = 0   # parameter estimation starts at this analysis step
    start_bias: int = 0    # bias-aware update starts at this analysis step
    activate_parameter_estimation: bool = True

    inflation_factor: float = 1.0
    inflation_factor_rejection: float = 1.002
    # None marks the filter as bias-blind; bias-aware filters (rBA_EnKF) set it.
    regularization_factor: float | None = None

    std_phi: float = 0.001
    std_alpha: float | dict[str, float | list[float]] = 0.001

    distribution_phi: str = 'normal'
    distribution_alpha: str = 'uniform'


    results_folder: str | None = None

    _keys_to_print = [
        'm', 'est_phi', 'est_alpha', 'est_bias', 'Na',
        'regularization_factor', 'inflation_factor', 'inflation_factor_rejection',
    ]


    def __init__(self, **kwargs) -> None:
        # Record what was consumed so subclasses can exclude these keys from
        # whatever they forward the remaining kwargs to (e.g. the model ctor).
        self._consumed_kwargs: set = set()
        keys = list(kwargs.keys())
        for attr in keys:
             # Derived read-only properties (e.g. EnsembleEstimator.m, which reads
             # the model's history) must not be assigned; they are consumed
             # downstream, e.g. by Model.init_ensemble.
             class_attr = getattr(self.__class__, attr, None)
             if isinstance(class_attr, property) and class_attr.fset is None:
                 continue
             # hasattr, not __class__.__dict__: the config fields are declared on
             # Estimator, so a leaf class (EnSRKF, EnKF, ...) has none of them in its
             # own __dict__ and every config kwarg was silently dropped.
             if hasattr(self.__class__, attr):
                 setattr(self, attr, kwargs.pop(attr))
                 self._consumed_kwargs.add(attr)

    # ── Identity / introspection ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ── Model / Bias accessors ───────────────────────────────────────────────

    @property
    def model(self) -> Model:
        """The Model instance used for the forecast step."""
        model = getattr(self, "_model", None)
        if model is not None:
            return model
        raise AttributeError("Estimator has no model attribute.")

    @property
    def bias(self) -> Bias | None:
        """Bias instance, or None if not configured."""
        return getattr(self, "_bias", None)

    @bias.setter
    def bias(self, value: Bias | None) -> None:
        """Set the bias model (must be a Bias instance or None)."""
        if value is not None and not isinstance(value, Bias):
            raise ValueError("Bias must be a Bias instance or None.")
        self._bias = value

    # ──  Current state accessor ───────────────────────────────────
    @property
    def current_state(self) -> np.ndarray:
        """Current state (delegates to model)."""
        return self.model.current_state

    @property
    def current_time(self) -> float:
        """Current time (delegates to model)."""
        return self.model.current_time

    @property
    def current_bias_estimate(self) -> np.ndarray | None:
        """Current bias estimate, or None if no bias model."""
        if self.bias is not None:

            state =  self.bias.current_state
            return self.bias.get_bias(state, mean=True)[0, ...]  # (Nq,1)

        return np.zeros((self.model.Nq,1))  # No bias model; return zero bias estimate



    # ── Model config helper ───────────────────────────────────────────────────

    @property
    def Na(self) -> int:
        return self.model.Na


    @property
    def est_alpha(self) -> list[str]:
        return self.model.est_alpha

    @property
    def Nphi(self) -> int:
        """Size of the model state vector."""
        return self.model.Nphi

    @property
    def Nq(self) -> int:
        """Number of observable dimensions."""
        return self.model.Nq

    @property
    def M(self) -> np.ndarray:
        return self.model.M

    @property
    def rng(self) -> np.random.Generator:
        if getattr(self, '_rng', None) is None:
            self._rng = self.model.rng
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        self._rng = value


    # ── Measurement-operator wrapper ───────────────────────────────────

    def _MA(self, A: np.ndarray) -> np.ndarray:
        """Map ensemble A (N, m) through M --> (Nq, m)."""
        if callable(self.M) and not isinstance(self.M, np.ndarray):
            return np.asarray(
                np.column_stack([self.M(A[:, i]) for i in range(A.shape[1])])
            )
        M = self.M
        if A.shape[0] != M.shape[1]:
            # Parameters excluded from the analysis: drop the alpha columns of M,
            # keeping the state and the trailing observable ones (cf. Filter.get_M
            # in the legacy data_assimilation module).
            Nq = M.shape[0]
            M = np.hstack((M[:, :A.shape[0] - Nq], M[:, -Nq:]))
        return M @ A


    # ── History update (delegates, also updates bias) ────────────────────────

    def update_history(
        self,
        psi: np.ndarray,
        t=None,
        b=None,
        modify_saved_states: bool = False,
        reset: bool = False,
    ) -> None:
        """Update model (and bias) history."""
        self.model.update_history(psi, t, reset=reset, modify_saved_states=modify_saved_states)
        # b is None when the caller only advances the model state; there is no
        # new bias sample to record, so leave the bias history untouched.
        if self.bias is not None and b is not None:
            self.bias.update_history(b, t, reset=reset, modify_saved_states=modify_saved_states)


    # ── Bias-awareness helpers ────────────────────────────────────────────────


    @property
    def is_bias_aware(self) -> bool:
        """True when the estimator carries an explicit bias correction."""
        return self.regularization_factor is not None

    # ── Assimilated data log ──────────────────────────────────────────────────

    @property
    def assimilated_data(self):
        """Namedtuple with fields *data* and *times* of all assimilated obs."""
        if not hasattr(self, "_assimilated_data"):
            self._assimilated_data = []
            self._assimilated_times = []
        AssimilatedData = namedtuple("AssimilatedData", ["data", "times"])
        return AssimilatedData(data=self._assimilated_data, times=self._assimilated_times)

    @assimilated_data.setter
    def assimilated_data(self, value: tuple) -> None:
        """Append (y_obs, t_obs) to the assimilation log."""
        if not hasattr(self, "_assimilated_data"):
            self._assimilated_data = []
            self._assimilated_times = []
        y_obs, t_obs = value
        self._assimilated_data.append(y_obs)
        self._assimilated_times.append(t_obs)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self):
        return deepcopy(self)

    def print_parameters(self, indent: int = 0) -> None:
        print(f'\n{self.name}')
        print('=' * len(self.name))
        for key in self._keys_to_print:
            print(f'{" " * indent}{key} = {getattr(self, key, "N/A")}')

    # ── Forecast step (shared implementation) ────────────────────────────────

    def forecast_step(
        self,
        t_end=None,
        reset: bool = False,
        close: bool = False,
        output_forecast: bool = False,
        **kwargs,
    ) -> np.ndarray | None:
        """Advance model (and bias, if present) in time.

        Parameters
        ----------
        t_end : float, optional
            Target time; Nt is derived from it if provided.
        reset : bool
            Whether to reset model history on update.
        close : bool
            Close the integrator after stepping.
        output_forecast : bool
            If True, return the raw forecast array.

        Returns
        -------
        ndarray or None
        """
        pm = self.model

        if t_end is not None:
            t_end = round(t_end, pm.precision_t)
            # round, not truncate: the division is one short whenever it is inexact
            # in binary (1.18 / 0.01 = 117.999...), same fix as legacy ensemble.py
            Nt = int(np.round((t_end - pm.current_time) / pm.dt))
        else:
            Nt = kwargs.get("Nt", -1)

        # print(f"\nForecasting from t={pm.current_time:.3f} to t={t_end:.3f} (Nt={Nt})...")

        if Nt == 0:
            return  # Already at requested time; nothing to advance.
        assert Nt > 0, "Must specify positive Nt or t_end for forecast_step."

        psi, t = pm.time_integrate(Nt=Nt, averaged=kwargs.get("averaged", False))

        if t_end is not None:
            assert abs(t[-1] - t_end) < pm.dt, (
                f"Final time {t[-1]} does not match requested t_end {t_end}."
            )

        try:
            pm.update_history(psi, t, reset=reset)
        except ValueError as e:
            print("Solver didn't return a homogeneous psi. Check initial conditions.")
            raise e

        # Advance bias model if present
        if self.bias is not None:
            pb = self.bias
            b, t_b = pb.time_integrate(Nt=Nt)
            pb.update_history(b, t_b, reset=reset)
            if pm.current_time != pb.current_time:
                raise AssertionError(
                    f"Time mismatch: model {pm.current_time} vs bias {pb.current_time}"
                )

        if close:
            pm.close()

        if output_forecast:
            return psi

    # ── Bias initialisation (general DA, not ensemble-specific) ──────────────

    def _init_bias(
        self,
        parent_bias: Bias | type[Bias] | None = None,
        **Bdict,
    ) -> None:
        """Initialise the bias model and store it as ``self._bias``.

        Parameters
        ----------
        parent_bias : Bias instance, Bias subclass, or None
            If a subclass, it is instantiated using the model's current
            observable mean as the initial innovation.
        **Bdict
            Extra keyword arguments forwarded to the Bias constructor.
            Keys ``y``, ``t``, and ``dt`` are removed to avoid duplication.
        """
        if isinstance(parent_bias, Bias):
            self._bias = parent_bias.copy()
        elif parent_bias is None:
            self._bias = None
        else:
            assert isinstance(parent_bias, type) and issubclass(parent_bias, Bias), \
                "parent_bias must be a Bias instance or a subclass of Bias"
            pm = self.model
            try:
                y0_all = pm.get_observables()
                y0 = np.mean(y0_all, axis=-1, keepdims=True)
            except (AttributeError, IndexError):
                y0 = np.zeros((1, pm.Nq, 1))

            for key in ['y', 't', 'dt']:
                Bdict.pop(key, None)

            self._bias = parent_bias(
                innovation=y0,
                t=pm.current_time,
                dt=pm.dt,
                initial_capacity=pm.history._initial_capacity,
                rom=pm,
                **Bdict,
            )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def analysis_step(
        self,
        d: np.ndarray,
        Cdd: np.ndarray,
        return_analysis: bool = False,
    ) -> np.ndarray | None:
        """Perform the Bayesian analysis step given observation *d*.

        Implementations are responsible for building the augmented forecast
        state, running the filter update, validating the result, and updating
        the model history in-place.  Optionally return the analysed state when
        ``return_analysis=True``.
        """

