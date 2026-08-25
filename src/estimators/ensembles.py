"""
estimators/ensembles.py
=======================
Ensemble Kalman filter family of estimators.

Class hierarchy
---------------
EnsembleEstimator   (abstract base — owns model + ensemble + bias)
├── EnKF            (stochastic, perturbed-observation EnKF)
├── EnSRKF          (deterministic square-root EnKF)
└── rBA_EnKF        (regularised bias-aware EnKF)

Usage
-----
::
    from romda.estimators.ensembles import EnSRKF

    enkf = EnSRKF(
        parent_model=Lorenz63,   # class or instance
        m=50,
        std_phi=0.2,
        std_alpha={'rho': (25, 35), 'beta': (2, 4)},
        dt=0.015,                # forwarded to Lorenz63(...)
        observe_dims=[0, 2],
    )
    enkf.forecast_step(t_end=t_d)
    enkf.analysis_step(d=d, Cdd=Cdd)
"""

from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky, inv
from romda.bias_estimators import Bias, NoBias
from romda.estimators.base import Estimator
from romda.estimators.inflation import multiplicative_inflation
from romda.models import Model
from romda.observations import Observations
from romda.plotting import Palette
from romda.utils import (
    allowed_kwargs_for_func,
    cut_signals,
    interpolate,
    mean_vector_to_ensemble,
    normalized_alpha,
    normalized_time,
    normalized_y,
)
from typeguard import typechecked

__all__ = ['EnsembleEstimator', 'EnKF', 'EnSRKF', 'rBA_EnKF']


# ══════════════════════════════════════════════════════════════════════════════
# EnsembleEstimator — concrete base for ensemble Kalman filter family
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleEstimator(Estimator):
    """Abstract base for ensemble-based estimators (EnKF, EnSRKF, rBA-EnKF).

    Owns the model, ensemble, and bias.  Subclasses only need to implement
    ``_analysis_kernel(Af, d, Cdd, **kwargs) -> Aa``.

    Attributes
    ----------
    m : int
        Number of ensemble members.
    std_phi : float
        Fractional std for initial state perturbations.
    std_alpha : float or dict
        Std (or ``{name: (lo, hi)}``) for initial parameter perturbations.
    distribution_phi : str
        Sampling distribution for state members ("normal" or "uniform").
    distribution_alpha : str
        Sampling distribution for parameter members.
    ensure_mean_at_init : bool
        Force one member to equal the ensemble mean at initialisation.
    ensemble_psi0 : ndarray or None
        Pre-built initial ensemble; bypasses generation if provided.
    activate_parameter_estimation : bool
        Whether to include parameter rows in the analysis update.
    regularization_factor : float
        Bias-regularisation weight (used by rBA-EnKF; ignored otherwise).
    """

    # ── Ensemble-specific class-level defaults ────────────────────────────────
    ensure_mean_at_init: bool = False
    ensemble_psi0: np.ndarray | None = None
    #: Correct the ESN reservoir block (``model.N_dim : model.N_dim + model.N_units``)
    #: in the analysis? ``False`` leaves it at its forecast value -- a physical-state-only
    #: update. No-op for models without an ``N_units`` (i.e. not an ESN).
    update_reservoir: bool = True


    # ── Constructor ───────────────────────────────────────────────────────────

    @typechecked
    def __init__(
        self,
        parent_model: Model | type[Model],
        parent_bias: Bias | type[Bias] | None = None,
        **kwargs,
    ):
        """Initialise ensemble estimator.

        Parameters
        ----------
        parent_model : Model instance or Model subclass
            If a class is passed, remaining *kwargs* are forwarded to its
            constructor (e.g. ``dt``, ``psi0``, model-specific parameters).
        parent_bias : Bias instance, Bias subclass, or None
        **kwargs
            Ensemble config keys (``m``, ``std_phi``, ``std_alpha``, …) and/or
            model constructor keys are accepted here.
        """
        #  Apply general DA attributes from Estimator base.
        super().__init__(**kwargs)

        # Split the remaining kwargs: ensemble-generation keys go to
        # init_ensemble; everything else (dt, psi0, model parameters, ...) to
        # the model constructor. Keys consumed by Estimator.__init__ are
        # forwarded to neither.
        ensemble_kwargs = allowed_kwargs_for_func(parent_model.init_ensemble, kwargs)
        model_kwargs = {k: v for k, v in kwargs.items()
                        if k not in self._consumed_kwargs and k not in ensemble_kwargs}

        # Instantiate or copy the model.
        if isinstance(parent_model, Model):
            parent_model = parent_model.copy()
        else:
            parent_model = parent_model(**model_kwargs)

        # Generate the initial ensemble inside the model, falling back to the
        # estimator's config for keys not passed explicitly.
        for attr in ('std_phi', 'std_alpha', 'distribution_phi', 'distribution_alpha'):
            if attr not in ensemble_kwargs:
                ensemble_kwargs[attr] = getattr(self, attr)
        parent_model.init_ensemble(**ensemble_kwargs)

        self._model = parent_model

        self._init_bias(parent_bias)

    # ── Ensemble-specific observable helpers ─────────────────────────────────
    @property
    def m(self) -> int:
        """Ensemble size."""
        return self._model.m

    def get_observables(self, Nt: int = 1, **kwargs) -> np.ndarray:
        """Return ensemble observables, bias-corrected if applicable."""
        y_model = self.model.get_observables(Nt=Nt, **kwargs)
        if self.bias is None or isinstance(self.bias, NoBias):
            return y_model
        if Nt != 1:
            raise NotImplementedError(
                "Bias correction for get_observables with Nt > 1 is not yet implemented."
            )
        # unbiased = y + b, cf. _recover_unbiased_solution
        return y_model + self.current_bias_estimate

    def get_observable_hist(
        self, Nt: int = 0
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return ``(y_unbiased, y_model)`` history, interpolating bias if needed."""
        pb = self.bias
        y_model = self.model.get_observable_hist(Nt=Nt)

        if pb is None or isinstance(pb, NoBias):
            return None, y_model

        t_model = self.model.hist_t[-Nt:] if Nt else self.model.hist_t
        y_unbiased = self._recover_unbiased_solution(pb.hist_t, pb.hist, t_model, y_model)
        return y_unbiased, y_model

    @staticmethod
    def _recover_unbiased_solution(
        t_b: np.ndarray,
        b: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Subtract interpolated bias from model observable history."""
        if b.shape[-1] == 1 and y.shape[-1] > 1:
            b = np.repeat(b, y.shape[-1], axis=-1)
        if len(t_b) != len(t):
            b = interpolate(t_b, b, t, fill_values=None)
        return y + b

    # ── Analysis step orchestration ───────────────────────────────────────────

    def analysis_step(
        self,
        d: np.ndarray,
        Cdd: np.ndarray,
        return_analysis: bool = False,
    ) -> np.ndarray | None:
        """Bayesian analysis step.

        Builds the augmented forecast ensemble, delegates the filter update to
        ``_analysis_kernel``, applies inflation, validates parameters, and
        updates the model history in-place.

        Parameters
        ----------
        d : ndarray (Nq,)
            Observation vector.
        Cdd : ndarray (Nq, Nq)
            Observation noise covariance.
        return_analysis : bool
            If True, return the analysed state array.
        """
        assert self.rng is not None, "RNG not set; ensure __init__ completed."

        Af_state = self.current_state          # (Nphi+Na, m)

        # State-estimation-only warm-up, as in Ensemble.analysis_step.
        if self.start_param > 0:
            self.activate_parameter_estimation = (
                len(self.assimilated_data.times) >= self.start_param
            )

        # Optionally exclude parameters from the analysis update. The forecast
        # parameters are kept aside and re-appended below, so that the analysis
        # written to history always has the model's full Nphi+Na rows.
        Af_params = None
        if self.Na > 0 and not self.activate_parameter_estimation:
            Af_params = Af_state[self.Nphi:self.Nphi + self.Na, :].copy()
            Af_state = Af_state[:self.Nphi, :]

        # Append observables to form augmented state.
        y = self.model.get_observables()       # (Nq, m)
        Af_aug = np.vstack((Af_state, y))      # (Nphi+Na+Nq, m)  or  (Nphi+Nq, m)

        # ── Call the subclass filter kernel ──────────────────────────────────


        Aa = self._analysis_kernel(Af_aug, d, Cdd)


        # ── Covariance inflation (Evensen 2009, Chap. 15) ────────────────────
        rho = self.inflation_factor
        if rho != 1.0:
            Aa = multiplicative_inflation(Aa, rho)
            self.inflation_history.times.append(self.current_time)
            self.inflation_history.factors.append(rho)

        # ── Spread and parameter validity checks ─────────────────────────────
        if not self.has_valid_spread(Aa[:self.model.Nphi, :]):
            self.rejected_analysis = (self.current_time, 'Invalid analysis spread')

        if self.Na > 0 and self.alpha_limits_matrix is not None:
            Aa_alpha = Aa[self.Nphi:self.Nphi + self.Na, :]
            is_physical, idx_alpha, _ = self.has_valid_params(
                Aa_alpha, self.alpha_limits_matrix, get_deltas=False
            )
            if not is_physical:
                self.rejected_analysis = (
                    self.current_time,
                    f'Non-physical parameters at indices {idx_alpha}',
                )
                Aa = multiplicative_inflation(Af_aug, self.inflation_factor_rejection)

        # An EnKF-type gain corrects each state row from its own covariance with y,
        # independent of what other rows are present -- so leaving the reservoir out
        # of the analysis is the same as running it and discarding its correction.
        if not self.update_reservoir:
            N_units = getattr(self.model, 'N_units', 0)
            if N_units > 0:
                N_dim = self.model.N_dim
                Aa[N_dim:N_dim + N_units, :] = Af_aug[N_dim:N_dim + N_units, :]

        # ── Update model history ─────────────────────────────────────────────
        # Only the state rows go to the model; the observables are derived.
        Aa_psi = Aa[:Af_state.shape[0], :]
        if Af_params is not None:
            Aa_psi = np.vstack((Aa_psi, Af_params))   # parameters left at their forecast

        self.update_history(
            Aa_psi,
            self.current_time,
            modify_saved_states=True,
        )
        self.assimilated_data = (d, self.current_time)

        # ── Update the bias estimator with the analysis innovation ───────────
        # i^a = d - y^a, evaluated on the *analysed* observables. Ported from
        # Ensemble.update_history_analysis; smooth's EnsembleEstimator dropped
        # this, which left the bias estimator frozen across analysis steps.
        if self.bias is not None:
            Ya = self.model.get_observables()               # (Nq, m)
            ia = d[:, np.newaxis] - Ya                      # (Nq, m)
            updated_state = self.bias.update_state_from_innovation(ia)
            self.bias.update_history(
                updated_state, t=self.bias.current_time, modify_saved_states=True
            )

        if return_analysis:
            # The analysed *state* rows, matching model.hist and the convention in
            # DeterministicEstimator.analysis_step. The trailing rows of Aa are the
            # augmented observables, which are derived rather than stored.
            return Aa_psi

    # ── Abstract filter kernel (implemented by EnKF, EnSRKF, rBA_EnKF) ───────

    @abstractmethod
    def _analysis_kernel(self,
                         Af: np.ndarray,
                         d: np.ndarray,
                         Cdd: np.ndarray,
                         **kwargs
                         ) -> np.ndarray:
        """Pure mathematical filter update.

        Parameters
        ----------
        Af  : (N, m)    augmented forecast ensemble ``[phi; alpha; y]``
        d   : (Nq,)     observation vector
        Cdd : (Nq, Nq)  observation noise covariance
        **kwargs        extra arguments for bias-aware kernels (b, J)

        Returns
        -------
        Aa : (N, m)  analysis ensemble
        """

    # ── Ensemble utilities ────────────────────────────────────────────────────

    def reshape_ensemble(self, m: int | None = None, reset: bool = True) -> None:
        """Re-perturb the current ensemble around its mean."""
        pm = self.model
        if m is None:
            m = self.m
        if m == 1:
            raise ValueError('Ensemble size m must be greater than 1.')
        current_psi = pm.current_state
        mean_psi = np.mean(current_psi, axis=-1)
        std_psi = np.std(current_psi, axis=-1)
        new_ensemble = mean_vector_to_ensemble(pm.rng, mean_psi, std_psi, m, method='normal')
        pm.update_history(psi=new_ensemble, t=pm.current_time, reset=reset)

    @staticmethod
    def inflate(
        A: np.ndarray,
        rho: float,
        d: np.ndarray | None = None,
        additive: bool = True,
    ) -> np.ndarray:
        """Deprecated alias for `romda.estimators.inflation.multiplicative_inflation`."""
        return multiplicative_inflation(A, rho)

    @staticmethod
    def has_valid_spread(A: np.ndarray, tol: float = 1e-6) -> bool:
        """Return True if ensemble spread is non-degenerate."""
        return True  # placeholder — detailed check commented out in original

    @staticmethod
    def has_valid_params(
        A_alpha: np.ndarray,
        alpha_limits_matrix: np.ndarray,
        get_deltas: bool = False,
    ) -> tuple[bool, list[int], np.ndarray]:
        """Check whether all parameter ensemble members lie within bounds.

        Parameters
        ----------
        A_alpha : (Na, m)
        alpha_limits_matrix : (2, Na, 1)  rows are [lower_bounds, upper_bounds]

        Returns
        -------
        is_physical : bool
        idx_alpha   : list of out-of-bounds parameter indices
        d_alpha     : allowed values array (only populated if get_deltas=True)
        """
        if alpha_limits_matrix is None:
            return True, [], np.array([])

        low_limits, high_limits = alpha_limits_matrix
        below = A_alpha < low_limits
        above = A_alpha > high_limits
        oob = np.any(below | above, axis=1)
        is_physical = not np.any(oob)
        idx_alpha = np.where(oob)[0].tolist()
        d_alpha: list[np.ndarray] = []

        if get_deltas and not is_physical:
            if np.any(above) and np.any(below):
                raise ValueError(
                    'Both above and below limits detected simultaneously; '
                    'check alpha_limits and A_alpha.'
                )
            allowed = A_alpha[~above & ~below]
            if np.any(below):
                d_alpha.append(
                    np.max(allowed, axis=1) if allowed.size > 0 else low_limits[:, 0]
                )
            elif np.any(above):
                d_alpha.append(
                    np.min(allowed, axis=1) if allowed.size > 0 else high_limits[:, 0]
                )

        return is_physical, idx_alpha, np.array(d_alpha)

    @property
    def alpha_limits_matrix(self) -> np.ndarray | None:
        if not hasattr(self, '_alpha_lims'):
            recompute = True
        else:
            if self._alpha_lims is None:
                recompute = True
            else:
                expected_shape = (2, self.Na, 1)
                if self._alpha_lims.shape != expected_shape:
                    print(f'Warning: alpha_limits_matrix has shape {self._alpha_lims.shape} but expected {expected_shape}. Recomputing.')
                    recompute = True
                else:
                    recompute = False
        if recompute:
            alpha_lims_dict = self.model.alpha_lims
            est_alpha_keys = sorted(self.model.est_alpha)
            alpha_lims_dict = {key: alpha_lims_dict.get(key, (None, None)) for key in est_alpha_keys}

            alpha_lims = np.array(
                [[(-np.inf if lo is None else lo),
                  ( np.inf if hi is None else hi)]
                 for (lo, hi) in alpha_lims_dict.values()],
                dtype=float,
            ).T  # (2, Na)

            if np.all(np.isinf(alpha_lims)):
                self._alpha_lims = None
            else:
                self._alpha_lims = alpha_lims[:, :, np.newaxis]  # (2, Na, 1)

        return self._alpha_lims

    @property
    def inflation_history(self):
        """Namedtuple with fields *times* and *factors* of each applied inflation."""
        if not hasattr(self, '_inflation_history'):
            InflationData = namedtuple('InflationData', ['times', 'factors'])
            self._inflation_history = InflationData(times=[], factors=[])
        return self._inflation_history

    @property
    def rejected_analysis(self):
        if not hasattr(self, '_rejected_analysis'):
            RejectedData = namedtuple('RejectedData', ['times', 'reasons'])
            self._rejected_analysis = RejectedData(times=[], reasons=[])
        return self._rejected_analysis

    @rejected_analysis.setter
    def rejected_analysis(self, value: tuple) -> None:
        _ = self.rejected_analysis  # ensure initialised
        time, reason = value
        self._rejected_analysis.times.append(time)
        self._rejected_analysis.reasons.append(reason)
        n_rej = len(self._rejected_analysis.times)
        n_total = len(self.assimilated_data.times) + 1
        print(f'Non-physical analysis: {n_rej}/{n_total}')

    # ── Printing ──────────────────────────────────────────────────────────────

    def print_parameters(self, indent: int = 0) -> None:
        super().print_parameters(indent=indent)
        print(f'{" " * indent}=== Model parameters: ===')
        self.model.print_parameters(show_header=False, indent=indent*2)
        print(f'{" " * indent}=== Bias parameters: ===')
        if self.bias is not None:
            self.bias.print_bias_parameters(indent=indent*2)
        else:
            print(f'{" " * indent*2}No bias model.')


    def visualize_history(self, **kwargs) -> None:
        """Plot observable and parameter histories."""
        kwargs_obs = allowed_kwargs_for_func(plot_observable_history, kwargs)
        plot_observable_history(ensemble=self, **kwargs_obs)
        if self.Na > 0:
            kwargs_alpha = allowed_kwargs_for_func(plot_alpha_history, kwargs)
            plot_alpha_history(ensemble=self, **kwargs_alpha)

    def visualize_state(self, **kwargs) -> None:
        """Plot ensemble state distributions."""
        func = self.model.visualize_state
        kwargs_state = allowed_kwargs_for_func(func, kwargs)
        func(**kwargs_state)


# ══════════════════════════════════════════════════════════════════════════════
# Concrete filter kernels
# ══════════════════════════════════════════════════════════════════════════════

class EnKF(EnsembleEstimator):
    r"""Stochastic Ensemble Kalman Filter (perturbed-observation variant).

    Each ensemble member assimilates a randomly perturbed copy of the observation,
    $\mathbf{d}_j \sim \mathcal{N}(\mathbf{d}, \mathbf{C}_{dd})$. Writing
    $\boldsymbol{\Psi}^\mathrm{f} = \mathbf{A}^\mathrm{f} - \overline{\mathbf{A}^\mathrm{f}}$
    for the mean-subtracted forecast ensemble and $\mathbf{S} = \mathbf{M}\boldsymbol{\Psi}^\mathrm{f}$:

    $$
    \mathbf{C} = (m-1)\,\mathbf{C}_{dd} + \mathbf{S}\mathbf{S}^\mathrm{T}, \qquad
    \mathbf{D} = \mathbf{d}\mathbf{1}^\mathrm{T} + \mathrm{chol}(\mathbf{C}_{dd})\,\boldsymbol{\varepsilon},
    \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbb{I}),
    $$

    $$
    \mathbf{A}^\mathrm{a} = \mathbf{A}^\mathrm{f}
    + \boldsymbol{\Psi}^\mathrm{f}\mathbf{S}^\mathrm{T}\mathbf{C}^{-1}
    \left(\mathbf{D} - \mathbf{M}\mathbf{A}^\mathrm{f}\right).
    $$

    Notes
    -----
    Algebraically equivalent to the textbook Kalman-gain form
    $\mathbf{K} = \mathbf{C}_{\psi\psi}\mathbf{C}_{yy}^{-1}$ with
    $\mathbf{C}_{\psi\psi} = \boldsymbol{\Psi}^\mathrm{f}\mathbf{S}^\mathrm{T}/(m{-}1)$,
    $\mathbf{C}_{yy} = \mathbf{S}\mathbf{S}^\mathrm{T}/(m{-}1) + \mathbf{C}_{dd}$, but
    rearranged to avoid forming the covariance matrices explicitly. The implementation
    equivalently expresses the update as a member-space transform
    $\mathbf{A}^\mathrm{a} = \mathbf{A}^\mathrm{f}(\mathbb{I}_m + \mathbf{X})$,
    $\mathbf{X} = \mathbf{S}^\mathrm{T}\mathbf{C}^{-1}(\mathbf{D}-\mathbf{M}\mathbf{A}^\mathrm{f})$,
    which coincides with the equation above because $\mathbf{S}$ has zero column-sum.

    References
    ----------
    Evensen (2003). The ensemble Kalman filter: theoretical formulation and practical
    implementation. *Ocean Dynamics*, 53, 343-367, Eq. (9.27).
    """

    def _analysis_kernel(
        self,
        Af: np.ndarray,
        d: np.ndarray,
        Cdd: np.ndarray,
        **kwargs,
    ) -> np.ndarray:

        m   = Af.shape[1]
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)
        # Create an ensemble of observations
        D = d[:, None] + cholesky(Cdd) @ self.rng.standard_normal((Cdd.shape[0], m))

        # Mapped forecast matrix M(Af) and mapped deviations M(Af')
        Psi_f   = Af  - Af.mean(1,  keepdims=True)
        S       = self._MA(Psi_f)
        Y       = self._MA(Af)

        C = (m - 1) * Cdd + np.dot(S, S.T)
        X = S.T @ inv(C) @ (D - Y)

        # Member-space transform X̂ with Aa = Af X̂; stashed so the adaptive
        # inflation can push a random ensemble through the SAME update
        # ([Aa; Ba] = [Af; Bf] X̂, Evensen 2009, Eq. 15.1).
        self._analysis_transform = np.eye(m) + X
        Aa = Af @ self._analysis_transform

        # Complex-state models (e.g. KS Fourier modes) legitimately produce
        # complex Aa; the realness guard only applies to real-state models.
        return Aa if np.iscomplexobj(Af) or np.isreal(Aa).all() else Af


class EnSRKF(EnsembleEstimator):
    r"""Ensemble Square-Root Kalman Filter (no observation perturbations).

    Updates the ensemble mean with the Kalman gain and transforms the ensemble
    deviations with a symmetric square-root transform, so no stochastic observation
    perturbations are needed. Writing $\boldsymbol{\Psi}^\mathrm{f}$ for the
    mean-subtracted forecast ensemble and $\mathbf{S} = \mathbf{M}\boldsymbol{\Psi}^\mathrm{f}$:

    $$
    \mathbf{C} = (m-1)\,\mathbf{C}_{dd} + \mathbf{S}\mathbf{S}^\mathrm{T}, \qquad
    \mathbf{K} = \boldsymbol{\Psi}^\mathrm{f}\mathbf{S}^\mathrm{T}\mathbf{C}^{-1},
    $$

    $$
    \overline{\boldsymbol{\psi}}^\mathrm{a} = \overline{\boldsymbol{\psi}}^\mathrm{f}
    + \mathbf{K}\left(\mathbf{d} - \mathbf{M}\overline{\boldsymbol{\psi}}^\mathrm{f}\right),
    \qquad
    \boldsymbol{\Psi}^\mathrm{a} = \boldsymbol{\Psi}^\mathrm{f}\,\mathbf{T}^{1/2},
    \qquad
    \mathbf{T} = \mathbb{I}_m - \mathbf{S}^\mathrm{T}\mathbf{C}^{-1}\mathbf{S},
    $$

    with $\mathbf{T}^{1/2}$ the symmetric square root of $\mathbf{T}$ (computed via
    its eigendecomposition, since $\mathbf{T}$ is symmetric).

    References
    ----------
    Tippett, Anderson, Bishop, Hamill & Whitaker (2003). Ensemble square root
    filters. *Mon. Wea. Rev.*, 131, 1485-1490.
    """

    def _analysis_kernel(
        self,
        Af: np.ndarray,
        d: np.ndarray,
        Cdd: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m   = Af.shape[1]
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)

        psi_f_m = np.mean(Af, 1, keepdims=True)   # (N, 1)
        Psi_f   = Af - psi_f_m                    # (N, m)

        y = self._MA(psi_f_m)   # (Nq, 1)
        S = self._MA(Psi_f)     # (Nq, m)

        C    = (m - 1) * Cdd + S @ S.T   # (Nq, Nq)
        Cinv = inv(C)

        # Mean update — d[:, None] makes (Nq,1) to match y shape
        ma = psi_f_m + Psi_f @ S.T @ Cinv @ (d[:, None] - y)   # (N, 1)

        # Square-root update: T = I - S.T Cinv S
        # (equivalent to Tippett 2003 with A_=Psi_f/√(m-1), Cyy=C/(m-1))
        T        = np.eye(m) - S.T @ Cinv @ S                   # (m, m)
        ev, evec = np.linalg.eigh(T)
        sqrtT    = evec @ np.diag(np.sqrt(np.maximum(ev, 0.0))) @ evec.T

        Aa = ma + Psi_f @ sqrtT   # (N, m)

        # The same update as a member-space transform Aa = Af X̂ (for the
        # adaptive inflation): mean projector + centred (gain + square-root) part.
        ones_m = np.ones((m, m)) / m
        w = S.T @ Cinv @ (d[:, None] - y)   # (m, 1)
        self._analysis_transform = (
            ones_m + (np.eye(m) - ones_m) @ (w @ np.ones((1, m)) + sqrtT)
        )
        return Aa if np.iscomplexobj(Af) or np.isreal(Aa).all() else Af


class rBA_EnKF(EnsembleEstimator):
    r"""Regularized bias-aware ensemble Kalman filter (r-EnKF).

    Extends the stochastic `EnKF` with an explicit correction for the (estimated)
    observation bias $\mathbf{b}$ and its Jacobian
    $\mathbf{J} = \mathrm{d}\mathbf{b}/\mathrm{d}(\mathbf{M}\boldsymbol{\psi})$,
    weighted by the regularization factor $\gamma \ge 0$ ($\gamma=0$ recovers the
    standard `EnKF`). Writing $\mathbf{Y} = \mathbf{M}\mathbf{A}^\mathrm{f} + \mathbf{B}$
    for the bias-corrected forecast observables, and $\mathbf{C}_{\psi q}$,
    $\mathbf{C}_{qq}$ for the (sample) forecast cross- and auto-covariances of the
    state and the mapped observables:

    $$
    \mathbf{C}_{yy} = (m-1)\,\mathbf{C}_{dd}
    + (\mathbb{I}+\mathbf{J})^\mathrm{T}(\mathbb{I}+\mathbf{J})\,\mathbf{C}_{qq}
    + \gamma\, \mathbf{J}^\mathrm{T}\mathbf{J}\,\mathbf{C}_{qq},
    \qquad
    \mathbf{K} = \mathbf{C}_{\psi q}\,\mathbf{C}_{yy}^{-1},
    $$

    $$
    \mathbf{A}^\mathrm{a} = \mathbf{A}^\mathrm{f} + \mathbf{K}
    \left[(\mathbb{I}+\mathbf{J})^\mathrm{T}(\mathbf{D} - \mathbf{Y})
    - \gamma\, \mathbf{J}^\mathrm{T}\mathbf{b}\right],
    $$

    where $\mathbf{D}$ is the perturbed-observation ensemble and $\mathbf{b}$ the
    current (mean) bias estimate, broadcast over the ensemble. The bias covariance
    is fixed to $\mathbf{C}_{bb} = \mathbf{C}_{dd}$, so the weight
    $\mathbf{W} = \mathbf{C}_{dd}\mathbf{C}_{bb}^{-1}$ of the reference paper
    reduces to the identity and is omitted.

    Notes
    -----
    $\mathbf{C}_{\psi q}$ and $\mathbf{C}_{qq}$ here are the sample cross- and
    auto-covariances (divided by $m-1$), unlike the unnormalized cross-moments used
    by the legacy `romda.legacy.data_assimilation.rBA_EnKF` implementation of the same
    filter. If observations are biased (`Bias.biased_observations`), $\mathbf{d}$ is
    shifted by the mean bias-innovation gap before the update. During the first
    `start_bias` analysis steps the plain `EnKF` update is applied instead
    (bias-blind warm-up window).

    Parameters
    ----------
    gamma : float, optional
        Bias-regularization factor (default 1.0), aliased as `regularization_factor`.
        Larger values give more weight to the bias norm in the cost function.

    References
    ----------
    Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
    ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
    [DOI: 10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502).
    """

    def __init__(
        self,
        parent_model,
        parent_bias=None,
        gamma: float = 1.0,
        **kwargs,
    ):
        # regularization_factor is the canonical attribute; gamma is the alias.
        kwargs.setdefault('regularization_factor', gamma)
        super().__init__(parent_model, parent_bias, **kwargs)

    @property
    def gamma(self) -> float:
        return self.regularization_factor

    def _analysis_kernel(
        self,
        Af: np.ndarray,
        d: np.ndarray,
        Cdd: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # Bias-blind warm-up window: the first `start_bias` analysis steps use
        # the plain EnKF update, as in the legacy Ensemble.analysis_step.
        if len(self.assimilated_data.times) < self.start_bias:
            return EnKF._analysis_kernel(self, Af, d, Cdd)

        assert self.bias is not None, (
            "Bias-aware filter selected but no bias instance found. "
            "Initialise with a Bias instance."
        )

        b = np.asarray(self.current_bias_estimate)  # (Nq, 1)
        J = self.bias.state_derivative()            # (Nq, Nq)

        if self.bias.biased_observations:
            d = d + np.mean(b - self.bias.current_innovations, axis=-1)

        m   = Af.shape[1]
        d   = np.atleast_1d(d)
        Cdd = np.atleast_2d(Cdd)
        Nq  = len(d)
        b   = b.reshape(Nq, 1)

        MEf = self._MA(Af)
        Y   = MEf + b
        D   = d[:, None] + cholesky(Cdd) @ self.rng.standard_normal((Nq, m))

        Af_p  = Af  - Af.mean(1,  keepdims=True)
        MEf_p = MEf - MEf.mean(1, keepdims=True)

        Cpp_xy = Af_p @ MEf_p.T / (m - 1)
        Cqq    = MEf_p @ MEf_p.T / (m - 1)

        # Cbb is fixed to Cdd, so the legacy weight W = Cdd @ inv(Cbb) is the
        # identity and drops out of both the covariance and the update.
        IpJ = np.eye(Nq) + J
        Cyy = (m - 1) * Cdd + IpJ.T @ IpJ @ Cqq + self.gamma * J.T @ J @ Cqq

        Cyy_inv = inv(Cyy)
        K  = Cpp_xy @ Cyy_inv
        G  = IpJ.T @ (D - Y) - self.gamma * (J.T @ b)   # (Nq, m)
        Aa = Af + K @ G

        # Member-space transform Aa = Af X̂ (for the adaptive inflation):
        # K = Af' MEf'ᵀ Cyy⁻¹ / (m-1) and Af' = Af (I - 11ᵀ/m).
        ones_m = np.ones((m, m)) / m
        self._analysis_transform = (
            np.eye(m) + (np.eye(m) - ones_m) @ (MEf_p.T @ Cyy_inv @ G) / (m - 1)
        )

        return Aa if np.isreal(Aa).all() else Af

    def print_parameters(self, indent: int = 0) -> None:
        super().print_parameters(indent=indent)
        print(f'{" " * indent}gamma (regularisation) = {self.gamma}')

# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary plotting functions  (used in visualisation methods)
# ══════════════════════════════════════════════════════════════════════════════




def plot_observable_history(
    ensemble: EnsembleEstimator,
    truth: Observations | None = None,
    plot_members: bool = False,
    reference_y=1.0,
    reference_t: float = 1.0,
    max_time=None,
    dims='all',
) -> None:
    C = Palette()
    pm, _pb = ensemble.model, ensemble.bias

    y_unbiased, y_model = ensemble.get_observable_hist()

    (y_unbiased, y_model), y_labels = normalized_y(
        reference_y, pm.obs_labels, y_unbiased, y_model
    )

    (t, t_margin), t_label = normalized_time(reference_t, pm.hist_t.copy(), np.array(pm.t_CR))

    t_obs = None
    if len(ensemble.assimilated_data.times) > 0:
        t_obs = normalized_time(reference_t, np.array(ensemble.assimilated_data.times))[0][0]
        max_time = max_time if max_time is not None else t[-1]
        min_time = t_obs[0] - 0.25 * t_margin
    else:
        min_time, max_time = t[0], t[-1]

    t, (y_model, y_unbiased) = cut_signals(t, y_model, y_unbiased, min_time=min_time, max_time=max_time)

    Nq = pm.Nq

    # Partial observation: map each model observable to its column in the observed
    # data. `ensemble.M` holds rows of the model's [0 | I] operator, so the position
    # of each row's 1 in the trailing Nq block is the model dim that row measures.
    # A callable M offers no mapping; assume full observation there.
    M = ensemble.M
    obs_col = {}
    if isinstance(M, np.ndarray) and M.ndim == 2:
        for j, row in enumerate(M):
            p = int(np.flatnonzero(row)[-1]) - (M.shape[1] - Nq)
            if 0 <= p < Nq:
                obs_col[p] = j
    else:
        obs_col = {qi: qi for qi in range(Nq)}

    # y-limits from the truth for every row when one is given (a diverging prediction
    # would squash the panels); dims without their own truth column get the truth's
    # global range. Only a truth-less call falls back to the model.
    y_margin = 0.15 * np.mean(abs(y_model), axis=(0, 2))
    max_y = np.max(y_model, axis=(0, 2))
    min_y = np.min(y_model, axis=(0, 2))

    if truth is not None:
        y_raw   = np.asarray(truth.y_raw.copy())
        y_true  = np.asarray(truth.y_true.copy())
        t_true  = normalized_time(reference_t, np.asarray(truth.t_true.copy()))[0][0]
        (y_raw, y_true), _ = normalized_y(reference_y, pm.obs_labels, y_raw, y_true)
        t_true, (y_raw, y_true) = cut_signals(t_true, y_raw, y_true, min_time=min_time, max_time=max_time)
        if len(t) != len(t_true):
            y_raw  = interpolate(t_true, y_raw,  t)
            y_true = interpolate(t_true, y_true, t)
        # truth arrays carry either every observable or exactly the measured ones,
        # in the operator's order
        truth_col = (lambda qi: qi) if y_true.shape[1] == Nq else obs_col.get
        y_margin = np.full(Nq, 0.15 * np.mean(abs(y_true)))
        max_y = np.full(Nq, np.max(y_true))
        min_y = np.full(Nq, np.min(y_true))
        for qi in range(Nq):
            jt = truth_col(qi)
            if jt is not None:
                y_margin[qi] = 0.15 * np.mean(abs(y_true[:, jt]))
                max_y[qi] = np.max(y_true[:, jt])
                min_y[qi] = np.min(y_true[:, jt])
    else:
        y_true, y_raw = None, None
        truth_col = obs_col.get

    if t_obs is not None:
        y_obs = normalized_y(reference_y, pm.obs_labels,
                             np.array(ensemble.assimilated_data.data)[..., np.newaxis])[0][0]

    dims_list: list[int] = (
        list(range(Nq)) if dims == 'all' else
        [dims] if isinstance(dims, int) else
        [int(d) for d in dims]
    )

    fig1 = plt.figure(figsize=(12, 2 * len(dims_list)), layout='constrained')
    ax_all = fig1.subplots(
        nrows=len(dims_list), ncols=3, sharey='row', sharex='col', width_ratios=[1, 1, 3]
    )
    if len(dims_list) == 1:
        ax_all = ax_all[np.newaxis, :]

    if t_obs is not None:
        x_lims = [
            [t_obs[0] - 0.25 * t_margin, t_obs[0] + 0.75 * t_margin],
            [t_obs[-1] - 0.25 * t_margin, min(t_obs[-1] + 0.75 * t_margin, max_time)],
            [t[0], max_time],
        ]
    else:
        x_lims = [
            [min_time, min_time + t_margin],
            [max_time - t_margin, max_time],
            [t[0], max_time],
        ]

    for row_i, qi in enumerate(dims_list):
        j = obs_col.get(qi)
        jt = truth_col(qi) if y_true is not None else None
        yl = [min_y[qi] - y_margin[qi], max_y[qi] + y_margin[qi]]
        for col_i, (ax, xl) in enumerate(zip(ax_all[row_i], x_lims)):
            if y_true is not None and jt is not None:
                ax.plot(t, y_true[:, jt, :], **C.true_props)
                if y_raw is not None:
                    ax.plot(t, y_raw[:, jt], **C.true_noisy_props)

            if isinstance(y_unbiased, np.ndarray) and y_unbiased.ndim == 3:
                if plot_members:
                    ax.plot(t, y_unbiased[:, qi, :], **C.y_unbias_props)
                else:
                    ax.plot(t, np.mean(y_unbiased[:, qi], axis=-1), **C.y_unbias_props)

            m_mean = np.mean(y_model[:, qi], axis=-1)
            ax.plot(t, m_mean, **C.y_biased_mean_props)

            # always show the ensemble spread; members are drawn on top when requested
            s = np.std(y_model[:, qi], axis=-1)
            ax.fill_between(t, m_mean + s, m_mean - s, color=C.get_color('BIASED', 0.5))
            if plot_members:
                ax.plot(t, y_model[:, qi, :], **C.y_biased_props)

            if t_obs is not None and j is not None:
                ax.plot(t_obs, y_obs[:, j], **C.obs_props)

            if col_i == 0:
                ax.set(ylabel=y_labels[qi])
                if row_i == 0:
                    fig1.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=6, frameon=False)
            ax.set(ylim=yl, xlim=xl)
            if row_i == len(dims_list) - 1:
                ax.set(xlabel=t_label)




def plot_alpha_history(
    ensemble: EnsembleEstimator,
    plot_members: bool = False,
    reference_a=None,
    reference_t: float = 1.0,
    max_time=None,
) -> None:
    pm = ensemble.model
    C = Palette()
    c1 = C.get_color_params(n=ensemble.Na, alpha=1)
    c2 = C.get_color_params(n=ensemble.Na, alpha=0.2)

    (t, t_margin), t_lbl = normalized_time(reference_t, pm.hist_t, np.array(pm.t_CR))
    hist_alpha, alpha_lbls = normalized_alpha(
        pm.hist[:, -pm.Na:],
        alpha_keys=ensemble.est_alpha,
        alpha_labels=pm.alpha_labels,
        reference_a=reference_a,
    )
    mean_alpha = np.mean(hist_alpha, axis=-1)
    std_alpha  = np.std(hist_alpha,  axis=-1)

    t_obs = np.array(ensemble.assimilated_data.times)
    if len(t_obs) > 0:
        t_obs = normalized_time(reference_t, t_obs)[0][0]
        max_time = max_time if max_time is not None else t[-1]
        min_time = t_obs[0] - 0.25 * t_margin
        x_lims = [
            [t_obs[0] - 0.25 * t_margin, t_obs[0] + 0.75 * t_margin],
            [t_obs[-1] - 0.25 * t_margin, min(t_obs[-1] + 0.75 * t_margin, max_time)],
            [min_time, max_time],
        ]
    else:
        min_time, max_time = t[0], t[-1]
        x_lims = [
            [min_time, min_time + t_margin],
            [max_time - t_margin, max_time],
            [min_time, max_time],
        ]

    fig = plt.figure(figsize=(12, 2 * ensemble.Na), layout='constrained')
    axs = fig.subplots(ensemble.Na, 3, sharex='col', sharey='row', width_ratios=[1, 1, 3])
    if ensemble.Na == 1:
        axs = [axs]

    for row_i, axs_row, p in zip(range(ensemble.Na), axs, ensemble.est_alpha):
        avg, s, all_h = [xx[:, row_i] for xx in [mean_alpha, std_alpha, hist_alpha]]
        for col_i, ax in enumerate(axs_row):
            if plot_members:
                ax.plot(t, all_h, color=c2[row_i], lw=1.0)
            ax.fill_between(t, avg + 2 * abs(s), avg - 2 * abs(s), alpha=0.2, color=c2[row_i])
            ax.plot(t, avg, color=c1[row_i], lw=2, dashes=(5, 1))
            if col_i == 0:
                ax.set(ylabel=alpha_lbls[p],
                       ylim=[min(avg) - 3 * max(s), max(avg) + 3 * max(s)])
            if row_i == ensemble.Na - 1:
                ax.set(xlabel=t_lbl, xlim=x_lims[col_i])
