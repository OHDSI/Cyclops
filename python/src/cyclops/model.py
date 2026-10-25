"""Fitting: :class:`Prior`, :class:`Control`, :class:`FitResult`, :class:`CyclopsModel`.

This layer is a typed, documented view of the C++ facade. It adds no numerics —
the estimators in :mod:`cyclops.estimators` are built on it, and it is the right
entry point when the estimator API is too narrow (streaming loads, profile
likelihoods, per-covariate priors).
"""

from __future__ import annotations

import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cyclops import _cyclops
from cyclops._enums import (
    ALGORITHM_KINDS,
    CONVERGENCE_KINDS,
    NOISE_LEVELS,
    PRIOR_KINDS,
    SELECTOR_KINDS,
    resolve,
)
from cyclops.data import CyclopsData

__all__ = ["Prior", "Control", "FitResult", "CyclopsModel", "CyclopsError"]

CyclopsError = _cyclops.CyclopsError

#: Cyclops reserves two seed values: -1 means "deterministic, do not shuffle" and
#: -99 means "seed from the wall clock" (see drivers/AbstractSelector.cpp).
_RESERVED_SEEDS = (-1, -99)

_SEED_MODULUS = 2**31 - 1


def _portable_seed(value: int) -> int:
    """Fold ``value`` into the range every platform's C ``long`` can hold.

    Keeps the two values Cyclops treats as sentinels intact, and maps everything
    else into ``[0, 2**31 - 2]`` so the same ``random_state`` produces the same
    cross-validation folds on 32-bit-``long`` platforms (Windows) as on 64-bit
    ones.
    """
    if value in _RESERVED_SEEDS:
        return value
    return abs(value) % _SEED_MODULUS


@dataclass
class Prior:
    """Regularization specification.

    Cyclops parameterizes both penalties by *variance*: a Laplace prior with
    variance ``v`` corresponds to an L1 rate of ``sqrt(2 / v)``, and a normal
    prior with variance ``v`` to an L2 penalty of ``1 / (2 v)``. Larger variance
    means weaker shrinkage.

    Args:
        kind: ``"none"``, ``"laplace"`` (``"l1"``), ``"normal"`` (``"l2"``),
            ``"barupdate"`` or ``"jeffreys"``.
        variance: Prior variance, or ``"cv"`` to select it by cross-validation.
        exclude: Covariate ids left unpenalized.
        force_intercept: By default the intercept is not penalized; set this to
            penalize it. Matches ``createPrior(forceIntercept=)``.
        kinds, variances: Per-covariate overrides; both must be given, each with
            one entry per covariate in covariate-id order.
    """

    kind: str = "none"
    variance: float | str = 1.0
    exclude: Sequence[int] | None = None
    force_intercept: bool = False
    kinds: Sequence[str] | None = None
    variances: Sequence[float] | None = None

    def _to_options(self) -> "_cyclops.PriorOptions":
        options = _cyclops.PriorOptions()
        options.kind = resolve(PRIOR_KINDS, self.kind, "prior")
        options.force_intercept = self.force_intercept
        options.exclude = [] if self.exclude is None else [int(i) for i in self.exclude]

        if isinstance(self.variance, str):
            if self.variance.strip().lower() != "cv":
                raise ValueError(
                    f"variance must be a number or 'cv', got {self.variance!r}"
                )
            options.use_cross_validation = True
            options.variance = 1.0  # starting point; the search overwrites it
        else:
            options.variance = float(self.variance)

        if (self.kinds is None) != (self.variances is None):
            raise ValueError("`kinds` and `variances` must be supplied together")
        if self.kinds is not None and self.variances is not None:
            options.kinds = [resolve(PRIOR_KINDS, k, "prior") for k in self.kinds]
            options.variances = [float(v) for v in self.variances]

        return options


@dataclass
class Control:
    """Mode-finding and cross-validation controls.

    Defaults match R's ``createControl()``, so the two packages take the same
    iterate path on the same data.
    """

    max_iterations: int = 1000
    tolerance: float = 1e-6
    convergence: str = "gradient"
    algorithm: str = "ccd"
    initial_bound: float = 2.0
    max_bound_count: int = 5
    use_kkt_swindle: bool = False
    swindle_multiplier: int = 10
    do_it_all: bool = True

    # Cross-validation
    cv_search: str = "auto"     #: "auto" (Brent) or "grid"
    fold: int = 10
    cv_repetitions: int = 1
    lower_limit: float = 0.01
    upper_limit: float = 20.0
    grid_steps: int = 10
    starting_variance: float = -1.0
    selector: str = "auto"
    min_cv_data: int = 100

    # Runtime
    noise: str = "silent"
    threads: int = 1
    seed: int | None = None
    reset_coefficients: bool = False
    retry_on_poor_blr_step: bool = True

    def _to_options(self) -> "_cyclops.FitOptions":
        if self.cv_search not in ("auto", "grid"):
            raise ValueError(
                f"cv_search must be 'auto' or 'grid', got {self.cv_search!r}"
            )
        if self.threads != -1 and self.threads < 1:
            raise ValueError("threads must be -1 or >= 1")
        if self.starting_variance != -1 and self.starting_variance <= 0:
            raise ValueError("starting_variance must be -1 or positive")

        options = _cyclops.FitOptions()
        options.max_iterations = int(self.max_iterations)
        options.tolerance = float(self.tolerance)
        options.convergence = resolve(
            CONVERGENCE_KINDS, self.convergence, "convergence"
        )
        options.algorithm = resolve(ALGORITHM_KINDS, self.algorithm, "algorithm")
        options.initial_bound = float(self.initial_bound)
        options.max_bound_count = int(self.max_bound_count)
        options.use_kkt_swindle = bool(self.use_kkt_swindle)
        options.swindle_multiplier = int(self.swindle_multiplier)
        options.do_it_all = bool(self.do_it_all)

        options.auto_search = self.cv_search == "auto"
        options.fold = int(self.fold)
        options.cv_repetitions = int(self.cv_repetitions)
        options.lower_limit = float(self.lower_limit)
        options.upper_limit = float(self.upper_limit)
        options.grid_steps = int(self.grid_steps)
        options.starting_variance = float(self.starting_variance)
        options.selector = resolve(SELECTOR_KINDS, self.selector, "selector")
        options.min_cv_data = int(self.min_cv_data)

        options.noise = resolve(NOISE_LEVELS, self.noise, "noise level")
        options.threads = int(self.threads)
        # R substitutes `as.integer(Sys.time())`; do the same so an unset seed
        # still varies between runs of a cross-validated fit.
        raw_seed = int(_time.time()) if self.seed is None else int(self.seed)
        # CCDArguments::seed is a C `long`, which is 32-bit on Windows and
        # 64-bit elsewhere. Fold it into the signed 32-bit range here so a given
        # random_state selects the same folds on every platform, rather than
        # overflowing the binding on Windows and narrowing silently in the core.
        options.seed = _portable_seed(raw_seed)
        options.reset_coefficients = bool(self.reset_coefficients)
        options.retry_on_poor_blr_step = bool(self.retry_on_poor_blr_step)
        return options


@dataclass
class FitResult:
    """Outcome of a fit. Field names follow R's ``cyclopsFit`` object."""

    covariate_ids: np.ndarray
    coefficients: np.ndarray
    log_likelihood: float
    log_prior: float
    return_flag: str
    iterations: int
    prior_info: str
    variance: list[float] = field(default_factory=list)
    cross_validation_info: str = ""
    covariate_count: int = 0
    fit_seconds: float = 0.0

    @property
    def converged(self) -> bool:
        return self.return_flag == "SUCCESS"

    @classmethod
    def _from_native(cls, native: "_cyclops.FitResult") -> "FitResult":
        return cls(
            covariate_ids=native.covariate_ids,
            coefficients=native.coefficients,
            log_likelihood=native.log_likelihood,
            log_prior=native.log_prior,
            return_flag=native.return_flag,
            iterations=native.iterations,
            prior_info=native.prior_info,
            variance=list(native.variance),
            cross_validation_info=native.cross_validation_info,
            covariate_count=native.covariate_count,
            fit_seconds=native.fit_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"FitResult(return_flag={self.return_flag!r}, "
            f"iterations={self.iterations}, "
            f"log_likelihood={self.log_likelihood:.6g}, "
            f"n_coefficients={len(self.coefficients)})"
        )


class CyclopsModel:
    """An optimizer bound to a :class:`~cyclops.data.CyclopsData`.

    The data object is kept alive for as long as the model exists; the C++ core
    holds it by reference.
    """

    def __init__(
        self,
        data: CyclopsData,
        *,
        compute_device: str = "native",
        prior: Prior | None = None,
        control: Control | None = None,
    ):
        if not data.is_finalized:
            raise ValueError("Call CyclopsData.finalize() before fitting")
        self._data = data
        self._handle = _cyclops.Model.create(data._handle, compute_device)
        self._prior = prior or Prior()
        self._control = control or Control()
        self._handle.set_prior(self._prior._to_options())
        self._handle.set_options(self._control._to_options())

    # -- configuration ------------------------------------------------------

    @property
    def data(self) -> CyclopsData:
        return self._data

    @property
    def prior(self) -> Prior:
        return self._prior

    @property
    def control(self) -> Control:
        return self._control

    def set_prior(self, prior: Prior) -> "CyclopsModel":
        self._prior = prior
        self._handle.set_prior(prior._to_options())
        return self

    def set_control(self, control: Control) -> "CyclopsModel":
        self._control = control
        self._handle.set_options(control._to_options())
        return self

    def set_weights(self, weights) -> "CyclopsModel":
        """Row weights in *stored* row order (see :attr:`CyclopsData.row_order`)."""
        self._handle.set_weights(np.ascontiguousarray(weights, dtype=np.float64))
        return self

    def set_censor_weights(self, weights) -> "CyclopsModel":
        """Fine-Gray censoring weights, each in ``[0, 1]``."""
        self._handle.set_censor_weights(
            np.ascontiguousarray(weights, dtype=np.float64)
        )
        return self

    def set_start_values(self, beta) -> "CyclopsModel":
        """Starting coefficients, one per covariate in covariate-id order."""
        self._handle.set_start_values(np.ascontiguousarray(beta, dtype=np.float64))
        return self

    def set_fixed(self, fixed) -> "CyclopsModel":
        """Hold coefficients at their starting value."""
        self._handle.set_fixed([bool(f) for f in fixed])
        return self

    # -- fitting ------------------------------------------------------------

    def fit(self, control: Control | None = None) -> FitResult:
        """Find the mode, cross-validating the prior variance if requested."""
        if control is not None:
            self.set_control(control)
        return FitResult._from_native(self._handle.fit())

    # -- inference ----------------------------------------------------------

    @property
    def coefficients(self) -> np.ndarray:
        return self._handle.coefficients

    @property
    def log_likelihood(self) -> float:
        return self._handle.log_likelihood

    @property
    def log_prior(self) -> float:
        return self._handle.log_prior

    def predict(self) -> np.ndarray:
        """Response-scale fitted values for the training rows, in stored order.

        Probabilities for logistic models, rates for Poisson, and the linear
        predictor for the conditional and survival models — the same convention
        as ``predict.cyclopsFit``.
        """
        return self._handle.predict()

    def gradient(self) -> np.ndarray:
        """``d log L / d beta`` at the current coefficients."""
        return self._handle.gradient()

    def hessian_diagonal(self, ids: Sequence[int]) -> np.ndarray:
        """Diagonal of the log-likelihood Hessian; negative at a maximum.

        Negate to get the observed information.
        """
        return self._handle.hessian_diagonal([int(i) for i in ids])

    def standard_errors(self, ids: Sequence[int] | None = None) -> np.ndarray:
        """Asymptotic standard errors from the inverse Fisher information.

        Only strictly valid when computed for all covariates at once, matching
        the caveat in R's ``getSEs()``.
        """
        return self._handle.standard_errors(
            [] if ids is None else [int(i) for i in ids]
        )

    def fisher_information(self, ids: Sequence[int] | None = None) -> np.ndarray:
        return self._handle.fisher_information(
            [] if ids is None else [int(i) for i in ids]
        )

    def is_regularized(self) -> list[bool]:
        return self._handle.is_regularized()

    def confidence_intervals(
        self,
        ids: Sequence[int] | None = None,
        *,
        threads: int = 1,
        threshold: float = 1.920729,
        include_penalty: bool = False,
    ):
        """Adaptive-bisection likelihood-profile intervals.

        Returns a list of objects with ``covariate_id``, ``lower``, ``upper`` and
        ``evaluations``. The default ``threshold`` is the 95% chi-square cutoff
        used by ``confint.cyclopsFit``.
        """
        return self._handle.profile(
            [] if ids is None else [int(i) for i in ids],
            threads,
            threshold,
            include_penalty,
        )

    def profile_likelihood(
        self,
        covariate_id: int,
        points,
        *,
        threads: int = 1,
        include_penalty: bool = False,
        with_derivatives: bool = False,
    ):
        """Profile log-likelihood evaluated at the given values of one coefficient."""
        return self._handle.profile_curve(
            int(covariate_id),
            np.ascontiguousarray(points, dtype=np.float64),
            threads,
            include_penalty,
            with_derivatives,
        )

    def take_log(self) -> list[str]:
        """Drain buffered progress output (empty unless ``noise != "silent"``)."""
        return self._handle.take_log()

    def __repr__(self) -> str:
        return (
            f"CyclopsModel(model_type={self._data.model_type!r}, "
            f"n_rows={self._data.n_rows}, n_covariates={self._data.n_covariates})"
        )
