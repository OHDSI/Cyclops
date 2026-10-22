"""Tests for :mod:`cyclops.model` and the binding boundary itself."""

from __future__ import annotations

import numpy as np
import pytest

import cyclops
from cyclops import Control, CyclopsData, CyclopsError, CyclopsModel, Prior, _cyclops
from cyclops.data import INTERCEPT_ID


def _logistic(dataset, **kwargs):
    data = CyclopsData.from_arrays(
        dataset.X, dataset.y, "lr", add_intercept=True, **kwargs
    )
    return data, CyclopsModel(data)


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_version_matches_r_package():
    """The wheel must report the Cyclops version it was built from."""
    assert cyclops.__version__ == _cyclops.version()
    assert cyclops.__version__.count(".") >= 2


def test_gpu_devices_is_a_list():
    assert isinstance(cyclops.list_gpu_devices(), list)


def test_model_capability_predicates():
    assert _cyclops.requires_strata(_cyclops.ModelKind.CONDITIONAL_LOGISTIC)
    assert not _cyclops.requires_strata(_cyclops.ModelKind.LOGISTIC)
    assert _cyclops.removes_intercept(_cyclops.ModelKind.COX)
    assert not _cyclops.removes_intercept(_cyclops.ModelKind.LOGISTIC)
    assert _cyclops.requires_offset(
        _cyclops.ModelKind.SELF_CONTROLLED_CASE_SERIES
    )
    assert _cyclops.requires_time(_cyclops.ModelKind.COX)


def test_all_model_types_are_constructible():
    for model_type in cyclops.MODEL_TYPES:
        assert CyclopsData.create(model_type).model_type == model_type


# ---------------------------------------------------------------------------
# Fit lifecycle
# ---------------------------------------------------------------------------


def test_fit_result_fields(small):
    _, model = _logistic(small)
    result = model.fit()

    assert result.return_flag == "SUCCESS"
    assert result.converged
    assert result.iterations > 0
    assert result.coefficients.shape == (small.n_features + 1,)
    assert result.covariate_ids.shape == result.coefficients.shape
    assert result.covariate_count == small.n_features + 1
    assert result.fit_seconds >= 0.0
    assert result.log_prior == 0.0  # no prior
    assert "FitResult(" in repr(result)


def test_unfinalized_data_is_rejected(small):
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_dense_covariate(1, small.X[:, 0])
    with pytest.raises(ValueError, match="finalize"):
        CyclopsModel(data)


def test_model_without_covariates_is_rejected(small):
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.finalize()
    with pytest.raises(CyclopsError, match="incompletely loaded"):
        CyclopsModel(data)


def test_refitting_is_idempotent(small):
    _, model = _logistic(small)
    first = model.fit()
    second = model.fit()
    np.testing.assert_allclose(first.coefficients, second.coefficients, rtol=1e-12)
    np.testing.assert_allclose(
        first.log_likelihood, second.log_likelihood, rtol=1e-14
    )


def test_prior_can_be_changed_between_fits(small):
    _, model = _logistic(small)
    unpenalized = model.fit()
    assert unpenalized.log_prior == 0.0

    model.set_prior(Prior(kind="laplace", variance=0.01))
    penalized = model.fit()
    assert np.abs(penalized.coefficients).sum() < np.abs(unpenalized.coefficients).sum()
    # log_prior is the normalised Laplace log density, so its sign depends on the
    # rate; what must change is that it is now contributing at all.
    assert penalized.log_prior != 0.0


def test_control_can_be_changed_between_fits(small):
    _, model = _logistic(small)
    model.fit(Control(max_iterations=1))
    assert model.fit(Control(max_iterations=1)).return_flag == "MAX_ITERATIONS"
    assert model.fit(Control(max_iterations=1000)).return_flag == "SUCCESS"


def test_data_outlives_the_model(small):
    """Cyclops holds the data by reference; the binding must keep it alive."""
    import gc

    data, model = _logistic(small)
    reference = model.fit().coefficients
    del data
    gc.collect()
    np.testing.assert_allclose(model.fit().coefficients, reference, rtol=1e-12)


# ---------------------------------------------------------------------------
# Fixed coefficients and weights
# ---------------------------------------------------------------------------


def test_fixed_coefficients_stay_at_their_start(small):
    data, model = _logistic(small)
    start = np.zeros(data.n_covariates)
    start[2] = 0.75
    fixed = np.zeros(data.n_covariates, dtype=bool)
    fixed[2] = True

    model.set_start_values(start)
    model.set_fixed(fixed)
    result = model.fit()
    assert result.coefficients[2] == pytest.approx(0.75)


def test_start_values_and_fixed_are_sized_by_estimated_coefficients(small):
    """Both exclude the offset column, matching `coefficients()`."""
    data = CyclopsData.from_arrays(
        small.X, small.counts, "pr", offset=small.offset, add_intercept=True
    )
    model = CyclopsModel(data)
    estimated = data.n_covariates - 1  # the offset is not estimated
    assert estimated == small.n_features + 1

    model.set_start_values(np.zeros(estimated))
    model.set_fixed([False] * estimated)
    assert model.fit().coefficients.shape == (estimated,)

    with pytest.raises(CyclopsError, match="each estimated coefficient"):
        model.set_start_values(np.zeros(data.n_covariates))
    with pytest.raises(CyclopsError, match="each estimated coefficient"):
        model.set_fixed([False] * data.n_covariates)


def test_weights_length_is_validated(small):
    _, model = _logistic(small)
    with pytest.raises(CyclopsError, match="weight for each data row"):
        model.set_weights(np.ones(small.n_samples - 1))


def test_negative_weights_are_rejected(small):
    _, model = _logistic(small)
    weights = np.ones(small.n_samples)
    weights[0] = -1.0
    with pytest.raises(CyclopsError, match="non-negative"):
        model.set_weights(weights)


def test_censor_weights_range_is_validated(small):
    _, model = _logistic(small)
    weights = np.ones(small.n_samples)
    weights[0] = 1.5
    with pytest.raises(CyclopsError, match=r"\[0, 1\]"):
        model.set_censor_weights(weights)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def test_prior_info_describes_the_penalty(small):
    _, model = _logistic(small)
    model.set_prior(Prior(kind="laplace", variance=2.0))
    # Cyclops parameterises Laplace by rate = sqrt(2 / variance).
    assert model.fit().prior_info.startswith("Laplace(1)")


def test_per_covariate_priors(small):
    data, model = _logistic(small)
    n = data.n_covariates
    model.set_prior(
        Prior(
            kind="laplace",
            kinds=["none"] + ["laplace"] * (n - 1),
            variances=[1.0] + [1e-4] * (n - 1),
        )
    )
    result = model.fit()
    regularized = model.is_regularized()
    assert not regularized[0]
    assert all(regularized[1:])
    assert result.coefficients[0] != 0.0


def test_per_covariate_priors_length_is_validated(small):
    _, model = _logistic(small)
    with pytest.raises(CyclopsError, match="one kind and one variance"):
        model.set_prior(Prior(kind="laplace", kinds=["laplace"], variances=[1.0]))


def test_prior_kinds_without_variances_raises(small):
    _, model = _logistic(small)
    with pytest.raises(ValueError, match="supplied together"):
        model.set_prior(Prior(kind="laplace", kinds=["laplace"]))


def test_invalid_variance_string_raises(small):
    _, model = _logistic(small)
    with pytest.raises(ValueError, match="number or 'cv'"):
        model.set_prior(Prior(kind="laplace", variance="auto"))


def test_excluding_an_unknown_covariate_raises(small):
    _, model = _logistic(small)
    with pytest.raises(CyclopsError, match="not found"):
        model.set_prior(Prior(kind="laplace", variance=1.0, exclude=[123456]))


# ---------------------------------------------------------------------------
# Control validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"cv_search": "brent"}, "cv_search"),
        ({"threads": 0}, "threads"),
        ({"starting_variance": 0.0}, "starting_variance"),
        ({"convergence": "nope"}, "Unknown convergence"),
        ({"selector": "nope"}, "Unknown selector"),
        ({"algorithm": "nope"}, "Unknown algorithm"),
        ({"noise": "nope"}, "Unknown noise"),
    ],
)
def test_control_validation(small, kwargs, message):
    _, model = _logistic(small)
    with pytest.raises(ValueError, match=message):
        model.set_control(Control(**kwargs))


def test_unset_seed_is_accepted(small):
    """R substitutes `as.integer(Sys.time())`; an unset seed must not reach C++."""
    _, model = _logistic(small)
    model.set_control(Control(seed=None))
    assert model.fit().return_flag == "SUCCESS"


def test_mm_reaches_the_same_mode_as_ccd(small):
    """Majorization-minimization is a different route to the same optimum.

    This also guards the `pid`-fallback fix in ModelData::binaryReductionByStratum:
    MM's METHOD_2 bound indexes by stratum, and unstratified data leave the
    stratum vector empty.
    """
    _, ccd = _logistic(small)
    reference = ccd.fit()

    _, mm = _logistic(small)
    # Keep the default tolerance: MM raises "Non-increasing!" when floating-point
    # noise near the optimum makes the objective dip, which a tolerance far below
    # 1e-6 reliably provokes.
    result = mm.fit(Control(algorithm="mm", max_iterations=5000))

    assert result.return_flag == "SUCCESS"
    np.testing.assert_allclose(
        result.coefficients, reference.coefficients, rtol=1e-4, atol=1e-6
    )


def test_kkt_swindle_reaches_the_same_mode(small):
    """Active-set screening is an optimisation, not a different model."""
    _, plain = _logistic(small)
    plain.set_prior(Prior(kind="laplace", variance=0.05))
    baseline = plain.fit(Control(tolerance=1e-10))

    _, screened = _logistic(small)
    screened.set_prior(Prior(kind="laplace", variance=0.05))
    swindled = screened.fit(Control(tolerance=1e-10, use_kkt_swindle=True))

    np.testing.assert_allclose(
        swindled.coefficients, baseline.coefficients, rtol=1e-5, atol=1e-7
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_fisher_information_is_symmetric_positive_definite(small):
    _, model = _logistic(small)
    model.fit()
    information = model.fisher_information()
    n = small.n_features + 1
    assert information.shape == (n, n)
    np.testing.assert_allclose(information, information.T, rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(information) > 0)


def test_standard_errors_match_inverse_information(small):
    _, model = _logistic(small)
    model.fit()
    expected = np.sqrt(np.diag(np.linalg.inv(model.fisher_information())))
    np.testing.assert_allclose(model.standard_errors(), expected, rtol=1e-8)


def test_hessian_diagonal_is_negative_at_the_mode(small):
    """The log-likelihood Hessian is negative definite at a maximum."""
    data, model = _logistic(small)
    model.fit()
    diagonal = model.hessian_diagonal(data.covariate_ids)
    assert np.all(diagonal < 0)
    # ...and it must agree with the Fisher information's diagonal.
    np.testing.assert_allclose(
        -diagonal, np.diag(model.fisher_information()), rtol=1e-8
    )


def test_profile_curve_with_derivatives(small):
    _, model = _logistic(small)
    result = model.fit()
    covariate = int(result.covariate_ids[1])
    estimate = result.coefficients[1]
    points = estimate + np.linspace(-0.4, 0.4, 9)

    curve = model.profile_likelihood(covariate, points, with_derivatives=True)
    assert curve.values.shape == points.shape
    assert curve.derivatives.shape == points.shape
    # The profile derivative changes sign across the maximum.
    assert curve.derivatives[0] > 0 > curve.derivatives[-1]


def test_confidence_intervals_widen_with_threshold(small):
    data, model = _logistic(small)
    model.fit()
    ids = [int(data.covariate_ids[1])]
    narrow = model.confidence_intervals(ids, threshold=0.5)[0]
    wide = model.confidence_intervals(ids, threshold=1.920729)[0]
    assert wide.lower < narrow.lower
    assert wide.upper > narrow.upper


def test_log_likelihood_accessors_agree(small):
    _, model = _logistic(small)
    result = model.fit()
    assert model.log_likelihood == pytest.approx(result.log_likelihood)
    assert model.log_prior == pytest.approx(result.log_prior)
    np.testing.assert_allclose(model.coefficients, result.coefficients)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_silent_by_default(small):
    _, model = _logistic(small)
    model.fit()
    assert model.take_log() == []


def test_noisy_mode_emits_progress(small):
    data = CyclopsData.from_arrays(
        small.X, small.y, "lr", add_intercept=True, silent=False
    )
    model = CyclopsModel(data, control=Control(noise="noisy"))
    model.fit()
    messages = model.take_log()
    assert messages, "expected progress output at noise='noisy'"
    assert model.take_log() == [], "take_log() must drain the buffer"


# ---------------------------------------------------------------------------
# Errors surface as Python exceptions, not crashes
# ---------------------------------------------------------------------------


def test_cyclops_error_is_a_runtime_error():
    assert issubclass(CyclopsError, RuntimeError)


def test_unknown_covariate_lookup_raises(small):
    _, model = _logistic(small)
    model.fit()
    with pytest.raises(CyclopsError, match="not found"):
        model.hessian_diagonal([999999])


def test_repr(small):
    _, model = _logistic(small)
    assert "CyclopsModel(model_type='lr'" in repr(model)


# ---------------------------------------------------------------------------
# Regressions from the code review
# ---------------------------------------------------------------------------


def test_poor_blr_retry_does_not_rewrite_the_caller_options(small):
    """The Lange retry must be scoped to the fit() that needed it.

    R recurses with a modified *local* control, so the caller's configuration
    survives. Leaving the model on Lange made a second fit() reach a different
    coefficient than a fresh model with identical stated options.
    """
    # A Jeffreys prior on a treatment-separated Cox arm trips POOR_BLR_STEP under
    # the gradient criterion (the R suite's test-finiteMLE.R case).
    rng = np.random.default_rng(123)
    n = 1000
    censor = rng.exponential(100.0, n)
    exposed = rng.random(n) < 0.5
    event_time = np.full(n, np.inf)
    event_time[~exposed] = rng.exponential(100.0, int((~exposed).sum()))
    time = np.minimum(censor, event_time)
    outcome = (event_time < censor).astype(float)
    X = exposed.astype(float).reshape(-1, 1)

    def build():
        data = CyclopsData.from_arrays(X, outcome, "cox", time=time)
        return CyclopsModel(
            data,
            prior=Prior(kind="jeffreys"),
            control=Control(convergence="gradient", max_iterations=100),
        )

    model = build()
    first = model.fit()
    fresh = build().fit()

    # The retry ran, so the flag is no longer POOR_BLR_STEP...
    assert first.return_flag == "SUCCESS"
    # ...and a pristine model with the same options agrees.
    np.testing.assert_allclose(
        first.coefficients, fresh.coefficients, rtol=1e-10, atol=1e-12
    )
    assert first.iterations == fresh.iterations
    # The caller's criterion is still in force for anything that follows.
    assert model.control.convergence == "gradient"


def test_per_covariate_priors_still_honour_exclusions(small):
    """`exclude` and the automatic intercept exclusion apply to both prior forms."""
    data = CyclopsData.from_arrays(
        small.X, small.y, "lr", add_intercept=True
    )
    model = CyclopsModel(data)
    n = data.n_covariates
    ids = data.covariate_ids

    model.set_prior(
        Prior(
            kind="laplace",
            kinds=["laplace"] * n,
            variances=[1e-4] * n,
            exclude=[int(ids[2])],
        )
    )
    regularized = model.is_regularized()
    assert not regularized[0], "intercept must stay unpenalized by default"
    assert not regularized[2], "explicitly excluded covariate must be unpenalized"
    assert regularized[1] and regularized[3], "the rest must be penalized"


def test_per_covariate_priors_can_force_the_intercept(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr", add_intercept=True)
    model = CyclopsModel(data)
    n = data.n_covariates
    model.set_prior(
        Prior(
            kind="laplace",
            kinds=["laplace"] * n,
            variances=[1e-4] * n,
            force_intercept=True,
        )
    )
    assert all(model.is_regularized())


def test_intercept_after_offset_is_rejected(small):
    """The offset must stay at column 0; inserting ahead of it corrupts indexing."""
    reserved = -(2**63)

    def build(intercept_first: bool):
        data = CyclopsData.create("pr")
        data.set_outcome(small.counts)
        data.add_dense_covariate(reserved, small.offset)
        data.add_covariates(small.X, covariate_ids=np.arange(1, small.n_features + 1))
        if intercept_first:
            data.add_intercept()
            data.set_offset(reserved)
        else:
            data.set_offset(reserved)
            data.add_intercept()
        return data

    good = build(intercept_first=True)
    assert good.covariate_ids[0] == -1        # offset
    assert good.intercept_label == INTERCEPT_ID

    with pytest.raises(CyclopsError, match="before promoting the offset"):
        build(intercept_first=False)


@pytest.mark.parametrize(
    "seed,expected",
    [(0, 0), (42, 42), (2**31 - 2, 2**31 - 2), (-1, -1), (-99, -99)],
)
def test_seed_passthrough_in_the_portable_range(small, seed, expected):
    from cyclops.model import _portable_seed

    assert _portable_seed(seed) == expected


def test_large_seed_is_folded_into_the_portable_range(small):
    """`long` is 32-bit on Windows, so a wide seed must be narrowed here.

    Otherwise the same random_state either overflows the binding or narrows
    differently per platform, and cross-validation picks different folds.
    """
    from cyclops.model import _portable_seed

    for seed in (2**40, -(2**40), 2**63 - 1):
        folded = _portable_seed(seed)
        assert 0 <= folded < 2**31 - 1, folded

    # And it must actually reach the optimizer without raising on any platform.
    data = CyclopsData.from_arrays(small.X, small.y, "lr", add_intercept=True)
    model = CyclopsModel(data, control=Control(seed=2**40))
    assert model.fit().return_flag == "SUCCESS"
