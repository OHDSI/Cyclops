"""Parity with the R Cyclops package.

The R package is the reference implementation: same C++ core, years of use, and
its own test-suite validated against ``glm``, ``survival::coxph``, ``gnm`` and
friends. These tests fit identical data through both front-ends and compare.

Because both paths reach the same optimizer, agreement should be far tighter
than statistical tolerance — coefficients to ~1e-10 and log likelihoods to
~1e-12. The tolerances below are deliberately near machine precision so that a
regression in the binding layer (a mis-ordered row, a dropped option, a
different warm start) fails loudly instead of hiding inside a loose 1e-4.

Run with:  pytest tests/ -m parity
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import fit
from cyclops import Control, Prior
from r_bridge import fit_in_r

pytestmark = pytest.mark.parity

#: Both front-ends run the same coordinate descent, so any disagreement beyond
#: accumulated floating-point noise is a binding bug.
COEF_TOL = 1e-9
LL_TOL = 1e-10


def assert_matches(
    result,
    r_result,
    *,
    coefficients=None,
    coef_tol: float = COEF_TOL,
    ll_tol: float = LL_TOL,
    check_iterations: bool = True,
):
    """Compare a Python :class:`~cyclops.FitResult` against an :class:`~r_bridge.RFit`.

    ``coefficients`` defaults to the fit's own vector. Both front-ends label the
    intercept 0 and the covariates 1..p and report them in id order, so the two
    vectors line up without reordering.
    """
    if coefficients is None:
        coefficients = np.asarray(result.coefficients)

    assert result.return_flag == r_result.return_flag, (
        f"convergence flag differs: python={result.return_flag} "
        f"r={r_result.return_flag}"
    )
    assert coefficients.shape == r_result.coefficients.shape, (
        f"coefficient count differs: python={coefficients.shape[0]} "
        f"r={r_result.coefficients.shape[0]}"
    )
    np.testing.assert_allclose(
        coefficients, r_result.coefficients, rtol=coef_tol, atol=coef_tol
    )
    np.testing.assert_allclose(
        result.log_likelihood, r_result.log_likelihood, rtol=ll_tol, atol=ll_tol
    )
    np.testing.assert_allclose(
        result.log_prior, r_result.log_prior, rtol=ll_tol, atol=ll_tol
    )
    if check_iterations:
        assert result.iterations == r_result.iterations, (
            f"iteration count differs: python={result.iterations} "
            f"r={r_result.iterations}"
        )


# ---------------------------------------------------------------------------
# Unconditional models
# ---------------------------------------------------------------------------


def test_logistic_unregularized(small):
    """The base case: intercept warm start, gradient convergence, no penalty."""
    data, _, result = fit(small.X, small.y, "lr", add_intercept=True)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True)

    assert_matches(result, reference)
    # R labels the intercept 0 and the covariates 1..p.
    np.testing.assert_array_equal(result.covariate_ids, reference.covariate_ids)
    np.testing.assert_array_equal(data.covariate_ids, reference.covariate_ids)


def test_logistic_no_intercept(small):
    _, _, result = fit(small.X, small.y, "lr", add_intercept=False)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=False)
    assert_matches(result, reference)


@pytest.mark.parametrize("variance", [0.01, 0.1, 1.0, 10.0])
def test_logistic_laplace(small, variance):
    """L1 at several strengths; also checks the automatic intercept exclusion."""
    _, _, result = fit(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior=Prior("laplace", variance=variance),
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        prior_variance=variance,
    )
    assert_matches(result, reference)
    assert result.prior_info == reference.prior_info


@pytest.mark.parametrize("variance", [0.05, 2.0])
def test_logistic_normal(small, variance):
    _, _, result = fit(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior=Prior("normal", variance=variance),
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="normal",
        prior_variance=variance,
    )
    assert_matches(result, reference)


def test_logistic_excluded_covariates(small):
    """Unpenalized columns must land on the same coefficients."""
    # Default ids are 1..p, so column c is id c + 1 on both sides.
    excluded_ids = [1, 3]
    _, _, result = fit(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior=Prior("laplace", variance=0.1, exclude=excluded_ids),
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        prior_variance=0.1,
        exclude=excluded_ids,
    )
    assert_matches(result, reference)


def test_logistic_sample_weights(small):
    rng = np.random.default_rng(5)
    weights = rng.integers(0, 3, size=small.n_samples).astype(np.float64)
    _, _, result = fit(small.X, small.y, "lr", add_intercept=True, weights=weights)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True, weights=weights)
    assert_matches(result, reference)


def test_logistic_predictions_match(small):
    """`Model::predict` must come back through the binding unchanged."""
    _, model, _ = fit(small.X, small.y, "lr", add_intercept=True)
    reference = fit_in_r(
        small.X, small.y, "lr", add_intercept=True, want_predictions=True
    )
    np.testing.assert_allclose(
        model.predict(), reference.predictions, rtol=1e-9, atol=1e-12
    )


def test_logistic_standard_errors(small):
    _, model, result = fit(small.X, small.y, "lr", add_intercept=True)
    reference = fit_in_r(
        small.X, small.y, "lr", add_intercept=True, want_standard_errors=True
    )
    assert reference.standard_errors is not None
    # `result.covariate_ids` omits any offset column, which has no standard error.
    np.testing.assert_allclose(
        model.standard_errors(list(result.covariate_ids)),
        reference.standard_errors,
        rtol=1e-8,
        atol=1e-10,
    )


def test_poisson_with_offset(small):
    _, _, result = fit(
        small.X, small.counts, "pr", add_intercept=True, offset=small.offset
    )
    reference = fit_in_r(
        small.X, small.counts, "pr", add_intercept=True, time=small.offset
    )
    assert_matches(result, reference)


def test_poisson_without_offset(small):
    """An offset of 1 is the only way to express "no offset" through R's
    `convertToCyclopsData`, which always promotes one for `pr`."""
    ones = np.ones(small.n_samples)
    _, _, result = fit(small.X, small.counts, "pr", add_intercept=True, offset=ones)
    reference = fit_in_r(small.X, small.counts, "pr", add_intercept=True, time=ones)
    assert_matches(result, reference)

    # ...and it must agree with omitting the offset entirely.
    _, _, plain = fit(small.X, small.counts, "pr", add_intercept=True)
    np.testing.assert_allclose(
        plain.coefficients, result.coefficients, rtol=1e-9, atol=1e-9
    )


def test_least_squares(small):
    _, _, result = fit(small.X, small.continuous, "ls", add_intercept=True)
    reference = fit_in_r(small.X, small.continuous, "ls", add_intercept=True)
    assert_matches(result, reference)


# ---------------------------------------------------------------------------
# Stratified models
# ---------------------------------------------------------------------------


def test_conditional_logistic(small):
    data, _, result = fit(small.X, small.y, "clr", strata=small.strata)
    reference = fit_in_r(small.X, small.y, "clr", strata=small.strata)
    assert_matches(result, reference)
    assert data.n_strata == reference.n_strata


def test_conditional_logistic_regularized(small):
    _, _, result = fit(
        small.X,
        small.y,
        "clr",
        strata=small.strata,
        prior=Prior("laplace", variance=0.5),
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "clr",
        strata=small.strata,
        prior="laplace",
        prior_variance=0.5,
    )
    assert_matches(result, reference)


def test_conditional_poisson(small):
    _, _, result = fit(
        small.X, small.counts, "cpr", strata=small.strata, offset=small.offset
    )
    reference = fit_in_r(
        small.X, small.counts, "cpr", strata=small.strata, time=small.offset
    )
    assert_matches(result, reference)


def test_self_controlled_case_series(small):
    """SCCS consumes the offset as its `time` vector, not as a design column."""
    _, _, result = fit(
        small.X, small.counts, "sccs", strata=small.strata, offset=small.offset
    )
    reference = fit_in_r(
        small.X,
        small.counts,
        "sccs",
        strata=small.strata,
        time=small.offset,
    )
    assert_matches(result, reference)


def test_strata_out_of_order(small):
    """Unsorted strata must be permuted internally, not silently mis-grouped."""
    rng = np.random.default_rng(3)
    shuffle = rng.permutation(small.n_samples)
    X, y, strata = small.X[shuffle], small.y[shuffle], small.strata[shuffle]

    _, _, result = fit(X, y, "clr", strata=strata)
    reference = fit_in_r(X, y, "clr", strata=strata)  # R sorts too
    assert_matches(result, reference)

    _, _, sorted_result = fit(small.X, small.y, "clr", strata=small.strata)
    np.testing.assert_allclose(
        result.coefficients, sorted_result.coefficients, rtol=1e-9, atol=1e-9
    )


# ---------------------------------------------------------------------------
# Survival models
# ---------------------------------------------------------------------------


def test_cox(small):
    _, _, result = fit(small.X, small.y, "cox", time=small.time)
    reference = fit_in_r(small.X, small.y, "cox", time=small.time)
    assert_matches(result, reference)


def test_cox_regularized(small):
    _, _, result = fit(
        small.X,
        small.y,
        "cox",
        time=small.time,
        prior=Prior("laplace", variance=0.2),
    )
    reference = fit_in_r(
        small.X, small.y, "cox", time=small.time, prior="laplace", prior_variance=0.2
    )
    assert_matches(result, reference)


def test_cox_stratified(small):
    _, _, result = fit(small.X, small.y, "cox", time=small.time, strata=small.strata)
    reference = fit_in_r(
        small.X, small.y, "cox", time=small.time, strata=small.strata
    )
    assert_matches(result, reference)


def test_cox_time_out_of_order(small):
    """Cox needs rows in descending-time order; the binding must impose it."""
    rng = np.random.default_rng(11)
    shuffle = rng.permutation(small.n_samples)
    _, _, shuffled = fit(
        small.X[shuffle], small.y[shuffle], "cox", time=small.time[shuffle]
    )
    _, _, ordered = fit(small.X, small.y, "cox", time=small.time)
    np.testing.assert_allclose(
        shuffled.coefficients, ordered.coefficients, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        shuffled.log_likelihood, ordered.log_likelihood, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Sparse designs and cross-validation
# ---------------------------------------------------------------------------


def test_sparse_indicator_design(sparse_binary):
    X, y, _ = sparse_binary
    _, _, result = fit(
        X, y, "lr", add_intercept=True, prior=Prior("laplace", variance=0.1)
    )
    reference = fit_in_r(
        X, y, "lr", add_intercept=True, prior="laplace", prior_variance=0.1
    )
    assert_matches(
        result,
        reference,
        # 300 penalized coefficients accumulate more floating-point noise than
        # the 4-covariate cases.
        coef_tol=1e-7,
        ll_tol=1e-9,
    )


def test_sparse_and_dense_agree(small):
    """The same numbers stored sparsely must give the same fit."""
    import scipy.sparse as sp

    _, _, dense = fit(
        small.X, small.y, "lr", add_intercept=True,
        prior=Prior("laplace", variance=0.3),
    )
    _, _, sparse = fit(
        sp.csc_matrix(small.X), small.y, "lr", add_intercept=True,
        prior=Prior("laplace", variance=0.3),
    )
    np.testing.assert_allclose(
        dense.coefficients, sparse.coefficients, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        dense.log_likelihood, sparse.log_likelihood, rtol=1e-12, atol=1e-12
    )


@pytest.mark.slow
def test_cross_validated_variance(medium):
    """A cross-validated search must land on the same hyperparameter.

    Fold assignment is seeded, so a shared seed makes this deterministic across
    both front-ends.
    """
    seed = 4242
    _, _, result = fit(
        medium.X,
        medium.y,
        "lr",
        add_intercept=True,
        prior=Prior(kind="laplace", variance="cv"),
        control=Control(seed=seed, fold=10, cv_repetitions=1, selector="byrow"),
    )

    reference = fit_in_r(
        medium.X,
        medium.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        use_cross_validation=True,
        seed=seed,
        fold=10,
        cv_repetitions=1,
        selector="byRow",
    )

    np.testing.assert_allclose(
        result.variance, reference.variance, rtol=1e-6, atol=1e-8
    )
    assert_matches(result, reference, coef_tol=1e-7, ll_tol=1e-9)


# ---------------------------------------------------------------------------
# Non-convergent and degenerate cases
# ---------------------------------------------------------------------------


def test_separable_data_reports_same_flag():
    """A separable design has no finite MLE; both sides must say so identically."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    _, _, result = fit(
        X, y, "lr", add_intercept=True, control=Control(max_iterations=100)
    )
    reference = fit_in_r(X, y, "lr", add_intercept=True, max_iterations=100)

    assert result.return_flag == reference.return_flag
    assert not result.converged
    np.testing.assert_allclose(
        result.coefficients, reference.coefficients, rtol=1e-7, atol=1e-7
    )


def test_max_iterations_respected(small):
    _, _, result = fit(
        small.X, small.y, "lr", add_intercept=True, control=Control(max_iterations=2)
    )
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True, max_iterations=2)
    assert result.return_flag == reference.return_flag == "MAX_ITERATIONS"
    assert_matches(result, reference)


@pytest.mark.parametrize("convergence", ["lange", "zhang"])
def test_alternative_convergence_criteria(small, convergence):
    _, _, result = fit(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        control=Control(convergence=convergence),
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        convergence=convergence,
    )
    assert_matches(result, reference)
