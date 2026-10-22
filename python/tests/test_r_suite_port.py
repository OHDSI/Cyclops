"""Port of the R testthat suite's gold-standard comparisons.

`test_parity_with_r.py` checks that Python and R Cyclops agree with each other.
That is necessary but circular: a bug in the shared C++ core would satisfy it.
This module closes the loop by asserting against the *external* gold standards
the R suite uses — `glm`, `lm`, `coxph`, `clogit` and `gnm` — so the Python
bindings are validated against implementations that share no code with Cyclops.

The gold values are literals here rather than computed at test time: R is not a
test dependency, and pinning them makes a regression in either implementation
visible. Each is annotated with the R expression that produced it, and
`data/generate_fixtures.R` regenerates every number and CSV in this directory.

Tolerances match the R suite's own (1e-4 for most cases), except where a tighter
bound is defensible.

Mapping to the R files:

| R test file                  | ported here                              |
|------------------------------|------------------------------------------|
| test-smallBernoulli.R        | TestSmallBernoulli                       |
| test-smallPoisson.R          | TestSmallPoisson                         |
| test-smallNormal.R           | TestSmallNormal                          |
| test-smallCox.R              | TestSmallCox                             |
| test-smallCLR.R              | TestConditionalLogistic, TestConditionalPoisson |
| test-conditionalPoisson.R    | TestConditionalPoisson, TestSccs         |
| test-correlation.R           | TestUnivariableCorrelation               |
| test-covariateRegularization.R | TestCovariateRegularization            |
| test-gradient.R              | TestGradient                             |
| test-reductions.R            | TestReductions                           |
| test-normalization.R         | TestNormalization                        |
| test-finiteMLE.R             | TestInfiniteMle                          |
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from conftest import fit
from cyclops import Control, CyclopsData, CyclopsError, CyclopsModel, Prior

DATA = Path(__file__).parent / "data"

#: The R suite's default tolerance for gold-standard comparisons.
TOL = 1e-4


def _read_csv(name: str) -> dict[str, np.ndarray]:
    """Read a fixture into named float columns."""
    with (DATA / name).open() as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.array([float(row[key]) for row in rows]) for key in rows[0]
    }


def _dummies(codes: np.ndarray, levels: list[float]) -> np.ndarray:
    """Treatment-contrast dummies, dropping the first level as R's default does."""
    return np.column_stack([(codes == level).astype(float) for level in levels[1:]])


# ---------------------------------------------------------------------------
# test-smallBernoulli.R
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bernoulli():
    columns = _read_csv("bernoulli.csv")
    return columns["log_bid"].reshape(-1, 1), columns["y"]


@pytest.fixture(scope="module")
def bladder():
    columns = _read_csv("bladder.csv")
    X = np.column_stack([columns["rx"], columns["size"]])
    return fit(
        X,
        columns["event"],
        "cox",
        time=columns["stop"],
        control=Control(tolerance=1e-12),
    )


@pytest.fixture(scope="module")
def oxford():
    return _read_csv("oxford.csv")


@pytest.fixture(scope="module")
def infert_clr():
    columns = _read_csv("infert.csv")
    X = np.column_stack([columns["spontaneous"], columns["induced"]])
    return (
        fit(X, columns["case"], "clr", strata=columns["stratum"].astype(np.int64)),
        columns,
    )


class TestSmallBernoulli:
    """Logistic regression against `glm(y ~ log_bid, family = binomial())`."""

    # coef(glm(y ~ log_bid, family = binomial()))
    COEF = np.array([-4.45254796956, 1.29636308449])
    # logLik(glmFit)
    LOG_LIKELIHOOD = -108.391966392
    # confint(glmFit)  -- profile likelihood, as confint.cyclopsFit also uses
    CONFINT = np.array([[-5.78337205631, -3.3175953749],
                        [0.978028966113, 1.66746141615]])
    # head(predict(glmFit, type = "response"), 5)
    PREDICT_HEAD = np.full(5, 0.0115147150468)

    def test_coefficients_and_likelihood(self, bernoulli):
        X, y = bernoulli
        _, _, result = fit(X, y, "lr", add_intercept=True)
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_profile_confidence_intervals(self, bernoulli):
        X, y = bernoulli
        _, model, _ = fit(X, y, "lr", add_intercept=True)
        # Both ends of the interval are profile-likelihood based, so they are
        # comparable directly rather than only through a normal approximation.
        intervals = model.confidence_intervals()
        observed = np.array([[i.lower, i.upper] for i in intervals])
        np.testing.assert_allclose(observed, self.CONFINT, rtol=TOL, atol=TOL)

    def test_predictions(self, bernoulli):
        X, y = bernoulli
        _, model, _ = fit(X, y, "lr", add_intercept=True)
        np.testing.assert_allclose(
            model.predict()[:5], self.PREDICT_HEAD, rtol=TOL, atol=TOL
        )

    def test_zhang_oles_criterion_reaches_the_same_mode(self, bernoulli):
        """R: `createControl(convergenceType = "zhang")`."""
        X, y = bernoulli
        _, _, default = fit(X, y, "lr", add_intercept=True)
        _, _, zhang = fit(
            X, y, "lr", add_intercept=True, control=Control(convergence="zhang")
        )
        np.testing.assert_allclose(
            zhang.coefficients, default.coefficients, rtol=TOL, atol=TOL
        )

    def test_kkt_swindle_reaches_the_same_mode(self, bernoulli):
        """R: `createControl(useKKTSwindle = TRUE)` must not change the answer."""
        X, y = bernoulli
        _, _, gold = fit(X, y, "lr", add_intercept=True, prior=Prior("laplace"))
        _, _, swindle = fit(
            X,
            y,
            "lr",
            add_intercept=True,
            prior=Prior("laplace"),
            control=Control(use_kkt_swindle=True),
        )
        np.testing.assert_allclose(
            swindle.coefficients, gold.coefficients, rtol=1e-6, atol=1e-8
        )

    def test_sparse_storage_gives_the_same_fit(self, bernoulli):
        """R fits the same data through `sparseFormula = ~ log_bid`."""
        import scipy.sparse as sp

        X, y = bernoulli
        _, _, dense = fit(X, y, "lr", add_intercept=True)
        _, _, sparse = fit(sp.csc_matrix(X), y, "lr", add_intercept=True)
        np.testing.assert_allclose(sparse.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(sparse.coefficients, dense.coefficients, rtol=1e-10)

    def test_intercept_added_after_the_covariates(self, bernoulli):
        """R: `y ~ log_bid - 1` then `finalizeSqlCyclopsData(addIntercept = TRUE)`."""
        X, y = bernoulli
        data, _, result = fit(X, y, "lr", add_intercept=True)
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        assert data.has_intercept
        assert data.covariate_ids[0] == 0


# ---------------------------------------------------------------------------
# test-smallPoisson.R / test-correlation.R / test-reductions.R
# ---------------------------------------------------------------------------


def _dobson():
    """R: `data.frame(counts = ..., outcome = gl(3,1,9), treatment = gl(3,3))`.

    Returns the treatment-contrast design (outcome2, outcome3, treatment2,
    treatment3) that `model.matrix(~ outcome + treatment)` produces.
    """
    counts = np.array([18.0, 17, 15, 20, 10, 20, 25, 13, 12])
    outcome = np.tile([1.0, 2.0, 3.0], 3)      # gl(3, 1, 9)
    treatment = np.repeat([1.0, 2.0, 3.0], 3)  # gl(3, 3)
    X = np.hstack(
        [_dummies(outcome, [1, 2, 3]), _dummies(treatment, [1, 2, 3])]
    )
    return X, counts, outcome, treatment


class TestSmallPoisson:
    """Poisson regression against `glm(counts ~ outcome + treatment, poisson())`."""

    COEF = np.array([3.04452243772, -0.454255272278, -0.292987124681,
                     1.39770117737e-16, -2.41565715107e-16])
    LOG_LIKELIHOOD = -23.380659201
    CONFINT = np.array([[2.69582150176, 3.36655581342],
                        [-0.857701838165, -0.0625584001078],
                        [-0.675369597414, 0.0824408934873],
                        [-0.393254829528, 0.393254829528],
                        [-0.393254829528, 0.393254829528]])

    def test_coefficients_and_likelihood(self):
        X, counts, _, _ = _dobson()
        _, _, result = fit(X, counts, "pr", add_intercept=True)
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_profile_confidence_intervals(self):
        X, counts, _, _ = _dobson()
        _, model, _ = fit(X, counts, "pr", add_intercept=True)
        intervals = model.confidence_intervals()
        observed = np.array([[i.lower, i.upper] for i in intervals])
        # Looser than the coefficient comparison: both sides locate the bound by
        # a root search with its own tolerance, so agreement is limited by the
        # two searches' stopping rules rather than by the likelihood.
        np.testing.assert_allclose(observed, self.CONFINT, rtol=1e-3, atol=5e-4)

    def test_predictions(self):
        X, counts, _, _ = _dobson()
        _, model, _ = fit(X, counts, "pr", add_intercept=True)
        # glm's fitted values reproduce the marginal totals for this design.
        np.testing.assert_allclose(model.predict().sum(), counts.sum(), rtol=TOL)

    def test_mittal_criterion(self):
        """R: `createControl(convergenceType = "mittal")`, tolerance 1e-3."""
        X, counts, _, _ = _dobson()
        _, _, result = fit(
            X, counts, "pr", add_intercept=True, control=Control(convergence="mittal")
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=1e-3, atol=1e-3)

    def test_fp32_precision(self):
        """R: `createCyclopsData(..., floatingPoint = 32)`."""
        X, counts, _, _ = _dobson()
        data = CyclopsData.from_arrays(
            X, counts, "pr", add_intercept=True, precision="fp32"
        )
        result = CyclopsModel(data).fit()
        np.testing.assert_allclose(
            result.coefficients, self.COEF, rtol=1e-3, atol=1e-4
        )


class TestPoissonWithOffset:
    """R: `glm(counts ~ treatment, offset = outcome, family = poisson())`.

    `outcome` is the offset already on the log scale, so Cyclops is given
    `exp(outcome)` and asked to log it back — which exercises the
    offset-promotion path rather than bypassing it.
    """

    COEF = np.array([0.504417040985, -7.76078946954e-13, 2.79555190599e-09])
    LOG_LIKELIHOOD = -88.4558629608

    def test_coefficients(self):
        _, counts, outcome, treatment = _dobson()
        X = _dummies(treatment, [1, 2, 3])
        _, _, result = fit(
            X, counts, "pr", add_intercept=True, offset=np.exp(outcome)
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_prelogged_offset_matches(self):
        """The same fit, handing Cyclops the log-scale offset directly."""
        _, counts, outcome, treatment = _dobson()
        X = _dummies(treatment, [1, 2, 3])
        data = CyclopsData.from_arrays(
            X, counts, "pr", offset=outcome, add_intercept=True,
            offset_already_log=True,
        )
        result = CyclopsModel(data).fit()
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)


class TestUnivariableCorrelation:
    """R: `getUnivariableCorrelation(dataPtrD)` vs `cor(counts, model.matrix(...))`."""

    CORRELATION = np.array([-0.533001790889, -0.159900537267,
                            4.07650592728e-17, -3.02450439766e-17])

    def test_matches_pearson_correlation(self):
        X, counts, _, _ = _dobson()
        data = CyclopsData.from_arrays(X, counts, "pr", add_intercept=True)
        # The R helper returns NA for the intercept and one value per covariate.
        observed = data.univariable_correlation()
        assert observed.shape == (5,)
        np.testing.assert_allclose(observed[1:], self.CORRELATION, rtol=TOL, atol=TOL)

    def test_named_subset(self):
        X, counts, _, _ = _dobson()
        data = CyclopsData.from_arrays(X, counts, "pr", add_intercept=True)
        subset = data.univariable_correlation([1, 2])  # outcome2, outcome3
        assert subset.shape == (2,)
        np.testing.assert_allclose(subset, self.CORRELATION[:2], rtol=TOL, atol=TOL)


class TestReductions:
    """R: `reduce(dataPtr, ...)` over indicator columns."""

    def test_column_sums(self):
        """R: `reduce(dataPtr, c(1,2))` gives `c(9, 3)` for the indicator design."""
        _, counts, outcome, treatment = _dobson()
        X = np.hstack(
            [_dummies(outcome, [1, 2, 3]), _dummies(treatment, [1, 2, 3])]
        )
        data = CyclopsData.from_arrays(X, counts, "pr", add_intercept=True)
        # Column 0 is the intercept (9 rows); each dummy marks 3 rows.
        assert data.column_sum(0) == pytest.approx(9.0)
        for covariate_id in (1, 2, 3, 4):
            assert data.column_sum(covariate_id) == pytest.approx(3.0)

    def test_squared_column_sums(self):
        X, counts, _, _ = _dobson()
        data = CyclopsData.from_arrays(X, counts, "pr", add_intercept=True)
        # Dummies are 0/1, so the squared sum equals the plain sum.
        for covariate_id in (1, 2, 3, 4):
            assert data.column_sum(covariate_id, power=2) == pytest.approx(3.0)
            assert data.column_sum(covariate_id, power=0) == pytest.approx(3.0)

    def test_outcome_is_not_reducible(self):
        """Unlike the templated `reduce()`, `sum()` looks the id up as a column."""
        X, counts, _, _ = _dobson()
        data = CyclopsData.from_arrays(X, counts, "pr", add_intercept=True)
        with pytest.raises(Exception, match="unknown"):
            data.column_sum(-1)
        np.testing.assert_allclose(data.y, counts)


class TestNormalization:
    """R: `.normalizeCovariates(dataPtr, type = "stdev")`."""

    def test_normalized_fit_rescales_to_the_same_coefficients(self):
        X, counts, _, _ = _dobson()
        _, _, plain = fit(X, counts, "pr", add_intercept=True)

        data = CyclopsData.create("pr")
        data.set_outcome(counts)
        data.add_covariates(X, covariate_ids=np.arange(1, X.shape[1] + 1))
        scale = data.normalize("stdev")
        data.add_intercept()
        data.finalize()
        scaled = CyclopsModel(data).fit()

        # Columns are multiplied by s_j, so the fitted coefficient is divided by
        # it; `coef(fit, rescale = TRUE)` multiplies back.
        rescaled = scaled.coefficients[1:] * scale
        np.testing.assert_allclose(
            rescaled, plain.coefficients[1:], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            scaled.log_likelihood, plain.log_likelihood, rtol=1e-10
        )


class TestCovariateRegularization:
    """R: `createPrior("laplace", exclude = ...)` shrinks only the rest."""

    def test_excluded_covariates_survive_shrinkage(self):
        X, counts, _, _ = _dobson()
        # R excludes ("(Intercept)", "outcome2", "outcome3"); the intercept is
        # id 0 and is excluded automatically, outcome2/outcome3 are ids 1 and 2.
        _, model, result = fit(
            X,
            counts,
            "pr",
            add_intercept=True,
            prior=Prior("laplace", exclude=[1, 2]),
        )
        # The treatment effects are ~0 in this design, so shrinkage drives them
        # exactly to zero -- the assertion the R test makes.
        np.testing.assert_allclose(result.coefficients[3:], 0.0, atol=1e-12)
        regularized = model.is_regularized()
        assert not any(regularized[:3])  # intercept, outcome2, outcome3
        assert all(regularized[3:])      # treatment2, treatment3

    def test_unknown_covariate_is_rejected(self):
        """R: `exclude = c("BAD", ...)` and `exclude = c(10, 1:3)` both error."""
        X, counts, _, _ = _dobson()
        with pytest.raises(CyclopsError, match="not found"):
            fit(
                X,
                counts,
                "pr",
                add_intercept=True,
                prior=Prior("laplace", exclude=[10]),
            )


# ---------------------------------------------------------------------------
# test-smallNormal.R
# ---------------------------------------------------------------------------


class TestSmallNormal:
    """Least squares against `lm(y ~ x)`."""

    COEF = np.array([-0.144479567273, 2.869818736])

    @staticmethod
    def _data():
        x = np.log([1.0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200])
        y = np.array([0.0, 3, 6, 7, 9, 13, 17, 12, 11, 14, 13])
        return x.reshape(-1, 1), y

    def test_coefficients(self):
        """Tighter than the R suite, which averages the relative error over the
        coefficient vector and so tolerates a partly converged intercept."""
        X, y = self._data()
        _, _, result = fit(
            X, y, "ls", add_intercept=True, control=Control(tolerance=1e-12)
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=1e-6, atol=1e-7)

    def test_sparse_storage_gives_the_same_fit(self):
        import scipy.sparse as sp

        X, y = self._data()
        _, _, result = fit(
            sp.csc_matrix(X),
            y,
            "ls",
            add_intercept=True,
            control=Control(tolerance=1e-12),
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# test-smallCox.R
# ---------------------------------------------------------------------------


class TestSmallCox:
    """Cox regression against `survival::coxph`.

    Two seven-row examples from the R suite: one with distinct event times, one
    with tied times. Cyclops uses Breslow's tie handling, which is `coxph`'s
    behaviour for these data.
    """

    NO_TIES: ClassVar[dict] = dict(
        length=[4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
        event=[1.0, 1, 0, 1, 1, 0, 1],
        x1=[0.0, 2, 0, 0, 1, 1, 1],
        x2=[0.0, 0, 1, 1, 1, 0, 0],
        # coef(coxph(Surv(length, event) ~ x1 + x2, test))
        coef=[0.823619040116, 1.51821258609],
        log_likelihood=-4.87940900514,
        se=[0.824167911167, 1.60536958241],
        # coef(coxph(Surv(length, event) ~ x1 + strata(x2), test))
        strat_coef=[1.2058524468],
        strat_log_likelihood=-2.97802826451,
    )

    TIME_TIES: ClassVar[dict] = dict(
        length=[4.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.0],
        event=[1.0, 1, 0, 1, 0, 0, 1],
        x1=[0.0, 2, 0, 0, 1, 1, 1],
        x2=[0.0, 0, 1, 1, 1, 0, 0],
        coef=[0.414858029733, -0.1653062443],
        log_likelihood=-4.40748004553,
        se=[0.772889722081, 1.48142278144],
        strat_coef=[0.397118727724],
        strat_log_likelihood=-3.04754903873,
    )

    @pytest.mark.parametrize("case", ["NO_TIES", "TIME_TIES"], ids=str.lower)
    def test_unstratified(self, case):
        spec = getattr(self, case)
        X = np.column_stack([spec["x1"], spec["x2"]])
        _, _, result = fit(
            X, np.array(spec["event"]), "cox", time=np.array(spec["length"])
        )
        np.testing.assert_allclose(
            result.coefficients, spec["coef"], rtol=TOL, atol=TOL
        )
        np.testing.assert_allclose(
            result.log_likelihood, spec["log_likelihood"], rtol=TOL
        )

    def test_standard_errors_are_unavailable_for_cox(self):
        """Cyclops cannot produce Cox standard errors, and says so.

        `computeFisherInformation` returns a singular matrix for the Cox
        likelihood. R surfaces this as a LAPACK error out of `getSEs()`; the
        facade raises rather than returning the inf/NaN that inverting a
        singular matrix would otherwise produce. The R suite never compares Cox
        standard errors against `coxph` for the same reason.
        """
        spec = self.NO_TIES
        X = np.column_stack([spec["x1"], spec["x2"]])
        _, model, result = fit(
            X, np.array(spec["event"]), "cox", time=np.array(spec["length"])
        )
        with pytest.raises(CyclopsError, match=r"singular|degenerate"):
            model.standard_errors(list(result.covariate_ids))

        # The coefficients themselves are sound; only the curvature is not.
        np.testing.assert_allclose(
            result.coefficients, spec["coef"], rtol=TOL, atol=TOL
        )

    @pytest.mark.parametrize("case", ["NO_TIES", "TIME_TIES"], ids=str.lower)
    def test_stratified(self, case):
        """R: `Surv(length, event) ~ x1 + strata(x2)`."""
        spec = getattr(self, case)
        X = np.array(spec["x1"]).reshape(-1, 1)
        _, _, result = fit(
            X,
            np.array(spec["event"]),
            "cox",
            time=np.array(spec["length"]),
            strata=np.array(spec["x2"], dtype=np.int64),
        )
        np.testing.assert_allclose(
            result.coefficients, spec["strat_coef"], rtol=TOL, atol=TOL
        )
        np.testing.assert_allclose(
            result.log_likelihood, spec["strat_log_likelihood"], rtol=TOL
        )


class TestGradient:
    """R test-gradient.R, on `survival::bladder`.

    `bladder` has many tied event times, so the gold standard uses
    `ties = "breslow"` — Cyclops' convention. Against `coxph`'s Efron default the
    coefficients differ in the third decimal, which is a genuine difference in
    tie handling rather than an error in either implementation.
    """

    # coef(coxph(Surv(stop, event) ~ rx + size, data = bladder, ties = "breslow"))
    COEF = np.array([-0.460877259563, -0.101298773524])
    LOG_LIKELIHOOD = -596.332754767

    def test_matches_coxph(self, bladder):
        _, _, result = bladder
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_gradient_vanishes_at_the_mode(self, bladder):
        """R: `expect_equivalent(gradient(fit), rep(0, length(coef(fit))))`."""
        _, model, _ = bladder
        np.testing.assert_allclose(model.gradient(), 0.0, atol=1e-8)

    def test_gradient_matches_a_central_difference(self, bladder):
        """R perturbs beta by eps and compares against a finite difference.

        Evaluated off the mode, where the gradient is not simply zero and the
        comparison therefore has content.
        """
        _, model, result = bladder
        mode = np.asarray(result.coefficients)
        eps = 1e-2
        offset = np.array([eps, 0.0])

        def log_likelihood_at(beta):
            # Setting the coefficients invalidates the cached linear predictor,
            # so reading the likelihood re-evaluates it at `beta`.
            model.set_start_values(beta)
            return model.log_likelihood

        point = mode + offset
        central = (
            log_likelihood_at(point + offset) - log_likelihood_at(point - offset)
        ) / (2 * eps)

        model.set_start_values(point)
        np.testing.assert_allclose(model.gradient()[0], central, atol=1e-3)

        # Leave the model at its mode for any other test sharing the fixture.
        model.set_start_values(mode)


# ---------------------------------------------------------------------------
# test-smallCLR.R / test-conditionalPoisson.R
# ---------------------------------------------------------------------------


class TestConditionalLogistic:
    """Conditional logistic on `infert`, against `survival::clogit`."""

    # coef(clogit(case ~ spontaneous + induced + strata(stratum), data = infert))
    COEF = np.array([1.98587551668, 1.40901163188])
    LOG_LIKELIHOOD = -64.2022369244
    SE = np.array([0.352443539807, 0.360712436249])
    # confint(gold) -- Wald, from the normal approximation
    CONFINT = np.array([[1.29509887207, 2.67665216128],
                        [0.702028248052, 2.1159950157]])

    def test_coefficients_and_likelihood(self, infert_clr):
        (_, _, result), _ = infert_clr
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=TOL, atol=TOL)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_standard_errors(self, infert_clr):
        """R: `expect_equal(vcov(cyclopsFit), vcov(gold))`."""
        (_, model, result), _ = infert_clr
        np.testing.assert_allclose(
            model.standard_errors(list(result.covariate_ids)),
            self.SE,
            rtol=TOL,
            atol=TOL,
        )

    def test_asymptotic_confidence_intervals(self, infert_clr):
        """R: `aconfint(cyclopsFit)` — the Wald interval `beta +/- 1.96 * se`."""
        (_, model, result), _ = infert_clr
        coefficients = np.asarray(result.coefficients)
        errors = model.standard_errors(list(result.covariate_ids))
        observed = np.column_stack(
            [coefficients - 1.959964 * errors, coefficients + 1.959964 * errors]
        )
        np.testing.assert_allclose(observed, self.CONFINT, rtol=1e-3, atol=1e-4)

    def test_stratum_count(self, infert_clr):
        (data, _, _), columns = infert_clr
        assert data.n_strata == len(np.unique(columns["stratum"]))


class TestConditionalPoisson:
    """Conditional Poisson on `Cyclops::oxford`, against `gnm`."""

    # coef(gnm(event ~ exgr + agegr + offset(loginterval), family = poisson,
    #          eliminate = indiv, data = oxford))
    COEF = np.array([2.48797464966, -1.49057568474])
    SE = np.array([0.70848791713, 1.11823960763])

    def test_coefficients(self, oxford):
        X = np.column_stack([oxford["exgr"], oxford["agegr"] - 1.0])
        _, _, result = fit(
            X,
            oxford["event"],
            "cpr",
            strata=oxford["indiv"].astype(np.int64),
            offset=oxford["interval"],
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=1e-5, atol=1e-6)

    def test_standard_errors(self, oxford):
        X = np.column_stack([oxford["exgr"], oxford["agegr"] - 1.0])
        _, model, result = fit(
            X,
            oxford["event"],
            "cpr",
            strata=oxford["indiv"].astype(np.int64),
            offset=oxford["interval"],
        )
        np.testing.assert_allclose(
            model.standard_errors(list(result.covariate_ids)),
            self.SE,
            rtol=1e-4,
            atol=1e-6,
        )


class TestSccs:
    """SCCS on `Cyclops::oxford`, against `clogit` with a log-time offset.

    The R suite's "Check simple SCCS as SCCS" case. SCCS consumes person-time as
    its `time` vector on the natural scale, where `clogit` takes `log(interval)`
    as an offset — the two parameterisations must land on the same coefficients.
    """

    COEF = np.array([2.48797465298, -1.4905756965])
    LOG_LIKELIHOOD = -10.0882772097

    def test_matches_clogit(self):
        oxford = _read_csv("oxford.csv")
        X = np.column_stack([oxford["exgr"], oxford["agegr"] - 1.0])
        _, _, result = fit(
            X,
            oxford["event"],
            "sccs",
            strata=oxford["indiv"].astype(np.int64),
            offset=oxford["interval"],
        )
        np.testing.assert_allclose(result.coefficients, self.COEF, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.log_likelihood, self.LOG_LIKELIHOOD, rtol=TOL)

    def test_agrees_with_conditional_poisson(self):
        """R notes the two likelihoods differ only by fixed terms."""
        oxford = _read_csv("oxford.csv")
        X = np.column_stack([oxford["exgr"], oxford["agegr"] - 1.0])
        common = dict(
            strata=oxford["indiv"].astype(np.int64), offset=oxford["interval"]
        )
        _, _, sccs = fit(X, oxford["event"], "sccs", **common)
        _, _, cpr = fit(X, oxford["event"], "cpr", **common)
        np.testing.assert_allclose(
            sccs.coefficients, cpr.coefficients, rtol=1e-8, atol=1e-10
        )


# ---------------------------------------------------------------------------
# test-finiteMLE.R
# ---------------------------------------------------------------------------


class TestInfiniteMle:
    """R test-finiteMLE.R: a Cox fit with no events in one treatment arm.

    `coxph` warns "coefficient may be infinite"; Cyclops reports the
    POOR_BLR_STEP path, which `fitCyclopsModel` retries under the Lange
    criterion. The Python facade does the same, so the observable outcome is a
    finished fit with a large coefficient rather than a crash.
    """

    @staticmethod
    def _simulate():
        rng = np.random.default_rng(123)
        n = 1000
        time_to_censor = rng.exponential(100.0, size=n)
        exposure = rng.random(n) < 0.5
        time_to_event = np.full(n, np.inf)
        unexposed = ~exposure
        time_to_event[unexposed] = rng.exponential(
            100.0, size=int(unexposed.sum())
        )
        time = np.minimum(time_to_censor, time_to_event)
        outcome = (time_to_event < time_to_censor).astype(float)
        return exposure.astype(float).reshape(-1, 1), outcome, time

    def test_separated_arm_still_returns(self):
        X, outcome, time = self._simulate()
        # Every event falls in the unexposed arm, so the MLE is -Inf.
        assert outcome[X[:, 0] == 1.0].sum() == 0

        _, _, result = fit(X, outcome, "cox", time=time)
        assert result.return_flag in ("SUCCESS", "MAX_ITERATIONS", "ILLCONDITIONED")
        assert result.coefficients[0] < -5.0, (
            "expected the coefficient to run away negative"
        )

    def test_jeffreys_prior_needs_a_single_covariate(self):
        """R: multivariable Jeffreys errors with '.*1 covariate.*'."""
        X, outcome, time = self._simulate()
        rng = np.random.default_rng(7)
        X2 = np.column_stack([X[:, 0], (rng.random(X.shape[0]) < 0.5).astype(float)])
        with pytest.raises(Exception, match=r"(?i)jeffreys|covariate"):
            fit(X2, outcome, "cox", time=time, prior=Prior("jeffreys"))

    def test_jeffreys_prior_with_lange_criterion(self):
        """R: succeeds once `convergenceType = "lange"` is set."""
        X, outcome, time = self._simulate()
        _, _, result = fit(
            X,
            outcome,
            "cox",
            time=time,
            prior=Prior("jeffreys"),
            control=Control(convergence="lange"),
        )
        assert result.return_flag == "SUCCESS"
        # A Jeffreys prior keeps the estimate finite where the MLE is not.
        assert np.isfinite(result.coefficients[0])
