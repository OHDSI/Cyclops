"""Regression tests for pre-existing C++ segfaults in ModelData and cox_raw."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cyclops import Control, CyclopsData, CyclopsModel, Prior


@pytest.mark.parametrize("model_type", ["ls", "lr", "pr"])
def test_sum_by_stratum_unstratified_does_not_segfault(model_type: str):
    """Unstratified models leave `pid` empty; each row is its own stratum."""
    X = sp.csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 2.5],
                [1.0, -1.5],
                [0.0, 0.0],
                [1.0, 3.0],
            ]
        )
    )
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    data = CyclopsData.from_arrays(X, y, model_type, add_intercept=True)

    assert data.n_strata == 5
    np.testing.assert_array_equal(
        data.sum_by_stratum(1, power=1), [1.0, 0.0, 1.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        data.sum_by_stratum(2, power=1), [0.0, 2.5, -1.5, 0.0, 3.0]
    )
    np.testing.assert_allclose(
        data.sum_by_stratum(2, power=2), [0.0, 6.25, 2.25, 0.0, 9.0]
    )
    np.testing.assert_array_equal(
        data.sum_by_stratum(2, power=0), [0.0, 1.0, 1.0, 0.0, 1.0]
    )


@pytest.mark.parametrize("kind", ["median", "q95"])
@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_normalize_median_q95_empty_sparse_column(kind: str, precision: str):
    """Empty sparse columns must not dereference end() in median/quantile."""
    X = sp.csc_matrix(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [0.0, -4.0],
                [0.0, 0.0],
                [0.0, 6.0],
            ]
        )
    )
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    data = CyclopsData.create("lr", precision=precision)
    data.set_outcome(y)
    data.add_covariates(X, covariate_ids=[1, 2], force_sparse=True)
    scales = data.normalize(kind)
    data.add_intercept()
    data.finalize()

    assert scales.shape == (2,)
    assert np.isfinite(scales[1])
    res = CyclopsModel(data, prior=Prior("normal", variance=2.0)).fit()
    assert res.converged


def test_cox_raw_tied_event_times_and_zero_weights():
    """cox_raw must index denominators via hPid (not k) when N < K."""
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    event = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    time = np.array([4.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.0])

    raw_data = CyclopsData.from_arrays(X, event, "cox_raw", time=time)
    tied_data = CyclopsData.from_arrays(X, event, "cox", time=time)

    prior = Prior("normal", variance=2.0)
    ctrl = Control(convergence="kkt", tolerance=1e-12)
    raw_model = CyclopsModel(raw_data, prior=prior, control=ctrl)
    raw_res = raw_model.fit()
    tied_res = CyclopsModel(tied_data, prior=prior, control=ctrl).fit()

    assert raw_res.converged
    np.testing.assert_allclose(
        raw_res.coefficients, tied_res.coefficients, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        raw_res.log_likelihood, tied_res.log_likelihood, rtol=1e-10
    )
    _ = raw_model.predict()
    _ = raw_model.gradient()

    # Zero-weight rows (as in cross-validation folds) and all-zero weights edge case
    w = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    w_res = CyclopsModel(raw_data, prior=prior, control=ctrl).set_weights(w).fit()
    assert w_res.converged

    # All-zero weights must not read past weights[K]
    zero_model = CyclopsModel(raw_data, prior=prior, control=ctrl)
    zero_model.set_weights(np.zeros(len(event)))
    _ = zero_model.fit()
