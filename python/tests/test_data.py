"""Tests for :mod:`cyclops.data` — layout, formats and the loader invariants."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cyclops import CyclopsError
from cyclops.data import INTERCEPT_ID, CyclopsData


def test_create_and_load_dense(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr")
    assert data.n_rows == small.n_samples
    assert data.n_covariates == small.n_features
    assert data.is_finalized
    assert not data.has_intercept
    np.testing.assert_array_equal(
        data.covariate_ids, np.arange(1, small.n_features + 1)
    )
    np.testing.assert_allclose(data.y, small.y)


def test_intercept_is_prepended_with_id_zero(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr", add_intercept=True)
    assert data.has_intercept
    assert data.n_covariates == small.n_features + 1
    assert data.covariate_ids[0] == INTERCEPT_ID
    assert data.intercept_label == INTERCEPT_ID


def test_column_formats_follow_content(small):
    """Cyclops picks its storage format from the values, and that is observable."""
    n = small.n_samples
    indicator = np.zeros((n, 1))
    indicator[::3] = 1.0
    valued = np.zeros((n, 1))
    valued[::3] = 2.5

    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(sp.csc_matrix(indicator), covariate_ids=[10])
    data.add_covariates(sp.csc_matrix(valued), covariate_ids=[11])
    data.add_dense_covariate(12, small.X[:, 0])
    data.finalize()

    assert data.covariate_formats == ["indicator", "sparse", "dense"]


def test_force_sparse_overrides_indicator(small):
    indicator = np.zeros((small.n_samples, 1))
    indicator[::3] = 1.0
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(sp.csc_matrix(indicator), covariate_ids=[1], force_sparse=True)
    data.finalize()
    assert data.covariate_formats == ["sparse"]


def test_all_zero_sparse_column_is_kept(small):
    """An empty column must survive as a column, not vanish or become an intercept."""
    matrix = sp.csc_matrix((small.n_samples, 3), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[1, 2] = 1.0  # column 1 stays empty
    data = CyclopsData.from_arrays(matrix.tocsc(), small.y, "lr")
    assert data.n_covariates == 3
    assert not data.has_intercept
    np.testing.assert_array_equal(data.covariate_ids, [1, 2, 3])


def test_strata_are_grouped_and_counted(small):
    data = CyclopsData.from_arrays(
        small.X, small.y, "clr", strata=small.strata
    )
    assert data.n_strata == len(np.unique(small.strata))
    # `strata` reports Cyclops' internal 0-based stratum index, which must be
    # non-decreasing for the conditional likelihood to group rows correctly.
    assert np.all(np.diff(data.strata) >= 0)


def test_unsorted_strata_are_permuted(small):
    rng = np.random.default_rng(0)
    shuffle = rng.permutation(small.n_samples)
    data = CyclopsData.from_arrays(
        small.X[shuffle], small.y[shuffle], "clr", strata=small.strata[shuffle]
    )
    assert data.row_order is not None
    np.testing.assert_array_equal(data.y, small.y[shuffle][data.row_order])
    assert np.all(np.diff(data.strata) >= 0)


def test_sorted_input_needs_no_permutation(small):
    data = CyclopsData.from_arrays(
        small.X, small.y, "clr", strata=small.strata
    )
    assert data.row_order is None


def test_cox_rows_sorted_by_descending_time(small):
    rng = np.random.default_rng(7)
    shuffle = rng.permutation(small.n_samples)
    data = CyclopsData.from_arrays(
        small.X[shuffle], small.y[shuffle], "cox", time=small.time[shuffle]
    )
    assert np.all(np.diff(data.time) <= 0)


def test_cox_defaults_to_a_single_stratum(small):
    data = CyclopsData.from_arrays(small.X, small.y, "cox", time=small.time)
    assert data.n_strata == 1


def test_conditional_model_requires_strata(small):
    with pytest.raises(ValueError, match="requires `strata`"):
        CyclopsData.from_arrays(small.X, small.y, "clr")


def test_cox_requires_time(small):
    with pytest.raises(ValueError, match="requires `time`"):
        CyclopsData.from_arrays(small.X, small.y, "cox")


def test_offset_is_promoted_and_log_transformed(small):
    data = CyclopsData.from_arrays(
        small.X, small.counts, "pr", offset=small.offset, add_intercept=True
    )
    assert data.has_offset
    # Offset first, then intercept, then the covariates.
    assert data.covariate_ids[0] == -1
    assert data.covariate_ids[1] == INTERCEPT_ID
    assert data.n_covariates == small.n_features + 2


def test_sccs_offset_becomes_the_time_vector(small):
    """SCCS reads the offset out of `time`, so no offset column appears."""
    data = CyclopsData.from_arrays(
        small.X, small.counts, "sccs", strata=small.strata, offset=small.offset
    )
    assert not data.has_offset
    assert data.n_covariates == small.n_features
    np.testing.assert_allclose(np.sort(data.time), np.sort(small.offset))


def test_sccs_rejects_both_time_and_offset(small):
    with pytest.raises(ValueError, match="one or the other"):
        CyclopsData.from_arrays(
            small.X,
            small.counts,
            "sccs",
            strata=small.strata,
            time=small.time,
            offset=small.offset,
        )


def test_sccs_rejects_prelogged_offset(small):
    with pytest.raises(ValueError, match="natural scale"):
        CyclopsData.from_arrays(
            small.X,
            small.counts,
            "sccs",
            strata=small.strata,
            offset=small.offset,
            offset_already_log=True,
        )


def test_reserved_covariate_ids_are_rejected(small):
    ids = np.arange(small.n_features)  # includes 0, the intercept id
    with pytest.raises(ValueError, match="reserved"):
        CyclopsData.from_arrays(
            small.X, small.y, "lr", covariate_ids=ids, add_intercept=True
        )


def test_custom_covariate_ids_are_preserved(small):
    ids = np.array([1001, 1002, 2001, 2002], dtype=np.int64)[: small.n_features]
    data = CyclopsData.from_arrays(small.X, small.y, "lr", covariate_ids=ids)
    np.testing.assert_array_equal(data.covariate_ids, ids)


def test_duplicate_covariate_ids_raise(small):
    ids = np.ones(small.n_features, dtype=np.int64)
    with pytest.raises(CyclopsError, match="already exists"):
        CyclopsData.from_arrays(small.X, small.y, "lr", covariate_ids=ids)


def test_shape_mismatch_raises(small):
    with pytest.raises(ValueError, match="rows"):
        CyclopsData.from_arrays(small.X[:-1], small.y, "lr")


def test_covariates_before_outcome_raise(small):
    data = CyclopsData.create("lr")
    with pytest.raises(CyclopsError, match="set_outcome"):
        data.add_dense_covariate(1, small.X[:, 0])


def test_loading_after_finalize_raises(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr")
    with pytest.raises(CyclopsError, match="already finalized"):
        data.add_dense_covariate(999, small.X[:, 0])


def test_row_index_out_of_range_raises(small):
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    with pytest.raises(CyclopsError, match="Row index out of range"):
        data.add_covariates_csc = None  # not the API under test
        data._handle.add_covariates_csc(
            [0, 1], [small.n_samples + 5], [1.0], [1], False
        )


def test_unknown_model_type_raises(small):
    with pytest.raises(ValueError, match="Unknown model type"):
        CyclopsData.create("not-a-model")


def test_streaming_load_matches_single_load(small):
    """Loading in column chunks must produce the same object as one call."""
    whole = CyclopsData.from_arrays(small.X, small.y, "lr")

    chunked = CyclopsData.create("lr")
    chunked.set_outcome(small.y)
    half = small.n_features // 2
    chunked.add_covariates(
        sp.csc_matrix(small.X[:, :half]), covariate_ids=np.arange(1, half + 1)
    )
    chunked.add_covariates(
        sp.csc_matrix(small.X[:, half:]),
        covariate_ids=np.arange(half + 1, small.n_features + 1),
    )
    chunked.finalize()

    np.testing.assert_array_equal(whole.covariate_ids, chunked.covariate_ids)
    assert whole.n_rows == chunked.n_rows
    assert whole.n_covariates == chunked.n_covariates


def test_univariable_correlation_matches_numpy(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr")
    expected = np.array(
        [np.corrcoef(small.X[:, j], small.y)[0, 1] for j in range(small.n_features)]
    )
    np.testing.assert_allclose(data.univariable_correlation(), expected, rtol=1e-10)


def test_column_sum_and_sum_by_stratum(small):
    data = CyclopsData.from_arrays(
        small.X, small.y, "clr", strata=small.strata
    )
    order = data.row_order if data.row_order is not None else np.arange(small.n_samples)
    column = small.X[order][:, 0]
    assert data.column_sum(1) == pytest.approx(column.sum())
    assert data.column_sum(1, power=2) == pytest.approx((column**2).sum())

    per_stratum = data.sum_by_stratum(1)
    assert per_stratum.shape == (data.n_strata,)
    assert per_stratum.sum() == pytest.approx(column.sum())


def test_make_dense_converts_named_columns(small):
    """`make_dense` addresses columns by covariate id, not by position."""
    sparse = np.zeros((small.n_samples, 3))
    sparse[::4, :] = 2.5

    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(sp.csc_matrix(sparse), covariate_ids=[10, 20, 30])
    assert data.covariate_formats == ["sparse"] * 3

    data.make_dense([20])
    assert data.covariate_formats == ["sparse", "dense", "sparse"]


def test_incremental_ids_continue_numbering(small):
    """Auto-assigned ids pick up after the columns already loaded."""
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(small.X[:, :2])
    data.add_covariates(small.X[:, 2:])
    np.testing.assert_array_equal(
        data.covariate_ids, np.arange(1, small.n_features + 1)
    )


def test_aux_vector_length_is_validated(small):
    with pytest.raises(ValueError, match="one element per row"):
        CyclopsData.from_arrays(
            small.X, small.y, "cox", time=small.time[:-1]
        )


def test_covariate_id_count_is_validated(small):
    with pytest.raises(ValueError, match="covariate_ids must have"):
        CyclopsData.from_arrays(small.X, small.y, "lr", covariate_ids=[1, 2])


def test_enum_members_pass_through_resolve():
    """The string layer is a convenience, not a requirement."""
    from cyclops import _cyclops
    from cyclops._enums import MODEL_KINDS, resolve

    assert resolve(MODEL_KINDS, "lr", "model type") is _cyclops.ModelKind.LOGISTIC
    assert (
        resolve(MODEL_KINDS, _cyclops.ModelKind.COX, "model type")
        is _cyclops.ModelKind.COX
    )
    with pytest.raises(TypeError, match="Invalid model type"):
        resolve(MODEL_KINDS, 42, "model type")


def test_normalize_returns_divisors(small):
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(small.X, covariate_ids=np.arange(1, small.n_features + 1))
    scale = data.normalize("stdev")
    data.finalize()
    assert scale.shape == (small.n_features,)
    assert np.all(scale > 0)
    np.testing.assert_allclose(data.scale, scale)


def test_repr_is_informative(small):
    data = CyclopsData.from_arrays(small.X, small.y, "lr")
    text = repr(data)
    assert "CyclopsData" in text and "n_rows=200" in text
