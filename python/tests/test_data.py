"""Tests for :mod:`cyclops.data` — layout, formats and the loader invariants."""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from cyclops import CyclopsError, CyclopsModel
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
    # stored as R's convertToCyclopsData() stores it (not as a dense column of
    # ones), so the fitters that look for an intercept column, such as the
    # adaptive intercept, find it
    assert data.covariate_formats[0] == "intercept"


def test_making_the_intercept_dense_keeps_it(small):
    """An intercept column is dense already; converting it used to crash the process."""
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_dense_covariate(1, small.X[:, 0])
    data.add_intercept()
    data.make_dense([INTERCEPT_ID])
    assert data.covariate_formats == ["intercept", "dense"]


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


def test_sparse_batch_with_a_bad_id_is_rejected_whole(small):
    """Ids are checked before any column is loaded, so a rejected batch leaves the
    data unchanged."""
    X = sp.csc_matrix((small.X[:, :3] > 0).astype(np.float64))
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(X[:, :1], covariate_ids=[5])
    with pytest.raises(CyclopsError, match="Variable 7 already exists"):
        data.add_covariates(X, covariate_ids=[6, 7, 7])  # repeated within the batch
    with pytest.raises(CyclopsError, match="Variable 5 already exists"):
        data.add_covariates(X, covariate_ids=[6, 8, 5])  # already loaded
    assert data.n_covariates == 1
    data.add_covariates(X, covariate_ids=[6, 7, 8])
    np.testing.assert_array_equal(data.covariate_ids, [5, 6, 7, 8])


def test_malformed_sparse_batch_is_rejected_whole(small):
    """A decreasing indptr or a repeated row within a column is found before
    anything is loaded."""
    n = small.n_samples
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(sp.csc_matrix(np.eye(n, 1)), covariate_ids=[1])
    rows = np.array([0, 1, 2], dtype=np.int64)
    ones = np.ones(3)
    with pytest.raises(CyclopsError, match="non-decreasing"):
        data._handle.add_covariates_csc(
            np.array([0, 2, 1, 3], dtype=np.int64), rows, ones,
            np.array([2, 3, 4]), False,
        )
    # row 1 twice in column 2
    repeated = sp.csc_matrix(
        (ones, np.array([0, 1, 1]), np.array([0, 1, 3])), shape=(n, 2)
    )
    with pytest.raises(CyclopsError, match="Repeated row-column entry"):
        data.add_covariates(repeated, covariate_ids=[2, 3])
    np.testing.assert_array_equal(data.covariate_ids, [1])


def _load_seconds(n_rows, y, p):
    rng = np.random.default_rng(3)
    rows = rng.integers(0, n_rows, size=2 * p)
    X = sp.csc_matrix(
        (np.ones(2 * p), (rows, np.repeat(np.arange(p), 2))), shape=(n_rows, p)
    )
    X.data[:] = 1.0  # duplicates were summed by the conversion
    ids = np.arange(1000, 1000 + p, dtype=np.int64)
    data = CyclopsData.create("lr")
    data.set_outcome(y)
    start = time.perf_counter()
    data.add_covariates(X, covariate_ids=ids)
    seconds = time.perf_counter() - start
    np.testing.assert_array_equal(data.covariate_ids, ids)
    return seconds


def test_wide_sparse_batch_loads_in_linear_time(small):
    """FeatureExtraction-sized batches keep their ids, and loading time grows
    linearly with the columns.

    Checking each new id against every existing column made loading quadratic:
    1.6 s for 50,000 columns and 40 s for 200,000, against 0.04 s and 0.11 s now
    (one laptop core). Four times the columns should take about four times as
    long; quadratic loading takes 16 times (26 measured). The ratio does not
    depend on the speed of the machine.
    """
    _load_seconds(small.n_samples, small.y, 1_000)  # warm up
    small_batch = _load_seconds(small.n_samples, small.y, 50_000)
    large_batch = _load_seconds(small.n_samples, small.y, 200_000)
    ratio = large_batch / max(small_batch, 1e-3)
    assert ratio < 10.0, (
        f"4x the columns took {ratio:.1f}x as long "
        f"({small_batch:.2f} s, {large_batch:.2f} s)"
    )


def test_row_ids_label_rows_without_moving_covariates():
    """Sparse covariates are read by position whatever the row ids (they were read
    as ids: a permutation of 0..n-1 silently moved them to other rows, and other ids
    failed)."""
    rng = np.random.default_rng(2)
    n, p = 300, 6
    X = sp.random(n, p, density=0.3, format="csc", random_state=rng)
    X.data[:] = 1.0
    eta = X @ np.linspace(-1, 1, p) - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)

    def coefficients(row_ids):
        data = CyclopsData.create("lr")
        data.set_outcome(y, row_ids=row_ids)
        data.add_covariates(X)
        data.add_intercept()
        data.finalize()
        return CyclopsModel(data).fit().coefficients

    plain = coefficients(None)
    np.testing.assert_array_equal(coefficients(np.arange(n)[::-1].copy()), plain)
    np.testing.assert_array_equal(coefficients(np.arange(1001, 1001 + n)), plain)
    # a row id at the bottom of the range is an id like any other
    lowest = np.iinfo(np.int64).min
    np.testing.assert_array_equal(coefficients(np.arange(lowest, lowest + n)), plain)


def test_duplicate_row_ids_raise(small):
    data = CyclopsData.create("lr")
    ids = np.arange(small.n_samples)
    ids[-1] = ids[0]
    with pytest.raises(CyclopsError, match="distinct"):
        data.set_outcome(small.y, row_ids=ids)


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


# -- buffer-protocol dtypes, indicator fast paths and CSR loading ------------


def test_csc_accepts_int32_and_int64_indices_and_various_value_dtypes():
    """Native int32/int64 indices and compact value dtypes load identically to fp64."""
    rng = np.random.default_rng(11)
    n, p = 150, 8
    dense = np.zeros((n, p), dtype=np.float64)
    for j in range(4):
        dense[rng.choice(n, size=35, replace=False), j] = 1.0
    for j in range(4, p):
        dense[rng.choice(n, size=35, replace=False), j] = rng.integers(1, 5, size=35)
    y = rng.binomial(1, 0.35, size=n).astype(np.float64)

    ref_csc = sp.csc_matrix(dense)
    ref_data = CyclopsData.from_arrays(ref_csc, y, "lr", add_intercept=True)
    ref_fit = CyclopsModel(ref_data).fit()

    val_dtypes = (np.float64, np.float32, np.int8, np.uint8, np.int32, np.int64)
    for idx_dtype in (np.int32, np.int64):
        for val_dtype in val_dtypes:
            indptr = ref_csc.indptr.astype(idx_dtype)
            indices = ref_csc.indices.astype(idx_dtype)
            values = ref_csc.data.astype(val_dtype)
            ids = np.arange(1, p + 1, dtype=idx_dtype)

            data = CyclopsData.create("lr")
            data.set_outcome(y.astype(np.float32) if val_dtype is np.float32 else y)
            data.add_covariates_csc(indptr, indices, values, covariate_ids=ids)
            data.add_intercept()
            data.finalize()

            assert data.covariate_formats == ref_data.covariate_formats
            fit = CyclopsModel(data).fit()
            np.testing.assert_allclose(
                fit.coefficients, ref_fit.coefficients, rtol=1e-12
            )


def test_bool_sparse_matrix_and_explicit_zeros():
    """Boolean sparse arrays drop explicit False entries and store indicator columns."""
    n, p = 40, 4
    indptr = np.array([0, 3, 5, 5, 7], dtype=np.int32)
    indices = np.array([1, 5, 10, 2, 8, 0, 9], dtype=np.int32)
    # Explicit False at position 1 (row 5 in col 0) and position 3 (row 2 in col 1)
    values = np.array([True, False, True, False, True, True, True], dtype=np.bool_)
    csc = sp.csc_array((values, indices, indptr), shape=(n, p))
    csr = csc.tocsr()
    y = np.zeros(n, dtype=np.float64)
    y[::2] = 1.0

    for mat in (csc, csr):
        data = CyclopsData.from_arrays(mat, y, "lr")
        assert data.covariate_formats == ["indicator"] * p
        assert data.column_sum(1) == 2.0
        assert data.column_sum(2) == 1.0
        assert data.column_sum(3) == 0.0
        assert data.column_sum(4) == 2.0


def test_values_none_indicator_fast_path_csc_and_csr():
    """values=None loads pure indicator columns without materializing a 1.0 array."""
    rng = np.random.default_rng(12)
    n, p = 120, 10
    X = sp.random(n, p, density=0.25, format="csc", random_state=rng)
    X.data[:] = 1.0
    X_csr = X.tocsr()
    y = rng.binomial(1, 0.4, size=n).astype(np.float64)

    ref = CyclopsData.from_arrays(X, y, "lr", add_intercept=True)
    ref_fit = CyclopsModel(ref).fit()

    csc_none = CyclopsData.create("lr")
    csc_none.set_outcome(y)
    csc_none.add_covariates_csc(X.indptr, X.indices, None)
    csc_none.add_intercept()
    csc_none.finalize()
    assert csc_none.covariate_formats == ref.covariate_formats
    np.testing.assert_allclose(
        CyclopsModel(csc_none).fit().coefficients, ref_fit.coefficients, rtol=1e-13
    )

    csr_none = CyclopsData.create("lr")
    csr_none.set_outcome(y)
    csr_none.add_covariates_csr(X_csr.indptr, X_csr.indices, None, n_covariates=p)
    csr_none.add_intercept()
    csr_none.finalize()
    assert csr_none.covariate_formats == ref.covariate_formats
    np.testing.assert_allclose(
        CyclopsModel(csr_none).fit().coefficients, ref_fit.coefficients, rtol=1e-13
    )

    # force_sparse=True with values=None produces SPARSE columns of 1.0s
    for loader in ("csc", "csr"):
        d = CyclopsData.create("lr")
        d.set_outcome(y)
        if loader == "csc":
            d.add_covariates_csc(X.indptr, X.indices, None, force_sparse=True)
        else:
            d.add_covariates_csr(
                X_csr.indptr, X_csr.indices, None, n_covariates=p, force_sparse=True
            )
        d.finalize()
        assert d.covariate_formats == ["sparse"] * p
        for j in range(1, p + 1):
            assert d.column_sum(j) == ref.column_sum(j)


def test_is_indicator_mask_skips_values_for_binary_columns():
    """Columns flagged in is_indicator skip scanning values in both CSC and CSR."""
    n = 20
    # 3 columns: col 0 binary (flagged), col 1 continuous, col 2 binary (flagged).
    # Put sentinel values in cols 0 and 2 to prove values are not scanned when flagged.
    indptr_csc = np.array([0, 3, 6, 8], dtype=np.int32)
    rows_csc = np.array([0, 4, 9, 1, 5, 8, 2, 12], dtype=np.int32)
    vals_csc = np.array(
        [99.0, np.nan, 0.0, 2.5, 0.0, -1.5, 42.0, 7.0], dtype=np.float64
    )
    mask = np.array([True, False, True], dtype=np.bool_)
    y = np.zeros(n, dtype=np.float64)
    y[::3] = 1.0

    data_csc = CyclopsData.create("lr")
    data_csc.set_outcome(y)
    data_csc.add_covariates_csc(
        indptr_csc, rows_csc, vals_csc, covariate_ids=[10, 20, 30], is_indicator=mask
    )
    data_csc.finalize()
    assert data_csc.covariate_formats == ["indicator", "sparse", "indicator"]
    assert data_csc.column_sum(10) == 3.0
    # 2.5 + (-1.5); explicit 0 dropped
    assert data_csc.column_sum(20) == pytest.approx(1.0)
    assert data_csc.column_sum(30) == 2.0

    # Same test in CSR layout
    csr = sp.csc_matrix((vals_csc, rows_csc, indptr_csc), shape=(n, 3)).tocsr()
    data_csr = CyclopsData.create("lr")
    data_csr.set_outcome(y)
    data_csr.add_covariates(csr, covariate_ids=[10, 20, 30], is_indicator=mask)
    data_csr.finalize()
    assert data_csr.covariate_formats == ["indicator", "sparse", "indicator"]
    assert data_csr.column_sum(10) == 3.0
    assert data_csr.column_sum(20) == pytest.approx(1.0)
    assert data_csr.column_sum(30) == 2.0

    # force_sparse=True overrides is_indicator and reads the actual values
    clean_vals = np.array([1.0, 1.0, 1.0, 2.5, 0.0, -1.5, 1.0, 1.0], dtype=np.float64)
    forced = CyclopsData.create("lr")
    forced.set_outcome(y)
    forced.add_covariates_csc(
        indptr_csc,
        rows_csc,
        clean_vals,
        covariate_ids=[10, 20, 30],
        force_sparse=True,
        is_indicator=mask,
    )
    forced.finalize()
    assert forced.covariate_formats == ["sparse", "sparse", "sparse"]


def test_csr_from_arrays_matches_csc_and_dense_for_cox_and_clr():
    """CSR inputs sort rows in C++ via row_order without transposition or copy."""
    rng = np.random.default_rng(19)
    n, p = 240, 12
    dense = np.zeros((n, p), dtype=np.float64)
    for j in range(8):
        dense[rng.choice(n, size=50, replace=False), j] = 1.0
    for j in range(8, p - 1):
        dense[rng.choice(n, size=60, replace=False), j] = rng.normal(size=60)
    # Leave last column all-zero to verify empty column handling across formats
    X_csr = sp.csr_matrix(dense)
    X_csc = sp.csc_matrix(dense)
    X_csr_arr = sp.csr_array(dense)

    y = rng.binomial(1, 0.3, size=n).astype(np.float64)
    time_vec = rng.exponential(scale=5.0, size=n)
    strata = rng.integers(0, 12, size=n, dtype=np.int32)

    # Cox with unsorted time and multiple strata
    d_dense = CyclopsData.from_arrays(dense, y, "cox", time=time_vec, strata=strata)
    d_csc = CyclopsData.from_arrays(X_csc, y, "cox", time=time_vec, strata=strata)
    d_csr = CyclopsData.from_arrays(X_csr, y, "cox", time=time_vec, strata=strata)
    d_arr = CyclopsData.from_arrays(X_csr_arr, y, "cox", time=time_vec, strata=strata)

    assert d_csr.row_order is not None
    np.testing.assert_array_equal(d_csr.row_order, d_dense.row_order)
    assert d_csr.covariate_formats == d_csc.covariate_formats
    assert d_arr.covariate_formats == d_csc.covariate_formats

    fit_csc = CyclopsModel(d_csc).fit()
    fit_csr = CyclopsModel(d_csr).fit()
    fit_arr = CyclopsModel(d_arr).fit()
    np.testing.assert_allclose(fit_csr.coefficients, fit_csc.coefficients, rtol=1e-12)
    np.testing.assert_allclose(fit_arr.coefficients, fit_csc.coefficients, rtol=1e-12)
    assert fit_csr.log_likelihood == pytest.approx(fit_csc.log_likelihood, rel=1e-12)

    # Conditional logistic regression with unsorted strata
    clr_csc = CyclopsData.from_arrays(X_csc, y, "clr", strata=strata)
    clr_csr = CyclopsData.from_arrays(X_csr, y, "clr", strata=strata)
    assert clr_csr.row_order is not None
    np.testing.assert_allclose(
        CyclopsModel(clr_csr).fit().coefficients,
        CyclopsModel(clr_csc).fit().coefficients,
        rtol=1e-12,
    )


def test_csr_row_order_supports_row_subset_and_unsorted_columns():
    """row_order supports CV fold row subsets and CSR rows with unsorted columns."""
    rng = np.random.default_rng(23)
    n, p = 180, 10
    X = sp.random(n, p, density=0.3, format="csr", random_state=rng)
    # Deliberately scramble column order within each CSR row
    indices = X.indices.copy()
    data_vals = X.data.copy()
    for r in range(n):
        s, e = X.indptr[r], X.indptr[r + 1]
        if e - s > 1:
            perm = rng.permutation(e - s)
            indices[s:e] = indices[s:e][perm]
            data_vals[s:e] = data_vals[s:e][perm]
    y = rng.binomial(1, 0.4, size=n).astype(np.float64)

    # Select a shuffled 60% subset (like a CV training fold)
    subset = rng.choice(n, size=110, replace=False).astype(np.int32)

    ref = CyclopsData.from_arrays(X[subset], y[subset], "lr", add_intercept=True)
    ref_fit = CyclopsModel(ref).fit()

    sub_data = CyclopsData.create("lr")
    sub_data.set_outcome(y[subset])
    sub_data.add_covariates_csr(
        X.indptr, indices, data_vals, n_covariates=p, row_order=subset
    )
    sub_data.add_intercept()
    sub_data.finalize()

    sub_fit = CyclopsModel(sub_data).fit()
    np.testing.assert_allclose(sub_fit.coefficients, ref_fit.coefficients, rtol=1e-12)
    assert sub_fit.log_likelihood == pytest.approx(ref_fit.log_likelihood, rel=1e-12)


def test_strided_dense_and_negative_stride_views():
    """add_dense_covariate reads strided and reversed views without corrupting data."""
    rng = np.random.default_rng(29)
    X_c = np.ascontiguousarray(rng.normal(size=(80, 4)), dtype=np.float64)
    X_f = np.asfortranarray(X_c)
    y = rng.binomial(1, 0.5, size=80).astype(np.float64)

    d_c = CyclopsData.from_arrays(X_c, y, "lr")
    d_f = CyclopsData.from_arrays(X_f, y, "lr")
    np.testing.assert_allclose(
        CyclopsModel(d_c).fit().coefficients,
        CyclopsModel(d_f).fit().coefficients,
        rtol=1e-13,
    )

    # Negative-stride reversed view
    rev = X_c[:, 0][::-1]
    assert rev.strides[0] < 0
    d_rev = CyclopsData.create("lr")
    d_rev.set_outcome(y)
    d_rev.add_dense_covariate(1, rev)
    d_rev.finalize()
    d_copy = CyclopsData.create("lr")
    d_copy.set_outcome(y)
    d_copy.add_dense_covariate(1, rev.copy())
    d_copy.finalize()
    np.testing.assert_allclose(
        CyclopsModel(d_rev).fit().coefficients,
        CyclopsModel(d_copy).fit().coefficients,
        rtol=1e-13,
    )


def test_csr_and_csc_atomic_validation_errors(small):
    """Malformed CSC/CSR batches fail cleanly and leave ModelData untouched."""
    n = small.n_samples
    data = CyclopsData.create("lr")
    data.set_outcome(small.y)
    data.add_covariates(sp.csc_matrix(np.eye(n, 1)), covariate_ids=[1])

    # Unsorted CSC row_indices within a column
    with pytest.raises(CyclopsError, match="ascending"):
        data._handle.add_covariates_csc(
            np.array([0, 2], dtype=np.int32),
            np.array([5, 2], dtype=np.int32),
            None,
            np.array([2], dtype=np.int32),
        )

    # Non-adjacent duplicate column in a single CSR row
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = 3
    with pytest.raises(CyclopsError, match="Repeated row-column entry"):
        data._handle.add_covariates_csr(
            indptr,
            np.array([1, 0, 1], dtype=np.int32),
            None,
            np.array([2, 3], dtype=np.int32),
        )

    # Out-of-range CSR column index
    with pytest.raises(CyclopsError, match="Column index out of range"):
        data._handle.add_covariates_csr(
            indptr,
            np.array([0, 1, 2], dtype=np.int32),
            None,
            np.array([2, 3], dtype=np.int32),
        )

    # Out-of-range CSR row_order index
    bad_order = np.arange(n, dtype=np.int32)
    bad_order[-1] = n
    with pytest.raises(CyclopsError, match="row_order index out of range"):
        data._handle.add_covariates_csr(
            np.zeros(n + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2, 3], dtype=np.int32),
            bad_order,
        )

    # Explicit empty row_order when n_rows > 0
    with pytest.raises(CyclopsError, match="row_order must have length"):
        data._handle.add_covariates_csr(
            np.zeros(n + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2, 3], dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    # Wrong is_indicator length
    with pytest.raises(CyclopsError, match="is_indicator must have length"):
        data._handle.add_covariates_csc(
            np.array([0, 0, 0], dtype=np.int32),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2, 3], dtype=np.int32),
            False,
            np.array([True], dtype=np.bool_),
        )

    # Float array rejected for integer index buffers
    with pytest.raises(TypeError, match="integer dtype"):
        data._handle.add_covariates_csc(
            np.array([0.0, 0.0], dtype=np.float64),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2], dtype=np.int32),
        )

    # Non-boolean array rejected for is_indicator
    with pytest.raises(TypeError, match="boolean dtype"):
        data._handle.add_covariates_csc(
            np.array([0, 0], dtype=np.int32),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2], dtype=np.int32),
            False,
            np.array([1], dtype=np.int32),
        )

    # Out-of-range int64 row indices (> 2^32) must not truncate into valid rows
    for values_arg, mask_arg in (
        (None, None),
        (np.array([1.0]), np.array([True], dtype=np.bool_)),
    ):
        with pytest.raises(CyclopsError, match="Row index out of range"):
            data._handle.add_covariates_csc(
                np.array([0, 1], dtype=np.int64),
                np.array([(1 << 32) + 1], dtype=np.int64),
                values_arg,
                np.array([2], dtype=np.int64),
                False,
                mask_arg,
            )

    # Nothing mutated by any of the failed calls above
    np.testing.assert_array_equal(data.covariate_ids, [1])


def test_strata_without_row_ids_and_atomic_outcome_update():
    """set_outcome with strata and no row_ids works cleanly and fails atomically."""
    y = np.array([0.0, 1.0, 0.0, 1.0])
    strata = np.array([10, 10, 20, 20], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0, 4.0])

    data = CyclopsData.create("clr")
    data.set_outcome(y, strata=strata)
    assert data.n_strata == 2
    np.testing.assert_array_equal(data.strata, [0, 0, 1, 1])

    # A rejected second set_outcome with strata must not overwrite y
    with pytest.raises(CyclopsError, match="already loaded"):
        data.set_outcome(np.array([1.0, 0.0, 1.0, 0.0]), strata=strata)
    np.testing.assert_array_equal(data.y, y)

    # Updating y alone is allowed
    y2 = np.array([1.0, 0.0, 1.0, 0.0])
    data.set_outcome(y2)
    np.testing.assert_array_equal(data.y, y2)

    data.add_dense_covariate(1, x)
    data.finalize()
    np.testing.assert_allclose(data.sum_by_stratum(1), [3.0, 7.0])


def test_fp32_precision_csc_csr_and_dense():
    """FP32 ModelData loads cleanly through CSC, CSR+row_order, and dense paths."""
    rng = np.random.default_rng(31)
    n, p = 100, 6
    X = sp.random(n, p, density=0.35, format="csr", random_state=rng)
    y = rng.binomial(1, 0.4, size=n).astype(np.float32)
    time_vec = rng.exponential(scale=3.0, size=n).astype(np.float32)

    d_csr = CyclopsData.from_arrays(X, y, "cox", time=time_vec, precision="fp32")
    d_csc = CyclopsData.from_arrays(
        X.tocsc(), y, "cox", time=time_vec, precision="fp32"
    )
    fit_csr = CyclopsModel(d_csr).fit()
    fit_csc = CyclopsModel(d_csc).fit()
    np.testing.assert_allclose(fit_csr.coefficients, fit_csc.coefficients, rtol=1e-6)


def test_unaligned_and_readonly_buffers_load_safely():
    """Unaligned and read-only NumPy/SciPy buffers (including unsorted CSC) work."""
    raw = bytearray(256)
    u_indptr = np.ndarray((3,), dtype=np.int32, buffer=raw, offset=1)
    u_indptr[:] = [0, 2, 4]
    u_rows = np.ndarray((4,), dtype=np.int32, buffer=raw, offset=17)
    u_rows[:] = [0, 3, 1, 2]
    u_vals = np.ndarray((4,), dtype=np.float32, buffer=raw, offset=49)
    u_vals[:] = [1.5, 2.5, 1.0, 1.0]
    u_ids = np.ndarray((2,), dtype=np.int64, buffer=raw, offset=81)
    u_ids[:] = [10, 20]
    y = np.array([0.0, 1.0, 0.0, 1.0])

    data = CyclopsData.create("lr")
    data.set_outcome(y)
    data.add_covariates_csc(u_indptr, u_rows, u_vals, covariate_ids=u_ids)
    data.finalize()
    assert data.covariate_formats == ["sparse", "indicator"]
    assert data.column_sum(10) == pytest.approx(4.0)
    assert data.column_sum(20) == pytest.approx(2.0)

    # Read-only unsorted CSC must not mutate caller or fail sort_indices()
    ro_csc = sp.csc_matrix(
        (
            np.array([1.0, 1.0]),
            np.array([3, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        ),
        shape=(4, 1),
    )
    ro_csc.indices.setflags(write=False)
    ro_csc.data.setflags(write=False)
    d_ro = CyclopsData.from_arrays(ro_csc, y, "lr")
    assert d_ro.column_sum(1) == pytest.approx(2.0)
    np.testing.assert_array_equal(ro_csc.indices, [3, 1])  # caller untouched


def test_index_edge_cases_whole_floats_uint64_and_row_checks(small):
    """Whole-number float ids work, uint64 overflow raises, and mismatches raise."""
    n = small.n_samples
    # Whole-number float strata and covariate_ids (e.g. from pandas joins)
    d = CyclopsData.from_arrays(
        small.X[:, :2],
        small.y,
        "clr",
        strata=small.strata.astype(np.float64),
        covariate_ids=np.array([5.0, 6.0]),
    )
    np.testing.assert_array_equal(d.covariate_ids, [5, 6])

    # Fractional float ids rejected
    with pytest.raises(ValueError, match="whole-number"):
        CyclopsData.from_arrays(
            small.X[:, :2], small.y, "lr", covariate_ids=np.array([1.5, 2.0])
        )

    # uint64 exceeding int64 range rejected rather than wrapping negative
    with pytest.raises(OverflowError):
        CyclopsData.from_arrays(
            small.X[:, :2],
            small.y,
            "lr",
            covariate_ids=np.array([1, 2**63 + 5], dtype=np.uint64),
        )
    with pytest.raises(OverflowError):
        d2 = CyclopsData.create("lr")
        d2.set_outcome(small.y)
        d2._handle.add_covariates_csc(
            np.array([0, 0], dtype=np.int32),
            np.empty(0, dtype=np.int32),
            None,
            np.array([2**63 + 5], dtype=np.uint64),
        )

    # CSC row-count mismatch rejected consistently with CSR
    d3 = CyclopsData.create("lr")
    d3.set_outcome(small.y)
    with pytest.raises(ValueError, match="rows"):
        d3.add_covariates(sp.csc_matrix(np.eye(n - 5, 2)))

    # is_indicator on dense input raises instead of silently ignoring
    with pytest.raises(ValueError, match="sparse"):
        CyclopsData.from_arrays(
            small.X[:, :2], small.y, "lr", is_indicator=[True, False]
        )

    # NaN weights rejected
    d4 = CyclopsData.from_arrays(small.X, small.y, "lr")
    model = CyclopsModel(d4)
    w = np.ones(n)
    w[0] = np.nan
    with pytest.raises(CyclopsError, match="non-negative"):
        model.set_weights(w)
    with pytest.raises(CyclopsError, match=r"\[0, 1\]"):
        model.set_censor_weights(w)


def test_sparse_formats_and_bootstrap_row_order():
    """Other SciPy formats (coo, bsr, 1-D csr_array) and duplicate row_order work."""
    rng = np.random.default_rng(37)
    n, p = 60, 4
    dense = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        dense[rng.choice(n, size=18, replace=False), j] = 1.0
    y = rng.binomial(1, 0.4, size=n).astype(np.float64)
    time_vec = rng.exponential(scale=4.0, size=n)

    ref = CyclopsData.from_arrays(sp.csc_matrix(dense), y, "cox", time=time_vec)
    ref_fit = CyclopsModel(ref).fit()

    for ctor in (
        sp.coo_matrix,
        sp.lil_matrix,
        lambda m: sp.bsr_matrix(m, blocksize=(2, 2)),
        sp.dia_matrix,
    ):
        d = CyclopsData.from_arrays(
            ctor(dense),
            y,
            "cox",
            time=time_vec,
            is_indicator=np.ones(p, dtype=np.bool_),
        )
        np.testing.assert_allclose(
            CyclopsModel(d).fit().coefficients, ref_fit.coefficients, rtol=1e-12
        )

    # Mixed float/bool object-dtype dense array (as from pandas DataFrames)
    obj_X = np.array(
        [[1.0, True], [2.5, False], [0.0, True], [1.5, False]], dtype=object
    )
    d_obj = CyclopsData.from_arrays(obj_X, [0.0, 1.0, 0.0, 1.0], "lr")
    assert d_obj.column_sum(1) == pytest.approx(5.0)
    assert d_obj.column_sum(2) == pytest.approx(2.0)

    # 1-D csr_array reshaped to single column like 1-D dense
    d_1d = CyclopsData.from_arrays(sp.csr_array(dense[:, 0]), y, "lr")
    assert d_1d.n_covariates == 1
    assert d_1d.column_sum(1) == pytest.approx(dense[:, 0].sum())

    # Duplicate rows in row_order (bootstrap sample)
    boot = rng.integers(0, n, size=n, dtype=np.int32)
    csr = sp.csr_matrix(dense)
    d_boot = CyclopsData.create("lr")
    d_boot.set_outcome(y[boot])
    d_boot.add_covariates(csr, row_order=boot)
    d_boot.finalize()
    d_ref = CyclopsData.from_arrays(csr[boot], y[boot], "lr")
    np.testing.assert_allclose(
        CyclopsModel(d_boot).fit().coefficients,
        CyclopsModel(d_ref).fit().coefficients,
        rtol=1e-12,
    )
