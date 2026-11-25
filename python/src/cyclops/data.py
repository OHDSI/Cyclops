"""Design-matrix container.

:class:`CyclopsData` owns the C++ ``ModelData`` object and enforces the two
invariants the kernels rely on but do not check:

* rows must be **grouped by stratum**, and for the survival models additionally
  **ordered by descending time** within a stratum;
* sparse columns must list their row indices in ascending order.

Both are handled by :meth:`from_arrays`, which returns the row permutation it
applied so callers can map per-row results back to their input order. This
mirrors ``convertToCyclopsData()`` and ``cyclopsData$sortOrder`` in R.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from cyclops import _cyclops
from cyclops._enums import MODEL_KINDS, NORMALIZATION_KINDS, PRECISIONS, resolve

__all__ = ["CyclopsData", "INTERCEPT_ID", "as_matrix"]

#: Covariate id Cyclops assigns to the intercept column.
INTERCEPT_ID = 0

#: Temporary covariate id for an offset supplied as a vector.
#:
#: ``ModelData::setOffsetCovariate`` treats -1 as "promote the time vector", so
#: an offset loaded as an ordinary column cannot use that id. Promotion relabels
#: the column to -1 anyway, so this value is never observable; it just has to be
#: outside the range a caller could plausibly use.
_OFFSET_LOAD_ID = -(2**63)

#: Ids the loader reserves; user-supplied covariate ids may not use them.
_RESERVED_IDS = frozenset({INTERCEPT_ID, -1, _OFFSET_LOAD_ID})

_STRATIFIED = frozenset({"clr", "clr_exact", "clr_efron", "cpr", "sccs"})
_SURVIVAL = frozenset({"cox", "cox_raw", "cox_time", "fgr"})
_UNSTRATIFIED = frozenset({"lr", "pr", "ls"})

#: Model types whose kernel consumes the ``time`` vector as a multiplicative
#: offset (``offs[k] * exp(x'beta)``) rather than through a design column.
#:
#: ``SelfControlledCaseSeries`` is the only such model — every other
#: ``getOffsExpXBeta`` overload in engine/ModelSpecifics.h ignores its ``offs``
#: argument. Routing an SCCS offset through a design column instead would both
#: leave ``hOffs`` empty (a null dereference in the kernel) and double-count the
#: exposure.
_TIME_IS_OFFSET = frozenset({"sccs"})


_NATIVE_VALUE_DTYPES = frozenset(
    {
        np.dtype(np.float64),
        np.dtype(np.float32),
        np.dtype(np.bool_),
        np.dtype(np.int8),
        np.dtype(np.uint8),
        np.dtype(np.int32),
        np.dtype(np.int64),
    }
)
_NATIVE_INT_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))
_INT64_MAX = np.iinfo(np.int64).max
# Exact float64 bounds of int64: [-2^63, 2^63). float64(2^63 - 1) rounds up to 2^63.
# Use np.float64 scalars so float16 comparisons promote up instead of overflowing.
_INT64_MIN_FLOAT = np.float64(-9223372036854775808.0)
_INT64_MAX_EXCL_FLOAT = np.float64(9223372036854775808.0)


def _as_1d_values(values, name: str, *, allow_strided: bool = False) -> np.ndarray:
    """Return a 1-D numeric array, avoiding copies for C++-native dtypes."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size > 0 and array.dtype.kind not in ("f", "i", "u", "b", "O"):
        raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
    if array.dtype in _NATIVE_VALUE_DTYPES and (
        allow_strided or array.flags.c_contiguous
    ):
        return array
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_1d_indices(
    values, name: str, *, allow_whole_floats: bool = True
) -> np.ndarray:
    """Return a 1-D contiguous int32/int64 array without widening int32."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    kind = array.dtype.kind
    if kind in ("i", "u"):
        if kind == "u" and array.dtype.itemsize >= 8 and bool(
            np.any(array > _INT64_MAX)
        ):
            raise OverflowError(
                f"{name} contains uint64 values exceeding int64 range"
            )
        if array.dtype in _NATIVE_INT_DTYPES and array.flags.c_contiguous:
            return array
        return np.ascontiguousarray(array, dtype=np.int64)
    if allow_whole_floats and kind == "f":
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{name} must contain finite integer values")
        if not bool(np.all(array == np.trunc(array))):
            raise ValueError(f"{name} must contain whole-number values")
        if bool(np.any((array < _INT64_MIN_FLOAT) | (array >= _INT64_MAX_EXCL_FLOAT))):
            raise OverflowError(f"{name} values exceed int64 range")
        return np.ascontiguousarray(array, dtype=np.int64)
    if allow_whole_floats and kind == "O":
        converted: list[int] = []
        for item in array:
            if isinstance(item, bool):
                raise TypeError(f"{name} must have an integer dtype, got bool")
            if isinstance(item, (int, np.integer)):
                val = int(item)
            elif isinstance(item, (float, np.floating)):
                fval = float(item)
                if not np.isfinite(fval) or fval != np.trunc(fval):
                    raise ValueError(f"{name} must contain whole-number values")
                val = int(fval)
            else:
                raise TypeError(f"{name} must have an integer dtype, got object")
            if val < -(2**63) or val > _INT64_MAX:
                raise OverflowError(f"{name} values exceed int64 range")
            converted.append(val)
        return np.asarray(converted, dtype=np.int64)
    raise TypeError(f"{name} must have an integer dtype, got {array.dtype}")


def _clashing_reserved_ids(ids: np.ndarray) -> list[int]:
    if ids.size == 0:
        return []
    candidates = (
        (-1, INTERCEPT_ID)
        if ids.dtype == np.dtype(np.int32)
        else sorted(_RESERVED_IDS)
    )
    scalar = ids.dtype.type
    return [r for r in candidates if bool(np.any(ids == scalar(r)))]


def as_matrix(X):
    """Coerce ``X`` to a 2-D NumPy array or a 2-D SciPy sparse matrix/array.

    Sparse input passes through untouched (converted only when required at load
    time), dense array-likes — lists included — become ndarrays, and a 1-D input
    is treated as a single column.

    Callers that need ``X.shape`` before handing it on must go through here
    first: an array-like such as a list of lists has no ``.shape``.
    """
    if sp.issparse(X):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
        return X
    matrix = np.asarray(X)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {matrix.shape}")
    return matrix


def _row_order(model_type: str, y: np.ndarray, time, strata) -> np.ndarray | None:
    """Row permutation required by ``model_type``, or None when already valid.

    Sort keys match ``NewDataConversion.R``: strata alone for the conditional
    models, and ``(stratum asc, time desc, y asc)`` for the survival models,
    whose risk-set recursions walk the rows in that order.
    """
    if model_type in _UNSTRATIFIED:
        return None

    n = y.shape[0]
    if model_type in _SURVIVAL:
        if time is None:
            raise ValueError(f"model type {model_type!r} requires `time`")
        # np.lexsort applies the last key first.
        order = np.lexsort((y, -np.asarray(time, dtype=np.float64), strata))
    else:
        order = np.argsort(strata, kind="stable")

    if np.array_equal(order, np.arange(n)):
        return None
    return order


class CyclopsData:
    """Outcome and covariates in the layout Cyclops expects.

    Prefer :meth:`from_arrays` for in-memory data; use the incremental API when
    streaming a design matrix that does not fit in memory.
    """

    def __init__(self, handle: "_cyclops.ModelData", model_type: str):
        self._handle = handle
        self._model_type = model_type
        self._row_order: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        model_type: str,
        *,
        precision: str = "fp64",
        silent: bool = True,
    ) -> "CyclopsData":
        """Create an empty container for ``model_type`` (``"lr"``, ``"cox"``, …)."""
        kind = resolve(MODEL_KINDS, model_type, "model type")
        handle = _cyclops.ModelData.create(
            kind, resolve(PRECISIONS, precision, "precision"), silent
        )
        return cls(handle, model_type.strip().lower())

    @classmethod
    def from_arrays(
        cls,
        X,
        y,
        model_type: str,
        *,
        time=None,
        strata=None,
        offset=None,
        covariate_ids: Sequence[int] | None = None,
        add_intercept: bool = False,
        offset_already_log: bool = False,
        force_sparse: bool = False,
        is_indicator: npt.ArrayLike | None = None,
        precision: str = "fp64",
        silent: bool = True,
    ) -> "CyclopsData":
        """Build a finalized container from arrays, sorting rows as required.

        ``X`` may be a NumPy array or any SciPy sparse matrix/array; CSC and CSR
        inputs are read directly without densification, and CSR inputs apply row
        sorting in C++ without transposing to CSC or copying ``X[order]``.

        When ``is_indicator`` is provided (a 1-D boolean mask of length
        ``X.shape[1]``, for sparse inputs only), columns marked ``True`` are
        loaded directly as binary ``indicator`` columns without scanning their
        values (assuming ``force_sparse=False``). Callers setting
        ``is_indicator[j] = True`` guarantee that every stored entry in column
        ``j`` is ``1`` with no explicit zeros.

        The applied row permutation is available as :attr:`row_order`.
        """
        model_type = model_type.strip().lower()
        y = _as_1d_values(y, "y")

        matrix = as_matrix(X)

        if matrix.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {matrix.shape[0]} rows but y has {y.shape[0]} elements"
            )

        time = None if time is None else _as_1d_values(time, "time")
        strata = None if strata is None else _as_1d_indices(strata, "strata")
        offset = None if offset is None else _as_1d_values(offset, "offset")
        for name, array in (("time", time), ("strata", strata), ("offset", offset)):
            if array is not None and array.shape[0] != y.shape[0]:
                raise ValueError(f"{name} must have one element per row of X")

        time_as_offset = model_type in _TIME_IS_OFFSET and offset is not None
        if time_as_offset:
            if time is not None:
                raise ValueError(
                    f"model type {model_type!r} consumes `offset` as its time "
                    "vector; pass one or the other, not both"
                )
            if offset_already_log:
                raise ValueError(
                    f"model type {model_type!r} log-transforms the offset in its "
                    "likelihood; supply it on the natural scale"
                )
            time, offset = offset, None

        if model_type in _STRATIFIED and strata is None:
            raise ValueError(f"model type {model_type!r} requires `strata`")
        if model_type in _SURVIVAL and strata is None:
            # The survival kernels index risk sets through the stratum vector, so
            # an unstratified fit still needs a single all-zero stratum. R does
            # the same (`outcomes$stratumId <- 0` in NewDataConversion.R).
            strata = np.zeros(y.shape[0], dtype=np.int32)

        order = _row_order(model_type, y, time, strata)
        sparse_row_order = None
        if order is not None:
            y = y[order]
            time = None if time is None else time[order]
            strata = None if strata is None else strata[order]
            offset = None if offset is None else offset[order]
            if sp.issparse(matrix):
                # Keep sparse inputs unpermuted and let add_covariates handle format
                # conversion (including BSR/DIA zero stripping) and C++ row_order.
                sparse_row_order = order
            else:
                matrix = matrix[order]

        self = cls.create(model_type, precision=precision, silent=silent)
        self._row_order = order
        self.set_outcome(y, time=time, strata=strata)

        n_features = matrix.shape[1]
        if covariate_ids is None:
            # 1-based so that id 0 stays free for the intercept.
            ids = np.arange(1, n_features + 1, dtype=np.int64)
        else:
            ids = _as_1d_indices(covariate_ids, "covariate_ids")
            if ids.shape != (n_features,):
                raise ValueError(
                    f"covariate_ids must have {n_features} elements, got {ids.shape[0]}"
                )
        clashing = _clashing_reserved_ids(ids)
        if clashing:
            raise ValueError(
                f"covariate ids {clashing} are reserved by Cyclops for the "
                "intercept and offset columns"
            )

        if offset is not None:
            # Cyclops promotes an existing covariate rather than accepting a bare
            # offset vector, so load it as an ordinary dense column and promote
            # it once every other column is in place.
            self.add_dense_covariate(_OFFSET_LOAD_ID, offset)

        self.add_covariates(
            matrix,
            covariate_ids=ids,
            force_sparse=force_sparse,
            is_indicator=is_indicator,
            row_order=sparse_row_order,
        )

        if add_intercept:
            self.add_intercept()
        if offset is not None:
            self.set_offset(_OFFSET_LOAD_ID, already_log=offset_already_log)

        self.finalize()
        return self

    # -- incremental loading ------------------------------------------------

    def set_outcome(self, y, *, time=None, strata=None, row_ids=None) -> None:
        """Load the outcome; establishes the row count and strata.

        Covariate matrices are always read by position: row ``k`` of a matrix
        belongs to ``y[k]``, whatever its row identifier.
        """
        y = _as_1d_values(y, "y")
        strata = None if strata is None else _as_1d_indices(strata, "strata")

        self._handle.set_outcome(
            y,
            None if time is None else _as_1d_values(time, "time"),
            strata,
            None if row_ids is None else _as_1d_indices(row_ids, "row_ids"),
        )

    def add_covariates(
        self,
        X,
        *,
        covariate_ids: Sequence[int] | None = None,
        force_sparse: bool = False,
        is_indicator: npt.ArrayLike | None = None,
        row_order: npt.ArrayLike | None = None,
    ) -> None:
        """Append covariate columns from a dense or sparse matrix.

        When ``is_indicator`` is provided (sparse inputs only), columns marked
        ``True`` skip scanning ``values`` and must contain only nonzero ``1``
        entries (no explicit zeros) unless ``force_sparse=True``.
        """
        if sp.issparse(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            elif X.ndim != 2:
                raise ValueError(f"X must be two-dimensional, got shape {X.shape}")

            order_arr = (
                None
                if row_order is None
                else _as_1d_indices(row_order, "row_order", allow_whole_floats=False)
            )
            if order_arr is None and self.n_rows > 0 and X.shape[0] != self.n_rows:
                raise ValueError(
                    f"X has {X.shape[0]} rows but outcome has {self.n_rows} rows"
                )

            strip_padded_zeros = is_indicator is not None and X.format in ("bsr", "dia")
            if order_arr is not None or X.format == "csr":
                csr = X if X.format == "csr" else X.tocsr()
                if strip_padded_zeros:
                    csr.eliminate_zeros()
                ids = self._resolve_ids(covariate_ids, csr.shape[1])
                self._handle.add_covariates_csr(
                    csr.indptr,
                    csr.indices,
                    csr.data,
                    ids,
                    row_order=order_arr,
                    force_sparse=force_sparse,
                    is_indicator=is_indicator,
                )
                return

            csc = X.tocsc(copy=False)
            if strip_padded_zeros:
                csc.eliminate_zeros()
            if not csc.has_sorted_indices:
                csc = csc.copy()
                csc.sort_indices()
            ids = self._resolve_ids(covariate_ids, csc.shape[1])
            self._handle.add_covariates_csc(
                csc.indptr,
                csc.indices,
                csc.data,
                ids,
                force_sparse=force_sparse,
                is_indicator=is_indicator,
            )
            return

        if is_indicator is not None:
            raise ValueError("is_indicator is only supported for sparse inputs")

        dense = np.asarray(X)
        if dense.size > 0 and dense.dtype.kind not in ("f", "i", "u", "b", "O"):
            raise TypeError(f"X must have a numeric dtype, got {dense.dtype}")
        if dense.dtype not in _NATIVE_VALUE_DTYPES:
            dense = np.asarray(dense, dtype=np.float64)
        if dense.ndim == 1:
            dense = dense.reshape(-1, 1)
        if dense.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {dense.shape}")
        if row_order is not None:
            order_arr = _as_1d_indices(row_order, "row_order", allow_whole_floats=False)
            if self.n_rows > 0 and order_arr.shape[0] != self.n_rows:
                raise ValueError(
                    "row_order must have length equal to the number of outcome rows"
                )
            if order_arr.size > 0 and (
                bool(np.any(order_arr < 0)) or bool(np.any(order_arr >= dense.shape[0]))
            ):
                raise ValueError("row_order index out of range")
            dense = dense[order_arr]
        ids = self._resolve_ids(covariate_ids, dense.shape[1])
        # A fully dense column is stored as DENSE; going through CSC would drop
        # structural zeros and change the storage format (and hence nothing
        # numerically, but it would waste an index vector per column).
        for column, covariate_id in enumerate(ids):
            self.add_dense_covariate(int(covariate_id), dense[:, column])

    def add_covariates_csc(
        self,
        indptr,
        row_indices,
        values=None,
        *,
        covariate_ids: Sequence[int] | None = None,
        force_sparse: bool = False,
        is_indicator: npt.ArrayLike | None = None,
    ) -> None:
        """Append covariates directly from CSC buffers without building a SciPy matrix.

        Pass ``values=None`` for pure binary 0/1 indicator columns, or supply a
        boolean ``is_indicator`` mask of length ``n_columns`` to skip scanning
        ``values`` for binary columns in a mixed batch (callers guarantee that
        flagged columns contain only ``1`` entries with no explicit zeros).
        """
        indptr_arr = _as_1d_indices(indptr, "indptr", allow_whole_floats=False)
        n_features = max(int(indptr_arr.shape[0]) - 1, 0)
        ids = self._resolve_ids(covariate_ids, n_features)
        self._handle.add_covariates_csc(
            indptr_arr,
            row_indices,
            values,
            ids,
            force_sparse=force_sparse,
            is_indicator=is_indicator,
        )

    def add_covariates_csr(
        self,
        indptr,
        col_indices,
        values=None,
        *,
        covariate_ids: Sequence[int] | None = None,
        n_covariates: int | None = None,
        row_order: npt.ArrayLike | None = None,
        force_sparse: bool = False,
        is_indicator: npt.ArrayLike | None = None,
    ) -> None:
        """Append covariates directly from CSR buffers with optional row permutation.

        ``row_order`` maps each outcome row ``r`` to CSR row ``row_order[r]``,
        avoiding CSR-to-CSC transposition and per-column sorting for survival and
        stratified models or row-subsetted cross-validation folds.

        When ``is_indicator`` is provided, flagged columns skip scanning
        ``values`` (when ``force_sparse=False``) and must contain only ``1``
        entries with no explicit zeros.
        """
        if n_covariates is not None and int(n_covariates) < 0:
            raise ValueError("n_covariates must be non-negative")
        if covariate_ids is None:
            if n_covariates is None:
                raise ValueError(
                    "Pass either `covariate_ids` or `n_covariates` for CSR input"
                )
            ids = self._resolve_ids(None, int(n_covariates))
        else:
            ids_arr = _as_1d_indices(covariate_ids, "covariate_ids")
            expected = (
                int(ids_arr.shape[0]) if n_covariates is None else int(n_covariates)
            )
            ids = self._resolve_ids(ids_arr, expected)
        self._handle.add_covariates_csr(
            indptr,
            col_indices,
            values,
            ids,
            row_order=row_order,
            force_sparse=force_sparse,
            is_indicator=is_indicator,
        )

    def add_dense_covariate(self, covariate_id: int, values) -> None:
        """Append one fully dense column."""
        self._handle.add_dense_covariate(
            int(covariate_id), _as_1d_values(values, "values", allow_strided=True)
        )

    def add_intercept(self) -> None:
        """Prepend an intercept column with covariate id 0."""
        self._handle.add_intercept()

    def set_offset(self, covariate_id: int, *, already_log: bool = False) -> None:
        """Promote a loaded covariate to the fixed-coefficient offset column.

        Pass ``covariate_id=-1`` to promote the outcome ``time`` vector instead.
        Unless ``already_log`` is set, the column is log-transformed in place.
        """
        self._handle.set_offset_covariate(int(covariate_id), already_log)

    def make_dense(self, covariate_ids: Iterable[int]) -> None:
        """Convert the named columns to dense storage."""
        self._handle.make_dense([int(i) for i in covariate_ids])

    def normalize(self, kind: str = "stdev") -> np.ndarray:
        """Rescale the dense and sparse covariates; return the factors applied.

        Indicator, intercept and offset columns are left alone (their factor is
        reported as 1.0). Columns are *multiplied* by the factor — ``1/sd`` for
        ``"stdev"`` — so a coefficient fitted afterwards is returned to the
        original scale by multiplying it by the same factor, as
        ``coef(fit, rescale = TRUE)`` does in R.
        """
        self._scale = self._handle.normalize(
            resolve(NORMALIZATION_KINDS, kind, "normalization")
        )
        return self._scale

    def finalize(self) -> "CyclopsData":
        """Freeze the container; required before fitting."""
        self._handle.finalize()
        return self

    # -- introspection ------------------------------------------------------

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def n_rows(self) -> int:
        return self._handle.row_count

    @property
    def n_covariates(self) -> int:
        return self._handle.covariate_count

    @property
    def n_strata(self) -> int:
        return self._handle.stratum_count

    @property
    def covariate_ids(self) -> np.ndarray:
        return self._handle.covariate_ids

    @property
    def covariate_formats(self) -> list[str]:
        return [fmt.name.lower() for fmt in self._handle.covariate_formats]

    @property
    def has_intercept(self) -> bool:
        return self._handle.has_intercept

    @property
    def has_offset(self) -> bool:
        return self._handle.has_offset

    @property
    def intercept_label(self) -> int:
        """Covariate id of the intercept column. Raises when there is none."""
        return self._handle.intercept_label

    @property
    def y(self) -> np.ndarray:
        return self._handle.outcome

    @property
    def time(self) -> np.ndarray:
        return self._handle.time

    @property
    def strata(self) -> np.ndarray:
        return self._handle.stratum_index

    @property
    def row_order(self) -> np.ndarray | None:
        """Permutation applied to input rows, or None if rows were left alone."""
        return self._row_order

    @property
    def scale(self) -> np.ndarray | None:
        """Factors from the last :meth:`normalize` call, if any."""
        return self._scale

    @property
    def is_finalized(self) -> bool:
        return self._handle.is_finalized

    def univariable_correlation(self, ids: Sequence[int] | None = None) -> np.ndarray:
        """Pearson correlation between each covariate and the outcome."""
        return self._handle.univariable_correlation(
            [] if ids is None else [int(i) for i in ids]
        )

    def column_sum(self, covariate_id: int, power: int = 1) -> float:
        """Sum of ``x**power`` over one covariate column.

        Raises for an unknown id. The outcome is not reducible this way; read it
        from :attr:`y`.
        """
        return self._handle.column_sum(int(covariate_id), power)

    def sum_by_stratum(self, covariate_id: int, power: int = 1) -> np.ndarray:
        return self._handle.sum_by_stratum(int(covariate_id), power)

    # -- internals ----------------------------------------------------------

    def _resolve_ids(
        self, covariate_ids: Sequence[int] | None, n_features: int
    ) -> np.ndarray:
        if covariate_ids is None:
            start = self.n_covariates + 1
            return np.arange(start, start + n_features, dtype=np.int64)
        ids = _as_1d_indices(covariate_ids, "covariate_ids")
        if ids.shape != (n_features,):
            raise ValueError(
                f"covariate_ids must have {n_features} elements, got {ids.shape[0]}"
            )
        return ids

    def __repr__(self) -> str:
        return (
            f"CyclopsData(model_type={self._model_type!r}, n_rows={self.n_rows}, "
            f"n_covariates={self.n_covariates}, n_strata={self.n_strata}, "
            f"finalized={self.is_finalized})"
        )
