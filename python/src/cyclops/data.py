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


def _as_1d(values, name: str, dtype=np.float64) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def as_matrix(X):
    """Coerce ``X`` to a 2-D NumPy array or a SciPy sparse matrix.

    Sparse input passes through untouched (converted to CSC only at load time),
    dense array-likes — lists included — become ndarrays, and a 1-D input is
    treated as a single column.

    Callers that need ``X.shape`` before handing it on must go through here
    first: an array-like such as a list of lists has no ``.shape``.
    """
    if sp.issparse(X):
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
        precision: str = "fp64",
        silent: bool = True,
    ) -> "CyclopsData":
        """Build a finalized container from arrays, sorting rows as required.

        ``X`` may be a NumPy array or any SciPy sparse matrix; sparse input is
        converted to CSC and passed through without densification.

        The applied row permutation is available as :attr:`row_order`.
        """
        model_type = model_type.strip().lower()
        y = _as_1d(y, "y")

        matrix = as_matrix(X)
        if sp.issparse(matrix):
            matrix = matrix.tocsc(copy=False)

        if matrix.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {matrix.shape[0]} rows but y has {y.shape[0]} elements"
            )

        time = None if time is None else _as_1d(time, "time")
        strata = None if strata is None else _as_1d(strata, "strata", dtype=np.int64)
        offset = None if offset is None else _as_1d(offset, "offset")
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
            strata = np.zeros(y.shape[0], dtype=np.int64)

        order = _row_order(model_type, y, time, strata)
        if order is not None:
            y = y[order]
            matrix = matrix[order]
            if sp.issparse(matrix):
                matrix = matrix.tocsc(copy=False)
            time = None if time is None else time[order]
            strata = None if strata is None else strata[order]
            offset = None if offset is None else offset[order]

        self = cls.create(model_type, precision=precision, silent=silent)
        self._row_order = order
        self.set_outcome(y, time=time, strata=strata)

        n_features = matrix.shape[1]
        if covariate_ids is None:
            # 1-based so that id 0 stays free for the intercept.
            ids = np.arange(1, n_features + 1, dtype=np.int64)
        else:
            ids = np.ascontiguousarray(covariate_ids, dtype=np.int64)
            if ids.shape != (n_features,):
                raise ValueError(
                    f"covariate_ids must have {n_features} elements, got {ids.shape[0]}"
                )
        clashing = sorted(_RESERVED_IDS.intersection(ids.tolist()))
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

        self.add_covariates(matrix, covariate_ids=ids, force_sparse=force_sparse)

        if add_intercept:
            self.add_intercept()
        if offset is not None:
            self.set_offset(_OFFSET_LOAD_ID, already_log=offset_already_log)

        self.finalize()
        return self

    # -- incremental loading ------------------------------------------------

    def set_outcome(self, y, *, time=None, strata=None, row_ids=None) -> None:
        """Load the outcome; establishes the row count and strata.

        ``ModelData::loadY`` walks ``row_id`` to build the stratum vector, so
        strata are silently ignored unless row identifiers are present. When
        ``strata`` is given and ``row_ids`` is not, positional identifiers are
        substituted. Row identifiers also give every row a text label, which
        costs one small string per row — hence they are not supplied
        unconditionally.
        """
        y = _as_1d(y, "y")
        strata = None if strata is None else _as_1d(strata, "strata", dtype=np.int64)
        if strata is not None and row_ids is None:
            row_ids = np.arange(y.shape[0], dtype=np.int64)

        self._handle.set_outcome(
            y,
            [] if time is None else _as_1d(time, "time"),
            [] if strata is None else strata,
            [] if row_ids is None else _as_1d(row_ids, "row_ids", dtype=np.int64),
        )

    def add_covariates(
        self,
        X,
        *,
        covariate_ids: Sequence[int] | None = None,
        force_sparse: bool = False,
    ) -> None:
        """Append covariate columns from a dense or sparse matrix."""
        if sp.issparse(X):
            csc = X.tocsc(copy=False)
            csc.sort_indices()
            n_features = csc.shape[1]
            ids = self._resolve_ids(covariate_ids, n_features)
            self._handle.add_covariates_csc(
                np.asarray(csc.indptr, dtype=np.int64),
                np.asarray(csc.indices, dtype=np.int64),
                np.asarray(csc.data, dtype=np.float64),
                ids,
                force_sparse,
            )
            return

        dense = np.asarray(X, dtype=np.float64)
        if dense.ndim == 1:
            dense = dense.reshape(-1, 1)
        ids = self._resolve_ids(covariate_ids, dense.shape[1])
        # A fully dense column is stored as DENSE; going through CSC would drop
        # structural zeros and change the storage format (and hence nothing
        # numerically, but it would waste an index vector per column).
        for column, covariate_id in enumerate(ids):
            self.add_dense_covariate(covariate_id, dense[:, column])

    def add_dense_covariate(self, covariate_id: int, values) -> None:
        """Append one fully dense column."""
        self._handle.add_dense_covariate(
            int(covariate_id), _as_1d(values, "values")
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
        ids = np.ascontiguousarray(covariate_ids, dtype=np.int64)
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
