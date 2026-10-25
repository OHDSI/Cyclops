"""Run the R Cyclops package and return its results to Python.

The R package is the numerical ground truth for these bindings, so the parity
tests need a way to fit the *same* data through `fitCyclopsModel` and read the
answer back. This module writes the design matrix to disk in the triplet layout
`convertToCyclopsData` expects, drives `Rscript`, and parses a JSON reply.

Data crosses the boundary as CSV and results as JSON so the comparison never
depends on R and Python agreeing on a binary layout. Coefficients are printed
with 17 significant digits, which round-trips an IEEE double exactly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

__all__ = ["RUnavailable", "r_cyclops_available", "RFit", "fit_in_r"]


class RUnavailable(RuntimeError):
    """Raised when R or the R Cyclops package is not usable."""


_R_SCRIPT = r"""
suppressMessages({
    library(Cyclops)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(args[1], simplifyVector = TRUE)

outcomes   <- read.csv(spec$outcomes_path)
covariates <- read.csv(spec$covariates_path)

# convertToCyclopsData() dispatches on which columns are present, so drop the
# ones this model does not use rather than passing along placeholder values.
for (column in c("stratumId", "time")) {
    if (!(column %in% spec$outcome_columns)) {
        outcomes[[column]] <- NULL
    }
}

data <- convertToCyclopsData(
    outcomes    = outcomes,
    covariates  = covariates,
    modelType   = spec$model_type,
    addIntercept = spec$add_intercept,
    checkRowIds = FALSE,
    quiet       = TRUE
)

# NOTE: convertToCyclopsData() already calls
# finalizeSqlCyclopsData(useOffsetCovariate = -1) for "pr" and "cpr", promoting
# the `time` column to a log-transformed offset. That is the only way to give
# those models an offset through this entry point, so callers pass person-time
# as `time` on the natural scale (see fit_in_r's docstring).

prior <- if (identical(spec$prior, "none")) {
    createPrior("none")
} else if (isTRUE(spec$use_cross_validation)) {
    createPrior(spec$prior, useCrossValidation = TRUE,
                exclude = if (length(spec$exclude)) spec$exclude else NULL)
} else {
    createPrior(spec$prior, variance = spec$prior_variance,
                exclude = if (length(spec$exclude)) spec$exclude else NULL)
}

control <- createControl(
    maxIterations   = spec$max_iterations,
    tolerance       = spec$tolerance,
    convergenceType = spec$convergence,
    noiseLevel      = "silent",
    threads         = 1,
    seed            = spec$seed,
    fold            = spec$fold,
    cvRepetitions   = spec$cv_repetitions,
    startingVariance = spec$starting_variance,
    selectorType    = spec$selector,
    minCVData       = spec$min_cv_data
)

fit <- fitCyclopsModel(
    data,
    prior    = prior,
    control  = control,
    weights  = if (length(spec$weights)) spec$weights else NULL,
    warnings = FALSE
)

# coef() refuses to return anything for a non-converged fit; the parity tests
# want to compare those cases too.
estimates <- fit$estimation

result <- list(
    covariate_ids  = as.character(estimates$column_label),
    coefficients   = as.numeric(estimates$estimate),
    log_likelihood = as.numeric(fit$log_likelihood),
    log_prior      = as.numeric(fit$log_prior),
    return_flag    = as.character(fit$return_flag),
    iterations      = as.integer(fit$iterations),
    prior_info     = as.character(fit$prior_info),
    variance       = as.numeric(fit$variance),
    n_covariates   = as.integer(getNumberOfCovariates(data)),
    n_rows         = as.integer(getNumberOfRows(data)),
    n_strata       = as.integer(getNumberOfStrata(data))
)

if (isTRUE(spec$want_predictions)) {
    result$predictions <- as.numeric(predict(fit))
}
if (isTRUE(spec$want_standard_errors)) {
    result$standard_errors <- as.numeric(
        tryCatch(Cyclops:::getSEs(fit, getCovariateIds(data)),
                 error = function(e) rep(NA_real_, length(result$coefficients)),
                 warning = function(w) suppressWarnings(
                     Cyclops:::getSEs(fit, getCovariateIds(data))))
    )
}

cat(toJSON(result, digits = 17, auto_unbox = TRUE, na = "null"))
"""


@dataclass
class RFit:
    """What the R package reported. Field names mirror ``cyclopsFit``."""

    covariate_ids: np.ndarray
    coefficients: np.ndarray
    log_likelihood: float
    log_prior: float
    return_flag: str
    iterations: int
    prior_info: str
    variance: np.ndarray
    n_covariates: int
    n_rows: int
    n_strata: int
    predictions: np.ndarray | None = None
    standard_errors: np.ndarray | None = None


def r_cyclops_available() -> bool:
    """True when ``Rscript`` can load Cyclops and jsonlite."""
    if shutil.which("Rscript") is None:
        return False
    probe = (
        'quit(status = if (requireNamespace("Cyclops", quietly=TRUE) && '
        'requireNamespace("jsonlite", quietly=TRUE)) 0L else 1L)'
    )
    try:
        return (
            subprocess.run(
                ["Rscript", "-e", probe], capture_output=True, timeout=120
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, OSError):
        return False


def _write_inputs(
    directory: Path,
    X,
    y: np.ndarray,
    *,
    time,
    strata,
    covariate_ids: np.ndarray,
) -> tuple[Path, Path, list[str]]:
    """Write outcomes and covariates in ``convertToCyclopsData`` triplet form."""
    n_rows = y.shape[0]
    row_ids = np.arange(1, n_rows + 1, dtype=np.int64)

    columns = ["rowId", "y"]
    outcome_data = {"rowId": row_ids, "y": y}
    if strata is not None:
        outcome_data["stratumId"] = np.asarray(strata, dtype=np.int64)
        columns.append("stratumId")
    if time is not None:
        outcome_data["time"] = np.asarray(time, dtype=np.float64)
        columns.append("time")

    outcomes_path = directory / "outcomes.csv"
    _write_csv(outcomes_path, outcome_data)

    csc = (X.tocsc(copy=False) if sp.issparse(X) else sp.csc_matrix(np.asarray(X)))
    csc.sort_indices()
    coo = csc.tocoo()
    keep = coo.data != 0.0
    covariate_data = {
        "rowId": row_ids[coo.row[keep]],
        "covariateId": covariate_ids[coo.col[keep]],
        "covariateValue": coo.data[keep],
    }
    if strata is not None:
        # convertToCyclopsData() sorts the covariate triplets by
        # (covariateId, stratumId, rowId) for the conditional models and reads
        # stratumId straight off this frame — unlike the Cox branch, it never
        # merges it in from `outcomes`. Omitting it is an "index out of bounds"
        # error in R.
        covariate_data["stratumId"] = np.asarray(strata, dtype=np.int64)[
            coo.row[keep]
        ]

    covariates_path = directory / "covariates.csv"
    _write_csv(covariates_path, covariate_data)
    return outcomes_path, covariates_path, columns


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    matrix = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in names]
    )
    with path.open("w") as handle:
        handle.write(",".join(names) + "\n")
        np.savetxt(handle, matrix, delimiter=",", fmt="%.17g")


def fit_in_r(
    X,
    y,
    model_type: str,
    *,
    time=None,
    strata=None,
    covariate_ids=None,
    add_intercept: bool = False,
    prior: str = "none",
    prior_variance: float = 1.0,
    use_cross_validation: bool = False,
    exclude=None,
    weights=None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    convergence: str = "gradient",
    seed: int = 123,
    fold: int = 10,
    cv_repetitions: int = 1,
    starting_variance: float = -1.0,
    selector: str = "auto",
    min_cv_data: int = 100,
    want_predictions: bool = False,
    want_standard_errors: bool = False,
    timeout: int = 600,
) -> RFit:
    """Fit ``(X, y)`` with the R package and return its results.

    ``covariate_ids`` defaults to ``1..n_features``, matching
    :meth:`cyclops.data.CyclopsData.from_arrays`, so coefficients line up
    positionally on both sides.

    ``time`` carries survival duration for the Cox family and person-time (on
    the natural scale) for ``"pr"``/``"cpr"``, which ``convertToCyclopsData``
    promotes to a log-transformed offset column. Those two model types *require*
    it: R aborts with a segmentation fault when it is absent, because
    ``setOffsetCovariate(-1)`` pushes an empty column. Pass ones for a Poisson
    fit with no offset.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_features = X.shape[1] if X.ndim == 2 else 1
    if covariate_ids is None:
        covariate_ids = np.arange(1, n_features + 1, dtype=np.int64)
    covariate_ids = np.asarray(covariate_ids, dtype=np.int64)

    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        outcomes_path, covariates_path, outcome_columns = _write_inputs(
            directory, X, y, time=time, strata=strata, covariate_ids=covariate_ids
        )

        spec = {
            "outcomes_path": str(outcomes_path),
            "covariates_path": str(covariates_path),
            "outcome_columns": outcome_columns,
            "model_type": model_type,
            "add_intercept": bool(add_intercept),
            "prior": prior,
            "prior_variance": float(prior_variance),
            "use_cross_validation": bool(use_cross_validation),
            "exclude": [] if exclude is None else [int(i) for i in exclude],
            "weights": [] if weights is None else [float(w) for w in weights],
            "max_iterations": int(max_iterations),
            "tolerance": float(tolerance),
            "convergence": convergence,
            "seed": int(seed),
            "fold": int(fold),
            "cv_repetitions": int(cv_repetitions),
            "starting_variance": float(starting_variance),
            "selector": selector,
            "min_cv_data": int(min_cv_data),
            "want_predictions": bool(want_predictions),
            "want_standard_errors": bool(want_standard_errors),
        }
        spec_path = directory / "spec.json"
        spec_path.write_text(json.dumps(spec))

        script_path = directory / "fit.R"
        script_path.write_text(_R_SCRIPT)

        environment = dict(os.environ, R_LIBS_USER=os.environ.get("R_LIBS_USER", ""))
        completed = subprocess.run(
            ["Rscript", "--vanilla", str(script_path), str(spec_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )

    if completed.returncode != 0:
        raise RUnavailable(
            f"R fit failed (exit {completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )

    payload = completed.stdout[completed.stdout.index("{") :]
    parsed = json.loads(payload)
    return RFit(
        covariate_ids=np.array(
            [int(float(i)) for i in _listify(parsed["covariate_ids"])]
        ),
        coefficients=np.asarray(_listify(parsed["coefficients"]), dtype=np.float64),
        log_likelihood=float(parsed["log_likelihood"]),
        log_prior=float(parsed["log_prior"]),
        return_flag=str(parsed["return_flag"]),
        iterations=int(parsed["iterations"]),
        prior_info=str(parsed["prior_info"]),
        variance=np.asarray(_listify(parsed["variance"]), dtype=np.float64),
        n_covariates=int(parsed["n_covariates"]),
        n_rows=int(parsed["n_rows"]),
        n_strata=int(parsed["n_strata"]),
        predictions=(
            np.asarray(_listify(parsed["predictions"]), dtype=np.float64)
            if "predictions" in parsed
            else None
        ),
        standard_errors=(
            np.asarray(_listify(parsed["standard_errors"]), dtype=np.float64)
            if "standard_errors" in parsed
            else None
        ),
    )


def _listify(value):
    """jsonlite unboxes length-1 vectors, so normalise back to a list."""
    if isinstance(value, list):
        return value
    return [value]
