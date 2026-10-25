"""String ↔ enum translation.

Cyclops' model-type codes (``"lr"``, ``"cox"``, …) are part of the OHDSI
vocabulary and are kept verbatim so that code and documentation transfer between
the R and Python packages. The other option strings mirror ``createControl()``
and ``createPrior()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from cyclops import _cyclops

__all__ = [
    "MODEL_KINDS",
    "PRIOR_KINDS",
    "CONVERGENCE_KINDS",
    "SELECTOR_KINDS",
    "ALGORITHM_KINDS",
    "NORMALIZATION_KINDS",
    "NOISE_LEVELS",
    "PRECISIONS",
    "resolve",
]

MODEL_KINDS: Mapping[str, "_cyclops.ModelKind"] = {
    "ls": _cyclops.ModelKind.NORMAL,
    "pr": _cyclops.ModelKind.POISSON,
    "lr": _cyclops.ModelKind.LOGISTIC,
    "clr": _cyclops.ModelKind.CONDITIONAL_LOGISTIC,
    "clr_exact": _cyclops.ModelKind.TIED_CONDITIONAL_LOGISTIC,
    "clr_efron": _cyclops.ModelKind.EFRON_CONDITIONAL_LOGISTIC,
    "cpr": _cyclops.ModelKind.CONDITIONAL_POISSON,
    "sccs": _cyclops.ModelKind.SELF_CONTROLLED_CASE_SERIES,
    "cox": _cyclops.ModelKind.COX,
    "cox_raw": _cyclops.ModelKind.COX_RAW,
    "cox_time": _cyclops.ModelKind.TIME_VARYING_COX,
    "fgr": _cyclops.ModelKind.FINE_GRAY,
}

PRIOR_KINDS: Mapping[str, "_cyclops.PriorKind"] = {
    "none": _cyclops.PriorKind.NONE,
    "laplace": _cyclops.PriorKind.LAPLACE,
    "normal": _cyclops.PriorKind.NORMAL,
    "barupdate": _cyclops.PriorKind.BAR_UPDATE,
    "jeffreys": _cyclops.PriorKind.JEFFREYS,
    # Familiar aliases.
    "l1": _cyclops.PriorKind.LAPLACE,
    "l2": _cyclops.PriorKind.NORMAL,
}

CONVERGENCE_KINDS: Mapping[str, "_cyclops.ConvergenceKind"] = {
    "gradient": _cyclops.ConvergenceKind.GRADIENT,
    "lange": _cyclops.ConvergenceKind.LANGE,
    "mittal": _cyclops.ConvergenceKind.MITTAL,
    "onestep": _cyclops.ConvergenceKind.ONE_STEP,
    "zhangoles": _cyclops.ConvergenceKind.ZHANG_OLES,
    # R spells this one "zhang"; accept both.
    "zhang": _cyclops.ConvergenceKind.ZHANG_OLES,
}

SELECTOR_KINDS: Mapping[str, "_cyclops.SelectorKind"] = {
    "auto": _cyclops.SelectorKind.AUTO,
    "byrow": _cyclops.SelectorKind.BY_ROW,
    "bypid": _cyclops.SelectorKind.BY_PID,
    "bystratum": _cyclops.SelectorKind.BY_PID,
}

ALGORITHM_KINDS: Mapping[str, "_cyclops.AlgorithmKind"] = {
    "ccd": _cyclops.AlgorithmKind.CCD,
    "mm": _cyclops.AlgorithmKind.MM,
}

NORMALIZATION_KINDS: Mapping[str, "_cyclops.NormalizationKind"] = {
    "stdev": _cyclops.NormalizationKind.STANDARD_DEVIATION,
    "max": _cyclops.NormalizationKind.MAX,
    "median": _cyclops.NormalizationKind.MEDIAN,
    "q95": _cyclops.NormalizationKind.Q95,
}

NOISE_LEVELS: Mapping[str, "_cyclops.NoiseLevel"] = {
    "silent": _cyclops.NoiseLevel.SILENT,
    "quiet": _cyclops.NoiseLevel.QUIET,
    "noisy": _cyclops.NoiseLevel.NOISY,
}

PRECISIONS: Mapping[str, "_cyclops.Precision"] = {
    "fp64": _cyclops.Precision.FP64,
    "fp32": _cyclops.Precision.FP32,
    "double": _cyclops.Precision.FP64,
    "single": _cyclops.Precision.FP32,
}

_T = TypeVar("_T")


def resolve(table: Mapping[str, _T], value: "str | _T", what: str) -> _T:
    """Map a user-supplied string onto its enum member.

    Enum members pass through unchanged so callers can bypass the string layer.
    """
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "").replace("_", "")
        # Model-type codes keep their underscores ("clr_exact", "cox_time").
        for candidate in (value.strip().lower(), key):
            if candidate in table:
                return table[candidate]
        options = ", ".join(sorted(table))
        raise ValueError(f"Unknown {what} {value!r}; expected one of: {options}")
    if value in table.values():
        return value  # type: ignore[return-value]
    raise TypeError(f"Invalid {what}: {value!r}")
