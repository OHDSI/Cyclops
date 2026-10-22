"""Cyclops: cyclic coordinate descent for large-scale regularized regression.

Python bindings for the OHDSI Cyclops C++ engine. Fits logistic, Poisson,
least-squares, conditional (stratified) and survival regressions with L1/L2
regularization on sparse designs with hundreds of thousands of covariates.

The API is the data/model pair — :class:`~cyclops.data.CyclopsData` holds the
outcome and design matrix, :class:`~cyclops.model.CyclopsModel` configures and
runs the optimizer::

    from cyclops import CyclopsData, CyclopsModel, Prior

    data = CyclopsData.from_arrays(X, y, "lr", add_intercept=True)
    result = CyclopsModel(data, prior=Prior("laplace", variance=0.1)).fit()
    result.coefficients
"""

from __future__ import annotations

from cyclops import _cyclops
from cyclops.data import INTERCEPT_ID, CyclopsData
from cyclops.model import Control, CyclopsError, CyclopsModel, FitResult, Prior

__version__ = _cyclops.version()

#: Cyclops model-type codes accepted by :meth:`CyclopsData.create`.
MODEL_TYPES = (
    "ls", "pr", "lr", "clr", "clr_exact", "clr_efron", "cpr",
    "sccs", "cox", "cox_raw", "cox_time", "fgr",
)


def list_gpu_devices() -> list[str]:
    """GPU devices visible to this build; empty for the CPU-only wheels."""
    return _cyclops.list_gpu_devices()


__all__ = [
    "__version__",
    "MODEL_TYPES",
    "CyclopsData",
    "CyclopsModel",
    "Prior",
    "Control",
    "FitResult",
    "INTERCEPT_ID",
    "CyclopsError",
    "list_gpu_devices",
]
