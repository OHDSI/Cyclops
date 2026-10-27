"""Shared fixtures: synthetic datasets, a fit helper, and the R availability gate."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from cyclops import CyclopsData, CyclopsModel

sys.path.insert(0, str(Path(__file__).parent))

from r_bridge import r_cyclops_available


def fit(
    X,
    y,
    model_type,
    *,
    prior=None,
    control=None,
    strata=None,
    time=None,
    offset=None,
    add_intercept=False,
    covariate_ids=None,
    weights=None,
    censor_weights=None,
    start_values=None,
):
    """Build a `CyclopsData`, fit it, and hand back all three objects.

    Test-only scaffolding over the two-step `CyclopsData` / `CyclopsModel` API,
    so a test that only cares about the numbers is one line. The weight vectors
    are permuted by `row_order` because `from_arrays` re-sorts rows for the
    conditional and survival models, while the caller supplies them in input
    order.

    Returns:
        (data, model, result)
    """
    data = CyclopsData.from_arrays(
        X,
        y,
        model_type,
        time=time,
        strata=strata,
        offset=offset,
        covariate_ids=covariate_ids,
        add_intercept=add_intercept,
    )
    model = CyclopsModel(data, prior=prior, control=control)

    order = data.row_order
    for vector, setter in (
        (weights, model.set_weights),
        (censor_weights, model.set_censor_weights),
    ):
        if vector is not None:
            vector = np.ascontiguousarray(vector, dtype=np.float64)
            setter(vector if order is None else vector[order])

    if start_values is not None:
        model.set_start_values(np.ascontiguousarray(start_values, dtype=np.float64))

    return data, model, model.fit()


def pytest_collection_modifyitems(config, items):
    """Skip `parity` tests when R or the R Cyclops package is unavailable."""
    if any(item.get_closest_marker("parity") for item in items) and not (
        r_cyclops_available()
    ):
        skip = pytest.mark.skip(
            reason="R with the Cyclops and jsonlite packages is required"
        )
        for item in items:
            if item.get_closest_marker("parity"):
                item.add_marker(skip)


@dataclass
class Dataset:
    """A synthetic problem, plus whatever auxiliary vectors its model needs."""

    X: np.ndarray
    y: np.ndarray
    beta: np.ndarray
    strata: np.ndarray
    time: np.ndarray
    offset: np.ndarray
    counts: np.ndarray
    continuous: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def _make_dataset(n_samples: int, n_features: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = rng.normal(scale=0.5, size=n_features)
    linear = 0.25 + X @ beta

    strata = np.repeat(np.arange(n_samples // 5), 5)[:n_samples]
    if strata.shape[0] < n_samples:  # n_samples not divisible by 5
        strata = np.concatenate(
            [strata, np.full(n_samples - strata.shape[0], strata[-1])]
        )

    offset = rng.uniform(0.5, 3.0, size=n_samples)
    return Dataset(
        X=X,
        y=(rng.random(n_samples) < 1.0 / (1.0 + np.exp(-linear))).astype(np.float64),
        beta=beta,
        strata=strata.astype(np.int64),
        # Distinct times: ties are handled differently by Breslow vs Efron and
        # would make an exact cross-implementation comparison ambiguous.
        time=np.sort(rng.permutation(np.arange(1, n_samples + 1).astype(np.float64))
                     + rng.uniform(0, 0.5, size=n_samples))[::-1].copy(),
        offset=offset,
        counts=rng.poisson(np.exp(linear) * offset).astype(np.float64),
        continuous=linear + rng.normal(scale=0.3, size=n_samples),
    )


@pytest.fixture(scope="session")
def small() -> Dataset:
    """200 x 4 — fast enough to use in most tests."""
    return _make_dataset(200, 4, seed=20240730)


@pytest.fixture(scope="session")
def medium() -> Dataset:
    """1000 x 12 — exercises cross-validation and multi-stratum paths."""
    return _make_dataset(1000, 12, seed=17)


@pytest.fixture(scope="session")
def sparse_binary():
    """A wide sparse indicator design — the shape Cyclops is built for.

    The first 10 columns carry signal and are made dense enough (~15% of rows)
    that recovery is actually identifiable; the remaining 290 are noise at 2%
    density. A uniformly sparse design at 2% gives each column only ~16
    informative rows, which is not enough signal for any estimator to find.
    """
    rng = np.random.default_rng(99)
    n_samples, n_features, n_signal = 800, 300, 10

    columns = []
    for index in range(n_features):
        density = 0.15 if index < n_signal else 0.02
        count = max(1, round(density * n_samples))
        rows = rng.choice(n_samples, size=count, replace=False)
        column = np.zeros(n_samples)
        column[rows] = 1.0
        columns.append(column)
    X = sp.csc_matrix(np.column_stack(columns))
    X.sort_indices()

    signal = np.zeros(n_features)
    signal[:n_signal] = rng.choice([-1.5, 1.5], size=n_signal)
    probability = 1.0 / (1.0 + np.exp(-(X @ signal - 1.0)))
    y = (rng.random(n_samples) < probability).astype(np.float64)
    return X, y, signal
