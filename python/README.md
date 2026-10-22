# Cyclops for Python

Python bindings for [OHDSI Cyclops](https://github.com/OHDSI/Cyclops) — cyclic
coordinate descent for large-scale regularized regression on observational
healthcare data.

The numerics are the same C++ implementation the R package uses. This is a
binding, not a reimplementation.

## Why

`scikit-learn` covers penalized logistic regression well, but not the models that
OHDSI studies actually need: conditional (stratified) logistic regression, Cox
proportional hazards with hundreds of thousands of sparse covariates,
self-controlled case series, Fine-Gray competing risks. Cyclops does those, at
that scale, with cross-validated regularization built in.

## Install

Not on PyPI yet — install from GitHub. This builds the C++ core, so it needs a
C++17 compiler and takes a few minutes:

```bash
pip install "git+https://github.com/OHDSI/Cyclops.git#subdirectory=python"
```

Needs a C++17 compiler and CMake >= 3.19. Eigen is used if installed
(`apt install libeigen3-dev`, `brew install eigen`); otherwise CMake downloads a
pinned copy, so the first build needs network access. Linux and macOS only.

## Quick start

```python
import numpy as np
from cyclops import LogisticRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 20))
y = (rng.random(1000) < 1 / (1 + np.exp(-X @ rng.normal(size=20)))).astype(float)

model = LogisticRegression(prior="laplace", prior_variance=0.1)
model.fit(X, y)

model.coef_          # (20,)
model.intercept_
model.predict_proba(X)[:, 1]
model.log_likelihood_
```

Sparse designs are passed through without densification:

```python
import scipy.sparse as sp

X = sp.random(100_000, 50_000, density=1e-4, format="csc")
LogisticRegression(prior="laplace", prior_variance="cv").fit(X, y)
```

`prior_variance="cv"` runs Cyclops' cross-validated hyperparameter search
(auto-search by default, or a fixed grid).

## Models

| Class | Cyclops model | Extra `fit` arguments |
|---|---|---|
| `LogisticRegression` | `lr` | — |
| `PoissonRegression` | `pr` | `offset=` |
| `LinearRegression` | `ls` | — |
| `ConditionalLogisticRegression` | `clr` | `strata=` |
| `ConditionalPoissonRegression` | `cpr` | `strata=` |
| `SelfControlledCaseSeries` | `sccs` | `strata=`, `offset=` |
| `CoxRegression` | `cox` | `time=`, `event=` |
| `FineGrayRegression` | `fgr` | `time=`, `event=`, `censor_weights=` |

## Lower-level API

The estimator classes are a convenience layer over a thin wrapper of the C++
facade. For streaming very wide designs, or for control the estimators do not
expose, use it directly:

```python
from cyclops.data import CyclopsData
from cyclops.model import CyclopsModel, FitOptions, Prior

data = CyclopsData.create("cox")
data.set_outcome(y=event, time=time, strata=None)
for chunk in covariate_chunks:            # scipy.sparse.csc_matrix
    data.add_covariates(chunk, covariate_ids=chunk_ids)
data.finalize()

model = CyclopsModel(data)
model.set_prior(Prior("laplace", variance=0.1))
result = model.fit(FitOptions(max_iterations=2000, threads=4))
```

## Testing

Two independent layers of validation:

- **`tests/test_r_suite_port.py`** ports the R testthat suite's comparisons
  against *external* gold standards — `glm`, `lm`, `coxph`, `clogit` and `gnm`.
  These share no code with Cyclops, so they catch errors that Python-vs-R
  agreement cannot. The expected values are pinned literals;
  `tests/data/generate_fixtures.R` regenerates them.
- **`tests/test_parity_with_r.py`** fits identical data through the R package and
  compares coefficients, log likelihood, convergence flag, iteration count,
  predictions and standard errors at ~1e-10.

```bash
pip install -e ".[test]"
pytest tests/ -m "not parity"      # no R required
pytest tests/ -m parity            # needs R with Cyclops and jsonlite
```

## Layout

- `src/cyclops/` — the Python package
- `src/bindings.cpp` — pybind11 bindings over the C++ facade in `../src/cyclops/api/`
- `tests/` — unit tests, plus R-parity tests behind the `parity` marker
