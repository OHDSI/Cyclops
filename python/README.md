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
from cyclops import Control, CyclopsData, CyclopsModel, Prior

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 20))
y = (rng.random(1000) < 1 / (1 + np.exp(-X @ rng.normal(size=20)))).astype(float)

data = CyclopsData.from_arrays(X, y, "lr", add_intercept=True)
result = CyclopsModel(data, prior=Prior("laplace", variance=0.1)).fit()

result.coefficients      # intercept first, then one per column of X
result.covariate_ids     # [0, 1, ..., 20]; the intercept is id 0
result.log_likelihood
```

Sparse designs are passed through without densification, and
`variance="cv"` runs Cyclops' cross-validated hyperparameter search:

```python
import scipy.sparse as sp

X = sp.random(100_000, 50_000, density=1e-4, format="csc")
data = CyclopsData.from_arrays(X, y, "lr", add_intercept=True)
CyclopsModel(data, prior=Prior("laplace", variance="cv")).fit()
```

Very wide designs can be streamed in column-chunks instead of built from one
matrix:

```python
data = CyclopsData.create("cox")
data.set_outcome(y=event, time=time)
for chunk, chunk_ids in covariate_chunks:   # scipy.sparse.csc_matrix
    data.add_covariates(chunk, covariate_ids=chunk_ids)
data.finalize()

model = CyclopsModel(data, control=Control(max_iterations=2000, threads=4))
result = model.fit()
```

## Models

`model_type` is the first positional argument to `CyclopsData.from_arrays` /
`CyclopsData.create`.

| `model_type` | Model | Required arguments |
|---|---|---|
| `lr` | Logistic | — |
| `pr` | Poisson | `offset=` (optional) |
| `ls` | Least squares | — |
| `clr`, `clr_exact`, `clr_efron` | Conditional logistic | `strata=` |
| `cpr` | Conditional Poisson | `strata=` |
| `sccs` | Self-controlled case series | `strata=`, `offset=` |
| `cox`, `cox_raw`, `cox_time` | Cox proportional hazards | `time=` |
| `fgr` | Fine-Gray competing risks | `time=`, `set_censor_weights()` |

A scikit-learn-style estimator layer (`LogisticRegression`, `CoxRegression`
and friends, with `coef_`/`predict_proba`/`get_params`) is deferred to a
follow-up; it is a pure convenience wrapper over the two classes above.

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
