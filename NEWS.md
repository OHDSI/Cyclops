Cyclops v3.3.1
==============

Changes:

1. fix uninitialized value in detected in computeAsymptoticPrecisionMatrix(); value was priorType
2. fix memory leak caused by call to ::Rf_error()
3. fix line-endings on Makevar on windows

Cyclops v3.3.0
==============

Changes:

1. bump for R 4.2
2. fix CRAN warnings
    a. used `minValues`
3. fix CRAN notes
    a. remove explicit dependence on C++11 (except for R <= 4.0)   

Cyclops v3.2.0
==============

Changes:

1. fixed likelihood profiling when non-convex due to numerical instability
2. fixed parsing of 64-bit covariate IDs
3. fix BLR convergence criterion when there are 0 (survival) events
3. add Jeffrey's prior for single regression coefficient

Cyclops v3.1.2
==============

Changes:

1. adaptive likelihood profiling when objective function is concave.

Cyclops v3.1.1
==============

Changes:

1. specify new option `nocenter` in `survival` package when testing predicted hazard function

Cyclops v3.1.0
==============

Changes:

1. implement Fine-Gray competing risks regression
2. fixed `getCyclopsProfileLogLikelihood` when starting with extreme coefficients


Cyclops v3.0.0
==============

Changes:

1. switch to `Andromeda` from `ff` to hold large datasets.  This change breaks API

Cyclops v2.0.4
==============

Changes:

1. removed, unused variable imputation functions that contained a `std::exit`

Cyclops v2.0.3
==============

Changes:

1. fix computation under conditional Poisson models by reverting to v1.3.4-style loop
2. fix several unit-tests for compatibility with `R 4.0` factors
3. add ability to profile likelihood function in parallel
4. add initial infrastructure for competing risks models

Cyclops v2.0.2
==============

Changes:

1. use `RNGversion("3.5.0")` in unit-tests to reproduce old RNG behavior
2. fix prior-type checks when specifying multiple types

Cyclops 2.0.1
=============

Changes:

1. patch two memory leaks in `ModelData.cpp` and `ModelSpecifics.hpp`

Cyclops 2.0.0
=============

Changes:

1. simplify internal transformation-reductions loops
2. implemented non-negative weights
3. allow for run-time selection of 32-/64-bit reals
4. remove dependence on GNUmake
5. temporarily remove dependence on `RcppParallel` (until TBB is again R-compliant)

Cyclops 1.3.4
=============

Changes:

1. fix undeclared dependencies in unit-tests: `MASS` and `microbenchmarks`
2. fix issues with ATLAS compilation
3. add contexts to testthat files
4. fix ASAN errors in `AbstractModelSpecifics`

Cyclops 1.3.3
=============

Changes:

1. fix testthat expected error message

Cyclops 1.3.2
=============

Changes:

1. explicitly includes `<complex>` header, needed for `R` 3.5 builds
2. remove `pragma` statements used to quiet `RcppEigen` and `RcppParallel`

Cyclops 1.3.1
=============

Changes:

1. fixes covariate indices returned from `.checkCovariates` when excluding covariates from regularization

Cyclops 1.3.0
=============

Changes:

1. implements specialized priors through callbacks for use, for example, in the BrokenAdaptiveRidge package to provide L0-based model selection
2. implements specialized control through callbacks for use, for example, auto-and-grid-based cross-validation hyperparameter searches
3. removes src/boost that clashes with BH 1.65.0

Cyclops 1.2.3
=============

Changes:

1. fixed `predict` error with `ff.data.frame` with size == 0

Cyclops 1.2.2
=============

Changes:

1. fixed `solaris` build errors
2. added compatibility for C++14 (make_unique)
3. fixed multiple ASan warnings

Cyclops 1.2.0
=============

Changes: initial submission to CRAN
