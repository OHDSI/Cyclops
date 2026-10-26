Cyclops v3.8.0
==============

Changes:

1. add `modelType = "cox_counting"` for counting-process data, `Surv(start, stop, event)` or
   `convertToCyclopsData()` with `outcomes$startTime` (and the end as `time` or `endTime`): delayed
   entry (left truncation) and time-varying covariates, with Breslow ties, strata, weights,
   profile-likelihood confidence intervals, `vcov()`, Schoenfeld residuals, `testProportionality()`
   and `survfit()`
2. `cox` fits use one risk-set group per event time, range-limited denominator updates and
   prefix-sum gradients: 60-700x faster in benchmarks with 200,000 rows, with results that differ
   from the previous version only in the last bits
3. `cox` requires start times of 0 (use `cox_counting` for other start times)
4. correct `vcov()` for `cox` models
5. keep all rows of a subject together in cross-validation and bootstrap: `cluster()` in the
   formula or `outcomes$subjectId`, and `createControl(selectorType = "bySubject")`
6. `survfit()` for `cox_counting` returns one curve per stratum
7. `createControl(threads = )` also parallelizes Cox fits (results identical to one thread)
8. single-precision Cox fits use only single-precision arithmetic, and are more accurate
9. `residuals()` returns Schoenfeld residuals for several covariates (a matrix, or a subset through
   `parm`) and scaled Schoenfeld residuals (`type = "scaledsch"`); `type` is now the second
   argument, as in `residuals.coxph()`
10. `testProportionality()` leaves rows with zero weight out of the `km` and `rank` transforms, as
    `cox.zph()` does for a fit on the rows with positive weight
11. the final fit after cross-validation uses the weights given to `fitCyclopsModel()` (it used all
    rows with weight 1; `createWeightBasedSearchControl()` still refits on all rows), and
    `runBootstrap()` leaves the weights and coefficients of the fit (it left those of the last
    replicate)
12. `runBootstrap()` resamples only rows (or strata, or subjects) with non-zero weight; it also drew
    held-out ones, so each replicate had a random number of draws from the fit
13. `createControl(fitter = "legacy")` fits `cox` models with the engine of Cyclops 3.7, to reproduce
    earlier results; the default, `"efficient"`, is the new engine
14. `cpr`, `sccs` and `clr` fits are about 2.3x faster, with identical results: per-stratum numerators
    that only Cox models read are no longer filled, the gradient and the `xBeta` update walk each
    column's entries one stratum at a time, and the convergence check of every model computes the
    log-likelihood and prior only when it prints them
15. single-precision (`floatingPoint = 32`) `cpr`, `sccs` and `clr` fits no longer leave out the last
    entries of a column with more than 2^24 (about 16.8 million) entries
16. `cpr`, `sccs` and `clr` fits update `exp(xBeta)` with one `exp()` per indicator column, or one per
    distinct value of a sparse column with few values (such as spline bases), instead of one per
    entry: about 3.5x faster than Cyclops 3.7 together with item 14. Results change in the last bits
    (coefficients by about 1e-11 relative, with the same number of iterations); single-precision fits
    move by less than their distance from double precision and can take a different number of
    iterations
17. `vcov()`, `getSEs()` and `aconfint()` for `cpr`, `sccs` and `clr` models weight each stratum by
    its number of events, as the likelihood does. Standard errors were too large with several events
    per stratum (by about the square root of the mean number) and too small when some strata had no
    events. `vcov()` after refitting the same data also reused stratum terms from the earlier fit
18. `cpr`, `sccs` and `clr` fits re-sum the stratum denominators when coefficients have moved far.
    Updating them by differences lost all precision when a denominator shrank by many orders of
    magnitude (for example, age and calendar-time splines with one outcome per case, or a distant
    starting value), giving wrong estimates or `ILLCONDITIONED`
19. a fit no longer reports convergence after a cycle in which a step was cut short by its bound.
    After `ILLCONDITIONED` retries shrank `initialBound`, the first cycle barely moved, and the fit
    returned `SUCCESS` with coefficients near their starting values
20. `ILLCONDITIONED` retries keep fixed coefficients at their values, including the one held by
    `confint()` or `getCyclopsProfileLogLikelihood()`. The retry reset it to its starting value, so
    a profile point could report the likelihood at 0 instead. For an exposure without events,
    `confint()` returned an upper bound next to the estimate (about -157 instead of -5)
21. `getCyclopsProfileLogLikelihood(maxDrop = )` refines the adaptive profile only where it lies
    within `maxDrop` of its maximum; the default, `Inf`, refines everywhere as before. With wide
    bounds, `maxDrop = 50` saves most evaluations and leaves the profile accurate to `tolerance`
    within 50 of its maximum, as evidence synthesis needs
22. `getCyclopsProfileLogLikelihood()` starts the fit at each point from the solution at the
    nearest point already evaluated and no longer refits at a coefficient of 0 first, so adaptive
    profiles need far fewer iterations. Results no longer depend on `threads`, and differ from the
    previous version only within the convergence tolerance. With more than one thread, profiles also
    keep `fixedCoefficients` fixed (they let them vary)
23. `confint()` starts each bound at its Wald estimate and takes safeguarded Newton steps on the
    profile likelihood, falling back to the previous bracketing search; it needs 2-6 times fewer
    fits, and bounds move by less than 1e-3. The fallback also covers regularized covariates with
    `includePenalty = TRUE`, any covariate with `includePenalty = FALSE` when others are
    regularized, and nearly flat profiles. Each bound starts from the estimate, so bounds no longer
    depend on their order or on `threads`; with more than one thread, `fixedCoefficients` also stay
    fixed
24. `lr` fits keep the fitted probability of each row: a coefficient's gradient and Hessian are sums of
    probabilities over its column, and an update scales the odds of the rows in its column, with one
    `exp()` per indicator column. Estimates agree with the engine of Cyclops 3.7 up to rounding. On
    simulated data with 25,000 to 1.6 million rows and 2,000 to 32,000 binary covariates, a single
    L1 fit is 1.7-3.5x faster and automatic cross-validation 2.1-4.1x, with the same iterations and
    log-likelihood. `createControl(fitter = "legacy")` keeps the engine of Cyclops 3.7 for `lr`
    models too. GPU fits are unchanged

Cyclops v3.7.1
==============

Changes:

1. more robust unit-test for bootstrapping

Cyclops v3.7.0
==============

Changes:

1. implement Schoenfeld residuals for Cox models with ties (Breslow) and strata
2. implement score-test of Cox proportionality for models with ties (Breslow) and strata
3. fix `covariate` type in `confint()` output 
4. remove use of `Rf_error()` that can cause memory leaks

Cyclops v3.6.0
==============

Changes:

1. resurrect boot-strapping
2. add JNI interface for manipulating Cyclops objects in an across-language persistent cache
3. add `cyclopsGetLogLikelihoodGradient()`

Cyclops v3.5.2
==============

1. fix cyclic dependency with `SelfControlledCaseSeries` in a test-unit


Cyclops v3.5.1
==============

Changes:

1. check for negative curvature before computing CIs
   a. change to "lange"-convergence when needed
2. fix `vcov` when model has an offset
3. fix profiling when in bad initial state
4. sort output of `predict` for compatibility with `Andromeda:duckbd`

Cyclops v3.5.0
==============

Changes:

1. provide optional (`optimalWarmStart = FALSE`) more parallelization when profiling likelihood
2. make `maxResets` a function parameter
3. ensure compatibility with R v4.4

Cyclops v3.4.1
==============

Changes:

1. fix some NaN/-Inf inconsistencies with likelihood profiling

Cyclops v3.4.0
==============

Changes:

1. remove dependence on `BH`
2. improvements on adaptive likelihood profiling
3. add `auto` option to `cvRepetitions`
4. bumped explicit C++11 requirement up to R v4.1
5. removed deprecated use of `dbplyr:::$.tbl_lazy`
   a. breaking change in `dbplyr v2.4.0`

Cyclops v3.3.1
==============

Changes:

1. fix uninitialized value in detected in `computeAsymptoticPrecisionMatrix()`; value was priorType
2. fix memory leak caused by call to `::Rf_error()`
3. fix line-endings on Makevar on windows

Cyclops v3.3.0
==============

1. bump for R 4.2
2. fix CRAN warnings
    a. used `minValues`
3. fix CRAN notes
    a. remove explicit dependence on C++11 (except for R <= 4.0)   

Cyclops v3.2.1
==============

Changes:

1. fix small memory leak caused by direct call to '::Rf_error()'
2. disable JVM calls on CRAN due to uninitialized memory in Java JVM


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
