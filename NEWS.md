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
13. `createControl(fitter = )` chooses the fitting code. `"default"` runs the current code. `"legacy"`
    runs the code of Cyclops 3.7, to reproduce earlier results: the `cox` engine, the `exp(xBeta)`
    updates of `lr`, `cpr`, `sccs`, `clr`, `clr_efron` and `clr_exact` (items 16 and 24), held-out rows
    in `lr` cross-validation (item 25), the cross-validation score of `lr` (item 26), and the
    profile-likelihood and `confint()` searches (items 22 and 23; their values then depend on `threads`
    again). The bug fixes in this release apply to every fitter. `"super"` runs the code of
    `"default"` and changes how `lr` models are fit (item 33). `cox_counting` models need a fitter
    other than `"legacy"`
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
24. `lr` fits are 1.3-2.4x faster per cycle, with the same number of cycles: an indicator column
    updates `exp(xBeta)` by one `exp()` factor, the gradient is computed from `exp(xBeta)` alone, and
    continuous columns use tables of `exp()` factors (few distinct values) or a vectorizable `exp()`.
    Coefficients change in the last bits (by about 1e-13). Denominators are set to `1 + exp(xBeta)`
    instead of being updated by differences, which drifted after large moves (such as a distant
    starting value) to a wrong optimum reported as `SUCCESS`, and rows with `xBeta` above about 709
    no longer give `NaN`
25. `lr` cross-validation fold fits leave out their held-out rows (they processed them with weight
    0). Together with item 24, cross-validated `lr` fits are 3.3-3.7x faster in benchmarks with
    20,000-100,000 rows and 4,000-13,000 covariates (8x where `threads > 1` also met the problem of
    item 27). Fold results change in the last bits
26. `lr` cross-validation scores held-out rows by their Bernoulli log-likelihood. It counted only the
    held-out cases (`y log p`), leaving out the controls, so the selected variance changes (by -1.2% to
    +3.0% in benchmarks with propensity-score data). `getCyclopsPredictiveLogLikelihood()` changes in
    the same way. `fitter = "legacy"` keeps the earlier score
27. cross-validation with `threads > 1` gives the same results as with one thread when
    `resetCoefficients = TRUE`: the model copies of other threads started the intercept at 0 and let
    `fixedCoefficients` vary. With `resetCoefficients = TRUE`, folds also go to threads as they become
    free (10 folds on 8 threads ran on 5)
28. `threads > 1` no longer slows down fits of models other than `cox` and `cox_counting` (by 8-60%,
    and several-fold for fits of a few cycles, such as the final fit after cross-validation)
29. cross-validation compares hyperparameters over the folds computed at all of them, and warns when
    fold fits fail. A hyperparameter at which no fold fit converged ends the search at the best one
    tried; the final model was fit at a variance of `NaN`. When no fold fit converges at all,
    `fitCyclopsModel()` stops with an error. Fold fits that stop with `POOR_BLR_STEP` continue under
    the Lange criterion, and when the final fit stops with `POOR_BLR_STEP`, only the final fit is
    repeated (the whole cross-validation was)
30. single-precision fits sum the convergence objective and the log-likelihood in double precision
31. a coefficient whose steps of at least 1 keep reversing without shrinking (seen with nearly
    saturated rows) has its step bound reduced, so the fit converges instead of running to
    `maxIterations`
32. `convertToCyclopsData(checkRowIds = FALSE)` stops for covariate rowIds that are not among the
    outcome rowIds (they were assigned to the first row), and `fixedCoefficients` given to one fit no
    longer stay fixed in later fits on the same data
33. `createControl(fitter = "super")` fits `lr` models to a stopping rule that measures the distance
    to the optimum, and cross-validates them faster; other models are fit as under `"default"`.
    `convergenceType`, `tolerance` and `maxIterations` now default to `NULL`: `"gradient"`, 1e-6 and
    1000 as before, or `"kkt"`, 1e-4 and 5000 for `lr` under `"super"`:
    - the new `convergenceType = "kkt"` stops when the largest minimum-norm subgradient of the log
      posterior, in standard-error units, is below `tolerance` (default 1e-4) over two full cycles and
      then at the point returned; `maxIterations` defaults to 5000 under it. The `"gradient"` criterion
      can stop at a turning point far from the optimum and report `SUCCESS`. Under `"super"`, a
      `tolerance` given without `convergenceType` is ignored with a warning, since `"kkt"` measures it
      in other units
    - cross-validation fold fits stop by `"kkt"` at 3e-3 and start from their own solution at the
      nearest variance tried; the auto-search steps down first from `startingVariance` and refines its
      result by up to three local quadratic fits
    - with 1000 or more covariates, the cycles between full sweeps skip coefficients that are 0 or
      barely moved
    - the intercept takes extra Newton steps during a cycle when they pay for their work and its
      gradient exceeds the tolerance. Updated once per cycle, it lags behind thousands of
      covariates; on high-prevalence data fits converge 5-19x faster (one that reached 5000 cycles
      now converges in 290)
    - each of these settings is also a new `createControl()` argument, which applies under every
      fitter and to every model when given: `cvConvergenceType`, `cvTolerance`, `cvMaxIterations`,
      `cvMinIterations`, `cvSearchDirection`, `cvRefine`, `cvWarmStart`, `activeSetEvery`,
      `activeSetThreshold` and `adaptiveIntercept`. Their default, `NULL`, leaves the choice to the
      fitter
    - in benchmarks with propensity-score data (20,000-100,000 rows, 4,000-13,000 covariates),
      cross-validation as `CohortMethod::createPs()` runs it is about 11-14x faster than under
      `"default"` (2.6x when the optimum lies between the first two variances tried), as
      PatientLevelPrediction runs it 2-5x, single fits 3-5x, and the selected variance lies closer
      to the optimum of the cross-validation curve (within 3-17% against 30-46% below it)
34. cross-validation with `resetCoefficients = FALSE` (the default, which PatientLevelPrediction
    uses) starts each fold fit from that fold's solution at the nearest variance already tried
    (`cvWarmStart` now defaults to `TRUE` there), so results no longer depend on `threads`: each
    model copy continued from whichever fold fit it ran last. Results change within the convergence
    tolerance, also with one thread, at about the same speed. `fitter = "legacy"` keeps the
    earlier behaviour
35. correct `vcov()` and `getSEs()` for `cox` models fit with `fitter = "legacy"` and for `cox_time`
    models (their Fisher information summed each event-time group as its own stratum, so the
    variances were far too large or not positive definite)
36. errors raised inside the C++ code report their message (several reported "c++ exception (unknown
    reason)", e.g. `vcov()` after a fit with weights)
37. `clr_exact` fits and Schoenfeld residuals of `cox` fits with `fitter = "legacy"` no longer read past
    the end of a sparse or indicator column
38. correct `vcov()` and `getSEs()` for `fgr` models: the Fisher information now sums over the
    Fine-Gray risk sets, including subjects with an earlier competing event weighted by the ratio
    of censoring weights (it returned meaningless values)
39. stratified `fgr` models keep each subject with a competing event in the risk sets of its own
    stratum only; the backwards scans of the denominator, gradient and Fisher information carried
    them into other strata, biasing estimates and standard errors
40. `finalizeSqlCyclopsData(useOffsetCovariate = )` removes the offset's name from the coefficient
    names. It removed the name at the position equal to the offset's covariate id, so `coef()`
    mislabeled coefficients when ids were not positions, as after `convertToCyclopsData()` with an
    intercept
41. `getSEs()` is exported. It names each standard error by its covariate; it took covariate ids for
    positions, so after `convertToCyclopsData()` the names were shifted or `NA`
42. all functions name coefficients the same way, in one place: by the names loaded with the data,
    else by covariate id with `"(Intercept)"` for id 0, as `coef()` did. For data loaded without
    names, `vcov()`, `confint()`, `getSEs()`, `summary()` and the univariable functions now return
    names, and covariates can be given by these names. `gradient()` also works for a fit that did not
    converge
43. `lr` fits with a Laplace prior skip the visits to an indicator column at zero whose step is
    provably zero: a bound on how far its gradient can have moved since it was last computed shows
    that it is still within the penalty (#100). Results are unchanged; with `fitter = "super"`,
    intercept steps are applied to the rows once per cycle, which changes results only by rounding. In
    benchmarks with 20,000-400,000 rows, L1 fits are 1.3-2.5x faster and cross-validation 1.1-2.4x;
    with 1.6 million rows, 1.0-1.1x

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
