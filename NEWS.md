Cyclops v4.0.2 (development version)
====================================

Changes:

1. `vcov()`, `getSEs()` and the Fisher information of `cox` and `cox_counting` fits come from one
   Hessian computation instead of one pass over the data per pair of covariates: with 571 covariates and
   13,152 rows, 24 s becomes 0.04 s, with results equal to 1e-13 (relative)
2. `vcov()` and `getSEs()` invert the information through a pivoted Cholesky factorization. When the
   information is singular (aliased covariates, such as exact duplicates), they return `NA` for the
   aliased coefficients, with a warning, instead of an error from `solve()`; the other entries are
   unchanged. Results of identified models differ from earlier versions only by rounding
3. the Fisher information of `cox` and `cox_counting` fits with integer case weights (including 0/1
   weights) no longer stops with "Weights are not yet implemented"; non-integer weights still do
4. `createControl(algorithm = "auto")` fits `cox` and `cox_counting` models with normal or no priors by
   coordinate descent that switches to dense Newton steps when an operation count says they are
   cheaper, and back to coordinate descent if they fail; `algorithm = "newton"` takes Newton steps after
   three coordinate-descent cycles. With 571 covariates and 13,152 rows (`cox_counting`), a ridge fit takes
   0.14 s instead of 1.3 s and reaches the optimum, where the default stopped 0.013 log-likelihood units
   short (`survival::coxph()`: 63 s). On large designs where Newton steps cost more than coordinate
   descent, `"auto"` stays with coordinate descent
5. `createControl(convergenceType = "gap")` stops when a certified bound on the distance to the optimum
   of the log posterior falls below `tolerance` (default 1e-6, scaled by the mean positive weight; 1e-5
   for cross-validation folds). It is the default under `algorithm = "auto"` or `"newton"`. Fits report
   the path and the bound in `fit$algorithm` and `fit$convergence`, and a fit that reaches `maxIterations`
   warns with its estimated gap
6. `fitter = "super"` fits `cox` and `cox_counting` models with `algorithm = "auto"`. Under the default
   fitter nothing changes, and an explicit `convergenceType` other than `"gap"` keeps `"auto"` on
   coordinate descent, with the results of earlier versions. A fit that cannot take Newton steps stops
   with an error under `"newton"` or an explicit `"gap"`, and otherwise runs coordinate descent with the
   criterion it would have had
7. a `"ccd"` fit after an `"mm"` fit on the same data object runs coordinate descent (it kept running
   `"mm"`)
8. separation: an unpenalized coefficient whose likelihood is monotone in it alone (`cox` and `cox_counting`:
   every event at its risk set's minimum or maximum of the covariate; `cpr`, `sccs` and `clr`: likewise
   within each stratum; `lr`: every row with a non-zero value has the same outcome, by sign) is detected
   before fitting and held at its limit while the other coefficients are fit, which then equal those of
   the limiting model. With no prior on any coefficient the fit is `ILLCONDITIONED`; otherwise it is
   `SUCCESS`. Either way the divergent coefficients stay at their limit (so the log likelihood, profiles
   and the other coefficients' `confint()` are those of the limiting model), the fit warns, and
   `fit$divergentCoefficients` names them; `confint()` returns `NA` for them in an `ILLCONDITIONED` fit.
   Earlier versions returned `SUCCESS` with large finite values or `ILLCONDITIONED` with `NaN`, depending
   on the data and the convergence criterion, so results of separated fits change; code that found
   separated coefficients by `is.nan(coef(fit, ignoreConvergence = TRUE))` should read
   `fit$divergentCoefficients`
9. Newton steps on `cox` and `cox_counting` designs solve with conjugate gradients (Newton-CG) when an
   operation count says that is cheaper than a dense factorization, or when the dense matrices of all
   concurrent fits would exceed `createControl(newtonMemoryBudget = )` (1024 MB by default, see 38). A
   rank-32 Woodbury preconditioner (the row part of the Hessian
   minus a binned approximation of its risk-set part) cuts the conjugate-gradient work 8-16 fold, and the
   stop is certified from an explicit-residual bracket. With 3,068 covariates and 13,152 rows a ridge fit
   to a gap of 1e-8 takes 1.5 s (dense Newton: 7.5 s; coordinate descent did not get there in 25,000
   cycles), and with 2,016 covariates 0.52 s (dense: 2.5 s)
10. the Hessian of `cox` and `cox_counting` models (used by `vcov()`, `getSEs()` and Newton steps) is
   computed on `createControl(threads = )` threads, with results that do not depend on their number
11. `createControl(mergeDuplicates = )` fits each group of exactly duplicated columns (same rows and values,
   same prior) as one coefficient: on duplicated designs coordinate descent no longer crawls along the
   flat direction between duplicates (571 distinct of 597 columns: 551 cycles to the `"kkt"` tolerance
   instead of more than 30,000). Normal priors report the equal split of the group's total, the unique
   maximum; Laplace and no priors put the total on the group's first column. `fit$aliasedColumns` names
   the groups. It is on by default with `algorithm = "auto"` or `"newton"`, and off otherwise
12. `algorithm = "auto"` and `"newton"` also fit `cox` and `cox_counting` models with Laplace priors, by
   prox-Newton: Newton steps on the working set of nonzero, violating and unpenalized coefficients, with
   the L1 model minimized by coordinate descent and then exactly on its signed active face. With 597
   covariates (13,152 rows) a Laplace fit to its optimum takes 0.14-0.35 s; the `"kkt"` criterion on
   coordinate descent took 3.4-94 s to a looser tolerance. Its stop, an exact model decrease below
   `tolerance`, is an estimate rather than a certified bound (`fit$convergence$certified` is `FALSE`)
13. `algorithm = "auto"` and `"newton"` also fit `cpr`, `sccs` and `clr` models (normal, Laplace or no
   priors). The SCCS design of the package vignette (20,000 cases, 1.18 million rows) fits in 0.6 s
   instead of 5.5 s and reaches the optimum, and designs with age and calendar-time splines, where
   coordinate descent reached its iteration limit in 63 of 66 cases, all converge in under 0.3 s.
   Designs where columns are constant within every stratum are fit to their minimum-norm solution
14. under `"auto"` and `"newton"`, profile likelihoods and `confint()` refit each point from the Hessian at
   the estimate (chord Newton steps, whose first step is the tangent of the profile), with a certificate
   that allows for the Hessian's lag; with SelfControlledCaseSeries' defaults, a fit with its profiles,
   confidence intervals and null likelihoods takes 1.8 s instead of 12.9 s
15. `algorithm = "auto"` and `"newton"` also fit `lr` models (normal, Laplace or no priors), by dense
   Newton, Newton-CG or prox-Newton, in cross-validation folds too. A propensity-score model with 1,755
   covariates and 197,753 rows reaches its optimum in 2.8-3.1 s with a normal prior and 1.5-3.3 s with
   a Laplace prior, where coordinate descent stopped 15-24 log-likelihood units short of a ridge optimum
   after 10-31 s; one with 19,591 covariates and a million rows reaches its Laplace optimum in 17 s, where
   coordinate descent stopped 21 units short after 35 s. `vcov()` of an `lr` fit with integer weights now treats them as row counts (it
   stopped with an error)
16. **The default algorithm is now `"auto"`** (with the `"gap"` criterion) for `cox`, `cox_counting`,
   `cpr`, `sccs`, `clr` and `lr` models. Across 57 benchmark problems it reached the optimum every time,
   where the earlier default stopped more than 1e-3 short of it on 31 (by up to 1,322 log-likelihood
   units) and reached its iteration limit on 5; it was a median 7.1 times faster where the earlier default
   fell short, and 10-15% slower where that default was already accurate and fast. Results therefore
   change, and cross-validated variances can move where earlier fold fits were not converged.
   `createControl(algorithm = "ccd")` reproduces earlier results exactly. Fits under `fitter = "legacy"`,
   other models, and the refits of priors with a `fitHook` (BrokenAdaptiveRidge,
   IterativeHardThresholding, `createNonSeparablePrior()`) keep coordinate descent
17. Newton-CG (used by `algorithm = "auto"` and `"newton"` when a dense Hessian is too large or costs more)
   certifies its `"gap"` stop when up to 32 coefficients have no prior, such as an unpenalized intercept,
   through a lower bound on the Hessian from the Schur complement of their block. Before, only fits with a
   normal prior on every coefficient could certify by Newton-CG, and with an unpenalized intercept and normal
   priors of very different variances the certification could not stop: on such a simulated logistic design
   with 3,000 rows `algorithm = "newton"` reached its iteration limit after 24 s, and now certifies in 0.1 s;
   with 13,334 rows and 18,933 covariates it had not finished after 25 minutes, and now certifies in 58 s.
   After a fallback to coordinate descent, `algorithm = "newton"` waits before its next Newton steps, as
   `"auto"` does. Fits with a normal prior on every coefficient do not change otherwise
18. `mergeDuplicates` (on by default under `algorithm = "auto"` and `"newton"`) compares columns with a prior on
   the rows with positive weight in logistic, Poisson and least-squares models, so columns that differ only on a
   cross-validation fold's held-out rows (or on rows given weight 0) are merged for that fit. In the folds of a real logistic design with 7,443 FeatureExtraction
   covariates, such pairs made every singular face of the Laplace fold fits: at variance 1 one fold took 1,218 s,
   and 164 s without them. Their total is reported as an equal split whatever the prior, so a fold's held-out
   predictions, and the cross-validation score, no longer depend on the order of the columns
19. Newton-CG builds a preconditioner once a fit's conjugate-gradient solves are expensive enough to repay it: a
   randomized Nystrom approximation of the likelihood Hessian (64 columns and the unpenalized coefficients) plus the
   prior and the likelihood curvature it misses, after Frangella, Tropp & Udell (2023), dropped again if its first solve
   costs more. Fits whose solves stay cheap are unchanged (`fit$convergence$preconditioner` says which ran). On a
   simulated design with 13,334 rows, an unpenalized intercept and normal variances spread over six orders of magnitude,
   `algorithm = "auto"` took 2.7 s instead of 30 s; ridge fits of real FeatureExtraction designs took 0.5-1.05 times as
   long. Certification solves now stop once the certificate holds under a preconditioner too (they ran to a tight
   residual)
20. Prox-Newton (Laplace and elastic-net priors) keeps the Cholesky factor of its face matrix across the rounds of a face
   solve, updating it as coordinates enter and leave, and stops its inner coordinate descent once the support has
   settled. A single Laplace fit of a real FeatureExtraction design (83k rows, 7,443 covariates) took 12.7 s instead
   of 23.0 s, and its 3-fold cross-validation 10.3 s instead of 14.7 s, at the same optimum. A face holding a column
   and a rescaled copy of it is now treated as singular. `fit$convergence$faceRounds` counts the face rounds
21. The logistic Hessian block that Newton and prox-Newton fits factor is built densest columns first, with contiguous
   writes, and on the fit's threads (`createControl(threads = )`); cross-validation gives each fold the threads beyond
   the folds it runs in parallel. Results are the same bit for bit for any number of threads. A single Laplace fit of
   a real FeatureExtraction design took 8.3 s instead of 11.0 s on one thread and 4.7 s instead of 9.7 s on four
22. L1 logistic regression (Laplace priors, at most the intercept unpenalized; fixed coefficients allowed) with the
   `"gap"` criterion is certified by its duality gap: `fit$convergence$certificate` is `"dual"` and `gapEstimate` the
   gap, a bound on the distance to the optimum. Such fits stopped on an estimate before, uncertified, and the
   estimate could read low: a real Laplace fit stopped with its objective 1.4e-6 from the optimum (1e-6 requested)
   where the estimate read 2.3e-9; it now takes two more Newton steps. The gap costs about one gradient and is
   computed only where the estimate would stop. A tolerance below what the gap can certify in floating point (about
   1e-14 relative) stops on the estimate as before
23. a fit without `weights` on a data object that an earlier fit had weighted is unweighted; it reused the
   earlier weights (also in 3.7.1), so it repeated the weighted fit
24. `createCyclopsData()` with outcome types (`type`, or `Multitype()` in the formula) of three or more levels
   gives each covariate one column per type; it merged two types and added empty columns, so the fit failed.
   A character `type`, or numeric codes other than 1 and 2, crashed R; `type` is now taken as a factor.
   Fits with two types do not change
25. `fitter = "super"` with `algorithm = "auto"` (the default) or `"newton"` fits its cross-validation folds as
   the final fit, with the `"gap"` criterion (fold tolerance 1e-5). They ran coordinate descent to a `"kkt"`
   tolerance of 3e-3, its setting for `algorithm = "ccd"`; on a real logistic design with 13,420 covariates 18
   of 30 fold fits reached the iteration limit, the variance search stopped early and chose a variance six times
   smaller, and the held-out AUC fell from 0.887 to 0.876. With this release's Newton changes the folds are also
   faster on real FeatureExtraction designs (3-fold super cross-validation 1.1-2.3x faster, the same variances).
   Under `algorithm = "ccd"` and with a given `cvConvergenceType` nothing changes
26. `createCyclopsData()` (a formula with an intercept), `finalizeSqlCyclopsData(addIntercept = TRUE)` and
   `add_intercept()` in the Python package store the intercept as an intercept column, as
   `convertToCyclopsData()` does, instead of a dense column of ones. The optimum is the same, but adaptive
   intercept steps now apply to such data too (logistic regression under `fitter = "super"` with
   `algorithm = "ccd"`, its cross-validation folds included, and least squares under `"default"` and
   `"super"`): a simulated sparse logistic design with 400 covariates took 21 iterations instead of 39, and
   over 269 such fits of simulated formula data the median fit took 2.6 times fewer iterations (never more).
   Asking for an intercept to be stored densely (`makeCovariatesDense`, or `make_dense()` in Python) leaves it
   as it is; on data from `convertToCyclopsData()` it crashed R
27. `fitCyclopsModel(cvFolds = )` cross-validates on given folds (a data frame of `rowId` and `fold`, or a vector in
   data order for formula data) instead of random ones, and `returnCvPredictions = TRUE` returns in
   `fit$cvPredictions` the out-of-fold prediction of each row, from a refit of its fold at the selected variance (for
   `"lr"`, `"pr"` and `"ls"`, on the scale of `predict()`; `NA` for rows with weight 0). The number of folds comes
   from `cvFolds` (a different `fold` given to `createControl()` is an error), and both need `cvRepetitions = 1`.
   PatientLevelPrediction refits each fold itself for these predictions
28. `convertToCyclopsData()` with no covariates (an intercept-only design) loads and fits; it crashed R (also in
   Cyclops 3.7.1)
29. subjects (`outcomes$subjectId` in `convertToCyclopsData()`, `cluster()` in a `createCyclopsData()` formula) work
   for every model type, not only `cox` and `cox_counting`: the rows of a subject stay in one cross-validation fold
   and are resampled together by `runBootstrap()` (`selectorType = "bySubject"`, which `"auto"` picks in place of
   `"byRow"` when the data have subjects). `convertToCyclopsData()` ignored `subjectId` for other models, and
   `cluster()` stopped. Fits of data with `subjectId`, such as `lr` designs with several rows per person, can change
30. new `evaluateLikelihood()`: the log-likelihood at given coefficients, with its gradient and the Fisher information
   (matrix or diagonal) of chosen covariates, under given weights (none: every row; never an earlier fit's weights).
   For packages that build estimators on Cyclops, such as IterativeHardThresholding, which used internal functions
31. the Fisher information of `pr` and `ls` models with weights (and of `lr` with weights that are not whole
   numbers) no longer stops with "Weights are not yet implemented"; `vcov()` and `getSEs()` of such fits invert the
   weighted information, as `glm()` does with prior weights (for `ls`, without a dispersion factor, as before)
32. the internal Fisher information after `.cyclopsSetBeta()` was computed at the previous coefficients
33. `fitCyclopsModel(returnCvPredictions = TRUE)` also works without a hyperparameter search: with a fixed prior or a
   prior with a `fitHook` (such as BrokenAdaptiveRidge's), each fold of `cvFolds` is fit on its training rows and
   predicts its held-out rows, next to the fit on all rows. A `fitHook` that names arguments `cvFolds` and
   `returnCvPredictions` gets them and returns the predictions itself. So PatientLevelPrediction no longer needs
   its own fold refits for any Cyclops model. A prior with `keepControl = TRUE` gives its `fitHook` the control as
   given; other hooks still get coordinate descent when no algorithm is set
34. `vcov()`, `getSEs()` and the Fisher information also accept non-integer case weights for `cox` and
   `cox_counting` fits (and `cpr`, `sccs` and `clr` under the default fitter): they return the inverse of the weighted
   information, as `coxph(weights = )$naive.var` (Breslow ties). `cox` fits under `fitter = "legacy"` still stop with
   "Weights are not yet implemented in Fisher Information calculations". `evaluateLikelihood()` likewise returns the
   weighted information, and stops only for a `cox` data object whose last fit used `fitter = "legacy"`
35. `vcov(fit, type = "robust")` returns the cluster-robust (sandwich) variance H^-1 (U'U) H^-1 of `lr`, `pr`, `cox` and
   `cox_counting` fits: H is the information of the log likelihood (the prior excluded), and each row of U sums the
   weighted score contributions of one subject (`cluster()` or `subjectId`; one row without subjects). Cox scores are
   the weighted score residuals with the risk-set compensator, for counting-process data and strata. Fixed
   coefficients are left out, and there is no small-sample factor: it matches `coxph(weights = , cluster = ,
   ties = "breslow")$var` and `sandwich::vcovCL(glm(weights = ), cluster = , type = "HC0", cadjust = FALSE)` to
   1e-6. `cox` fits under `fitter = "legacy"` are not supported. Under a prior it warns that it is not a validated
   variance of regularized coefficients, unless `overrideNoRegularization = TRUE`. `confint(fit, method = "robust")`
   gives the Wald intervals from it, and, as for profile intervals, stops for regularized parameters unless
   `overrideNoRegularization = TRUE`. The defaults (`type = "modelBased"`, `method = "profile"`) are unchanged
36. `convertToCyclopsData()` takes an `offset` column in the outcomes of `lr` models: an offset added to the linear
   predictor (log-odds scale), as `glm(offset = )`, held as a covariate with coefficient fixed at 1 (as the log time of
   `pr` models), so every algorithm, prior and cross-validation uses it. The data object comes back finalized. Before,
   an `lr` offset needed a covariate and `finalizeSqlCyclopsData(useOffsetCovariate = , offsetAlreadyOnLogScale = TRUE)`.
   `predict()` on new data adds the `offset` column of `newOutcomes`, if any. Offsets must be finite, and other model
   types warn that they ignore an `offset` column
37. new `modelType = "plr"` (pooled logistic regression) in `convertToCyclopsData()`: logistic regression over the
   intervals at risk of each subject without expanding the covariates to those rows. A row of the data is a subject;
   its intervals come as one outcome row per interval (explicit) or as `entry` and `exit` (implicit, see the new
   `convertToLongOutcome()`), with case weights and offsets per row or per subject. A `timeBasis` gives time
   covariates by interval, and `timeEffectMap` their interactions with covariates. Fits equal `glm(binomial())` and
   `"lr"` on the expanded rows; cross-validation folds and robust-variance clusters are subjects, or persons given
   as `outcomes$subjectId` (e.g. the clones of one person). `link = "cloglog"` fits the complementary log-log
   model (equal to `glm(binomial("cloglog"))` on the expanded rows), whose non-event rows factorize so a covariate
   update costs O(nonzeros). With the logit link the second-order kernels run `algorithm = "auto"` (the default, as
   for `"lr"`) and `"newton"` with certified `"gap"` stops (L1 by the duality gap); cloglog fits use coordinate
   descent, which on correlated time bases needs a tighter `tolerance` than the default. The cloglog `vcov()` is
   the observed information (`glm()` reports the expected one). Separated coefficients without a prior are
   reported at their limit with a warning, as for `"lr"`. The expanded rows are never built, so memory stays
   near that of one row per subject. Large Laplace fits take prox-Newton steps (the copy of the subjects'
   covariates by row no longer counts against the memory budget of dense Newton, which had left a 100,000-person
   CCW censoring model on coordinate descent, unconverged after 6,300 s; a fit whose active set is too wide for
   `newtonMemoryBudget`, where 24·|W|² alone exceeds it (about 6,700 columns by default, see 38), still falls back
   to coordinate descent, for `"lr"` too), and the Hessian block and coordinate
   gradients use `threads`, with the same result on any number. Separation is found in time-basis and interaction
   columns too (an interval without events in a cross-validation fold had made the selected variance differ from
   `"lr"`'s), and the cross-validation search starts from the default variance of the person-interval rows.
   `convertToCyclopsData()` loads the interactions in one pass over the covariates (3.5 s, was 111 s, for 1,907
   interactions of 23.8 million covariate values).
   Predictions are not yet implemented
38. new `createControl(newtonMemoryBudget = 1024)`: the memory budget, in MB, of one fit's dense Newton and
   prox-Newton work (24·|W|² for the active-set Hessian, plus the row-wise copies of X and the Nyström, Schur-bound
   and conjugate-gradient work), a ceiling rather than an allocation. Its default rises from a fixed 256 MB to
   1024 MB, which allows active sets of about 6,700 columns (was 3,300): a 100,000-person CCW censoring model with
   3,550 active columns that had alternated between Newton and coordinate descent, unconverged after 13 minutes,
   converges in 50 s with 9 Newton iterations
39. `survfit()` for `cox` and `cox_counting` fits uses the case weights of the fit in the Breslow baseline (event counts and
   risk sets; rows with weight 0 are left out), and gives one curve per stratum for `cox` fits too; it used unweighted
   counts and, for `cox`, ignored strata. `predict(fit, newOutcomes, newCovariates, times = )` returns the survival
   S(t | x) = exp(-Lambda0(t) exp(x'beta)) of new rows of `cox` and `cox_counting` fits at `times` (a matrix, a row
   per `rowId` and a column per time), with the baseline of each row's `stratumId` in stratified fits; it matches
   `survfit(coxph(..., weights = , ties = "breslow"), newdata, ctype = 1, stype = 2)`. `predict()` stopped for new
   Cox data before. Both stop once another model has been fitted to the same Cyclops data (fits share its interface;
   `survfit()` silently mixed the two fits before) unless that fit used `forceNewObject = TRUE`; fit objects keep
   their `weights`
39. `updateWeights()` and `updateOutcomes()` replace the row weights or the outcomes of a Cyclops data object in
   place, so that a model is refit (bootstrap frequency weights, other outcomes on the same design) without
   rebuilding the data; a refit equals a fit of rebuilt data. Weights and outcomes are given by `rowId` (a vector
   only when the data keep their rows in `rowId` order) or, for `createCyclopsData()` data, in data order. A row of
   weight 0 is left out: the fit equals that of the data without the row (`lr`, `pr`, `ls`, `cox`, `cox_counting`).
   Not yet in conditional models (`clr`, `clr_efron`, `clr_exact`, `cpr`, `sccs`), where a row of weight 0 still
   enters its stratum: `updateWeights()` needs weights of 0 to cover whole strata, and `fitCyclopsModel(weights = )`,
   and cross-validation or `runBootstrap()` with `selectorType = "byRow"`, warn there
40. `runBootstrap()` runs its replicates in parallel on `createControl(threads = )`. Each replicate starts from the
   coefficients of the fit (it started from the previous replicate) and draws the same samples on any number of
   threads, so results do not depend on the threads. A replicate whose fit fails is `NA` (it was the estimate at
   which the fit stopped), with a warning that counts them (`failed` in the result); the summary of a coefficient
   with `NA` samples is `NA` unless `na.rm = TRUE` drops them (it was an error)
41. `survfit()` for `cox` and `cox_counting` fits uses the case weights of the fit in the Breslow baseline (event counts and
   risk sets; rows with weight 0 are left out), and gives one curve per stratum for `cox` fits too; it used unweighted
   counts and, for `cox`, ignored strata. `predict(fit, newOutcomes, newCovariates, times = )` returns the survival
   S(t | x) = exp(-Lambda0(t) exp(x'beta)) of new rows of `cox` and `cox_counting` fits at `times` (a matrix, a row
   per `rowId` and a column per time), with the baseline of each row's `stratumId` in stratified fits; it matches
   `survfit(coxph(..., weights = , ties = "breslow"), newdata, ctype = 1, stype = 2)`. `predict()` stopped for new
   Cox data before. Both stop once another model has been fitted to the same Cyclops data (fits share its interface;
   `survfit()` silently mixed the two fits before) unless that fit used `forceNewObject = TRUE`, and always once
   `updateOutcomes()` has changed the data; fit objects keep their `weights`
42. A Laplace logistic fit under the `"gap"` criterion whose prox-Newton steps meet their own stop (an exact face solve
   with Newton decrement and scaled KKT residual within tau_eff) while the duality gap stays above tau_eff stops,
   uncertified (`fit$convergence$certificate` "none", the gap in `gapEstimate`), once a further such step does not halve
   the gap. With dense continuous columns the dual point can stay that far from the primal at the optimum, and the fit
   used to alternate 50-step Newton phases with coordinate descent until `maxIterations`: a 497,463-row CCW censoring
   model (KKT residual 2e-7, gap 4.7e-6 against 1e-6) ran over 20 minutes, and now stops after 12 Newton iterations
   (77 s with data conversion). Fits that the gap certifies are unchanged
43. A Laplace fit under `algorithm = "auto"` or `"newton"` that `createControl(newtonMemoryBudget = )` leaves without any
   Newton step (its prox-Newton work, 24·|W|² + 12 bytes per entry of the working-set columns, exceeds the budget)
   warns, with the memory it needed in `fit$convergence$budgetBytes`. Such fits run coordinate descent, which on
   correlated designs can need thousands of cycles: a 500,000-person CCW Cox censoring model (1.2·10^8 entries, 1.4 GB
   of prox-Newton work) ran 1,700 cycles in 25 minutes without converging under the 1024 MB default, and converged in
   24 iterations and 105 s with `newtonMemoryBudget = 4096`
44. `createControl(fitter = "super")` fits pooled logistic regression (`plr`) as it fits `lr`: the `"kkt"` criterion, an
   auto-search that steps down first, stays at or above the null-model variance and refines its result, per-fold warm
   starts, active-set cycling and adaptive intercept steps. `plr` fits used the settings of
   `fitter = "default"`, so a model fit with `plr` and with `lr` on its expanded rows could select different variances
   under `"super"`
45. The prox-Newton Hessian block of `plr` adds its pairs of columns in the order and cache tiles of `lr`'s (densest
   columns first), and a Newton iteration keeps the per-person layouts of both its free set and its working set
   instead of rebuilding them. Cross-validating a CCW censoring model of 25,000 persons (72,748 person-intervals,
   3,821 columns) took 218 s on 4 threads and 466 s on 1, and takes 128 s and 402 s (`lr` on the expanded rows:
   126 s and 395 s); the Hessian is the same up to rounding
46. Coordinate descent with `threads > 1` (`plr`, and `cox` with `activeSetEvery`) takes the steps it takes on one
   thread: its cycles skip the coordinates of the active set and take the extra intercept steps of
   `adaptiveIntercept` (it made full sweeps without extra intercept steps), so results no longer depend on
   `threads`. A `plr` fit by the super fitter's coordinate descent moved by up to 7e-6 on 2 threads
47. `plr` fits take the steps of `lr` on its expanded person-interval rows under every fitter, so both select the
   same variance and reach the same coefficients (to rounding, with equal iterations, on a 400-subject comparison of
   fitters `"legacy"`, `"default"` and `"super"` with algorithms `"ccd"`, `"newton"` and `"auto"`, fixed and
   cross-validated Laplace and normal priors; `"auto"` with `createControl(plrCostModel = "lr")`). Before,
   cross-validated variances differed by up to a factor of 3:
   - `fitter = "legacy"` scores a `plr` fold by `lr`'s legacy score (the held-out events' `y log p`); it used the
     full held-out likelihood
   - `createControl(plrCostModel = "lr")`: `algorithm = "auto"` switches between coordinate descent and Newton by
     `lr`'s cost model on the person-interval rows. The default, `"plr"`, keeps the cost model of `plr`'s own work,
     which switches earlier and is faster (cross-validating a CCW censoring model of 25,000 persons on 4 threads:
     128 s, against 210 s with `"lr"`; `lr` 137 s); its fits agree with `lr`'s to the convergence tolerance
   - the mean positive weight of the "gap" tolerance and the extra intercept steps of `adaptiveIntercept` count the
     person-interval rows
   - exact-duplicate columns (`mergeDuplicates`) merge in `plr` (equal in X and in their time-basis column)
   - `cvRepetitions = "auto"` counts the person-interval rows of `plr`, and `convergenceType = "zhang"` compares
     X beta on them
   - the auto-search's choice between cross-validation scores equal to 1e-12 of their size no longer rests on their
     last bits, which differ between `plr` and `lr` (and decided the search on the flat scores of the null model)
   Under `fitter = "legacy"`, a fold with a separated time-basis column still differs: `lr`'s legacy kernels overflow
   there and restart with a smaller bounding box
48. Dense Newton steps scale the system by its diagonal before solving it: with a normal prior of tiny variance
   (about 1e-9 or less, as cross-validation can select) next to unpenalized columns, the solve dropped the unpenalized
   directions as if singular, and the fit stopped away from its optimum (an intercept of -3.15 instead of -3.32)
49. Less memory per data object and per cross-validation or bootstrap copy, with the same results and speed: outcomes
   whose row IDs are consecutive (as `lr` data built from sorted tables, such as CCW's expanded rows) keep no row-ID
   map and no row labels (converting 3 million `lr` rows: 12.2 s instead of 15.3 s); and the engines no longer allocate the
   censoring weights of the two-way-scan models (Fine-Gray) for other models, or three per-row arrays only the GPU
   engines use (40 bytes per row in every copy)
50. `plr` fits by coordinate descent (`algorithm = "ccd"`) with a Laplace prior skip the visits to a column of 1s at
   zero whose step is provably zero, as `lr` fits do (item 43 of 3.8.0), with bounds on the person-interval rows
   (time-basis interactions of a 0/1 basis included), also on threads. Results are bitwise unchanged on any number of
   threads. A fixed-variance CCW censoring model of 25,000 persons (72,748 person-intervals, 3,821 columns, 3,445
   cycles, 1,105,682 skipped visits) took 96.6 s on 1 thread and 78.9 s on 4, and takes 65.0 s and 62.9 s; `"auto"`
   fits (2.2 s) and `lr` (13.7 s) are unchanged. The cloglog link, single precision, `fitter = "legacy"` and weights
   other than 0 and 1 keep every visit
51. Threads within one `lr` or `plr` fit (`createControl(threads = )`): a `plr` column of at least 4,096 subjects
   computes its coordinate-descent gradient and update on up to 4 threads, and the passes of Newton steps over rows
   and subjects (`lr`: row weights, the full gradient, the Hessian block's transposition, trial log likelihoods and
   steps; `plr`: the subjects' layout of the columns, the full gradient, the direction, log likelihoods, refreshes)
   run on the fit's threads. Terms are computed on the threads and added in the serial order, so results are
   bitwise identical on any number of threads and on one thread unchanged. A worker pool run wakes and waits for
   only the threads it uses. The ordered sums keep the terms of at most 524,288 rows at a time. Cross-validation
   folds, and bootstrap replicates on copies of the fit, keep coordinate descent on one thread.
   Fixed-variance CCW censoring models (25,000 persons, 72,748 person-intervals, 3,821 columns; medians of 3 runs,
   1 / 2 / 4 / 8 threads): `plr` `"ccd"` 66.0 / 63.8 / 63.9 / 64.8 s become 64.0 / 42.8 / 30.3 / 29.1 s; `plr`
   `"auto"` 4.27 / 2.84 / 2.16 / 1.81 s become 4.27 / 2.45 / 1.55 / 1.16 s; `lr` `"auto"` 6.54 / 3.91 / 2.68 / 2.05 s
   become 6.54 / 3.57 / 2.23 / 1.52 s; `lr` `"ccd"` (13.9 s) is unchanged. The cross-validated `plr` model (10
   folds) on 4 threads takes 130.4 s instead of 132.1 s
52. Fine-Gray (subdistribution hazard) models with counting-process data, weights and censoring distributions per
   group, without expanding rows: `convertToCyclopsData(..., fineGrayCensoring = )` with an outcome column
   `fineGrayGroup` for `"cox"` and `"cox_counting"` models. A row with `fineGrayGroup` g > 0 ended with a competing
   event at its stop time T and stays in the risk set of every later event time t of its stratum, with its weight
   multiplied by G_g(t-) / G_g(T-), where G_g is the step function of group g in `fineGrayCensoring` (columns
   `fineGrayGroup`, `time`, `survival`). The partial likelihood, coordinate descent, Newton steps (dense and
   conjugate-gradient, with the Woodbury preconditioner), `vcov()` (model-based and `type = "robust"`), Schoenfeld
   residuals, separation checks and cross-validation use one running sum per group in each risk set, so a fit costs
   O(rows + event times x groups) instead of one extra row per competing event and later event time. All of them
   equal the `cox_counting` fit on those expanded rows (tests to 1e-8 or better). Survival prediction and
   `survfit()` are not yet implemented for these models and stop with an error; they need double precision
53. X'y and the column dot products of `lr` Newton steps (the full gradient, Hessian-vector products and the
   diagonal) loop over one column format at a time and sum in a local, without a format test per entry; X'y no longer
   stores and reloads its running sum. The terms are added in the same order and rounded the same way, so results
   are bitwise unchanged. A normal-prior `lr` fit (100,000 rows, 13,000 columns, 18 million values; `"auto"`, Newton)
   takes 11.0 s instead of 12.3 s on 1 thread and 10.8 s instead of 12.2 s on 4; Laplace fits are unchanged
54. `convertToCyclopsData()` from data frames needs much less memory beyond its input: it reads the covariate vectors
   in place instead of copying them (into `integer64` vectors and again into C++), and sorts unsorted `lr` and `pr`
   covariates through a permutation instead of a sorted copy. With 26 million covariate values (100,000 rows,
   20,000 covariates) the process peak fell from 1.96 GB to 0.82 GB for covariates sorted by `covariateId` and
   `rowId`, and from 2.57 GB to 1.04 GB for covariates sorted by `rowId`. The batches of an `Andromeda` table are read
   in place too. The data, and so every fit, are the same as before

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
44. `createControl(cvNullModelBound = )`: with a Laplace prior, the auto-search does not evaluate variances
    below the null-model variance (the largest at which every penalized coefficient is 0), and starts there
    when `startingVariance` lies below it; when the null model scores best, it first looks up to 100 times
    its variance. On by default for `lr` under `fitter = "super"`, whose search steps down first: below that
    variance every fold fit is the same model, and the search could stop on the run of equal scores and return
    a model without covariates (often, with about 100 events and PatientLevelPrediction's starting variance of
    0.01). Fits on propensity-score data, where the search stays well above that variance, are unchanged
45. least-squares (`ls`) models: the auto-search starts at the null-model variance, steps up, fits its quadratic to
    the best point and its two neighbours and refines its result, so the selected variance no longer depends on the
    units of the outcome (the earlier start and first step did, and a quadratic through every point could land far
    from the optimum). Fits use the `"kkt"` criterion, measured in units of the outcome's standard deviation (the
    `"gradient"` criterion could stop far from the optimum, and a `tolerance` given without `convergenceType` is now
    in standard-error units), with active-set cycling, adaptive intercept steps and unbounded steps; they keep the
    residual of each row and update the intercept in constant time. `fitter = "super"` also fits the
    cross-validation folds to a `"kkt"` tolerance of 0.1, and `fitter = "legacy"` keeps the earlier settings. In
    benchmarks with 20,000-1,000,000 rows and 2,000-10,000 sparse binary covariates, cross-validation is 2.3-17x
    faster (`super`: 3.0-20x), with the same or lower held-out error
46. `createPrior("elasticnet", laplaceFraction = )` creates an elastic net prior: a mix of the Laplace and
    normal priors, with the penalty `lambda * (laplaceFraction * |beta| + (1 - laplaceFraction) / 2 * beta^2)`
    and `lambda = sqrt(2 / variance)`. `laplaceFraction` is `glmnet`'s `alpha`, and cross-validation chooses
    the variance for a fixed `laplaceFraction`; its null-model variance (`cvNullModelBound`) is
    `laplaceFraction^2` times the Laplace prior's. For `ls` models the normal part is divided by the outcome's
    standard deviation, as in `glmnet`, so that fits do not depend on the units of the outcome
47. cross-validation with `threads > 1` no longer writes shared state from several threads without
    synchronization (data races reported by ThreadSanitizer; every thread wrote the same value, so results do
    not change)
48. logistic regression with `fitter = "default"` or `"super"` on x86-64: the gradient keeps the running maximum
    of the weights by a comparison instead of `std::fmax()`, which GCC compiles to a library call per entry; fits
    take about half the time, with identical results (arm64 keeps `std::fmax()`, a single instruction there)

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
