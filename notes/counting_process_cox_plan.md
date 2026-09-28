# Counting-process Cox model (`cox_counting`): plan, validation and status

Branch: `right-censored`

## Summary

Row-level outcomes become `(start, stop, event)`: a row is at risk at event time `t` when
`start < t <= stop`, and the event (or censoring) happens at `stop`. Rows carry the usual
strata, case weights, and dense/sparse/indicator covariates. This is `Surv(start, stop, event)` in
the survival package. It covers delayed entry (left truncation) and time-varying covariates
(one row per interval). The `cox` model is the special case `start = 0`.

* **New model type** `"cox_counting"`. Its engine, `CountingProcessModelSpecifics<RealType>`, is a
  **subclass of the Cox engine** `ModelSpecifics<BreslowTiedCoxProportionalHazards<RealType>, RealType>`.
  It reuses the Cox gradient/Hessian scan, log-likelihood, weighting and cross-validation code, and
  overrides only how rows map to risk-set groups.
* **One risk-set group per event time**, for both `cox_counting` and `cox` on CPU. Censored rows join
  the next event group, so every `O(N)` pass runs over distinct event times instead of rows.
  With the range-limited updates and prefix-sum gradients of section 5, `cox` is 35–39x faster
  than on `main` in the benchmarks there, and about 560x faster for a sparse high-dimensional
  L1 fit.
* **Linear order is kept.** Each evaluation is `O(K + N)` (K rows, N distinct event times). The
  one-time setup is `O(K log N + nnz log nnz)` per weight vector (initialization, each CV fold).
* **Decisions (from review):**
  1. The name `cox_counting` stays.
  2. `cox` requires every start time to be 0.
  3. `cox` uses the one-group-per-event-time engine.
  4. Cross-validation keeps all rows of a subject in the same fold.
  5. Bootstrap by subject is an option (`selectorType = "bySubject"`, the default when subjects are given).
  6. Start times are available on the R side, so `survfit` and `testProportionality` work for
     `cox_counting`.
  7. Weighted `vcov` and GPU support are on hold.
  8. Single-precision fits use only single-precision arithmetic.
* **Other models are unchanged.** `ModelSpecifics.h/.hpp`, the GPU engine, `cox_time`, `fgr` and
  `cox_raw` are not modified; `fgr`, `lr`, `pr` and `clr` fits are bitwise identical to `main`.
  `cox` results differ from `main` only in the last bits (section 3.3).

## 1. Algorithm: (start, stop] risk sets as one prefix scan

The Cox engine sorts rows by `(stratum, -stop, event)`. It evaluates risk-set sums as a forward
(segmented) prefix scan: at event time `t`, the rows already visited are those with `stop >= t`.
With start times, the risk set is

    R(t) = { k : stop_k >= t }  \  { k : start_k >= t }      (the second set is inside the first)

**Groups.** The risk-set groups are the distinct event times of each stratum, in decreasing order,
`g = 0 .. N-1`. Every row `k` gets two group positions:

    entry(k) = first group with time <= stop_k    (the row joins the risk set here)
    exit(k)  = first group with time <= start_k   (the row has left the risk set here; may be "never")

Each row's contribution `v_k` is added at `entry(k)` and subtracted at `exit(k)`. The segmented
prefix sum over groups is then exactly the risk-set sum:

    S(g) = sum_{g' <= g} ( sum_{entry(k) = g'} v_k  -  sum_{exit(k) = g'} v_k )  =  sum_{k in R(t_g)} v_k

This is applied to `D` (`v = w e^eta`), `N_j` (`v = w e^eta x_j`) and `N2_j` (`v = w e^eta x_j^2`).
The Cox formulas then apply unchanged (Breslow ties; `d_g` is the weighted event count):

    loglik = sum_k w_k delta_k eta_k - sum_g d_g log D_g
    grad_j = sum_g d_g N_j,g / D_g - sum_k w_k delta_k x_kj                (negative log-likelihood)
    hess_j = sum_g d_g ( N2_j,g / D_g - (N_j,g / D_g)^2 )
    info_jl = sum_g d_g ( N_jl,g / D_g - N_j,g N_l,g / D_g^2 )             (Fisher information)

**Conventions** (same as `survival::coxph`):
* Intervals are open at `start`: a row that starts at an event time is not at risk then.
* A row censored at `t` is at risk at `t`.
* Tied events share one group.
* A row that is at risk at no event time contributes nothing. It points to an unused group `N`.
  Examples are `(1, 1.5]` with no event inside it, or a row censored after the last event.

## 2. Engine design

`src/cyclops/engine/CountingProcessModelSpecifics.hpp` is the CPU engine for `cox` (no start times,
so no exits) and `cox_counting`. On GPU, `cox` still uses `GpuModelSpecificsCox`, which has one
group per row.

| Cox `ModelSpecifics` member | new engine |
|---|---|
| `computeGradientAndHessian` (the scan), `computeThirdDerivative` | reused unchanged |
| `getLogLikelihood`, `getPredictiveLogLikelihood` | reused unchanged |
| `setWeights`, `computeXjY`, `computeXBeta`, `computeAccumlatedDenominator`, `initialize` | reused unchanged |
| `setPidForAccumulation` | **override**: event-time groups, `entry` (stored in `hPid`) and `exit` (`hExitPid`) per row, stratum resets, and sparse indices that include exit groups |
| `computeNumeratorForGradient` | **override**: base version, then subtract at exit groups |
| `updateXBeta` | **override**: a denominator change is applied at entry and exit |
| `computeRemainingStatistics` | **override**: base version, then subtract at exit groups |
| `computeSchoenfeldResiduals` (residuals and the proportionality score) | **override**: computed from the group risk-set sums (handles exits and case weights) |
| `computeFisherInformation` | **override**: the risk-set information above. The row-indexed Cox version gave wrong Cox variances (`test-believedBroken.R`); it now matches `coxph` |
| `clone` | **override**, so threaded CV copies the right engine |
| `computeMMGradientAndHessian` | **throws**: MM did not converge for Cox on `main` either |

Other changes:
* `ModelData`:
  * an optional `startTime` vector plus `loadStartTime()`, which checks `start < stop`;
  * an optional per-row subject index plus `loadSubjectId()`, with ids numbered by first appearance.
* **Cross-validation folds and bootstrap samples by subject.** A new selector type, `BY_SUBJECT`,
  runs the existing `BY_PID` selectors over subject indices. When every row is its own subject, this
  reproduces the by-row folds and draws exactly.
* `ModelType::COX_COUNTING` / `"cox_counting"`. CPU only; a GPU request stops with "Invalid model
  type", as for the other CPU-only models.
* R interface:
  * `createCyclopsData(Surv(start, stop, event) ~ ... + cluster(id), modelType = "cox_counting")`;
  * `convertToCyclopsData(outcomes, covariates, modelType = "cox_counting")` with optional
    `outcomes$startTime` and `outcomes$subjectId`, for both data frames and Andromeda;
  * `cox` accepts `cluster()` and `subjectId` as well, and stops with an error if any start time
    is non-zero;
  * `createControl(selectorType = "bySubject")` samples subjects for cross-validation and
    `runBootstrap()`. `"auto"` picks it wherever it would pick `"byRow"` and the data have
    subjects; `"byRow"` and `"byPid"` still force rows or strata;
  * `getStartTimeVector()` (internal, like `getTimeVector()`) returns the loaded start times;
  * `residuals()` gives Schoenfeld residuals for all covariates (a matrix, or a subset through
    `parm`) and scaled Schoenfeld residuals (`type = "scaledsch"`, which uses `vcov()`), for `cox`
    and `cox_counting`; events outside the fit (zero weight) have no residual;
  * `testProportionality()` and `survfit()` also work for `cox_counting`. They
    use the start times: the `km` transform uses Kaplan-Meier with delayed entry, and the Breslow
    baseline uses (start, stop] risk sets, one curve per stratum (`survfit` for `cox` still pools
    strata). Rows with zero weight stay out of the time transform, as in `cox.zph` on the rows
    with positive weight (which ignores the weights in the `km` and `rank` transforms);
  * `predict` on new data stops with an error, as for `cox`.

## 3. Validation

### 3.1 Algorithm prototype (`extras/CountingProcessCoxPrototype.R`)

This is an R implementation of the exact group/entry/exit layout above. It is compared against a
brute-force O(K·N) enumeration of every risk set and against `coxph(..., ties = "breslow")`
(log-likelihood, score and information from `coxph.detail`, and fitted coefficients). Values are
maximum relative differences:

| example | rows | event times | scan vs brute force | scan vs coxph | fit vs coxph | sparse prefix-sum vs scan (§5) |
|---|---|---|---|---|---|---|
| delayed entry (start = another row's event time) | 7 | 5 | 1e-16 | 4e-16 | 1e-12 | 1e-15 |
| failure/censoring/start ties | 10 | 4 | 6e-16 | 5e-16 | 7e-13 | 7e-16 |
| strata | 10 | 4 | 9e-16 | 5e-16 | 2e-16 | 9e-16 |
| case weights incl. a zero weight | 10 | 4 | 3e-16 | 2e-15 | 3e-16 | 2e-15 |
| `survSplit(lung)`: subjects span several rows | 673 | 138 | 3e-13 | 2e-13 | 3e-15 | 5e-13 |
| time-varying covariate + strata | 529 | 309 | 2e-15 | 2e-15 | 9e-16 | 5e-15 |
| left truncation + strata + weights | 1344 | 1088 | 9e-15 | 3e-14 | 1e-14 | 7e-15 |
| same data with start = 0 | 1344 | 1088 | 2e-14 | 2e-14 | 2e-14 | 2e-15 |

Splitting right-censored rows with `survSplit` changes the log-likelihood, gradient and Hessian by
exactly 0.

### 3.2 Engine tests (`tests/testthat/test-countingProcessCox.R`, 26 tests)

Against `coxph` (coefficients and log-likelihood to 1e-6):
* delayed entry, ties, strata, and case weights with a held-out row;
* left truncation with strata and weights (n = 1500);
* sparse and indicator covariates;
* a time-varying exposure through `convertToCyclopsData`, for both data-frame and Andromeda input;
* `vcov` for `cox` and `cox_counting`;
* `cox_counting` Schoenfeld residuals (delayed entry, failure ties, strata, case weights including
  a held-out event, `survSplit` rows, left truncation; several covariates, `parm` subsets, and
  scaled residuals);
* `cox_counting` baseline survival against `survfit.coxph`, with and without strata, and
  `testProportionality` against `cox.zph` with the `km` and `identity` transforms;
* weighted `testProportionality` with held-out events and censored rows, for `cox` and
  `cox_counting` and for 0/1 and gamma weights, against `cox.zph` on the rows with positive weight
  (`km`, `rank` and `identity` transforms);
* gradients and Hessian diagonals of sparse and indicator columns at an arbitrary beta, against
  `coxph.detail` (to 1e-10), for `cox` and `cox_counting`.

Invariances and other checks:
* `survSplit` data under `cox_counting` equals unsplit data under `cox`.
* `start = 0`, and a two-column `Surv`, give the same fit as `cox`.
* The predictive log-likelihood on held-out rows equals `coxph`'s log-likelihood.
* Profile-likelihood CI limits sit exactly `qchisq(0.95, 1) / 2` below the maximum.
* Single precision matches to 1e-3.
* Input checks: `start >= stop` is rejected, and `cox` rejects non-zero starts.
* **Subject-level folds:**
  * cross-validation on `survSplit` rows with `cluster(id)`, or with `outcomes$subjectId`,
    reproduces cross-validation on the unsplit data: hyperparameter equal to 10 digits,
    coefficients to 8e-14;
  * row-level folds on the same split rows select a different hyperparameter (1.77 vs 0.35).
* **Subject-level bootstrap:** on `survSplit` rows, `"auto"` and `"bySubject"` reproduce the
  bootstrap samples of the unsplit data; `"byRow"` does not. With 40 subjects held out (weight
  0), the bootstrap reproduces that of the data without them.

### 3.3 Existing behavior

* Full test suite, built with R 4.4.1:
  * `main`: 158 tests, 0 failures, 26 skipped (GPU etc.);
  * branch: 181 tests, 0 failures, 25 skipped;
  * every existing test keeps its status, except the re-enabled `vcov` check in
    `test-believedBroken.R`, which was empty (so skipped) and now passes.
* The same 13 representative fits under `main` and the branch:
  * `fgr`, `lr`, `pr` and `clr`: `identical()`;
  * `cox` (sparse covariates, Laplace, weights, 50 strata, formula interface, legacy
    three-column `Surv`): maximum relative difference ≤ 1e-14;
  * `cox` cross-validation (hyperparameter and coefficients): 3e-13; profile CIs: 3e-14;
  * `cox` stratified predictive log-likelihood: identical.
* `vcov` for `cox` changes from wrong (`main`: 4.57 vs `coxph` 0.647 on the `test-believedBroken.R`
  example) to correct (0.6471719).
* Compiler warnings are identical (the same 6).

### 3.4 Performance (Apple M-series, single thread, R 4.4.1)

Simulated data: K = 200,000 rows, 2% events, 2,000 indicator covariates at 0.5% density plus 1
dense covariate. Each run converges in 7 iterations, and the coefficients agree within 2.3e-14:

| prior | `cox` on `main` | `cox` on this branch | speedup |
|---|---|---|---|
| normal(1) | 19.5 s (2.78 s / it) | 0.66 s (0.094 s / it) | 29.5x |
| laplace(5) | 18.4 s (2.64 s / it) | 0.62 s (0.088 s / it) | 29.8x |

Scaling of `cox_counting` with real start times (200 indicators at 1% plus 1 dense, 2% events) is
linear in K:

| K | 50k | 100k | 200k | 400k | 800k |
|---|---|---|---|---|---|
| s / iteration | 0.009 | 0.021 | 0.040 | 0.094 | 0.209 |
| µs / row / iteration | 0.19 | 0.21 | 0.20 | 0.23 | 0.26 |

## 4. Remaining work (not implemented)

1. **Weighted Fisher information** (on hold): the risk-set sums are already weighted. The weighted
   `vcov` still throws, as on `main`.
2. **GPU (CUDA)** (on hold): the scan kernels are unchanged; add a scatter of exit contributions,
   and event groups for `cox`.
3. **Stratified baseline survival for `cox`**: `survfit` still pools strata for `cox` (done for
   `cox_counting`).
4. **Single precision with heavy row turnover.** Single-precision fits use only single-precision
   arithmetic (compiling the float engine with `-Wdouble-promotion` finds no promotion). FP32 `cox`
   is at least as accurate as on `main` (relative coefficient error 8e-7 vs 9e-6, and 2.5e-6 vs
   8.3e-6 on lung). `cox_counting` is comparable (1e-5 with left truncation) except when many rows
   enter and leave at the same event times: `survSplit(lung)` with 100 cuts gives 2.8e-4. At each
   cut, one interval's exit and the next interval's entry for the same subject cancel in float.
   Compensated (Kahan) summation, still single-precision arithmetic, would recover most of this.
5. `NEWS.md` entry, and the choice of release version.

## 5. Performance optimizations considered

Profile of the **`cox` engine on `main`** (instrumented build, the 200k-row benchmark above, 7
iterations):

| part | time | cause |
|---|---|---|
| `computeGradientAndHessian` | 9.25 s | the scan visits every group from a column's first non-zero to the end: O(N − first_j) |
| `updateXBeta` | 9.49 s | the accumulated denominators are recomputed over all N groups after every coordinate update |
| `computeNumeratorForGradient` | 0.10 s | O(nnz_j) |
| `getLogLikelihood` | 0.007 s | not a bottleneck |

With 2,000 strata the split is the same (10.6 s / 10.9 s): nothing is local to strata.

| # | optimization | applies to | results identical? | status / expected effect |
|---|---|---|---|---|
| 1 | **Event-time groups** (N = distinct event times) | `cox`, `cox_counting` | no (~1e-15) | **done**: 29.5–29.8x at 2% events; scales with the inverse event rate |
| 2 | **Sparse gradient/Hessian by prefix sums.** Between a column's entry/exit points its numerators are constant, so each segment contributes `N·ΔC` and `N^2·ΔP`, with `C = cumsum(d/D)` and `P = cumsum(d/D^2)` per stratum (in double), maintained with the denominators | both | no (≤ 1e-14; gradients and Hessians match `coxph.detail` to 1e-10) | **done**: the per-column cost drops from O(N − first_j) to O(#entry/exit points of j); 12x in the sparse L1 regime (table below) |
| 3 | **Range- and stratum-limited denominator updates.** After updating `beta_j`, recompute accumulated denominators only in strata that contain column j, and in `cox_counting` only on `[min entry_j, max exit_j)` (after the last exit the change cancels) | both | yes for `cox` (verified `identical()`); last bits with exits | **done**: 1.6x with 20k strata |
| 4 | **Skip strata with no non-zeros** in the sparse fill loop | both | yes | proportional to the fraction of strata a column misses; superseded by #2 |
| 5 | **Skip divisions at groups without events** | `cox` on `main` | yes | **measured: no gain** (9.39 s vs 9.25 s); the loop is bound by the traversal, not the division. Moot now that every group has events |
| 6 | **`exp(delta)` update for indicator columns** (`u_k *= e^delta` instead of `exp` per row; `computeRemainingStatistics` resets any drift) | both | no (≤ 1e-13; FP32 error unchanged) | **done**: 1.3–1.6x in A, C and D |
| 7 | **Numerators inside the gradient.** The planned fused merge of entries and exits was measured **slower** (7–33%): the scatter it replaced writes into a small, cached per-group array, while visiting exits in exit order reads per-row arrays at random. Kept instead: each column's numerator changes are scattered into a per-call workspace inside `computeGradientAndHessian`, and dense columns use the same segment sums as sparse ones | both | yes for `cox` | **done**: 1.04–1.36x; the model state is only read, so columns can be evaluated concurrently (#8) |
| 8 | **Multithreading** (`createControl(threads =)`) | both | yes: bitwise identical to one thread | **done**, see below: 1.5x on strongly regularized L1 fits, neutral otherwise |

Measured effect of #3 and #2 (seconds per iteration, same data and iteration counts; B has 108
non-zero coefficients out of 20,000):

| regime | `main` | before #3 | after #3 | after #3 + #2 |
|---|---|---|---|---|
| A: 200k rows, 2,001 covariates, normal(1) | 2.78 | 0.100 | 0.083 | 0.079 |
| B: 200k rows, 20,000 sparse covariates, laplace(0.1) | 13.3 | 0.295 | 0.286 | 0.024 |
| C: 200k rows in 20,000 strata, 2,000 covariates, normal(1) | 2.74 | 0.310 | 0.193 | 0.071 |
| D: `cox_counting`, 400k rows, 1,000 short-window covariates | — | 0.146 | 0.122 | 0.119 |

Adding #6 and #7 (double precision): A 0.046, B 0.020, C 0.039, D 0.066 seconds per iteration,
60x, 650x and 70x faster than `main` for A, B and C.

**Multithreading (#8).** Per-coordinate work at these sizes is 1–60 µs, too little to split
across threads profitably, and each update changes the state that the next gradient needs. So
threads go where the work is independent, and results stay bitwise identical to one thread:

* **Coefficients at zero.** With an L1 prior, most coordinates sit at zero and stay there
  (delta = 0), leaving the state unchanged. Runs of such coordinates have their gradients
  computed in parallel, each thread with its own workspace. The run is then processed in order
  exactly as the serial cycle and cut at the first coordinate that moves. How often candidates
  move is tracked, and runs are dispatched only when fewer than 1 move is expected in
  `8 × threads` candidates. A persistent pool, whose threads spin briefly before sleeping,
  keeps the hand-off cheap.
* **Setup.** Risk-set indices are built for all columns in parallel. Independently of threads,
  the layout is no longer rebuilt when the pattern of non-zero weights is unchanged (the base
  engine calls it twice per fit, and again for each predictive log-likelihood): 1.16–1.24x with
  one thread.
* CV keeps running folds in parallel; each fold fits with one thread, and the final fit uses
  all threads.

| seconds per iteration | 1 thread | 8 threads |
|---|---|---|
| B: 20k sparse covariates, laplace(0.1), 108 non-zero | 0.0184 | 0.0121 |
| 1M rows, 20k sparse covariates, laplace(0.1), 3,187 non-zero | 0.326 | 0.319 |
| A, C, D (every coordinate moves) | 0.045, 0.038, 0.054 | 0.043, 0.038, 0.054 |

Without the move-rate check, the 1M-row fit ran 3.4x slower at 8 threads: runs broke every
~6 coordinates. Larger gains would need parallel prefix sums in each update (the 1M-row fit
spends most of its time in O(N) passes over 20k groups), which change the order of additions.
Another option is parallel bootstrap replicates, which run serially today.

**Single precision.** Differences of running hazard sums cancel badly in float. So FP32 segments
sum the per-group increments `d/D` and `d/D^2` directly, as the Cox scan does, without the
divisions. FP32 therefore keeps O(N − first_j) per column; only double precision uses #2's
O(#entry/exit points). With prefix sums, FP32 `cox` failed to converge on a 2,500-row example
that `main` fits.

## 6. Decisions

1. Name: `cox_counting`.
2. `cox` requires start times of 0: an error in `createCyclopsData` (three-column `Surv`) and in
   `convertToCyclopsData` (`startTime` column).
3. `cox` uses one risk-set group per event time on CPU.
4. Cross-validation folds are by subject when subjects are given (`cluster()`, or
   `outcomes$subjectId`), for `cox` and `cox_counting`.
5. Bootstrap by subject through `selectorType = "bySubject"`; `"auto"` uses it wherever it would use
   `"byRow"` and the data have subjects.
6. Start times on the R side: `survfit` and `testProportionality` for `cox_counting`.
7. Weighted `vcov` and GPU support are on hold.
8. The Cox `vcov` check in `test-believedBroken.R` is re-enabled.
