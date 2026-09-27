# Counting-process Cox model (`cox_counting`): plan, validation and status

Branch: `right-censored`

## Summary

Row-level outcomes become `(start, stop, event)`: a row is at risk at event time `t` when
`start < t <= stop`, and the event (or censoring) happens at `stop`. Rows carry the usual
strata, case weights, and dense/sparse/indicator covariates. This is `Surv(start, stop, event)` in
the survival package. It covers delayed entry (left truncation) and time-varying covariates
(one row per interval). The existing `cox` model is the special case `start = 0`.

* **New model type** `"cox_counting"` with its own engine, `CountingProcessModelSpecifics<RealType>`.
  The engine is a **subclass of the existing Cox engine**
  `ModelSpecifics<BreslowTiedCoxProportionalHazards<RealType>, RealType>`. It reuses the Cox
  gradient/Hessian scan, log-likelihood, weighting and cross-validation code, and overrides only how
  rows map to risk-set groups.
* **Linear order is kept.** Each evaluation is `O(K + N)` (K rows, N distinct event times). The
  one-time setup is `O(K log N + nnz log nnz)` per weight vector (initialization, each CV fold).
* **Existing behavior is unchanged.** The Cox engine sources (`ModelSpecifics.h/.hpp`) are not
  modified. Existing code paths only gain new `case`/list entries. All existing tests pass, and 13
  representative fits are bitwise identical before and after (section 3.3).
* **Status:** phase 1 (engine, R interface, tests) is implemented and validated. The remaining work
  and the optimization proposals are in sections 4 and 5.

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

**Conventions** (same as `survival::coxph`):
* Intervals are open at `start`: a row that starts at an event time is not at risk then.
* A row censored at `t` is at risk at `t`.
* Tied events share one group.
* A row that is at risk at no event time contributes nothing. It points to an unused group `N`.
  Examples are `(1, 1.5]` with no event inside it, or a row censored after the last event.

**Event-time groups.** Censored rows join the next event group instead of forming their own group.
So `N` is the number of distinct event times, not the number of rows. Every `O(N)` pass shrinks by
the censoring ratio: denominator accumulation, dense scans, and the fill loops of sparse scans.
This is the main reason `cox_counting` is faster than `cox` on the same data (section 3.4).

## 2. Engine design

`src/cyclops/engine/CountingProcessModelSpecifics.hpp` (about 350 lines):

| Cox `ModelSpecifics` member | `cox_counting` |
|---|---|
| `computeGradientAndHessian` (the scan), `computeThirdDerivative` | reused unchanged |
| `getLogLikelihood`, `getPredictiveLogLikelihood` | reused unchanged |
| `setWeights`, `computeXjY`, `computeXBeta`, `computeAccumlatedDenominator`, `initialize` | reused unchanged |
| `setPidForAccumulation` | **override**: event-time groups, `entry` (stored in `hPid`) and `exit` (`hExitPid`) per row, stratum resets, and sparse indices that include exit groups |
| `computeNumeratorForGradient` | **override**: base version, then subtract at exit groups |
| `updateXBeta` | **override**: a denominator change is applied at entry and exit |
| `computeRemainingStatistics` | **override**: base version, then subtract at exit groups |
| `clone` | **override**, so threaded CV copies the right engine |
| `computeFisherInformation`, `computeSchoenfeldResiduals`, `computeMMGradientAndHessian` | **throw**: the Cox versions index rows, not (start, stop] risk sets |

Other changes (each only adds a case for the new type):
* `ModelData`: optional `startTime` vector plus `loadStartTime()`, which checks `start < stop`. When
  the vector is empty, every row is at risk from the origin (the model reduces to right-censored Cox).
* `ModelType::COX_COUNTING` / `"cox_counting"`. CPU factory only; a GPU request stops with "Invalid model
  type", as for the other CPU-only models. The type is added to the survival, remove-intercept and use-offset
  lists. CV folds are assigned by row, as for `cox`.
* R: `createCyclopsData(Surv(start, stop, event) ~ ..., modelType = "cox_counting")`, and
  `convertToCyclopsData(outcomes, covariates, modelType = "cox_counting")` with an optional
  `outcomes$startTime` column, for both data frames and Andromeda. `predict` on new data, `survfit`
  and `residuals` stop with an error for `cox_counting`, because they assume right-censored risk sets.
* `modelType = "cox"` with `Surv(start, stop, event)` keeps its current meaning (`time = stop - start`).

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

### 3.2 Engine tests (`tests/testthat/test-countingProcessCox.R`, 14 tests)

Against `coxph` (coefficients and log-likelihood to 1e-6): delayed entry; ties; strata; case weights
with a held-out row; left truncation with strata and weights (n = 1500); sparse and indicator
covariates; a time-varying exposure through `convertToCyclopsData`, for both data-frame and Andromeda
input.

Invariances and other checks:
* `survSplit` data under `cox_counting` equals unsplit data under `cox`.
* `start = 0`, and a two-column `Surv`, reproduce the `cox` engine.
* The predictive log-likelihood on held-out rows equals `coxph`'s log-likelihood.
* Laplace cross-validation with `start = 0` selects the same hyperparameter as `cox`.
* The profile-likelihood CI limits sit exactly `qchisq(0.95, 1) / 2` below the maximum.
* Single precision matches to 1e-3.
* Input checks: `start >= stop` is rejected; `residuals` and `survfit` stop with an error.

### 3.3 Existing behavior unchanged

* The full test suite, built with R 4.4.1:
  * baseline (`main`): 158 tests, 0 failures, 26 skipped (GPU etc.);
  * branch: 172 tests, 0 failures, 26 skipped;
  * the only differences are the 14 new tests.
* Thirteen fits give `identical()` coefficients and log-likelihoods under both builds:
  * `cox` with sparse covariates, Laplace, cross-validation (including the hyperparameter), weights,
    50 strata, the stratified predictive log-likelihood, the formula interface, profile CIs, and the
    legacy three-column `Surv`;
  * `fgr`, `lr`, `pr` and `clr`.
* Compiler warnings are identical (the same 6).

### 3.4 Performance (Apple M-series, single thread, R 4.4.1)

Simulated data: K = 200,000 rows, 2% events, 2,000 indicator covariates at 0.5% density plus 1
dense covariate. Both engines converge in the same number of iterations to coefficients that agree
within 1.7e-14:

| prior | `cox` | `cox_counting` (start = 0) | per-iteration speedup |
|---|---|---|---|
| normal(1) | 19.5 s (7 it) | 0.68 s (7 it) | 28.6x |
| laplace(5) | 18.6 s (7 it) | 0.69 s (7 it) | 26.9x |

Scaling of `cox_counting` with real start times (200 indicators at 1% plus 1 dense, 2% events) is
linear in K:

| K | 50k | 100k | 200k | 400k | 800k |
|---|---|---|---|---|---|
| s / iteration | 0.009 | 0.021 | 0.040 | 0.094 | 0.209 |
| µs / row / iteration | 0.19 | 0.21 | 0.20 | 0.23 | 0.26 |

## 4. Remaining work (not implemented)

1. **Fisher information / `vcov`**: needs risk-set cross terms. The same scan works with
   `N_jk = sum w e^eta x_j x_k`. Profile-likelihood CIs already work.
2. **Schoenfeld residuals** and a `cox.zph` analogue: residuals at each event time use the scanned
   `N_j / D`.
3. **Baseline hazard (`survfit`)**: Breslow increments `d_g / D_g` over the (start, stop] risk sets.
   The engine already computes these.
4. **GPU (CUDA)**: the scan kernels are unchanged; add a scatter of exit contributions (atomic add,
   or a reduce-by-key over rows pre-sorted by exit group).
5. **Subject-level CV folds**: the rows of one subject should share a fold. This needs a `subjectId`
   when a subject spans several rows. Folds are currently by row, as for `cox`.
6. **Single precision**: entry/exit prefix sums can cancel when risk sets turn over many times.
   FP64 agrees with `coxph` to 1e-14 on the examples; FP32 to about 1e-3. Options are long-double
   accumulation, or separate entry and exit arrays with a reset when the risk set empties.
7. `NEWS.md` entry, and the choice of release version.

## 5. Performance optimizations considered

Profile of the **existing `cox` engine** (instrumented build, the 200k-row benchmark above, 7
iterations):

| part | time | cause |
|---|---|---|
| `computeGradientAndHessian` | 9.25 s | the scan visits every group from a column's first non-zero to the end: O(N − first_j) |
| `updateXBeta` | 9.49 s | the accumulated denominators are recomputed over all N groups after every coordinate update |
| `computeNumeratorForGradient` | 0.10 s | O(nnz_j) |
| `getLogLikelihood` | 0.007 s | not a bottleneck |

With 2,000 strata the split is the same (10.6 s / 10.9 s): nothing is local to strata. In
`cox_counting` the scan takes 0.185 s, because N is about 4k event times instead of 200k rows.

| # | optimization | applies to | results identical? | expected effect |
|---|---|---|---|---|
| 1 | **Event-time groups** (N = distinct event times) | `cox_counting` (done); could be ported to `cox` | no (sums re-associated, ~1e-14) | measured 27–29x end to end at 2% events; scales with the inverse event rate |
| 2 | **Sparse gradient/Hessian by prefix sums.** Between a column's entry/exit points its numerators are constant, so each segment contributes `N·ΔC` and `N^2·ΔP`, with `C = cumsum(d/D)` and `P = cumsum(d/D^2)` built in the existing O(N) denominator pass | both | no (validated to 5e-13 in the prototype) | the per-column scan drops from O(N − first_j) to O(#entry/exit points of j); the largest gain for sparse, high-dimensional L1 fits where most coordinates don't move. Use per-stratum sums and FP64. |
| 3 | **Range- and stratum-limited denominator updates.** After updating `beta_j`, recompute accumulated denominators only in strata that contain column j, and in `cox_counting` only on `[min entry_j, max exit_j)`: after the last exit the change cancels | both | yes for `cox`; last bits for `cox_counting` | large for many small strata or short exposure windows; small for unstratified right-censored data |
| 4 | **Skip strata with no non-zeros** in the sparse fill loop (after a reset the contributions are exact zeros) | both | yes | proportional to the fraction of strata a column misses; superseded by #2 |
| 5 | **Skip divisions at groups without events** | `cox` | yes | **measured: no gain** (9.39 s vs 9.25 s). The loop is bound by the O(N) traversal, not the division, so shrinking or skipping groups (#1, #2) is what helps |
| 6 | **`exp(delta)` update for indicator columns** (`u_k *= e^delta` instead of `exp` per row; `computeRemainingStatistics` resets any drift each cycle) | both | no | removes one `exp` per non-zero in `updateXBeta` |
| 7 | **Fuse the numerator scatter into the scan** for sparse/indicator columns (the non-zeros are already sorted by group) | both | yes | removes the zero/scatter/gather passes over `numerPid` |
| 8 | **Parallelism** across strata (TBB), or GPU scans | both | yes / no | strata are independent; the scans are the existing CUB kernels |

Recommended order: #3 (identical results), then #2, then porting #1 to `cox` as an opt-in. Until
then, `cox_counting` with start times at 0 can be used for right-censored data wherever its missing
features (section 4) are not needed.

## 6. Open questions

1. Is `"cox_counting"` the right name (alternatives: `"cox_cp"`, `"cox_start_stop"`)?
2. `modelType = "cox"` with `Surv(start, stop, event)` silently fits `time = stop - start`. Should it
   warn when a start is non-zero, or switch to `cox_counting`? Either would change existing behavior.
3. Should event-time grouping (#1) move into `cox`? It is not bitwise identical (about 1e-14).
4. Should CV folds follow a `subjectId` for multi-row subjects?
