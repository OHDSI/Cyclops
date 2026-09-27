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
  the next event group, so every `O(N)` pass runs over distinct event times instead of rows: about
  30x faster in the benchmark below.
* **Linear order is kept.** Each evaluation is `O(K + N)` (K rows, N distinct event times). The
  one-time setup is `O(K log N + nnz log nnz)` per weight vector (initialization, each CV fold).
* **Decisions (from review):**
  1. The name `cox_counting` stays.
  2. `cox` requires every start time to be 0.
  3. `cox` uses the one-group-per-event-time engine.
  4. Cross-validation keeps all rows of a subject in the same fold.
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
* **Cross-validation folds.** When subject ids are present, the fold units are subjects: the
  existing `BY_PID` selector runs over subject indices. When every row is its own subject, this
  reproduces the current by-row folds exactly.
* `ModelType::COX_COUNTING` / `"cox_counting"`. CPU only; a GPU request stops with "Invalid model
  type", as for the other CPU-only models.
* R interface:
  * `createCyclopsData(Surv(start, stop, event) ~ ... + cluster(id), modelType = "cox_counting")`;
  * `convertToCyclopsData(outcomes, covariates, modelType = "cox_counting")` with optional
    `outcomes$startTime` and `outcomes$subjectId`, for both data frames and Andromeda;
  * `cox` accepts `cluster()` and `subjectId` as well, and stops with an error if any start time
    is non-zero;
  * `residuals()` works for `cox_counting`; `predict` on new data and `survfit` stop with an error.

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

### 3.2 Engine tests (`tests/testthat/test-countingProcessCox.R`, 18 tests)

Against `coxph` (coefficients and log-likelihood to 1e-6):
* delayed entry, ties, strata, and case weights with a held-out row;
* left truncation with strata and weights (n = 1500);
* sparse and indicator covariates;
* a time-varying exposure through `convertToCyclopsData`, for both data-frame and Andromeda input;
* `vcov` for `cox` and `cox_counting`;
* `cox_counting` Schoenfeld residuals.

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

### 3.3 Existing behavior

* Full test suite, built with R 4.4.1:
  * `main`: 158 tests, 0 failures, 26 skipped (GPU etc.);
  * branch: 176 tests, 0 failures, 26 skipped;
  * every existing test keeps its status.
* The same 13 representative fits under `main` and the branch:
  * `fgr`, `lr`, `pr` and `clr`: `identical()`;
  * `cox` (sparse covariates, Laplace, cross-validation including the hyperparameter, weights,
    50 strata, formula interface, legacy three-column `Surv`): maximum relative difference ≤ 1e-15;
  * `cox` profile CIs: 3e-14;
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

1. **Baseline hazard (`survfit`) for `cox_counting`**: Breslow increments `d_g / D_g` over the
   (start, stop] risk sets. `survfit` for `cox` is computed in R and is unchanged.
2. **`testProportionality` for `cox_counting`**: the engine score already handles start times,
   but the R `km` transform needs a Kaplan-Meier curve with delayed entry, i.e. start times on the
   R side.
3. **Bootstrap by subject**: the `BootstrapSelector` takes the same `ids`, so this is one line in
   `CcdInterface::runBootstrap`. It was not requested, so it is left as is.
4. **Weighted Fisher information**: the risk-set sums are already weighted. The weighted
   `vcov` still throws, as on `main`.
5. **GPU (CUDA)**: the scan kernels are unchanged; add a scatter of exit contributions, and event
   groups for `cox`.
6. **Single precision**: entry/exit prefix sums can cancel when risk sets turn over many times.
   FP64 agrees with `coxph` to 1e-14 on the examples; FP32 to about 1e-3. Options are long-double
   accumulation, or separate entry and exit arrays with a reset when the risk set empties.
7. `NEWS.md` entry, and the choice of release version.

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
| 2 | **Sparse gradient/Hessian by prefix sums.** Between a column's entry/exit points its numerators are constant, so each segment contributes `N·ΔC` and `N^2·ΔP`, with `C = cumsum(d/D)` and `P = cumsum(d/D^2)` built in the existing O(N) denominator pass | both | no (validated to 5e-13 in the prototype) | the per-column scan drops from O(N − first_j) to O(#entry/exit points of j); the largest gain for sparse, high-dimensional L1 fits where most coordinates don't move. Use per-stratum sums and FP64 |
| 3 | **Range- and stratum-limited denominator updates.** After updating `beta_j`, recompute accumulated denominators only in strata that contain column j, and in `cox_counting` only on `[min entry_j, max exit_j)` (after the last exit the change cancels) | both | yes for right-censored data; last bits with exits | large for many small strata or short exposure windows |
| 4 | **Skip strata with no non-zeros** in the sparse fill loop | both | yes | proportional to the fraction of strata a column misses; superseded by #2 |
| 5 | **Skip divisions at groups without events** | `cox` on `main` | yes | **measured: no gain** (9.39 s vs 9.25 s); the loop is bound by the traversal, not the division. Moot now that every group has events |
| 6 | **`exp(delta)` update for indicator columns** (`u_k *= e^delta` instead of `exp` per row; `computeRemainingStatistics` resets any drift each cycle) | both | no | removes one `exp` per non-zero in `updateXBeta` |
| 7 | **Fuse the numerator scatter into the scan** for sparse/indicator columns (the non-zeros are already sorted by group) | both | yes | removes the zero/scatter/gather passes over `numerPid` |
| 8 | **Parallelism** across strata (TBB), or GPU scans | both | yes / no | strata are independent; the scans are the existing CUB kernels |

Recommended next: #3, then #2.

## 6. Decisions

1. Name: `cox_counting`.
2. `cox` requires start times of 0: an error in `createCyclopsData` (three-column `Surv`) and in
   `convertToCyclopsData` (`startTime` column).
3. `cox` uses one risk-set group per event time on CPU.
4. Cross-validation folds are by subject when subjects are given (`cluster()`, or
   `outcomes$subjectId`), for `cox` and `cox_counting`.
