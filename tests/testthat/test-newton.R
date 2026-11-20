library("testthat")
library("survival")

# Newton plan Phase 3: algorithm = "auto" / "newton" and convergenceType = "gap" for cox and cox_counting

newtonProblem <- function(seed, K = 3000, p = 25, delayed = TRUE) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1:2] <- rnorm(2 * K)
    X[, 4] <- X[, 3] + rbinom(K, 1, 0.05) # a correlated pair slows coordinate descent
    eta <- drop(X %*% rnorm(p, 0, 0.3))
    frame <- data.frame(start = if (delayed) round(runif(K, 0, 3)) else 0, y = rbinom(K, 1, 0.7),
                        stratum = sort(sample.int(3, K, replace = TRUE)), w = sample(1:3, K, replace = TRUE), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(eta)))
    rhs <- paste(paste0("X", 1:p), collapse = " + ")
    formula <- as.formula(paste(if (delayed) "Surv(start, time, y)" else "Surv(time, y)", "~", rhs, "+ strata(stratum)"))
    list(frame = frame, formula = formula, model = if (delayed) "cox_counting" else "cox")
}

newtonFit <- function(problem, prior, ..., weights = NULL) {
    fitCyclopsModel(createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model),
                    prior = prior, control = createControl(noiseLevel = "silent", ...), weights = weights,
                    warnings = FALSE)
}

objective <- function(fit) -(fit$log_likelihood + fit$log_prior)

test_that("auto, newton and gap reach the optimum within the certified gap", {
    for (problem in list(newtonProblem(1), newtonProblem(2, delayed = FALSE))) {
        for (prior in list(createPrior("normal", variance = 0.5), createPrior("none"))) {
            reference <- newtonFit(problem, prior, convergenceType = "kkt", tolerance = 1e-12, maxIterations = 1e5)
            for (control in list(list(algorithm = "auto"), list(algorithm = "newton"), list(convergenceType = "gap"),
                                 list(fitter = "super"))) {
                fit <- do.call(newtonFit, c(list(problem, prior), control))
                expect_equal(fit$return_flag, "SUCCESS")
                expect_true(fit$convergence$certified)
                expect_lte(fit$convergence$gapEstimate, 1e-6) # the default tolerance
                expect_lte(objective(fit) - objective(reference), 1e-6 + 1e3 * .Machine$double.eps * abs(objective(fit)))
                if (identical(control, list(algorithm = "newton")) || identical(fit$algorithm, "newton")) {
                    expect_equal(fit$algorithm, "newton") # Newton finishes at the optimum
                    expect_equal(coef(fit), coef(reference), tolerance = 1e-6)
                }
            }
        }
    }
})

test_that("explicit criteria under auto leave fits as they were, and the default is auto", {
    problem <- newtonProblem(3)
    prior <- createPrior("normal", variance = 0.5)
    for (control in list(list(convergenceType = "kkt", tolerance = 1e-6), list(convergenceType = "gradient"))) {
        before <- do.call(newtonFit, c(list(problem, prior), control))
        under <- do.call(newtonFit, c(list(problem, prior), control, list(algorithm = "auto")))
        expect_identical(coef(under), coef(before))
        expect_identical(under$iterations, before$iterations)
        expect_null(before$convergence)
    }
    default <- newtonFit(problem, prior) # Newton plan D1
    expect_identical(coef(default), coef(newtonFit(problem, prior, algorithm = "auto")))
    expect_false(is.null(default$convergence))
})

test_that("ineligible fits error under newton or an explicit gap, and fall back under auto", {
    pr <- createCyclopsData(counts ~ outcome, data = data.frame(counts = c(18, 17, 15, 20, 10, 20, 25, 13, 12),
                                                               outcome = gl(3, 1, 9)), modelType = "pr")
    expect_error(fitCyclopsModel(pr, control = createControl(algorithm = "newton")), "not available")
    expect_error(fitCyclopsModel(pr, control = createControl(convergenceType = "gap")), "not available")

    problem <- newtonProblem(4)
    other <- createPrior("barupdate", variance = 1)
    default <- newtonFit(problem, other)
    auto <- newtonFit(problem, other, algorithm = "auto")
    expect_identical(coef(auto), coef(default))
    expect_identical(auto$iterations, default$iterations)
    expect_equal(auto$algorithm, "ccd")
    expect_match(auto$convergence$ineligible, "prior other than normal, Laplace, elastic net or none")
    expect_error(newtonFit(problem, other, algorithm = "newton"), "not available")

    # a tolerance given without a criterion applies to the fallback only
    looser <- newtonFit(problem, other, algorithm = "auto", tolerance = 1e-3)
    expect_identical(coef(looser), coef(newtonFit(problem, other, tolerance = 1e-3)))
})

test_that("explicit criteria map onto Newton iterations under newton", {
    problem <- newtonProblem(5)
    prior <- createPrior("normal", variance = 0.5)
    one <- newtonFit(problem, prior, algorithm = "newton", convergenceType = "onestep")
    expect_equal(one$convergence$newtonIterations, 1)
    expect_false(one$convergence$certified)
    kkt <- newtonFit(problem, prior, algorithm = "newton", convergenceType = "kkt", tolerance = 1e-6)
    expect_equal(kkt$return_flag, "SUCCESS")
    expect_gte(kkt$convergence$newtonIterations, 1)
})

test_that("the gap stop is invariant to rescaling weights, and Newton results do not depend on threads", {
    problem <- newtonProblem(6, delayed = FALSE)
    none <- createPrior("none")
    w <- problem$frame$w
    unit <- newtonFit(problem, none, algorithm = "auto", weights = w)
    tiny <- newtonFit(problem, none, algorithm = "auto", weights = w * 1e-6)
    expect_equal(coef(tiny), coef(unit), tolerance = 1e-8)

    prior <- createPrior("normal", variance = 0.5)
    one <- newtonFit(problem, prior, algorithm = "auto", threads = 1)
    two <- newtonFit(problem, prior, algorithm = "auto", threads = 2)
    expect_identical(coef(two), coef(one))
    expect_identical(two$iterations, one$iterations)
})

test_that("maxIterations under gap warns with the estimated gap", {
    problem <- newtonProblem(7)
    data <- createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model)
    expect_warning(fit <- fitCyclopsModel(data, prior = createPrior("normal", variance = 0.5),
                                          control = createControl(convergenceType = "gap", maxIterations = 3)),
                   "estimated gap")
    expect_equal(fit$return_flag, "MAX_ITERATIONS")
    expect_false(fit$convergence$certified)
})

test_that("a given gap is certified or reported unmet, never stopped on the kkt criterion alone", {
    # logistic "gap" runs coordinate descent to its kkt stop first; without a one-pass certificate (a ridge with a free
    # intercept), the fit must still certify the gap given, or report that it did not
    data <- createCyclopsData(y ~ x, data = data.frame(y = c(0, 1, 0, 1), x = c(.999, .999, 1.001, 1.001)), modelType = "lr")
    prior <- createPrior("normal", variance = 1e10, exclude = "(Intercept)")
    fit <- fitCyclopsModel(data, prior, createControl(algorithm = "ccd", convergenceType = "gap", tolerance = 1e-6,
                                                      maxIterations = 2000, noiseLevel = "silent"),
                           startingCoefficients = c(-10, 10), warnings = FALSE)
    expect_true(fit$return_flag != "SUCCESS" || fit$convergence$certified)
    auto <- fitCyclopsModel(data, prior, createControl(convergenceType = "gap", tolerance = 1e-6, noiseLevel = "silent"),
                            startingCoefficients = c(-10, 10), warnings = FALSE)
    expect_equal(auto$return_flag, "SUCCESS")
    expect_true(auto$convergence$certified)

    # a bound that stays above a tight tolerance is not taken for the rounding floor while the fit has not tightened
    X <- data.frame(y = c(0, 1, 0, 1), a = c(-1, -1, 1, 1), b = c(-1, -1, 1, 1))
    twins <- createCyclopsData(y ~ a + b - 1, data = X, modelType = "lr")
    tight <- fitCyclopsModel(twins, createPrior("normal", variance = 1e6),
                             createControl(convergenceType = "gap", tolerance = 1e-9, mergeDuplicates = FALSE,
                                           noiseLevel = "silent"),
                             startingCoefficients = c(.1, -.1), warnings = FALSE)
    expect_true(tight$convergence$certified)
    expect_lte(tight$convergence$gapEstimate, 1e-9)
})

test_that("active-set settings do not make Newton fits of Cox models depend on threads", {
    # findModeGap() sweeps every coordinate outside the kkt-guided logistic path, as concurrent (threaded) cycles do
    problem <- newtonProblem(8)
    prior <- createPrior("normal", variance = 0.5)
    fit <- function(threads) newtonFit(problem, prior, algorithm = "newton", convergenceType = "kkt", tolerance = 1e-8,
                                       activeSetEvery = 10, activeSetThreshold = 0, threads = threads)
    one <- fit(1)
    two <- fit(2)
    expect_identical(coef(two), coef(one))
    expect_identical(two$iterations, one$iterations)
})

test_that("logistic \"auto\" with an offset (PatientLevelPrediction's transfer learning) fits the same model as ccd", {
    # the offset is a covariate turned into Cyclops' offset on the log scale, as PatientLevelPrediction passes the source
    # model's linear predictor; 1,000 covariates take the active-set cycling and, for the normal prior, the shuffled order
    set.seed(42)
    n <- 4000
    p <- 1000
    X <- matrix(rbinom(n * p, 1, 0.05), n)
    offset <- drop(X %*% c(rnorm(50, 0, 0.5), rep(0, p - 50))) - 2
    y <- rbinom(n, 1, plogis(offset + drop(X %*% c(rep(0, 30), rnorm(30, 0, 0.3), rep(0, p - 60)))))
    entries <- which(X != 0, arr.ind = TRUE)
    covariates <- rbind(data.frame(rowId = entries[, 1], covariateId = entries[, 2], covariateValue = 1),
                        data.frame(rowId = 1:n, covariateId = -1e9, covariateValue = offset))
    data <- function() {
        d <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", quiet = TRUE)
        finalizeSqlCyclopsData(d, useOffsetCovariate = -1e9, offsetAlreadyOnLogScale = TRUE)
        d
    }
    for (type in c("laplace", "normal")) {
        prior <- createPrior(type, variance = 0.05, exclude = 0)
        auto <- fitCyclopsModel(data(), prior, createControl(noiseLevel = "silent"))
        ccd <- fitCyclopsModel(data(), prior, createControl(algorithm = "ccd", convergenceType = "gap", tolerance = 1e-10,
                                                            maxIterations = 1e5, noiseLevel = "silent"))
        expect_equal(auto$return_flag, "SUCCESS")
        expect_equal(coef(auto), coef(ccd), tolerance = 1e-3)
        expect_identical(coef(auto) != 0, coef(ccd) != 0)
    }
    # (the offset is used: without it the coefficients move by about 2)
    plain <- fitCyclopsModel(convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates[covariates$covariateId > 0, ],
                                                  modelType = "lr", quiet = TRUE),
                             createPrior("normal", variance = 0.05, exclude = 0), createControl(noiseLevel = "silent"))
    expect_gt(max(abs(coef(plain) - coef(auto))), 0.5)
})
