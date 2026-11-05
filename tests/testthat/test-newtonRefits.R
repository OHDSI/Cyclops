library("testthat")
library("survival")

# Newton plan Phase 3 safety tests (T3, T8, T9, T11, T13) for algorithm = "auto" / "newton"

refitProblem <- function(seed, K = 2000, p = 12) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1:2] <- rnorm(2 * K)
    eta <- drop(X %*% rnorm(p, 0, 0.3))
    frame <- data.frame(start = round(runif(K, 0, 3)), y = rbinom(K, 1, 0.7),
                        stratum = sort(sample.int(3, K, replace = TRUE)), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(eta)))
    frame
}
refitFormula <- function(p = 12) {
    as.formula(paste("Surv(start, time, y) ~", paste0("X", 1:p, collapse = " + "), "+ strata(stratum)"))
}
fitNewton <- function(frame, formula = refitFormula(), prior = createPrior("normal", variance = 0.5), ...) {
    fitCyclopsModel(createCyclopsData(formula, data = frame, modelType = "cox_counting"), prior = prior,
                    control = createControl(algorithm = "newton", noiseLevel = "silent"), warnings = FALSE, ...)
}

test_that("T8: a fold fit (0/1 weights) equals the fit on the subset", {
    frame <- refitProblem(1)
    keep <- rbinom(nrow(frame), 1, 0.8)
    fold <- fitNewton(frame, weights = keep)
    subset <- fitNewton(frame[keep == 1, ])
    expect_equal(coef(fold), coef(subset), tolerance = 1e-8)
})

test_that("T11: Newton fits do not depend on the order of rows or columns", {
    frame <- refitProblem(2)
    fit <- fitNewton(frame)
    shuffled <- frame[sample(nrow(frame)), ]
    expect_equal(coef(fitNewton(shuffled)), coef(fit), tolerance = 1e-9)
    reversed <- as.formula(paste("Surv(start, time, y) ~", paste0("X", 12:1, collapse = " + "), "+ strata(stratum)"))
    expect_equal(coef(fitNewton(frame, reversed))[names(coef(fit))], coef(fit), tolerance = 1e-9)
})

test_that("T13: fixed coefficients and an excluded (unpenalized) coefficient under Newton", {
    frame <- refitProblem(3)
    prior <- createPrior("normal", variance = 0.5, exclude = "X1")
    reference <- fitCyclopsModel(createCyclopsData(refitFormula(), data = frame, modelType = "cox_counting"),
                                 prior = prior, control = createControl(convergenceType = "kkt", tolerance = 1e-12,
                                                                        maxIterations = 1e5, noiseLevel = "silent"))
    expect_equal(coef(fitNewton(frame, prior = prior)), coef(reference), tolerance = 1e-7)

    start <- c(0.3, rep(0, 11))
    fixed <- fitNewton(frame, prior = prior, fixedCoefficients = c(TRUE, rep(FALSE, 11)), startingCoefficients = start)
    expect_equal(coef(fixed)[["X1"]], 0.3)
})

test_that("T13: cross-validation, profiles, confint and bootstrap under Newton", {
    frame <- refitProblem(4)
    data <- createCyclopsData(refitFormula(), data = frame, modelType = "cox_counting")
    cv <- fitCyclopsModel(data, prior = createPrior("normal", useCrossValidation = TRUE),
                          control = createControl(algorithm = "auto", cvType = "auto", fold = 5, seed = 7,
                                                  noiseLevel = "silent"))
    ccd <- fitCyclopsModel(createCyclopsData(refitFormula(), data = frame, modelType = "cox_counting"),
                           prior = createPrior("normal", useCrossValidation = TRUE),
                           control = createControl(cvType = "auto", fold = 5, seed = 7, noiseLevel = "silent"))
    expect_equal(cv$variance, ccd$variance, tolerance = 1e-2)
    expect_false(is.null(cv$convergence))

    fit <- fitNewton(frame, prior = createPrior("none"))
    profile <- getCyclopsProfileLogLikelihood(fit, parm = "X1", x = coef(fit)[["X1"]] + c(-0.05, 0, 0.05))
    expect_equal(which.max(profile$value), 2)
    ci <- confint(fit, parm = "X1")
    expect_true(ci[, 2] < coef(fit)[["X1"]] && coef(fit)[["X1"]] < ci[, 3])
    gold <- coxph(refitFormula(), data = frame, ties = "breslow")
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1e-6)

    boot <- runBootstrap(fit, replicates = 10)
    expect_equal(nrow(boot$summary), length(coef(fit)))
})

test_that("T3: an exact duplicate under no prior is not certified, and its sum equals the reduced fit", {
    frame <- refitProblem(5)
    frame$X13 <- frame$X4
    duplicated <- as.formula(paste("Surv(start, time, y) ~", paste0("X", 1:13, collapse = " + "), "+ strata(stratum)"))
    fit <- fitNewton(frame, duplicated, prior = createPrior("none"))
    reduced <- fitNewton(frame, prior = createPrior("none"))
    expect_false(fit$convergence$certified)
    expect_gte(fit$convergence$droppedDirections, 1)
    expect_equal(coef(fit)[["X4"]] + coef(fit)[["X13"]], coef(reduced)[["X4"]], tolerance = 1e-6)
    expect_equal(coef(fit)[paste0("X", c(1:3, 5:12))], coef(reduced)[paste0("X", c(1:3, 5:12))], tolerance = 1e-6)
})

test_that("T9: a mixture with Laplace priors is ineligible: auto falls back, newton errors", {
    frame <- refitProblem(6)
    data <- createCyclopsData(refitFormula(), data = frame, modelType = "cox_counting")
    prior <- createPrior(c("none", rep("laplace", 11)), variance = rep(1, 12))
    auto <- fitCyclopsModel(data, prior = prior, control = createControl(algorithm = "auto", noiseLevel = "silent"))
    expect_equal(auto$algorithm, "ccd")
    expect_true(nzchar(auto$convergence$ineligible))
    expect_error(fitCyclopsModel(data, prior = prior, control = createControl(algorithm = "newton"),
                                 forceNewObject = TRUE), "not available")
})
