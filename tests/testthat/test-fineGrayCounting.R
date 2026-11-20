library(testthat)

# Fine-Gray rows of "cox_counting" (outcome column fineGrayGroup and fineGrayCensoring): every fit must equal the
# Cox fit on expanded rows, in which each competing-event row is followed by one row per later event time t of
# its stratum, with weight w G_g(t-) / G_g(T-).

simulateFineGray <- function(n = 300, strata = 1, seed = 1) {
    # The same data whatever RNGversion() other test files set
    set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    persons <- data.frame(subjectId = seq_len(n), stratumId = sample(seq_len(strata), n, replace = TRUE),
                          x1 = rbinom(n, 1, 0.5), x2 = round(rnorm(n), 2), x3 = rbinom(n, 1, 0.2))
    stop <- sample(2:30, n, replace = TRUE)
    split <- ifelse(runif(n) < 0.4, pmax(1, stop - sample(1:5, n, replace = TRUE)), NA) # some persons have 2 rows
    type <- sample(0:2, n, replace = TRUE, prob = c(0.4, 0.35, 0.25)) # censored, outcome, competing
    rows <- list()
    for (i in seq_len(n)) {
        cuts <- if (is.na(split[i]) || split[i] >= stop[i]) c(0, stop[i]) else c(0, split[i], stop[i])
        m <- length(cuts) - 1
        rows[[i]] <- data.frame(subjectId = i, stratumId = persons$stratumId[i],
                                startTime = cuts[-length(cuts)], time = cuts[-1],
                                y = c(rep(0, m - 1), as.numeric(type[i] == 1)),
                                competing = c(rep(0, m - 1), as.numeric(type[i] == 2)),
                                weights = round(runif(m, 0.5, 2), 3))
    }
    rows <- do.call(rbind, rows)
    rows <- rows[order(rows$stratumId, rows$subjectId, rows$startTime), ]
    rows$rowId <- seq_len(nrow(rows))
    rows$x1 <- persons$x1[rows$subjectId]
    rows$x2 <- persons$x2[rows$subjectId]
    rows$x3 <- persons$x3[rows$subjectId]
    rows$fineGrayGroup <- ifelse(rows$competing == 1, 1 + rows$x1, 0) # two censoring groups
    # Censoring survival per group, with jumps on some event times (left limits matter there)
    censoring <- data.frame(fineGrayGroup = rep(1:2, each = 6),
                            time = c(3, 7, 10, 15, 21, 26, 5, 7, 12, 18, 22, 28),
                            survival = c(0.97, 0.9, 0.85, 0.7, 0.6, 0.45, 0.95, 0.88, 0.8, 0.66, 0.5, 0.4))
    return(list(rows = rows, censoring = censoring))
}

censoringBefore <- function(censoring, group, t) {
    step <- censoring[censoring$fineGrayGroup == group, ]
    step <- step[order(step$time), ]
    vapply(t, function(u) if (any(step$time < u)) step$survival[max(which(step$time < u))] else 1, numeric(1))
}

expandFineGray <- function(data) {
    rows <- data$rows
    tails <- list()
    for (k in which(rows$fineGrayGroup > 0)) {
        row <- rows[k, ]
        later <- sort(unique(rows$time[rows$y == 1 & rows$stratumId == row$stratumId & rows$time > row$time]))
        if (length(later) == 0) next
        tail <- row[rep(1, length(later)), ]
        tail$startTime <- c(row$time, later[-length(later)])
        tail$time <- later
        tail$y <- 0
        tail$weights <- row$weights * censoringBefore(data$censoring, row$fineGrayGroup, later) /
            censoringBefore(data$censoring, row$fineGrayGroup, row$time)
        tails[[length(tails) + 1]] <- tail
    }
    expanded <- rbind(rows, do.call(rbind, tails))
    expanded <- expanded[order(expanded$stratumId, expanded$subjectId, expanded$startTime), ]
    expanded$rowId <- seq_len(nrow(expanded))
    expanded$fineGrayGroup <- 0
    return(expanded)
}

toCyclops <- function(rows, censoring = NULL, strata = FALSE, covariates = c("x1", "x2", "x3")) {
    outcomes <- rows[, c("rowId", "subjectId", "startTime", "time", "y", "weights", "fineGrayGroup")]
    outcomes$stratumId <- if (strata) rows$stratumId else 1
    if (is.null(censoring)) outcomes$fineGrayGroup <- NULL
    covariateRows <- do.call(rbind, lapply(seq_along(covariates), function(j) {
        data.frame(rowId = rows$rowId, covariateId = j, covariateValue = rows[[covariates[j]]])
    }))
    covariateRows <- covariateRows[covariateRows$covariateValue != 0, ]
    convertToCyclopsData(outcomes, covariateRows, modelType = "cox_counting", fineGrayCensoring = censoring, quiet = TRUE)
}

fitBoth <- function(data, strata = FALSE, prior = createPrior("none"), control = createControl(), ...) {
    native <- fitCyclopsModel(toCyclops(data$rows, data$censoring, strata), prior = prior, control = control, ...)
    expanded <- fitCyclopsModel(toCyclops(expandFineGray(data), strata = strata), prior = prior, control = control, ...)
    list(native = native, expanded = expanded)
}

test_that("Fine-Gray rows equal the Cox model on expanded rows (coefficients, log-likelihood, variances)", {
    for (strata in c(FALSE, TRUE)) {
        data <- simulateFineGray(strata = if (strata) 3 else 1, seed = 1 + strata)
        fits <- fitBoth(data, strata = strata)
        expect_equal(fits$native$return_flag, "SUCCESS")
        expect_equal(coef(fits$native), coef(fits$expanded), tolerance = 1e-8)
        expect_equal(fits$native$log_likelihood, fits$expanded$log_likelihood, tolerance = 1e-10)
        expect_equal(unname(vcov(fits$native)), unname(vcov(fits$expanded)), tolerance = 1e-7)
        expect_equal(unname(vcov(fits$native, type = "robust")), unname(vcov(fits$expanded, type = "robust")), tolerance = 1e-7)
    }
})

test_that("Fine-Gray rows give the same fit under every algorithm and thread count", {
    data <- simulateFineGray(seed = 3)
    reference <- fitBoth(data)$expanded
    for (algorithm in c("ccd", "newton", "auto")) {
        for (threads in c(1, 4)) {
            control <- createControl(algorithm = algorithm, threads = threads, tolerance = 1e-10,
                                     convergenceType = if (algorithm == "auto") "gap" else "gradient")
            native <- fitCyclopsModel(toCyclops(data$rows, data$censoring), prior = createPrior("none"), control = control)
            expect_equal(coef(native), coef(reference), tolerance = 1e-6, info = paste(algorithm, threads))
        }
    }
})

test_that("Fine-Gray rows with a Laplace prior equal the expanded rows", {
    data <- simulateFineGray(seed = 4)
    for (algorithm in c("ccd", "auto")) {
        fits <- fitBoth(data, prior = createPrior("laplace", variance = 0.5),
                        control = createControl(algorithm = algorithm, tolerance = 1e-10,
                                                convergenceType = if (algorithm == "auto") "gap" else "gradient"))
        expect_equal(fits$native$return_flag, "SUCCESS")
        expect_true(any(coef(fits$native) != 0))
        expect_equal(coef(fits$native), coef(fits$expanded), tolerance = 1e-6, info = algorithm)
    }
})

test_that("A single covariate, as in a Fine-Gray MSM, matches the expanded rows", {
    data <- simulateFineGray(seed = 5)
    native <- fitCyclopsModel(toCyclops(data$rows, data$censoring, covariates = "x1"), prior = createPrior("none"),
                              control = createControl(algorithm = "newton", convergenceType = "gradient", tolerance = 1e-10))
    expanded <- fitCyclopsModel(toCyclops(expandFineGray(data), covariates = "x1"), prior = createPrior("none"),
                                control = createControl(algorithm = "newton", convergenceType = "gradient", tolerance = 1e-10))
    expect_equal(coef(native), coef(expanded), tolerance = 1e-8)
    expect_equal(unname(vcov(native, type = "robust")), unname(vcov(expanded, type = "robust")), tolerance = 1e-7)
    # Schoenfeld residuals use the Fine-Gray risk sets too
    expect_equal(residuals(native), residuals(expanded), tolerance = 1e-7)
})

test_that("Without competing-event rows, Fine-Gray data equal the plain Cox model", {
    data <- simulateFineGray(seed = 6)
    data$rows$fineGrayGroup <- 0
    native <- fitCyclopsModel(toCyclops(data$rows, data$censoring), prior = createPrior("none"))
    plain <- fitCyclopsModel(toCyclops(data$rows), prior = createPrior("none"))
    expect_equal(coef(native), coef(plain), tolerance = 1e-10)
})

test_that("Fine-Gray inputs are validated, and prediction is refused", {
    data <- simulateFineGray(seed = 7)
    bad <- data$rows
    bad$y[bad$fineGrayGroup > 0][1] <- 1
    expect_error(toCyclops(bad, data$censoring), "y = 0")
    expect_error(toCyclops(data$rows, data.frame(fineGrayGroup = 1, time = 1)), "columns fineGrayGroup, time and survival")
    expect_error(toCyclops(data$rows, transform(data$censoring, survival = survival + 1)), "\\[0, 1\\]")
    outcomes <- data$rows[, c("rowId", "time", "y", "fineGrayGroup")]
    expect_error(convertToCyclopsData(outcomes, data.frame(rowId = 1, covariateId = 1, covariateValue = 1),
                                      modelType = "lr", quiet = TRUE), "cox_counting")
    fit <- fitCyclopsModel(toCyclops(data$rows, data$censoring), prior = createPrior("none"))
    expect_error(survival::survfit(fit), "Fine-Gray")
})

fixedFineGray <- function(rows, censoring, strata, beta, threads) {
    data <- toCyclops(rows, censoring, strata)
    fitCyclopsModel(data, startingCoefficients = beta, fixedCoefficients = rep(TRUE, length(beta)),
                    prior = createPrior("none"), control = createControl(threads = threads))
    data
}

test_that("Fine-Gray second-order kernels equal those of the expanded rows", {
    data <- simulateFineGray(strata = 2, seed = 8)
    beta <- c(0.3, -0.2, 0.5)
    v <- c(0.7, -1.1, 0.4)
    for (threads in c(1, 4)) {
        native <- Cyclops:::.cyclopsSecondOrder(fixedFineGray(data$rows, data$censoring, TRUE, beta, threads)$cyclopsInterfacePtr,
                                                NULL, v, 0.3, 1)
        expanded <- Cyclops:::.cyclopsSecondOrder(fixedFineGray(expandFineGray(data), NULL, TRUE, beta, threads)$cyclopsInterfacePtr,
                                                  NULL, v, 0.3, 1)
        expect_true(native$supported)
        for (name in c("logLikelihood", "gradient", "hessian", "hv", "trial0", "trial", "committed")) {
            expect_equal(native[[name]], expanded[[name]], tolerance = 1e-9, info = paste(name, threads))
        }
    }
})

test_that("Fine-Gray rows with the Newton-CG solver (with and without Woodbury) equal the expanded rows", {
    data <- simulateFineGray(seed = 9)
    withEnvironment <- function(variables, expr) {
        old <- Sys.getenv(names(variables), unset = NA, names = TRUE)
        on.exit(for (name in names(old)) if (is.na(old[[name]])) Sys.unsetenv(name) else do.call(Sys.setenv, as.list(old[name])))
        do.call(Sys.setenv, as.list(variables))
        expr
    }
    control <- createControl(algorithm = "newton", convergenceType = "gap", tolerance = 1e-10)
    prior <- createPrior("normal", variance = 0.5)
    reference <- fitCyclopsModel(toCyclops(expandFineGray(data)), prior = prior, control = control)
    for (woodbury in c("1", "0")) {
        native <- withEnvironment(c(CYCLOPS_NEWTON_SOLVER = "cg", CYCLOPS_NEWTON_WOODBURY = woodbury),
                                  fitCyclopsModel(toCyclops(data$rows, data$censoring), prior = prior, control = control))
        expect_equal(native$convergence$solver, "cg")
        expect_equal(coef(native), coef(reference), tolerance = 1e-6, info = woodbury)
    }
})

test_that("A covariate seen only in competing-event rows behaves as in the expanded rows", {
    data <- simulateFineGray(seed = 10)
    data$rows$x3 <- as.numeric(data$rows$fineGrayGroup > 0 & data$rows$x1 == 1) # never on an outcome row
    native <- suppressWarnings(fitCyclopsModel(toCyclops(data$rows, data$censoring), prior = createPrior("none")))
    expanded <- suppressWarnings(fitCyclopsModel(toCyclops(expandFineGray(data)), prior = createPrior("none")))
    # Detected as separated (it never has an outcome) in both, with the same limit and other coefficients
    expect_equal(native$return_flag, expanded$return_flag)
    expect_equal(native$divergentCoefficients, expanded$divergentCoefficients)
    expect_equal(coef(native, ignoreConvergence = TRUE), coef(expanded, ignoreConvergence = TRUE), tolerance = 1e-5)
})

test_that("Cross-validation of Fine-Gray rows selects the variance of the expanded rows", {
    data <- simulateFineGray(n = 1500, seed = 13)
    # outcomes that depend on x1 and x2, and noise columns, so the search ends inside the grid
    set.seed(13, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    p <- plogis(-0.3 + 1.2 * data$rows$x1 + 0.8 * data$rows$x2)
    last <- !duplicated(data$rows$subjectId, fromLast = TRUE)
    flip <- last & data$rows$fineGrayGroup == 0 & runif(nrow(data$rows)) < 0.5
    data$rows$y[flip] <- rbinom(sum(flip), 1, p[flip])
    for (j in 4:8) data$rows[[paste0("x", j)]] <- rbinom(1500, 1, 0.3)[data$rows$subjectId]
    covariates <- paste0("x", 1:8)
    expanded <- expandFineGray(data)
    foldsOf <- function(rows) data.frame(rowId = rows$rowId, fold = rows$subjectId %% 5 + 1) # by person, tails included
    control <- createControl(cvType = "grid", lowerLimit = 1e-3, upperLimit = 10, gridSteps = 25, noiseLevel = "silent")
    prior <- createPrior("laplace", useCrossValidation = TRUE)
    native <- fitCyclopsModel(toCyclops(data$rows, data$censoring, covariates = covariates), prior = prior,
                              control = control, cvFolds = foldsOf(data$rows))
    reference <- fitCyclopsModel(toCyclops(expanded, covariates = covariates), prior = prior, control = control,
                                 cvFolds = foldsOf(expanded))
    expect_gt(native$variance[1], 1e-3) # inside the grid
    expect_lt(native$variance[1], 10)
    expect_equal(native$variance, reference$variance)
    expect_equal(coef(native), coef(reference), tolerance = 1e-6)
})
