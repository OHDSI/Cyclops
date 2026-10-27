library("testthat")

#
# Logistic regression kernel (createControl(fitter = "default")): saturated rows, large moves and
# cross-validation fold fits that skip their held-out rows
#

lrFit <- function(cyclopsData, prior, start = NULL, ...) {
    fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                    startingCoefficients = start,
                    control = createControl(noiseLevel = "silent", fitter = "default", ...))
}

test_that("Large intercept moves reach the optimum (denominators do not drift)", {
    set.seed(7)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 3000, ncovars = 20, effectSizeSd = 0.5, eCovarsPerRow = 3,
                               model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    prior <- createPrior("laplace", 0.01, exclude = 0)
    optimum <- lrFit(cyclopsData, prior, tolerance = 1E-10)
    for (intercept in c(30, 36, 40)) {
        fit <- lrFit(cyclopsData, prior, start = c(intercept, rep(0, 20)), tolerance = 1E-10, maxIterations = 3000)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$log_likelihood, optimum$log_likelihood, tolerance = 1E-8)
        expect_equal(coef(fit), coef(optimum), tolerance = 1E-5)
    }
})

test_that("Overflowed exp(xBeta) is never carried forward as a finite value", {
    # A continuous column is updated while the intercept sits near -720, then the Laplace prior pins it at 0
    set.seed(5)
    n <- 20000
    covariates <- list()
    eta <- rep(-0.5, n)
    for (j in 1:30) {
        rows <- which(runif(n) < 0.15)
        covariates[[j]] <- data.frame(rowId = rows, covariateId = j, covariateValue = 1)
        eta[rows] <- eta[rows] + rnorm(1, 0, 0.4)
    }
    for (j in 31:32) {
        rows <- which(runif(n) < 0.4)
        value <- rnorm(length(rows))
        covariates[[j]] <- data.frame(rowId = rows, covariateId = j, covariateValue = value)
        eta[rows] <- eta[rows] + (j == 31) * 0.3 * value
    }
    covariates <- do.call(rbind, covariates)
    covariates <- covariates[order(covariates$covariateId, covariates$rowId), ]
    outcomes <- data.frame(rowId = 1:n, y = rbinom(n, 1, plogis(eta)))
    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE)
    prior <- createPrior("laplace", variance = 1E-4, exclude = 0)
    optimum <- lrFit(cyclopsData, prior, tolerance = 1E-8, maxIterations = 3000)
    fit <- lrFit(cyclopsData, prior, start = c(-720, rep(0, 31), 0.5), tolerance = 1E-8, maxIterations = 3000)
    expect_equal(fit$return_flag, "SUCCESS")
    expect_equal(fit$log_likelihood, optimum$log_likelihood, tolerance = 1E-8)
    expect_equal(coef(fit), coef(optimum), tolerance = 1E-5)
})

test_that("Fully saturated starts recover instead of diverging", {
    # Every row has p exactly 0 or 1, so the curvature is 0 and the Newton step is infinite
    set.seed(1)
    n <- 500
    j <- 10
    x <- matrix(rbinom(n * j, 1, 0.2), n, j)
    z <- round(runif(n, 0, 1000))
    y <- rbinom(n, 1, plogis(-0.5 + x %*% rnorm(j, 0, 0.5) + 0.001 * (z - 500)))
    nz <- which(x != 0, arr.ind = TRUE)
    cyclopsData <- convertToCyclopsData(data.frame(rowId = 1:n, y = y),
                                        data.frame(rowId = c(nz[, 1], 1:n), covariateId = c(nz[, 2], rep(j + 1, n)),
                                                   covariateValue = c(rep(1, nrow(nz)), z)),
                                        modelType = "lr", quiet = TRUE)
    prior <- createPrior("laplace", variance = 0.1, exclude = 0)
    optimum <- lrFit(cyclopsData, prior, tolerance = 1E-8)
    for (intercept in c(700, 720, 800, -750, -800)) {
        fit <- lrFit(cyclopsData, prior, start = c(intercept, rep(0, j + 1)), tolerance = 1E-8, maxIterations = 3000)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$log_likelihood, optimum$log_likelihood, tolerance = 1E-6)
    }
})

test_that("Steps that oscillate without shrinking get a smaller bound", {
    # A continuous column started so that |xBeta| ~ 705: its Newton steps flip between two values
    set.seed(31)
    n <- 20000
    covariates <- list()
    eta <- rep(-0.5, n)
    for (j in 1:60) {
        rows <- which(runif(n) < 0.1)
        covariates[[j]] <- data.frame(rowId = rows, covariateId = j, covariateValue = 1)
        eta[rows] <- eta[rows] + rnorm(1, 0, 0.3)
    }
    for (j in 61:80) {
        rows <- which(runif(n) < 0.4)
        value <- rnorm(length(rows))
        covariates[[j]] <- data.frame(rowId = rows, covariateId = j, covariateValue = value)
        eta[rows] <- eta[rows] + 0.2 * value
    }
    covariates <- do.call(rbind, covariates)
    covariates <- covariates[order(covariates$covariateId, covariates$rowId), ]
    outcomes <- data.frame(rowId = 1:n, y = rbinom(n, 1, plogis(eta)))
    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE)
    prior <- createPrior("normal", variance = 0.05, exclude = 0)
    optimum <- lrFit(cyclopsData, prior, tolerance = 2E-7, maxIterations = 3000)
    start <- c(-0.5, rep(0, 80))
    start[1 + 61] <- 705 / max(abs(covariates$covariateValue[covariates$covariateId == 61]))
    fit <- lrFit(cyclopsData, prior, start = start, tolerance = 2E-7, maxIterations = 3000)
    expect_equal(fit$return_flag, "SUCCESS")
    expect_lt(fit$iterations, 3000)
    expect_equal(fit$log_likelihood, optimum$log_likelihood, tolerance = 1E-6)
})

test_that("Fold fits over their training rows equal weighted fits", {
    # With one fold per row, the folds do not depend on the permutation, so the cross-validation score
    # can be checked against leave-one-out weighted fits
    set.seed(5)
    n <- 120
    sim <- simulateCyclopsData(nstrata = 1, nrows = n, ncovars = 6, effectSizeSd = 0.5, eCovarsPerRow = 2,
                               model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    y <- sim$outcomes$y[order(sim$outcomes$rowId)]
    variance <- 0.5
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", exclude = 0, useCrossValidation = TRUE),
                           forceNewObject = TRUE,
                           control = createControl(cvType = "grid", lowerLimit = variance, upperLimit = variance,
                                                   gridSteps = 1, fold = n, minCVData = 1, seed = 1,
                                                   resetCoefficients = TRUE, convergenceType = "lange",
                                                   tolerance = 1E-12, noiseLevel = "silent"))
    heldOut <- sapply(1:n, function(i) {
        w <- rep(1, n)
        w[i] <- 0
        f <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", variance, exclude = 0), weights = w,
                             forceNewObject = TRUE,
                             control = createControl(convergenceType = "lange", tolerance = 1E-12,
                                                     noiseLevel = "silent"))
        p <- predict(f)[i]
        y[i] * log(p) + (1 - y[i]) * log(1 - p)
    })
    expect_equal(getCrossValidationInfo(fit)$ordinate, mean(heldOut), tolerance = 1E-6)
})
