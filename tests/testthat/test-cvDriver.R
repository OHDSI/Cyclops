library(testthat)

simulateLr <- function(seed = 123, nrows = 2000, ncovars = 40) {
    set.seed(seed)
    sim <- simulateCyclopsData(nstrata = 1, nrows = nrows, ncovars = ncovars, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "logistic")
    convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
}

cvControl <- function(threads, ...) {
    createControl(cvType = "auto", fold = 5, cvRepetitions = 2, seed = 1, resetCoefficients = TRUE,
                  startingVariance = 0.01, noiseLevel = "silent", threads = threads, ...)
}

test_that("Cross-validation does not depend on the number of threads", {
    cyclopsData <- simulateLr()
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    fits <- lapply(c(1, 2, 3), function(threads) {
        fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE, control = cvControl(threads))
    })
    for (i in 2:3) {
        expect_identical(fits[[i]]$variance, fits[[1]]$variance)
        expect_identical(coef(fits[[i]]), coef(fits[[1]]))
    }
})

test_that("Cross-validation keeps fixed coefficients fixed with threads", {
    cyclopsData <- simulateLr()
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    J <- getNumberOfCovariates(cyclopsData)
    fixed <- c(FALSE, TRUE, TRUE, rep(FALSE, J - 3))
    start <- c(-1, 0.5, -0.5, rep(0, J - 3))
    fits <- lapply(c(1, 3), function(threads) {
        fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE, control = cvControl(threads),
                        startingCoefficients = start, fixedCoefficients = fixed)
    })
    expect_identical(fits[[2]]$variance, fits[[1]]$variance)
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))
    expect_equivalent(coef(fits[[2]])[2:3], c(0.5, -0.5))
})

test_that("A final fit that fails the gradient criterion does not repeat the cross-validation", {
    # x1 only in controls and no intercept: the gradient criterion, sum(y * xBeta), cannot see x1 move
    set.seed(7)
    n <- 2000
    y <- rbinom(n, 1, 0.3)
    x1 <- ifelse(y == 0, rbinom(n, 1, 0.3), 0)
    x2 <- rbinom(n, 1, 0.3)
    covariates <- rbind(data.frame(rowId = which(x1 == 1), covariateId = 1, covariateValue = 1),
                        data.frame(rowId = which(x2 == 1), covariateId = 2, covariateValue = 1))
    cyclopsData <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr",
                                        addIntercept = FALSE, quiet = TRUE)
    expect_warning(log <- capture.output(
        fit <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", useCrossValidation = TRUE),
                               control = createControl(noiseLevel = "quiet", cvType = "grid", lowerLimit = 1e-4,
                                                       upperLimit = 1e-4, gridSteps = 1, fold = 5, seed = 1))),
        "BLR convergence")
    expect_equal(length(grep("pred log like = ", log)), 5) # each fold fit once
    expect_equal(length(grep("Not computed", log)), 0)
    expect_equal(fit$return_flag, "SUCCESS")
    expect_equal(fit$cv_failed_folds, 0)
    expect_equal(fit$variance, 1e-4)
})

test_that("Threads do not change a logistic regression fit", {
    set.seed(42)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 100, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    fits <- lapply(c(1, 4), function(threads) {
        fitCyclopsModel(data, prior = createPrior("laplace", 0.1, exclude = 0), forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(noiseLevel = "silent", threads = threads))
    })
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))
})

test_that("Cross-validation options of the fold fits do not depend on the number of threads", {
    cyclopsData <- simulateLr()
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    fits <- lapply(c(1, 3), function(threads) {
        fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE,
                        control = cvControl(threads, cvConvergenceType = "lange", cvTolerance = 1e-7,
                                            cvWarmStart = TRUE))
    })
    expect_identical(fits[[2]]$variance, fits[[1]]$variance)
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))

    # Defaults reproduce the explicit settings of the default fitter
    default <- fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE, control = cvControl(1))
    explicit <- fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE,
                                control = cvControl(1, cvWarmStart = FALSE))
    expect_identical(default$variance, explicit$variance)
    expect_identical(coef(default), coef(explicit))
})
