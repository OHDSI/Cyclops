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
                                                       upperLimit = 1e-4, gridSteps = 1, fold = 5, seed = 1,
                                                       algorithm = "ccd"))), # the "gradient" criterion's failure
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
                        control = cvControl(threads, cvSearchDirection = "down", cvConvergenceType = "lange",
                                            cvTolerance = 1e-7, cvWarmStart = TRUE, cvRefine = TRUE,
                                            cvNullModelBound = TRUE))
    })
    expect_identical(fits[[2]]$variance, fits[[1]]$variance)
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))

    # Defaults reproduce the explicit settings of the default fitter
    default <- fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE, control = cvControl(1))
    explicit <- fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE,
                                control = cvControl(1, cvSearchDirection = "auto", cvWarmStart = FALSE,
                                                    cvRefine = FALSE, cvNullModelBound = FALSE))
    expect_identical(default$variance, explicit$variance)
    expect_identical(coef(default), coef(explicit))
})

test_that("Cross-validation without resetCoefficients does not depend on the number of threads", {
    cyclopsData <- simulateLr()
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    fits <- lapply(c(1, 3), function(threads) { # as PatientLevelPrediction calls it
        fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE,
                        control = createControl(cvType = "auto", fold = 3, seed = 1, startingVariance = 0.01,
                                                noiseLevel = "silent", threads = threads))
    })
    expect_identical(fits[[2]]$variance, fits[[1]]$variance)
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))

    warmStart <- function(...) Cyclops:::.resolveControl(createControl(...), "lr", 10)$cvWarmStart
    expect_true(warmStart())
    expect_false(warmStart(resetCoefficients = TRUE))
    expect_false(warmStart(fitter = "legacy"))
    expect_false(warmStart(cvWarmStart = FALSE))
})

test_that("Cross-validation with the super fitter does not depend on the number of threads", {
    cyclopsData <- simulateLr()
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    fits <- lapply(c(1, 3), function(threads) {
        fitCyclopsModel(cyclopsData, prior = prior, forceNewObject = TRUE,
                        control = cvControl(threads, fitter = "super"))
    })
    expect_equal(fits[[1]]$return_flag, "SUCCESS")
    expect_identical(fits[[2]]$variance, fits[[1]]$variance)
    expect_identical(coef(fits[[2]]), coef(fits[[1]]))
})

test_that("The auto-search does not evaluate variances below the null-model variance", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 2000, ncovars = 40, effectSizeSd = 0.5, eCovarsPerRow = 2,
                               model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", addIntercept = TRUE,
                                        quiet = TRUE)
    # lambda max: the largest |gradient| of a covariate at the intercept-only fit
    y <- sim$outcomes$y[match(sim$covariates$rowId, sim$outcomes$rowId)]
    gradient <- tapply(sim$covariates$covariateValue * (mean(sim$outcomes$y) - y), sim$covariates$covariateId, sum)
    nullVariance <- 2 / max(abs(gradient))^2

    search <- function(...) {
        log <- capture.output(fit <- fitCyclopsModel(cyclopsData, forceNewObject = TRUE, warnings = FALSE,
            prior = createPrior("laplace", exclude = c(0), useCrossValidation = TRUE),
            control = createControl(cvType = "auto", fold = 5, seed = 1, fitter = "super", noiseLevel = "quiet",
                                    startingVariance = nullVariance / 1000, ...)))
        tried <- as.numeric(sub(".*Grid-point #[0-9]+ at ([0-9.eE+-]+).*", "\\1",
                                grep("Grid-point", log, value = TRUE)))
        list(fit = fit, tried = unique(tried), log = log)
    }
    bounded <- search()
    reported <- as.numeric(sub(".*null-model variance ([0-9.eE+-]+).*", "\\1",
                               grep("null-model variance", bounded$log, value = TRUE)))
    expect_equal(reported, nullVariance, tolerance = 1E-4)
    expect_equal(bounded$tried[1], reported, tolerance = 1E-4) # the start lay below it
    expect_true(all(bounded$tried >= reported * (1 - 1E-4)))
    expect_gt(sum(coef(bounded$fit)[-1] != 0), 0)

    unbounded <- search(cvNullModelBound = FALSE)
    expect_lt(unbounded$tried[1], nullVariance)
})

test_that("Cox cross-validation with threads > folds (setSecondOrderThreads) and fp32 threads are bit-exact", {
    skip_on_cran() # threads
    set.seed(141)
    K <- 2500
    p <- 80 # n >= 32 exercises parallel colRangeS2 inside CV folds when threads > fold
    X <- matrix(rbinom(K * p, 1, 0.15), K, p)
    X[, 71:80] <- matrix(rnorm(K * 10), K, 10)
    eta <- drop(X[, c(1:5, 71:73)] %*% c(0.4, -0.35, 0.3, -0.25, 0.2, 0.3, -0.25, 0.2))
    tObs <- ceiling(rexp(K, 0.05 * exp(eta)))
    y <- rbinom(K, 1, 0.65)
    nz <- which(X != 0, arr.ind = TRUE)
    covDf <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
    covDf <- covDf[order(covDf$rowId, covDf$covariateId), ]
    outDf <- data.frame(rowId = seq_len(K), y = y, time = tObs)

    cd64 <- convertToCyclopsData(outDf, covDf, modelType = "cox", floatingPoint = 64, quiet = TRUE)
    for (priorType in c("laplace", "normal")) {
        prior <- createPrior(priorType, useCrossValidation = TRUE)
        f1 <- fitCyclopsModel(cd64, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(cvType = "auto", fold = 3, seed = 7, startingVariance = 0.05,
                                                      convergenceType = "gap", tolerance = 1e-6,
                                                      threads = 1, noiseLevel = "silent"))
        f3 <- fitCyclopsModel(cd64, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(cvType = "auto", fold = 3, seed = 7, startingVariance = 0.05,
                                                      convergenceType = "gap", tolerance = 1e-6,
                                                      threads = 3, noiseLevel = "silent"))
        f6 <- fitCyclopsModel(cd64, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(cvType = "auto", fold = 3, seed = 7, startingVariance = 0.05,
                                                      convergenceType = "gap", tolerance = 1e-6,
                                                      threads = 6, noiseLevel = "silent"))
        expect_identical(f3$variance, f1$variance)
        expect_identical(coef(f3), coef(f1))
        expect_identical(f6$variance, f1$variance)
        expect_identical(coef(f6), coef(f1))
    }

    # fp32 single-precision Cox with threads = 1 vs 4 (exercises computeGradientsAndHessians !hazardSums gating)
    cd32 <- convertToCyclopsData(outDf, covDf, modelType = "cox", floatingPoint = 32, quiet = TRUE)
    fp1 <- fitCyclopsModel(cd32, prior = createPrior("laplace", 0.05), forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(threads = 1, noiseLevel = "silent"))
    fp4 <- fitCyclopsModel(cd32, prior = createPrior("laplace", 0.05), forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(threads = 4, noiseLevel = "silent"))
    expect_identical(coef(fp4), coef(fp1))
})
