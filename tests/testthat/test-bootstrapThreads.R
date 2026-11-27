library("testthat")
library("Cyclops")

bootstrapSamples <- function(cyclopsData, threads, seed = 123, replicates = 50, ...) {
    fit <- fitCyclopsModel(cyclopsData, ...,
                           control = createControl(seed = seed, threads = threads, noiseLevel = "silent",
                                                   tolerance = 1E-8))
    as.matrix(runBootstrap(fit, replicates = replicates)$samples)
}

test_that("Bootstrap samples are reproducible with a seed and do not depend on the threads", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, ncovars = 4, nrows = 500, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)

    serial <- bootstrapSamples(cyclopsData, threads = 1)
    expect_equal(bootstrapSamples(cyclopsData, threads = 1), serial, tolerance = 0)
    expect_equal(bootstrapSamples(cyclopsData, threads = 3), serial, tolerance = 1E-12)
    expect_false(isTRUE(all.equal(bootstrapSamples(cyclopsData, threads = 1, seed = 124), serial)))

    # More threads than replicates
    expect_equal(bootstrapSamples(cyclopsData, threads = 4, replicates = 2), serial[1:2, ], tolerance = 1E-12)
})

test_that("Cox bootstrap of subjects does not depend on the threads", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, ncovars = 4, nrows = 400, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "survival")
    sim$outcomes$subjectId <- (sim$outcomes$rowId + 1) %/% 2 # two rows per subject
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "cox", quiet = TRUE)

    serial <- bootstrapSamples(cyclopsData, threads = 1)
    expect_equal(bootstrapSamples(cyclopsData, threads = 3), serial, tolerance = 1E-10)
})

test_that("Cold-start bootstrap replicates do not depend on the threads", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, ncovars = 4, nrows = 500, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    coldSamples <- function(threads) {
        fit <- fitCyclopsModel(cyclopsData, control = createControl(seed = 123, threads = threads, tolerance = 1E-8,
                                                                    resetCoefficients = TRUE, noiseLevel = "silent"))
        as.matrix(runBootstrap(fit, replicates = 20)$samples)
    }
    serial <- coldSamples(1)
    expect_equal(coldSamples(3), serial, tolerance = 1E-12)
    expect_equal(serial, bootstrapSamples(cyclopsData, threads = 1, replicates = 20), tolerance = 1E-5) # same resamples
})

test_that("Bootstrap on threads leaves the weights and coefficients of the fit", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, ncovars = 4, nrows = 300, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "logistic")
    sim$outcomes$weights <- runif(nrow(sim$outcomes))
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, control = createControl(seed = 123, threads = 2, noiseLevel = "silent"))
    weights <- Cyclops:::.cyclopsGetWeights(fit$interface)
    prediction <- predict(fit)

    bootstrap <- runBootstrap(fit, replicates = 20)

    expect_equal(Cyclops:::.cyclopsGetWeights(fit$interface), weights)
    expect_equal(predict(fit), prediction)
})

test_that("Failed bootstrap replicates are NA, counted and dropped only on request", {
    set.seed(3)
    n <- 200 # quasi-separation: x1 = 1 has one row with y = 0, so many samples separate
    x1 <- rbinom(n, 1, 0.1)
    x2 <- rnorm(n)
    y <- rbinom(n, 1, plogis(-1 + x2))
    y[x1 == 1] <- 1
    y[which(x1 == 1)[1]] <- 0
    covariates <- data.frame(rowId = c(1:n, 1:n), covariateId = rep(1:2, each = n), covariateValue = c(x1, x2))
    cyclopsData <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates[covariates$covariateValue != 0, ],
                                        modelType = "lr", quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, control = createControl(seed = 1, noiseLevel = "silent"))

    expect_warning(bootstrap <- runBootstrap(fit, replicates = 40), "replicates failed and have NA estimates")
    expect_equal(nrow(bootstrap$samples), 40)
    expect_gt(bootstrap$failed, 0)
    expect_equal(bootstrap$failed, sum(!complete.cases(bootstrap$samples)))
    expect_true(all(is.na(bootstrap$summary$bpi_lower)))

    expect_warning(dropped <- runBootstrap(fit, replicates = 40, na.rm = TRUE), "they are dropped")
    expect_equal(dropped$failed, bootstrap$failed)
    expect_equal(nrow(dropped$samples), 40 - bootstrap$failed)
    expect_true(all(is.finite(dropped$summary$bpi_lower)))
})

test_that("A bootstrap by row of a conditional model warns", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 50, nrows = 300, ncovars = 2, model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "clr", quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, control = createControl(selectorType = "byRow", seed = 1, noiseLevel = "silent"))
    expect_warning(runBootstrap(fit, replicates = 5), "use selectorType = \"byPid\"")
    fit <- fitCyclopsModel(cyclopsData, control = createControl(selectorType = "byPid", seed = 1, noiseLevel = "silent"))
    messages <- character() # replicates of these sparse data can fail, with their own warning
    withCallingHandlers(runBootstrap(fit, replicates = 5), warning = function(w) {
        messages <<- c(messages, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    expect_false(any(grepl("selectorType", messages)))
})

test_that("Wide Cox bootstrap with threads > replicates uses second-order threads cleanly and restores state", {
    skip_on_cran() # threads
    set.seed(151)
    K <- 1500
    p <- 70 # exercises parallel colRangeS2 inside bootstrap replicates when threads > replicates
    X <- matrix(rbinom(K * p, 1, 0.15), K, p)
    X[, 65:70] <- matrix(rnorm(K * 6), K, 6)
    eta <- drop(X[, c(1:4, 65:66)] %*% c(0.4, -0.3, 0.25, -0.2, 0.3, -0.25))
    tObs <- ceiling(rexp(K, 0.05 * exp(eta)))
    y <- rbinom(K, 1, 0.7)
    nz <- which(X != 0, arr.ind = TRUE)
    covDf <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
    covDf <- covDf[order(covDf$rowId, covDf$covariateId), ]
    cd1 <- convertToCyclopsData(data.frame(rowId = seq_len(K), y = y, time = tObs), covDf,
                                modelType = "cox", quiet = TRUE)
    cd4 <- convertToCyclopsData(data.frame(rowId = seq_len(K), y = y, time = tObs), covDf,
                                modelType = "cox", quiet = TRUE)
    prior <- createPrior("laplace", 0.05)

    fit1 <- fitCyclopsModel(cd1, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(seed = 19, threads = 1, tolerance = 1e-6, noiseLevel = "silent"))
    fit4 <- fitCyclopsModel(cd4, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(seed = 19, threads = 4, tolerance = 1e-6, noiseLevel = "silent"))
    expect_identical(coef(fit4), coef(fit1))
    predBefore <- predict(fit4)

    s1 <- as.matrix(runBootstrap(fit1, replicates = 2)$samples)
    s4 <- as.matrix(runBootstrap(fit4, replicates = 2)$samples)
    expect_identical(s4, s1)
    expect_equal(predict(fit4), predBefore, tolerance = 1e-12)

    # Post-bootstrap refit on fit4's data object still matches serial bit-for-bit
    refit1 <- fitCyclopsModel(cd1, prior = createPrior("laplace", 0.02), warnings = FALSE,
                              control = createControl(threads = 1, tolerance = 1e-6, noiseLevel = "silent"))
    refit4 <- fitCyclopsModel(cd4, prior = createPrior("laplace", 0.02), warnings = FALSE,
                              control = createControl(threads = 4, tolerance = 1e-6, noiseLevel = "silent"))
    expect_identical(coef(refit4), coef(refit1))
})
