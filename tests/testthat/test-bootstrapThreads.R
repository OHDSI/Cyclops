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
