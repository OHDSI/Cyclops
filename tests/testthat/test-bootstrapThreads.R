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
