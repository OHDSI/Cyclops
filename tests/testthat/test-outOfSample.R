library("testthat")

context("test-outOfSample.R")
suppressWarnings(RNGversion("3.5.0"))

test_that("Out-of-sample optimization", {
    skip_on_cran()
    set.seed(666)

    nrows <- 10000
    outOfSampleProp <- 0.2

    simulant <- simulateCyclopsData(nstrata = 1,
                                    nrows = nrows,
                                ncovars = 5,
                                model = "logistic")

    nOutSample <- nrows * outOfSampleProp
    nInSample <- nrows - nOutSample

    weights <- c(rep(1, nInSample), rep(0, nOutSample))

    data <- convertToCyclopsData(simulant$outcomes,
                                 simulant$covariates,
                                 modelType = "lr",
                                        addIntercept = TRUE)

    prior <- createPrior("laplace", variance = 1, exclude = "(Intercept)",
                         useCrossValidation = TRUE)

    control <- createWeightBasedSearchControl(initialValue = 1,
                                              noiseLevel = "noisy")

    optimal <- fitCyclopsModel(data, prior, control, weights = weights)

})
