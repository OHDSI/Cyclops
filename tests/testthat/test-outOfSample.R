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

    grid <- c(0.01, 0.1, 0.3, 0.5, 1.0)

    lapply(grid, function(var) {
        prior <- createPrior("laplace", variance = var, exclude = "(Intercept)",
                             useCrossValidation = FALSE)
        control <- createControl()
        fit <- fitCyclopsModel(data, prior, control, weights = weights)
        Cyclops:::.cyclopsSetWeights(fit$interface, weights = 1 - weights)
        logLik1 <- Cyclops:::.cyclopsGetLogLikelihood(fit$interface)
        Cyclops:::.cyclopsSetWeights(fit$interface, weights = weights)
        logLik2 <- Cyclops:::.cyclopsGetLogLikelihood(fit$interface)
        paste(var, logLik1, logLik2)
    })

})
