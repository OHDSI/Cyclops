library("testthat")
library("Andromeda")

context("test-predict.R")

test_that("Test predict for Poisson regression", {

    sim <- simulateCyclopsData(nstrata = 1, nrows = 10000, ncovars = 2, eCovarsPerRow = 0.5, effectSizeSd = 1,model = "poisson")
    covariates <- sim$covariates
    outcomes <- sim$outcomes

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "pr", addIntercept = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"))
    predictOriginal <- predict(fit)

    # Test using data frames
    predictNew <- predict(fit, outcomes, covariates)
    expect_equal(predictOriginal, predictNew)

    # Test using Andromeda
    andr <- andromeda(outcomes = outcomes, covariates = covariates)
    predictNew <- predict(fit, andr$outcomes, andr$covariates)
    expect_equal(predictOriginal, predictNew)
})

test_that("Test predict for logistic regression", {

    sim <- simulateCyclopsData(nstrata = 1, nrows = 10000, ncovars = 2, eCovarsPerRow = 0.5, effectSizeSd = 1,model = "logistic")
    covariates <- sim$covariates
    outcomes <- sim$outcomes

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "lr", addIntercept = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"))
    predictOriginal <- predict(fit)

    # Test using data frames
    predictNew <- predict(fit, outcomes, covariates)
    expect_equal(predictOriginal, predictNew)

    # Test using Andromeda
    andr <- andromeda(outcomes = outcomes, covariates = covariates)
    predictNew <- predict(fit, andr$outcomes, andr$covariates)
    expect_equal(predictOriginal, predictNew)
})

test_that("Test predict for pr with all-zero betas", {

    sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 2, eCovarsPerRow = 0.5, effectSizeSd = 1,model = "poisson")
    covariates <- sim$covariates
    outcomes <- sim$outcomes

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "pr", addIntercept = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", variance = 0.0001, exclude = 0))
    predictOriginal <- predict(fit)

    # Test using data frames
    predictNew <- predict(fit, outcomes, covariates)
    expect_equal(predictOriginal, predictNew)

    # Test using Andromeda
    andr <- andromeda(outcomes = outcomes, covariates = covariates)
    predictNew <- predict(fit, andr$outcomes, andr$covariates)
    expect_equal(predictOriginal, predictNew)

    # newCovariates <- as.ffdf(covariates)
    # newOutcomes <- as.ffdf(outcomes)
    # object <- fit
})

test_that("Test predict for lr with all-zero betas", {

    sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 2, eCovarsPerRow = 0.5, effectSizeSd = 1,model = "logistic")
    covariates <- sim$covariates
    outcomes <- sim$outcomes

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "lr", addIntercept = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", variance = 0.0001, exclude = 0))
    predictOriginal <- predict(fit)

    # Test using data frames
    predictNew <- predict(fit, outcomes, covariates)
    expect_equal(predictOriginal, predictNew)

    # Test using Andromeda
    andr <- andromeda(outcomes = outcomes, covariates = covariates)
    predictNew <- predict(fit, andr$outcomes, andr$covariates)
    expect_equal(predictOriginal, predictNew)
})
