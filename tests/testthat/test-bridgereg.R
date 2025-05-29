library("testthat")

test_that("Check that bridge with exponent = 1 match laplace coefficients", {
    cyclopsData <- simulateCyclopsData(nstrata = 1,
                                       nrows = 1000,
                                       ncovars = 25,
                                       model = "logistic")
    cyclopsData <- convertToCyclopsData(cyclopsData$outcomes, cyclopsData$covariates,
                                        modelType = "lr", addIntercept = TRUE,
                                        quiet = FALSE)
    laplace <- createPrior("laplace")
    laplaceBridge <- createPrior("bridge",
                                 exponent = 1)

    cyclopsFit <- fitCyclopsModel(cyclopsData,
                                  prior = laplace)
    cyclopsFitBridge <-fitCyclopsModel(cyclopsData,
                                       prior = laplaceBridge)
    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), coef(cyclopsFitBridge), tolerance = tolerance)

    })

test_that("Check that bridge with exponent = 1 match normal coefficients", {
    cyclopsData <- simulateCyclopsData(nstrata = 1,
                                       nrows = 1000,
                                       ncovars = 25,
                                       model = "logistic")
    cyclopsData <- convertToCyclopsData(cyclopsData$outcomes, cyclopsData$covariates,
                                        modelType = "lr", addIntercept = TRUE,
                                        quiet = FALSE)
    normal <- createPrior("laplace")
    normalBridge <- createPrior("bridge",
                                 exponent = 2)

    cyclopsFit <- fitCyclopsModel(cyclopsData,
                                  prior = laplace)
    cyclopsFitBridge <-fitCyclopsModel(cyclopsData,
                                       prior = laplaceBridge)
    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), coef(cyclopsFitBridge), tolerance = tolerance)
    })

test_that("Check that smaller bridge exponent produces sparser results", {
    cyclopsData <- simulateCyclopsData(nstrata = 1,
                                       nrows = 1000,
                                       ncovars = 25,
                                       model = "logistic")
    cyclopsData <- convertToCyclopsData(cyclopsData$outcomes, cyclopsData$covariates,
                                        modelType = "lr", addIntercept = TRUE,
                                        quiet = FALSE)
    bridge <- createPrior("bridge",
                          exponent = 1)
    bridgeSmall <- createPrior("bridge",
                                exponent = 0.125)

    cyclopsFit <- fitCyclopsModel(cyclopsData,
                                  prior = bridge)
    cyclopsFitSmall <- fitCyclopsModel(cyclopsData,
                                       prior = bridgeSmall)
    expect_true(sum(coef(cyclopsFitSmall) == 0) > sum(coef(cyclopsFit) == 0))
})


