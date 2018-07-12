library("testthat")

context("test-floatingPoint.R")

#
# FP precision
#

test_that("Double precision", {
	binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
	binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
	binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

	log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
	y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

	tolerance <- 1E-4

	glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard

	dataPtrD <- createCyclopsData(y ~ log_bid, modelType = "lr",
	                              floatingPoint = 64)

	expect_equal(getFloatingPointSize(dataPtrD), 64)

	cyclopsFitD <- fitCyclopsModel(dataPtrD, prior = createPrior("none"),
	                       control = createControl(noiseLevel = "silent"))

	expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
})

test_that("Single precision", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    tolerance <- 1E-4

    glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard

    dataPtrS <- createCyclopsData(y ~ log_bid, modelType = "lr",
                                  floatingPoint = 32)

    expect_equal(getFloatingPointSize(dataPtrS), 32)

    cyclopsFitS <- fitCyclopsModel(dataPtrS, prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))

    expect_equal(coef(cyclopsFitS), coef(glmFit), tolerance = tolerance * 10)
})

test_that("Speed difference", {
    skip("for debugging purposes; simulation takes too long")

    set.seed(123)
    sim <- simulateCyclopsData(nstrata=1,
                               ncovars=1000,
                               nrows=100000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="logistic")

    dataD <- convertToCyclopsData(outcomes = sim$outcomes,
                                  covariates = sim$covariates, modelType = "lr",
                                  floatingPoint = 64)

    dataS <- convertToCyclopsData(outcomes = sim$outcomes,
                                  covariates = sim$covariates, modelType = "lr",
                                  floatingPoint = 32)

    system.time(
        fitD <- fitCyclopsModel(cyclopsData = dataD,
                                prior = createPrior("laplace", variance = 1),
                                control = createControl(fold = 10, cvRepetitions = 1,
                                                        startingVariance = 0.1, noiseLevel = "silent",
                                                        seed = 123),
                                forceNewObject = TRUE))

    system.time(
        fitS <- fitCyclopsModel(cyclopsData = dataS,
                                prior = createPrior("laplace", variance = 1),
                                control = createControl(fold = 10, cvRepetitions = 1,
                                                        startingVariance = 0.1, noiseLevel = "silent",
                                                        seed = 123),
                                forceNewObject = TRUE))

    expect_equal(coef(fitD)[1:10], coef(fitS)[1:10], tolerance = 1E-3)
})

