library("testthat")

context("test-largeBernoulli.R")
suppressWarnings(RNGversion("3.5.0"))

#
# Large Bernoulli regression
#

# test_that("Large Bernoulli CCD data file read", {
# 	tolerance <- 1E-4
# 	dataPtr <- readCyclopsData(system.file("extdata/CCD_LOGISTIC_TEST_17var.txt",
# 																		 package="Cyclops"), "lr")
# 	expect_equal(getNumberOfRows(dataPtr), 22296) # Reads rows
# 	expect_equal(getNumberOfStrata(dataPtr), 22296) # Generates unique ids
# 	expect_equal(getNumberOfCovariates(dataPtr), 18) # Adds intercept
#
# 	cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"),
# 												control = createControl(noiseLevel = "silent"))
# 	expect_equal(cyclopsFit$log_likelihood * 2, -1578.046, tolerance = tolerance) # SAS fit
# 	expect_named(coef(cyclopsFit)) # Reads covariate names from file
# 	expect_named(predict(cyclopsFit)) # Reads row names from file
# })

test_that("Separable covariates in logistic regression", {
    set.seed(666)

    simulant <- simulateCyclopsData(nstrata = 1,
                                ncovars = 5,
                                model = "logistic")

    successes <- simulant$outcomes$rowId[simulant$outcomes$y == 1]
    separable <- simulant$covariates[simulant$covariates$covariateId == 3 &
                                         simulant$covariates$rowId %in% successes,]
    separable$covariateId <- 6

    simulant$covariates <- rbind(simulant$covariates, separable)

    data <- convertToCyclopsData(simulant$outcomes, simulant$covariates, modelType = "lr",
                                        addIntercept = TRUE)

    fit <- fitCyclopsModel(data, prior = createPrior("none"))

    expect_error(coef(fit), "did not converge")

    # Use separability condition
    separability <- getUnivariableSeparability(data)
    expect_equal(sum(separability), 1.0)

    scFit <- fitCyclopsModel(data, prior = createPrior("none"),
                             forceNewObject = TRUE,
                             fixedCoefficients = separability)

    expect_equivalent(coef(scFit)[separability], 0.0)

    # Use estimate existence condition
    separability <- is.nan(coef(fit, ignoreConvergence = TRUE))

    ecFit <- fitCyclopsModel(data, prior = createPrior("none"),
                             forceNewObject = TRUE,
                             fixedCoefficients = separability)

    expect_equivalent(coef(ecFit)[separability], 0.0)

    # Use non-separable MLE prior
    nsFit <- fitCyclopsModel(data, prior = createNonSeparablePrior(),
                             forceNewObject = TRUE)
    expect_true(is.na(coef(nsFit)[7]))
})

test_that("Separable covariates in cox regression", {
    set.seed(666)

    simulant <- simulateCyclopsData(nstrata = 1,
                                    ncovars = 5,
                                    model = "survival")

    successes <- simulant$outcomes$rowId[simulant$outcomes$y == 1]
    separable <- simulant$covariates[simulant$covariates$covariateId == 3 &
                                         simulant$covariates$rowId %in% successes,]
    separable$covariateId <- 6

    simulant$covariates <- rbind(simulant$covariates, separable)

    data <- convertToCyclopsData(simulant$outcomes, simulant$covariates, modelType = "cox")

    fit <- fitCyclopsModel(data, prior = createPrior("none"))

    #expect_error(coef(fit), "did not converge") TODO

    separability <- getUnivariableSeparability(data)
    expect_equal(sum(separability), 1.0)

    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           forceNewObject = TRUE,
                           fixedCoefficients = separability)

    expect_equivalent(coef(fit)[separability], 0.0)
})
