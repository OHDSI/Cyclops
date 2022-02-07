library("testthat")
library("survival")
library("Andromeda")

context("test-pooledBernoulli.R")

test_that("Test data.frame to data for lr", {
    # library(Cyclops)
    # library("testthat")
    test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1, x2
                       0, 5, 0, 0, 1
                       0, 4, 1, 1, 2
                       0, 3, 0, 0, 1
                       0, 2, 0, 1, 0
                       0, 2, 1, 0, 0
                       0, 2, 1, 0, 1
                       0, 1, 0, 0, 0
                       0, 1, 1, 2, 1 ")
    testLong <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1, x2
                       0, 5, 0, 0, 1
                       0, 4, 0, 0, 1
                       0, 3, 0, 0, 1
                       0, 2, 0, 0, 1
                       0, 1, 0, 0, 1
                       0, 4, 1, 1, 2
                       0, 3, 0, 1, 2
                       0, 2, 0, 1, 2
                       0, 1, 0, 1, 2
                       0, 3, 0, 0, 1
                       0, 2, 0, 0, 1
                       0, 1, 0, 0, 1
                       0, 2, 0, 1, 0
                       0, 1, 0, 1, 0
                       0, 2, 1, 0, 0
                       0, 1, 0, 0, 0
                       0, 2, 1, 0, 1
                       0, 1, 0, 0, 1
                       0, 1, 0, 0, 0
                       0, 1, 1, 2, 1 ")
    tolerance <- 1E-4

    nCovars <- 2
    covariates <- data.frame(stratumId = 0,
                             rowId = rep(1:nrow(test),nCovars),
                             covariateId = rep(1:nCovars,each = nrow(test)),
                             covariateValue = c(test$x1,test$x2))
    outcomes <- data.frame(stratumId = 0,
                           rowId = 1:nrow(test),
                           y = test$status,
                           time = test$length,
                           timeEffects = test$length) # linear effect
    covariates <- covariates[covariates$covariateValue != 0,]

    covariatesLong <- data.frame(stratumId = c(rep(1, 5), rep(2, 4), rep(3, 3), rep(4, 2), rep(5, 2), rep(6, 2), 7, 8),
                                 rowId = rep(1:nrow(testLong),nCovars),
                                 covariateId = rep(1:nCovars,each = nrow(testLong)),
                                 covariateValue = c(testLong$x1,testLong$x2))
    outcomesLong <- data.frame(stratumId = c(rep(1, 5), rep(2, 4), rep(3, 3), rep(4, 2), rep(5, 2), rep(6, 2), 7, 8),
                               rowId = 1:nrow(testLong),
                               y = testLong$status,
                               time = testLong$length,
                               timeEffects = testLong$length)
    covariatesLong <- covariatesLong[covariatesLong$covariateValue != 0,]

    # real lr (long outcome and long covariates)
    cyclopsDataLR <- convertToCyclopsData(outcomesLong,covariatesLong,modelType = "lr")
    fitLR <- fitCyclopsModel(cyclopsDataLR)

    # efficient pooled lr (long outcome and short covariates)
    cyclopsDataPLR <- convertToCyclopsData(outcomesLong,covariates,modelType = "plr")
    fitPLR <- fitCyclopsModel(cyclopsDataPLR)

    expect_equal(coef(fitLR), coef(fitPLR), tolerance = tolerance)
    expect_equivalent(logLik(fitLR),logLik(fitPLR))
})
