library("testthat")
library("glmnet")

context("test-pooledBernoulli.R")

test_that("Test data.frame to data for plr", {
    tolerance <- 1E-4

    test <- read.table(header=T, sep = ",", text = "
                       time, status, x1, x2
                       5, 0, 0, 1
                       4, 1, 1, 2
                       3, 0, 0, 1
                       2, 0, 1, 0
                       2, 1, 0, 0
                       2, 1, 0, 1
                       1, 0, 0, 0
                       1, 1, 2, 1 ")

    nCovars <- 2
    sparseCov <- data.frame(stratumId = 0,
                            rowId = rep(1:nrow(test),nCovars),
                            covariateId = rep(1:nCovars,each = nrow(test)),
                            covariateValue = c(test$x1,test$x2))
    denseCov <- data.frame(rowId = 1:dim(test)[1],
                           x1 = test$x1,
                           x2 = test$x2)
    shortCov <- sparseCov[sparseCov$covariateValue != 0,]

    longOut <- convertToLongOutcome(time = test$time, status = test$status)
    longCov <- merge(longOut[,1:2], shortCov, by.x = "stratumId", by.y = "rowId", all.y = TRUE)
    wideData <- merge(longOut, denseCov, by.x = "stratumId", by.y = "rowId")

    # gold lr
    goldLR <- glm(y ~ x1 + x2, data = wideData, family = binomial())

    # real lr (long outcome and long covariates)
    cyclopsDataLR <- convertToCyclopsData(outcomes = longOut,
                                          covariates = longCov,
                                          modelType = "lr")
    fitLR <- fitCyclopsModel(cyclopsDataLR)

    # efficient pooled lr (long outcome and short covariates)
    cyclopsDataPLR <- convertToCyclopsData(outcomes = cbind(longOut, timeEffects = longOut$time), # linear effect
                                           covariates = shortCov,
                                           modelType = "plr")
    fitPLR <- fitCyclopsModel(cyclopsDataPLR)

    expect_equal(unname(coef(goldLR)), unname(coef(fitLR)), tolerance = tolerance)
    expect_equal(coef(fitLR), coef(fitPLR), tolerance = tolerance)
    expect_equivalent(logLik(fitLR),logLik(fitPLR))
})
