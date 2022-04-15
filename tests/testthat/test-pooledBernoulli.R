library("testthat")
library("glmnet")

context("test-pooledBernoulli.R")

test_that("Test data.frame to data for plr without time effects", {
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

    longOut <- convertToLongOutcome(time = test$time, status = test$status)$longOutcome
    longCov <- merge(longOut[,1:2], shortCov, by.x = "stratumId", by.y = "rowId", all.y = TRUE)
    wideData <- merge(longOut, denseCov, by.x = "stratumId", by.y = "rowId")

    # gold lr
    goldLR <- glm(y ~ x1 + x2, data = wideData, family = binomial())

    # real lr (long outcome and long covariates)
    cyclopsDataLR <- convertToCyclopsData(outcomes = longOut,
                                          covariates = longCov,
                                          modelType = "lr")
    # cyclopsDataLR <- createCyclopsData(y ~ x1 + x2,
    #                                    data = wideData,
    #                                    modelType = "lr")
    fitLR <- fitCyclopsModel(cyclopsDataLR)

    # efficient pooled lr (long outcome and short covariates)
    cyclopsDataPLR <- convertToCyclopsData(outcomes = longOut,
                                           covariates = shortCov,
                                           modelType = "plr")
    fitPLR <- fitCyclopsModel(cyclopsDataPLR)

    expect_equal(unname(coef(goldLR)), unname(coef(fitLR)), tolerance = tolerance)
    expect_equal(unname(coef(fitLR)), unname(coef(fitPLR)), tolerance = tolerance)
    expect_equivalent(logLik(fitLR),logLik(fitPLR))
})

test_that("Test data.frame to data for plr with multiple time-dependent covariates", {
    tolerance2 <- 1E-2
    tolerance4 <- 1E-4

    test <- read.table(header=T, sep = ",", text = "
                       time, status, x1, x2
                       5, 0, 0, 1
                       4, 1, 1, 2
                       3, 0, 0, 1
                       2, 0, 1, 0
                       2, 1, 0, 0
                       2, 1, 0, 1
                       1, 0, 0, 0
                       1, 1, 1, 1 ")
    longOut <- convertToLongOutcome(time = test$time, status = test$status, linearEffect = TRUE)$longOutcome
    longOut$timeCubic <- longOut$timeLinear^3
    denseCov <- data.frame(rowId = 1:dim(test)[1],
                           x1 = test$x1,
                           x2 = test$x2)

    wideData <- merge(longOut, denseCov, by.x = "stratumId", by.y = "rowId")

    nCovars <- 2
    shortCov <- data.frame(stratumId = 0,
                           rowId = rep(1:nrow(denseCov),nCovars),
                           covariateId = rep(1:nCovars,each = nrow(denseCov)),
                           covariateValue = c(denseCov$x1, denseCov$x2))


    # gold lr
    goldLR <- glm(y ~ x1 + x2 + timeLinear + timeCubic, data = wideData, family = binomial())

    # real lr (long outcome and long covariates)
    cyclopsDataLR <- createCyclopsData(y ~ x1 + x2 + timeLinear + timeCubic,
                                       data = wideData,
                                       modelType = "lr")
    fitLR <- fitCyclopsModel(cyclopsDataLR)

    # efficient pooled lr (long outcome and short covariates)
    timeEffects <- longOut[,-c(3:4)]
    timeEffects <- timeEffects[sample(1:nrow(timeEffects)), ] # shuffle rows
    cyclopsDataPLR <- convertToCyclopsData(outcomes = longOut,
                                           covariates = shortCov,
                                           timeEffects = timeEffects,
                                           modelType = "plr")
    fitPLR <- fitCyclopsModel(cyclopsDataPLR)

    expect_equal(unname(coef(goldLR)), unname(coef(fitLR)), tolerance = tolerance2)
    expect_equal(unname(coef(fitLR)), unname(coef(fitPLR)), tolerance = tolerance4)
    expect_equivalent(logLik(fitLR),logLik(fitPLR))
})

test_that("Test data.frame to data for plr with time effects as interaction terms", {
    tolerance <- 1E-3

    test <- read.table(header=T, sep = ",", text = "
                       time, status, x1, x2
                       5, 0, 0, 1
                       4, 1, 1, 2
                       3, 0, 0, 1
                       2, 0, 1, 0
                       2, 1, 0, 0
                       2, 1, 0, 1
                       1, 0, 0, 0
                       1, 1, 1, 1 ")
    longOut <- convertToLongOutcome(time = test$time, status = test$status, linearEffect = TRUE)$longOutcome
    denseCov <- data.frame(rowId = 1:dim(test)[1],
                           x1 = test$x1,
                           x2 = test$x2)

    wideData <- merge(longOut, denseCov, by.x = "stratumId", by.y = "rowId")

    nCovars <- 4
    longCov <- data.frame(stratumId = 0,
                          rowId = rep(1:nrow(wideData),nCovars),
                          covariateId = rep(1:nCovars,each = nrow(wideData)),
                          covariateValue = c(wideData$x1, wideData$x2, wideData$time, wideData$x1 * wideData$time))
    nCovars <- 2
    shortCov <- data.frame(stratumId = 0,
                           rowId = rep(1:nrow(denseCov),nCovars),
                           covariateId = rep(1:nCovars,each = nrow(denseCov)),
                           covariateValue = c(denseCov$x1, denseCov$x2))


    # gold lr
    goldLR <- glm(y ~ x1 + x2 + time + x1:time, data = wideData, family = binomial())

    # real lr (long outcome and long covariates)
    # cyclopsDataLR <- convertToCyclopsData(outcomes = longOut[, 1:4],
    #                                       covariates = longCov,
    #                                       modelType = "lr")
    cyclopsDataLR <- createCyclopsData(y ~ x1 + x2+ time + x1:time,
                                       data = wideData,
                                       modelType = "lr")
    fitLR <- fitCyclopsModel(cyclopsDataLR)

    # efficient pooled lr (long outcome and short covariates)
    timeEffects <- longOut[,1:3]
    colnames(timeEffects) <- c("rowId", "stratumId", "linear")
    timeEffects <- timeEffects[sample(1:nrow(timeEffects)), ] # shuffle rows
    timeMap <- data.frame(covariateId = 1,
                          timeEffectId = 0)
    cyclopsDataPLR <- convertToCyclopsData(outcomes = longOut,
                                           covariates = shortCov,
                                           timeEffects = timeEffects,
                                           timeEffectMap = timeMap,
                                           modelType = "plr")
    fitPLR <- fitCyclopsModel(cyclopsDataPLR)

    expect_equal(unname(coef(goldLR)), unname(coef(fitLR)), tolerance = tolerance)
    expect_equal(unname(coef(fitLR)), unname(coef(fitPLR)), tolerance = tolerance)
    expect_equivalent(logLik(fitLR),logLik(fitPLR))
})
