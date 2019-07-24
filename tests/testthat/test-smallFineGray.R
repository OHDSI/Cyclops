library("testthat")
library("survival")
library("cmprsk")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check very small Fine-Gray example with no ties", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,2,2,0
                       0, 3,  0,0,1
                       0, 2.5,2,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$length, test$event, cbind(test$x1, test$x2), variance = FALSE)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
})

test_that("Check very small Fine-Gray example with time ties, but no failure ties", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,1,0,0
                       0, 3,2,2,0
                       0, 3,2,0,1
                       0, 3,1,0,1
                       0, 2,0,1,0
                       0, 2,0,1,0
                       0, 2,1,1,1")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$length, test$event, cbind(test$x1, test$x2), variance = FALSE)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
})

test_that("Check very small Fine-Gray example with time ties and failure ties", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3, 1,2,0
                       0, 3,  1,0,1
                       0, 3, 2,0,1
                       0, 2,  0,1,1
                       0, 1, 2,1,0
                       0, 1, 0,1,1,
                       0, 1, 1,1,0,
                       0, 1, 1,1,1")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$length, test$event, cbind(test$x1, test$x2), variance = FALSE)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
})

test_that("Check very small Fine-Gray example with no ties (sparse vs. dense)", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,2,2,0
                       0, 3,  0,0,1
                       0, 2.5,2,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:7, time = test$length, y = test$event, weight = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "cox")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)
})


test_that("Check very small Fine-Gray example with time ties, but no failure ties (sparse vs. dense)", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,1,0,0
                       0, 3,2,2,0
                       0, 3,2,0,1
                       0, 3,1,0,1
                       0, 2,0,1,0
                       0, 2,0,1,0
                       0, 2,1,1,1")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:7, time = test$length, y = test$event, weight = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 6, 7, 7),
                             covariateId = c(1, 2, 2, 1, 1, 1, 2),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "cox")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)
})

test_that("Check very small Fine-Gray example with time ties and failure ties (sparse vs. dense)", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3, 1,2,0
                       0, 3,  1,0,1
                       0, 3, 2,0,1
                       0, 2,  0,1,1
                       0, 1, 2,1,0
                       0, 1, 0,1,1,
                       0, 1, 1,1,0,
                       0, 1, 1,1,1")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:9, time = test$length, y = test$event, weight = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "cox")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)
})