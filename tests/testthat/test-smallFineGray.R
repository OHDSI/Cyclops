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
    max(abs(goldFit$coef - coef(cyclopsFit)))

    tolerance <- 1E-4
    #expect_equal(coef(cyclopsFit), coef(goldRight), tolerance = tolerance)
})

test_that("Check very small Fine-Gray example with time-ties, but no failure ties", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,1,0,0
                       0, 3,2,2,0
                       0, 3,1,0,1
                       0, 3,2,0,1
                       0, 2,1,1,1
                       0, 2,0,1,0
                       0, 2,0,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", weights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$length, test$event, cbind(test$x1, test$x2), variance = FALSE)
    max(abs(goldFit$coef - coef(cyclopsFit)))

    tolerance <- 1E-4
    #expect_equal(coef(cyclopsFit), coef(goldRight), tolerance = tolerance)
})
