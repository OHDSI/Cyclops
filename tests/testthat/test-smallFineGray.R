library("testthat")
library("survival")
library("cmprsk")
library("crrSC")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check that Cox fits only have 2 outcome identifies", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,2,2,0
                       0, 3,  0,0,1
                       0, 2.5,2,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")
    outcomes <- data.frame(rowId = 1:7,
                           time = c(4, 3.5, 3, 2.5, 2, 1.5, 1),
                           y = c(1, 2, 0, 0, 1, 1, 1))

    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    expect_error(ptr <- convertToCyclopsData(outcomes, covariates, modelType = "cox"))

    andro <- Andromeda::andromeda(outcomes = outcomes, covariates = covariates)
    expect_error(ptr <- convertToCyclopsData(andro$outcomes, andro$covariates, modelType = "cox"))
})

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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$length, test$event, cbind(test$x1, test$x2), variance = FALSE)

    dataSurv <- finegray(Surv(time = test$length, event = test$event, type = 'mstate') ~ test$x1 + test$x2)
    survFit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ `test$x1` + `test$x2`,
                        weight = fgwt, data = dataSurv)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFit), survFit$coefficients, tolerance = tolerance)

})

test_that("Check very small stratified Fine-Gray example with no ties", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,2,2,0
                       0, 3,  0,0,1
                       0, 2.5,2,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    #Stratified outcome with non-stratified censor weights
    #Cyclops
    fgDatStrat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtrStrat <- Cyclops::createCyclopsData(fgDatStrat$surv ~ test$x1 + strata(test$x2),
                                               modelType = "fgr",
                                               censorWeights = fgDatStrat$weights)
    cyclopsFitStrat <- Cyclops::fitCyclopsModel(dataPtrStrat)

    #survival
    dataSurv <- finegray(Surv(time = length, event = event, type = 'mstate') ~ x1 + strata(x2),
                         data = test)
    dataSurv$strata <- sort(test$x2)
    survFitStrat <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x1 + strata(strata),
                          weight = fgwt, data = dataSurv)

    #Stratified outcome with stratified censor weights
    #Cyclops
    fgDatStratWeights <- Cyclops:::getFineGrayWeights(test$length, test$event, strata = strata(test$x2))
    dataPtrStratWeights <- Cyclops::createCyclopsData(fgDatStratWeights$surv ~ test$x1 + strata(test$x2),
                                                      modelType = "fgr",
                                                      censorWeights = fgDatStratWeights$weights)
    goldStrat <- crrSC::crrs(test$length, test$event, test$x1, strata = strata(test$x2), ctype = 1)
    cyclopsFitStratWeights <- Cyclops::fitCyclopsModel(dataPtrStratWeights)

    #survival
    doSurvStrat <-  function(df){
        surv <- finegray(Surv(time = length, event = event, type = 'mstate') ~ x1,
                         data = df, count = "rep")
        return(surv)
    }

    newTest <- test %>% split(f = test$x2)
    newTestStratWeights <- lapply(newTest, doSurvStrat) %>% bind_rows(.id = "strata")
    survFitStratWeights <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x1 + strata(strata),
                                 weight=fgwt, data=newTestStratWeights)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFitStrat), survFitStrat$coefficients, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStratWeights), survFitStratWeights$coefficients, tolerance = tolerance)


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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:7, time = test$length, y = test$event, censorWeights = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "fgr")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)

    woWeights <- data.frame(rowId = 1:7, time = test$length, y = test$event)
    woPtr <- convertToCyclopsData(woWeights, covariates, modelType = "fgr")
    woFit <- fitCyclopsModel(woPtr)
    expect_equal(coef(woFit), coef(sparseFit))
})

test_that("Check very small Fine-Gray example with no ties using Andromeda", {
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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:7, time = test$length, y = test$event, censorWeights = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    andro <- andromeda(outcomes = outcomes, covariates = covariates)

    dataPtr <- convertToCyclopsData(andro$outcomes, andro$covariates, modelType = "fgr")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)

    woWeights <- data.frame(rowId = 1:7, time = test$length, y = test$event)

    woAndro <- andromeda(outcomes = woWeights, covariates = covariates)

    woPtr <- convertToCyclopsData(woAndro$outcomes, woAndro$covariates, modelType = "fgr")
    woFit <- fitCyclopsModel(woPtr)
    expect_equal(coef(woFit), coef(sparseFit))
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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:7, time = test$length, y = test$event, censorWeights = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 6, 7, 7),
                             covariateId = c(1, 2, 2, 1, 1, 1, 2),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "fgr")
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
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    denseFit <- Cyclops:::fitCyclopsModel(dataPtr)

    outcomes <- data.frame(rowId = 1:9, time = test$length, y = test$event, censorWeights = fgDat$weights)
    covariates <- data.frame(rowId = c(2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9),
                             covariateId = c(1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2),
                             covariateValue = c(2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "fgr")
    sparseFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-8
    expect_equivalent(coef(denseFit), coef(sparseFit), tolerance = tolerance)
})

test_that("Check very small Fine-Gray versus Cox for 0/1 events (censorWeights undefined for Cox model).", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,0,2,0
                       0, 3,  0,0,1
                       0, 2.5,0,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    cyclopsFit1 <- Cyclops:::fitCyclopsModel(dataPtr)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox")
    cyclopsFit2 <- Cyclops:::fitCyclopsModel(dataPtr)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit1), coef(cyclopsFit2), tolerance = tolerance)
})

test_that("Check very small Fine-Gray versus Cox for 0/1 events (censorWeights defined, but not used, for Cox model).", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,0,2,0
                       0, 3,  0,0,1
                       0, 2.5,0,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "fgr", censorWeights = fgDat$weights)
    cyclopsFit1 <- Cyclops:::fitCyclopsModel(dataPtr)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox", censorWeights = fgDat$weights)
    cyclopsFit2 <- Cyclops:::fitCyclopsModel(dataPtr)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit1), coef(cyclopsFit2), tolerance = tolerance)
})

test_that("Check very small Cox example with weights and censor weights defined (the latter is unused for Cox models).", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,1,2,0
                       0, 3,  0,0,1
                       0, 2.5,1,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    fgDat <- Cyclops:::getFineGrayWeights(test$length, test$event)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox",
                                           weights = c(1, 1, 1, 0, 0, 1, 1), censorWeights = fgDat$weights)
    cyclopsFit1 <- Cyclops:::fitCyclopsModel(dataPtr)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$x1 + test$x2, modelType = "cox",
                                           weights = c(1, 1, 1, 0, 0, 1, 1))
    cyclopsFit2 <- Cyclops:::fitCyclopsModel(dataPtr)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit1), coef(cyclopsFit2), tolerance = tolerance)
})
