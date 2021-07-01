library("testthat")
library("survival")
library("cmprsk")
library("crrSC")

test_that("Check medium Fine-Gray example with no ties", {
    data(bce)
    test <- bce

    fgDat <- Cyclops:::getFineGrayWeights(test$time, test$type)
    dataPtr <- Cyclops:::createCyclopsData(fgDat$surv ~ test$trt + test$nnodes, modelType = "fgr", censorWeights = fgDat$weights)
    cyclopsFit <- Cyclops:::fitCyclopsModel(dataPtr)
    goldFit <- crr(test$time, test$type, cbind(test$trt, test$nnodes), variance = FALSE)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
})

test_that("Check medium stratified Fine-Gray example with no ties", {
    data(bce)
    test <- bce %>% arrange(age)

    fgDat <- Cyclops:::getFineGrayWeights(ftime = test$time, fstatus = test$type)
    dataPtr <- Cyclops::createCyclopsData(fgDat$surv ~ test$trt, modelType = "fgr", censorWeights = fgDat$weights)
    cyclopsFit <- Cyclops::fitCyclopsModel(dataPtr)

    fgDatStratWeights <- Cyclops:::getFineGrayWeights(ftime = test$time, fstatus = test$type, strata = strata(test$age)) #fix
    dataPtrStratWeights <- Cyclops::createCyclopsData(fgDatStrat$surv ~ test$trt + strata(test$age), modelType = "fgr", censorWeights = fgDatStrat$weights)
    cyclopsFitStratWeights <- Cyclops::fitCyclopsModel(dataPtrStratWeights)

    dataPtrStrat <- Cyclops::createCyclopsData(fgDatStrat$surv ~ test$trt + strata(test$age), modelType = "fgr", censorWeights = fgDatNonStrat$weights)
    cyclopsFitStrat <- Cyclops::fitCyclopsModel(dataPtrStrat)

    dataPtrWeights <- Cyclops::createCyclopsData(fgDatNonStrat$surv ~ test$trt, modelType = "fgr", censorWeights = fgDatStratWeights$weights)
    cyclopsFitWeights <- Cyclops::fitCyclopsModel(dataPtrWeights)

    goldFit <- cmprsk::crr(test$time, test$type, test$trt)
    goldFitStratWeights <- crrSC::crrs(test$time, test$type, test$trt, strata = strata(test$age), ctype = 1)
    goldFitStrat <- crrSC::crrs(test$time, test$type, test$trt, strata = strata(test$age), ctype = 2)

    survStrat <- finegray(Surv(time = time, event = type, type = 'mstate') ~ trt + strata(age),
                          data = test)
    survStrat$strata <- test %>% filter(test$age != 84, test$age != 65) %>% pull(age)
    time <- test %>% filter(test$age != 84, test$age != 65) %>% pull(time)
    survFitStrat <- coxph(Surv(fgstart, fgstop, fgstatus) ~ trt + strata(strata),
                     weight=fgwt, data=survStrat)

    surv <- finegray(Surv(time = time, event = type, type = 'mstate') ~ trt,
                     data = test)
    survFit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ trt,
                     weight=fgwt, data=surv)

    #Compare Cyclops to crrSC
    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFit), goldFit$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStratWeights), goldFitStratWeights$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStrat), goldFitStrat$coef, tolerance = tolerance)

    #Compare Cyclops to survival
    expect_equivalent(coef(cyclopsFit), survFit$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStrat), survFitStrat$coef, tolerance = tolerance)

    #Expect that all three models yield unique results
    expect_false(coef(cyclopsFit) == coef(cyclopsFitStrat))
    expect_false(coef(cyclopsFit) == coef(cyclopsFitStratWeights))
    expect_false(coef(cyclopsFitStratWeights) == coef(cyclopsFitStrat))
})
