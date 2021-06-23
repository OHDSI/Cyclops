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

    fgDatStrat <- Cyclops:::getFineGrayWeights(ftime = test$time, fstatus = test$type, strata = strata(test$age)) #fix
    fgDatNonStrat <- Cyclops:::getFineGrayWeights(ftime = test$time, fstatus = test$type)

    dataPtrStrat <- Cyclops::createCyclopsData(fgDatStrat$surv ~ test$trt + strata(test$age), modelType = "fgr", censorWeights = fgDatStrat$weights)
    goldStrat <- crrSC::crrs(test$time, test$type, test$trt, strata = strata(test$age))
    cyclopsFitStrat <- Cyclops::fitCyclopsModel(dataPtrStrat)

    goldNonStrat <- cmprsk::crr(test$time, test$type, test$trt)
    dataPtrNonStrat <- Cyclops::createCyclopsData(fgDatNonStrat$surv ~ test$trt, modelType = "fgr", censorWeights = fgDatNonStrat$weights)
    cyclopsFitNonStrat <- Cyclops::fitCyclopsModel(dataPtrNonStrat)

    dataPtrcheck <- Cyclops::createCyclopsData(fgDatStrat$surv ~ test$trt + strata(test$age), modelType = "fgr", censorWeights = fgDatStrat$weights)
    cyclopsFitcheck <- Cyclops::fitCyclopsModel(dataPtrcheck, startingCoefficients = c(0), fixedCoefficients = c(TRUE))

    survStrat <- finegray(Surv(time = time, event = type, type = 'mstate') ~ trt + strata(age),
                       data = test)
    survStrat$strata <- test %>% filter(test$age != 84, test$age != 65) %>% pull(age)
    time <- test %>% filter(test$age != 84, test$age != 65) %>% pull(time)
    survFitStrat <- coxph(Surv(fgstart, fgstop, fgstatus) ~ trt + strata(strata),
                     weight=fgwt, data=survfg)

    tolerance <- 1E-4
    expect_equivalent(coef(cyclopsFitNonStrat), goldNonStrat$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStrat), goldStrat$coef, tolerance = tolerance)
    expect_equivalent(coef(cyclopsFitStrat), survFitStrat$coefficients, tolerance = tolerance)
    expect_false(coef(cyclopsFitNonStrat) == goldStrat$coef)
})
