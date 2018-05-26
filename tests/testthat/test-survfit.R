library("testthat")
library("survival")

context("test-survfit.R")

test_that("Check very small Cox example with no ties, breslow baseline", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3.5,1,2,0
0, 3,  0,0,1
0, 2.5,1,0,1
0, 2,  1,1,1
0, 1.5,0,1,0
0, 1,  1,1,0
")

    goldFit <-  coxph(Surv(start, length, event) ~ x1 + x2, test, ties = "breslow")
    goldSurv <- survfit(goldFit)

    dataPtr <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(dataPtr)
    cyclopsSurv <- survfit(cyclopsFit, type="aalen")

    expect_equal(goldSurv$time, cyclopsSurv$time)
    tolerance <- 1E-4
    expect_equal(goldSurv$surv, cyclopsSurv$surv, tolerance = tolerance)
})

test_that("Check very small Cox example with time-ties, but no failure ties, breslow baseline", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  0,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    goldFit <-  coxph(Surv(start, length, event) ~ x1 + x2, test, ties = "breslow")
    goldSurv <- survfit(goldFit)

    dataPtr <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(dataPtr)
    cyclopsSurv <- survfit(cyclopsFit, type="aalen")

    expect_equal(goldSurv$time, cyclopsSurv$time)
    tolerance <- 1E-4
    expect_equal(goldSurv$surv, cyclopsSurv$surv, tolerance = tolerance)
})

test_that("Check very small Cox example with failure ties, no risk-set contribution after tie, breslow baseline", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 2,  1,0,0
")
    goldFit <-  coxph(Surv(start, length, event) ~ x1 + x2, test, ties = "breslow")
    goldSurv <- survfit(goldFit)

    dataPtr <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(dataPtr)
    cyclopsSurv <- survfit(cyclopsFit, type="aalen")

    expect_equal(goldSurv$time, cyclopsSurv$time)
    tolerance <- 1E-4
    expect_equal(goldSurv$surv, cyclopsSurv$surv, tolerance = tolerance)})

test_that("Check very small Cox example with failure ties, with risk-set contribution after tie, breslow baseline", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    goldFit <-  coxph(Surv(start, length, event) ~ x1 + x2, test, ties = "breslow")
    goldSurv <- survfit(goldFit)

    dataPtr <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(dataPtr)
    cyclopsSurv <- survfit(cyclopsFit, type="aalen")

    expect_equal(goldSurv$time, cyclopsSurv$time)
    tolerance <- 1E-4
    expect_equal(goldSurv$surv, cyclopsSurv$surv, tolerance = tolerance)
})

