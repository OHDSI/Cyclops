library("testthat")
library("survival")

test_that("Check very small Cox example with weighting", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 4,  0,0,1
                       0, 3.5,1,2,0
                       0, 3.5,1,2,0
                       0, 3.5,1,2,0
                       0, 3.5,0,2,0
                       0, 3.5,0,2,0
                       0, 3,  0,0,1
                       0, 3,  0,0,1
                       0, 2.5,1,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    gold <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")

    # Duplicate some rows, so we can reweigh later:
    #testDup <- rbind(test, test[c(1,2,3),])
    weights <- c(0.5, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1)

    weights = (1:12)/12

    goldDup <- coxph(Surv(length, event) ~ x1 + x2, testDup, ties = "breslow")

    expect_equal(coef(gold), coef(goldDup))

    cyclopsData <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData)

    expect_equal(coef(gold), coef(cyclopsFit))
    expect_equal(coef(goldDup), coef(cyclopsFit))
})

## unweighted with ties

library("Cyclops")
library("survival")
library("testthat")

test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1
                   0, 1,  1,2
                   0, 1,  0,0
                   0, 2,  1,1
                   0, 2,  1,1
                   0, 2,  1,0
                   0, 2,  0,1
                   0, 3,  0,0
                   0, 4,  1,1
                   0, 5,  0,0  ")

gold <- coxph(Surv(length, status) ~ x1, test, ties = "breslow")

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData)

coef(gold)
logLik(gold)

coef(cyclopsFit)
logLik(cyclopsFit)


## weighted without ties

test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1
                   0, 1,  1,1
                   0, 1,  0,1
                   0, 2,  1,2
                   0, 2,  0,1
                   0, 3,  0,0
                   0, 4,  1,1
                   0, 5,  0,0  ")

weights <- c(1,2,3,2,1,2,1)

gold <- coxph(Surv(length, status) ~ x1, test, weights = weights)

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights)

coef(gold)
logLik(gold)

coef(cyclopsFit)
logLik(cyclopsFit)


## weighted with ties

test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1
                   0, 1,  1,2
                   0, 1,  0,0
                   0, 2,  1,1
                   0, 2,  1,1
                   0, 2,  1,0
                   0, 2,  0,1
                   0, 3,  0,0
                   0, 4,  1,1
                   0, 5,  0,0  ")

weights <- c(1,2,3,4,3,2,1,2,1)

gold <- coxph(Surv(length, status) ~ x1, test, weights, ties = "breslow")

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData, weights=weights)

coef(gold)
logLik(gold)

coef(cyclopsFit)
logLik(cyclopsFit)

