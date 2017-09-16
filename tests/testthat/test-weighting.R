library("testthat")
library("survival")
library("Cyclops")


## Single covariate: unweighted with ties

test <- read.table(header=T, sep = ",", text = "
                   start, length, status, x1
                   0, 1,  1,2
                   0, 1,  0,0
                   0, 2,  1,0
                   0, 2,  1,0
                   0, 2,  0,1
                   0, 3,  0,0
                   0, 4,  1,1
                   0, 5,  0,0  ")

gold <- coxph(Surv(length, status) ~ x1, test, ties = "breslow")

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData)

expect_equal(coef(gold), coef(cyclopsFit))
logLik(gold)
logLik(cyclopsFit)



## Single covariate: weighted without ties

test <- read.table(header=T, sep = ",", text = "
                   start, length, status, x1
                   0, 1,  1,1
                   0, 1,  0,1
                   0, 2,  1,0
                   0, 2,  0,1
                   0, 3,  0,0
                   0, 4,  1,1
                   0, 5,  0,0 ")

weights <- c(1,2,3,2,1,2,1)

gold <- coxph(Surv(length, status) ~ x1, test, weights = weights)

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights)

expect_equal(coef(gold), coef(cyclopsFit))
logLik(gold)
logLik(cyclopsFit)


## Single covariate: weighted with ties

test <- read.table(header=T, sep = ",", text = "
                   start, length, status, x1
                   0, 5, 0, 0
                   0, 4, 1, 1
                   0, 3, 0, 0
                   0, 2, 0, 1
                   0, 2, 1, 0
                   0, 2, 1, 0
                   0, 1, 0, 0
                   0, 1, 1, 2 ")

weights <- c(1,2,1,2,4,3,2,1)

gold <- coxph(Surv(length, status) ~ x1, test, weights, ties = "breslow")

cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
cyclopsFit <- fitCyclopsModel(cyclopsData, weights=weights)

expect_equal(coef(gold), coef(cyclopsFit))
logLik(gold)
logLik(cyclopsFit)
