library("testthat")
library("survival")

test_that("Check very small Cox example with ties, but without weights",{
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

    gold <- coxph(Surv(length, status) ~ x1, test, ties = "breslow")

    cyclopsData <- createCyclopsData(Surv(length, status) ~ x1, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData)

    expect_equal(coef(gold), coef(cyclopsFit))
    expect_equivalent(logLik(gold),logLik(cyclopsFit))

})


test_that("Check very small Cox example with ties, with weights",{
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
    expect_equivalent(logLik(gold),logLik(cyclopsFit))

})
