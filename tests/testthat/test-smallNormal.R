library("testthat")

context("test-smallNormal.R")

#
# Small Normal regression
#

test_that("Small Normal dense regression with CCD algorithm", {

    x <- log(c(1,5,10,20,30,40,50,75,100,150,200))
    y <- c(0,3,6,7,9,13,17,12,11,14,13)

    tolerance <- 1E-4

    gold <- lm(y ~ x) # gold standard

    data <- createCyclopsData(y ~ x, modelType = "ls")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(tolerance = 1E-8))
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})

test_that("Small Normal sparse regression with CCD algorithm", {

    x <- log(c(1,5,10,20,30,40,50,75,100,150,200))
    y <- c(0,3,6,7,9,13,17,12,11,14,13)

    tolerance <- 1E-4

    gold <- lm(y ~ x) # gold standard

    data <- createCyclopsData(y ~ 1, sparseFormula = ~ x, modelType = "ls")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(tolerance = 1E-8))
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})
