library("testthat")

context("test-smallMM.R")

#
# Small MM regression
#


test_that("Small Bernoulli dense regression with MM algorithm", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    tolerance <- 1E-3

    gold <- glm(y ~ log_bid, family = binomial()) # gold standard

    data <- createCyclopsData(y ~ log_bid, modelType = "lr")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(algorithm = "mm"))
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})

test_that("Small Bernoulli sparse regression with MM algorithm", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    tolerance <- 1E-3

    gold <- glm(y ~ log_bid, family = binomial()) # gold standard

    data <- createCyclopsData(y ~ 1, sparseFormula = ~ log_bid, modelType = "lr")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(algorithm = "mm"))
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})

test_that("Small Bernoulli indicator regression with MM algorithm", {
    binomial_bid <- c(0,0,0,0,0,0,1,1,1,1,1)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    tolerance <- 1E-3

    gold <- glm(y ~ log_bid, family = binomial()) # gold standard

    data <- createCyclopsData(y ~ 1, indicatorFormula = ~ log_bid, modelType = "lr")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(algorithm = "mm"))
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})
