library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check evaluate profile likelihood with two variables", {
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
    data <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                              modelType = "cox")
    fit <- fitCyclopsModel(data)

    x <- seq(from = 0.8, to = 0.9, length = 100)
    out <- getCyclopsProfileLogLikelihood(fit, "x1", x)

    argMax <- out$point[which(out$value == max(out$value))]
    expect_equivalent(coef(fit)["x1"], argMax, tolerance = 0.01)
})

test_that("Check evaluate profile likelihood with multiple threads", {
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
    data <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                              modelType = "cox")
    fit <- fitCyclopsModel(data, control = createControl(threads = 2))

    x <- seq(from = 0.8, to = 0.9, length = 100)
    out <- getCyclopsProfileLogLikelihood(fit, "x1", x)

    argMax <- out$point[which(out$value == max(out$value))]
    expect_equivalent(coef(fit)["x1"], argMax, tolerance = 0.01)
})

test_that("Check evaluate profile likelihood with one variable; cold-start", {
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
    data <- createCyclopsData(Surv(length, event) ~ x1, data = test,
                              modelType = "cox")
    fit <- fitCyclopsModel(data, fixedCoefficients = c(TRUE))

    x <- seq(from = 0.3, to = 0.4, length = 100)
    out <- getCyclopsProfileLogLikelihood(fit, "x1", x)
    argMax <- out$point[which(out$value == max(out$value))]

    gold <- fitCyclopsModel(data, forceNewObject = TRUE)
    expect_equivalent(coef(gold)["x1"], argMax, tolerance = 0.01)
    expect_equivalent(coef(fit)["x1"], 0)
})

test_that("Check adapative profiling likelihood", {
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
    data <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                              modelType = "cox")
    fit <- fitCyclopsModel(data)

    expect_error(getCyclopsProfileLogLikelihood(fit, "x1"))
    expect_error(getCyclopsProfileLogLikelihood(fit, "x1", x = c(0,1), bounds = c(0.8, 0.9)))
    expect_error(getCyclopsProfileLogLikelihood(fit, "x1", bounds = c(0.8, 0.8)))

    out <- getCyclopsProfileLogLikelihood(fit, "x1", bounds = c(0, 2), initialGridSize = 10)
    expect_gt(nrow(out), 10)

    argMax <- out$point[which(out$value == max(out$value))]
    expect_equal(coef(fit)["x1"], argMax)
})

test_that("Check adapative profiling likelihood, other covariate is perfect predictor", {

    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,1
0, 4,1,2,1
0, 4,  0,0,0
0, 4,1,0,1
0, 4,  1,1,1
0, 4,0,1,0
0, 4,  1,1,1
")
    data <- createCyclopsData(event ~ x1 + x2, data = test,
                              modelType = "cpr")

    # Recreating combination of circumstances that lead to infinite log likelihood
    # Note: infinite variance would have been selected by cross-validation if we had enough data:
    fit <- fitCyclopsModel(data,
                           prior = createPrior("laplace", variance = Inf, exclude = "x1"))
    expect_warning(getCyclopsProfileLogLikelihood(fit, "x1", bounds = c(0, 2), initialGridSize = 10),
                   "Failing to compute likelihood at entire initial grid")

    # This just produces NaN likelihood:
    fit <- fitCyclopsModel(data)
    expect_warning(getCyclopsProfileLogLikelihood(fit, "x1", bounds = c(0, 2), initialGridSize = 10),
                   "Failing to compute likelihood at entire initial grid")
})

