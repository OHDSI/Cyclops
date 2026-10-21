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

test_that("Check adapative profiling likelihood with derivative information", {
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

    out <- getCyclopsProfileLogLikelihood(fit, "x1", bounds = c(0, 2), initialGridSize = 10,
                                          returnDerivatives = TRUE)

    pt <- which(out$value == max(out$value))
    argMax <- out[pt, ]
    expect_equal(0.0, argMax$derivative, tolerance = 1E-4)
    increasing <- out[pt - 1, ]
    expect_gt(increasing$derivative, 0.0)
})

test_that("Check adapative profiling likelihood, other covariate is perfect predictor", {
	skip("Needs debugging") # DEVELOP
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


test_that("Check adaptive profiling refines only within maxDrop of the maximum", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 200, nrows = 2000, ncovars = 2, effectSizeSd = 0.5,
                               zeroEffectSizeProp = 0, model = "poisson")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "cpr",
                                 addIntercept = FALSE, quiet = TRUE)
    fit <- fitCyclopsModel(data)

    expect_error(getCyclopsProfileLogLikelihood(fit, "1", bounds = c(-3, 3), maxDrop = 0),
                 "maxDrop")

    full <- getCyclopsProfileLogLikelihood(fit, "1", bounds = c(-3, 3), tolerance = 0.1)
    drop <- getCyclopsProfileLogLikelihood(fit, "1", bounds = c(-3, 3), tolerance = 0.1,
                                           maxDrop = 5)

    expect_lt(nrow(drop), nrow(full))
    expect_equal(range(drop$point), c(-3, 3))
    expect_equal(max(drop$value), max(full$value))

    # Within maxDrop of the maximum, interpolating the profile is as accurate as without maxDrop
    near <- full$value >= max(full$value) - 5
    interpolated <- approx(drop$point, drop$value, xout = full$point[near])$y
    expect_lt(max(abs(interpolated - full$value[near])), 0.1)
})

test_that("Check profile likelihood does not depend on the number of threads", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 200, nrows = 2000, ncovars = 4, effectSizeSd = 0.5,
                               zeroEffectSizeProp = 0, model = "poisson")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "cpr",
                                 addIntercept = FALSE, quiet = TRUE)
    x <- seq(-1, 1, length.out = 9)
    profiles <- lapply(c(1, 2), function(threads) {
        # Fixed coefficients stay fixed in every thread
        fit <- fitCyclopsModel(data, fixedCoefficients = c(FALSE, TRUE, FALSE, FALSE),
                               control = createControl(threads = threads), forceNewObject = TRUE)
        list(adaptive = getCyclopsProfileLogLikelihood(fit, "1", bounds = c(-3, 3), tolerance = 0.1),
             grid = getCyclopsProfileLogLikelihood(fit, "3", x = x),
             single = sapply(x, function(v) getCyclopsProfileLogLikelihood(fit, "3", x = v)$value))
    })
    expect_identical(profiles[[1]], profiles[[2]])

    # Warm starts reach the same profile as fits that each start from the estimate
    expect_equal(profiles[[1]]$grid$value, profiles[[1]]$single)
})
