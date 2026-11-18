library("testthat")

test_that("normal-prior logistic fits under fitter super shuffle their order deterministically and reach the cyclic optimum", {
    set.seed(42)
    n <- 3000
    base <- matrix(rbinom(n * 30, 1, 0.15), n, 30)
    X <- cbind(base, base[, 1:10] | base[, 11:20], base[, 1:5] | base[, 6:10] | base[, 21:25]) * 1 # nested rollups
    y <- rbinom(n, 1, plogis(-1.5 + drop(base[, 1:6] %*% rep(0.5, 6))))
    data <- createCyclopsData(y ~ X, modelType = "lr")
    prior <- createPrior("normal", variance = 1, exclude = "(Intercept)")
    fit <- function(fitter, threads = 1, tolerance = 1e-10)
        fitCyclopsModel(data, prior, createControl(algorithm = "ccd", fitter = fitter, convergenceType = "kkt",
                                                   tolerance = tolerance, threads = threads, noiseLevel = "silent"),
                        forceNewObject = TRUE, warnings = FALSE)
    one <- fit("super")
    expect_identical(coef(fit("super")), coef(one)) # seeded by the cycle number
    expect_identical(coef(fit("super", threads = 2)), coef(one))
    expect_equal(coef(one), coef(fit("default")), tolerance = 1e-7) # the same optimum as the cyclic order
})
