library("testthat")

test_that("normal-prior logistic fits under fitter super shuffle their order deterministically and reach the cyclic optimum", {
    set.seed(42)
    n <- 4000 # at least 1000 free covariates: smaller fits keep the cyclic order
    base <- matrix(rbinom(n * 900, 1, 0.02), n, 900)
    X <- cbind(base, base[, 1:300] | base[, 301:600], base[, 1:100] | base[, 101:200] | base[, 601:700]) * 1 # nested rollups
    y <- rbinom(n, 1, plogis(-1.5 + drop(base[, 1:6] %*% rep(0.8, 6))))
    data <- createCyclopsData(y ~ X, modelType = "lr")
    prior <- createPrior("normal", variance = 1, exclude = "(Intercept)")
    fit <- function(fitter, threads = 1, tolerance = 1e-8)
        fitCyclopsModel(data, prior, createControl(algorithm = "ccd", fitter = fitter, convergenceType = "kkt",
                                                   tolerance = tolerance, threads = threads, maxIterations = 1e5,
                                                   noiseLevel = "silent"),
                        forceNewObject = TRUE, warnings = FALSE)
    one <- fit("super")
    expect_identical(coef(fit("super")), coef(one)) # seeded by the cycle number
    expect_identical(coef(fit("super", threads = 2)), coef(one))
    cyclic <- fit("default")
    expect_true(one$iterations != cyclic$iterations) # the shuffled order is in use
    expect_equal(coef(one), coef(cyclic), tolerance = 1e-5) # the same optimum as the cyclic order
})
