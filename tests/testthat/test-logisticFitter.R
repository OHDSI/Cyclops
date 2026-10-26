library("testthat")
library("Matrix")

control <- function(fitter, ...) {
    createControl(noiseLevel = "silent", tolerance = 1E-10, fitter = fitter, ...)
}

expectSameFits <- function(fits, tolerance = 1E-8) {
    expect_equal(fits[[2]]$return_flag, fits[[1]]$return_flag)
    expect_equal(coef(fits[[2]]), coef(fits[[1]]), tolerance = tolerance)
    expect_equal(as.numeric(logLik(fits[[2]])), as.numeric(logLik(fits[[1]])), tolerance = tolerance)
}

# Indicator, sparse, dense and intercept columns
set.seed(123)
n <- 1500
x <- cbind(intercept = 1, age = runif(n), sex = rbinom(n, 1, 0.5), rare = rbinom(n, 1, 0.05),
           common = rbinom(n, 1, 0.4), count = rbinom(n, 1, 0.3) * rpois(n, 3))
y <- rbinom(n, 1, plogis(-1.5 + x[, "age"] + 0.5 * x[, "sex"] + x[, "rare"] - 0.3 * x[, "count"]))
mixedData <- function() {
    createCyclopsData(y = y, dx = Matrix(x[, c("intercept", "age")], sparse = FALSE),
                      ix = Matrix(x[, c("sex", "rare", "common")], sparse = TRUE),
                      sx = Matrix(x[, "count", drop = FALSE], sparse = TRUE), modelType = "lr")
}

test_that("The efficient and legacy logistic fitters agree through a sequence of fits on one engine", {
    ids <- getCovariateIds(mixedData())
    weightings <- list(rep(c(1, 2, 0, 0.5), length.out = n), # weighted views
                       rep(c(1, 0, 1), length.out = n),       # unit-weight views
                       rep(c(1, 2, 0, 0.5), length.out = n),  # the first views again
                       rep(c(2, 0.5), length.out = n),        # no zero weight: the columns
                       NULL)
    run <- function(fitter) {
        data <- mixedData()
        fit <- fitCyclopsModel(data, control = control(fitter),
                               startingCoefficients = rep(0.1, length(ids)))
        out <- list(coef(fit), vcov(fit), confint(fit, as.numeric(ids[c(2, 4)])),
                    getCyclopsProfileLogLikelihood(fit, parm = as.numeric(ids[3]), x = c(0, 0.5))$value,
                    Cyclops:::.cyclopsGetLogLikelihoodHessianDiagonal(data$cyclopsInterfacePtr, ids))
        # Weighted fits come after vcov(), which fails after weights in both fitters
        for (prior in list(createPrior("laplace", 0.05, exclude = as.numeric(ids[1])),
                           createPrior("normal", 0.5, exclude = as.numeric(ids[1])))) {
            for (weights in weightings) {
                fit <- fitCyclopsModel(data, prior = prior, weights = weights,
                                       control = control(fitter))
                out <- c(out, list(fit$return_flag, coef(fit), logLik(fit), predict(fit)))
            }
        }
        out
    }
    expect_equal(run("efficient"), run("legacy"), tolerance = 1E-8)
})

test_that("The efficient and legacy logistic fitters select the same variance by cross-validation", {
    # Ten real effects, so the cross-validation score has a clear maximum; a grid search moves only
    # with rounding there, whereas an automatic search on a flat score can stop elsewhere
    set.seed(42)
    x <- Matrix(rbinom(1000 * 50, 1, 0.2), 1000, sparse = TRUE)
    y <- rbinom(1000, 1, plogis(-1 + as.vector(x %*% rep(c(0.8, -0.6, 0), c(5, 5, 40)))))
    fits <- lapply(c("legacy", "efficient"), function(fitter) {
        data <- createCyclopsData(y = y, dx = Matrix(1, 1000, 1), ix = x, modelType = "lr")
        prior <- createPrior("laplace", exclude = as.numeric(getCovariateIds(data)[1]),
                             useCrossValidation = TRUE)
        fitCyclopsModel(data, prior = prior,
                        control = createControl(noiseLevel = "silent", cvType = "grid", fold = 5,
                                                seed = 1, threads = 2, tolerance = 1E-8,
                                                fitter = fitter))
    })
    expect_equal(fits[[2]]$variance, fits[[1]]$variance, tolerance = 1E-8)
    expectSameFits(fits, tolerance = 1E-6)
})

test_that("The efficient logistic fitter handles fractional outcomes, separation and extreme offsets", {
    set.seed(5)
    z <- rbinom(500, 1, 0.4)
    u <- rnorm(500)
    yz <- rbinom(500, 1, plogis(-1 + z))
    fitBoth <- function(formula, floatingPoint = 64, ...) {
        lapply(c("legacy", "efficient"), function(fitter) {
            fitCyclopsModel(createCyclopsData(formula, modelType = "lr", floatingPoint = floatingPoint),
                            ..., control = control(fitter, maxIterations = 300))
        })
    }
    fractional <- replace(yz, 1:50, 0.5)
    expectSameFits(fitBoth(fractional ~ z + u))

    # Every row with z = 1 has an event: the coefficient of z diverges and stops at a different large
    # value (p (1 - p) of these rows is only rounding error), with the same likelihood
    separated <- ifelse(z == 1, 1, rbinom(500, 1, 0.2))
    fits <- fitBoth(separated ~ z + u)
    expect_equal(fits[[2]]$return_flag, fits[[1]]$return_flag)
    expect_equal(as.numeric(logLik(fits[[2]])), as.numeric(logLik(fits[[1]])), tolerance = 1E-10)

    # exp(xBeta) of 0 and p of exactly 1; in single precision, the fitters differ by less than their
    # distance from double precision
    shift <- rep(c(-800, 0, 40), length.out = 500)
    yz <- ifelse(shift == 0, yz, shift > 0)
    prior <- createPrior("laplace", 0.1, exclude = 1)
    double <- fitBoth(yz ~ z + u + offset(shift), prior = prior)
    expectSameFits(double)
    single <- lapply(fitBoth(yz ~ z + u + offset(shift), floatingPoint = 32, prior = prior), coef)
    expect_lt(max(abs(single[[2]] - single[[1]])), max(abs(single[[1]] - coef(double[[1]]))))
})

test_that("Changing the logistic fitter rebuilds the engine", {
    data <- mixedData()
    efficient <- fitCyclopsModel(data, control = control("efficient"))
    expect_equal(data$fitter, "efficient")
    legacy <- fitCyclopsModel(data, control = control("legacy"))
    expect_equal(data$fitter, "legacy")
    expect_equal(coef(legacy), coef(efficient), tolerance = 1E-10)

    # A control from before fitter existed means "efficient"
    older <- control("legacy")
    older$fitter <- NULL
    fitCyclopsModel(data, control = older)
    expect_equal(data$fitter, "efficient")

    expect_error(createControl(fitter = "other"))
})

test_that("The efficient logistic fitter sees covariates added after a fit", {
    a <- rbinom(n, 1, 0.3)
    rows <- which(a == 1)
    run <- function(fitter) {
        data <- Cyclops:::createSqlCyclopsData(modelType = "lr")
        Cyclops:::loadNewSqlCyclopsDataY(data, NULL, as.numeric(1:n), y, NULL)
        Cyclops:::loadNewSqlCyclopsDataX(data, 1, as.numeric(1:n), rep(1, n), name = "(Intercept)")
        Cyclops:::loadNewSqlCyclopsDataMultipleX(data, rep(2, sum(rows <= n / 2)), rows[rows <= n / 2])
        fitCyclopsModel(data, control = control(fitter))
        Cyclops:::loadNewSqlCyclopsDataMultipleX(data, rep(2, sum(rows > n / 2)), rows[rows > n / 2],
                                                 append = TRUE)
        coef(fitCyclopsModel(data, control = control(fitter)))
    }
    expect_equal(run("efficient"), run("legacy"), tolerance = 1E-10)
})
