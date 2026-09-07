test_that("aggregation reuses ordinary sparse CCD and the shared prior once", {
    set.seed(20260907)
    x <- Matrix::Matrix(matrix(rbinom(600, 1, 0.4), ncol = 3), sparse = TRUE)
    colnames(x) <- c("a", "b", "c")
    y <- rbinom(nrow(x), 1, plogis(-0.2 + x[, 1] - 0.7 * x[, 2]))
    data <- createCyclopsData(y ~ 1, data = data.frame(y), sx = x, modelType = "lr")
    pooled <- createCyclopsData(y ~ 1, data = data.frame(y = rep(y, 2)),
                                sx = rbind(x, x), modelType = "lr")
    prior <- createPrior("laplace", variance = 1, exclude = "(Intercept)")
    control <- createControl(convergenceType = "lange", tolerance = 1e-12, seed = 20260907)
    reference <- fitCyclopsModel(pooled, prior, control, startingCoefficients = rep(0, 4))
    coordinates <- integer()
    control$aggregate <- function(coordinate, statistics) {
        coordinates <<- c(coordinates, coordinate)
        summed <- if (coordinate == 0) 1 else 1:2
        statistics[summed] <- 2 * statistics[summed]
        statistics
    }
    fit <- fitCyclopsModel(data, prior, control, startingCoefficients = rep(0, 4))
    expect_identical(fit$return_flag, "SUCCESS")
    expect_equal(coef(fit), coef(reference), tolerance = 1e-9)
    expect_equal(predict(fit), predict(reference)[seq_along(y)], tolerance = 1e-9, ignore_attr = TRUE)
    expect_equal(2 * fit$log_likelihood + fit$log_prior,
                 reference$log_likelihood + reference$log_prior, tolerance = 1e-9)
    expect_identical(fit$iterations, reference$iterations)
    expect_equal(coordinates, c(0L, rep(c(1:4, 0L), fit$iterations)))
})

test_that("failed aggregation releases the native callback", {
    x <- Matrix::Matrix(cbind(c(0, 1, 0, 1), c(1, 1, 0, 0)), sparse = TRUE)
    data <- createCyclopsData(y ~ 1, data = data.frame(y = c(0, 1, 1, 0)), sx = x, modelType = "lr")
    prior <- createPrior("laplace", variance = 1, exclude = "(Intercept)")
    control <- createControl(convergenceType = "lange", aggregate = function(coordinate, statistics) {
        statistics[1] <- NA_real_
        statistics
    })
    expect_error(fitCyclopsModel(data, prior, control, startingCoefficients = c(0, 0, 0)),
                 "Invalid aggregated statistics")
    control$aggregate <- NULL
    fit <- fitCyclopsModel(data, prior, control, startingCoefficients = c(0, 0, 0))
    expect_identical(fit$return_flag, "SUCCESS")
    expect_equal(unname(coef(fit)), c(0, 0, 0))
})
