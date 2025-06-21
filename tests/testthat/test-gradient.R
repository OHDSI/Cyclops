library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("gradient", {

    data <- Cyclops::createCyclopsData(Surv(stop, event) ~ (rx - 1) + size, data = bladder, modelType = "cox")

    fit <- Cyclops::fitCyclopsModel(data)

    gradientAtMode <- gradient(fit)

    expect_equivalent(gradientAtMode, rep(0, length(coef(fit))))

    mode <- coef(fit)
    eps <- 1E-2
    Cyclops:::.cyclopsSetBeta(fit$interface,
                              mode + c(-eps, 0))
    grad <- gradient(fit)

    Cyclops:::.cyclopsSetBeta(fit$interface,
                              mode + c(-2*eps, 0))
    h0 <- Cyclops:::.cyclopsGetLogLikelihood(fit$interface)
    central <- (fit$log_likelihood - h0) / (2 * eps)


    expect_equivalent(grad[1], central, tolerance = 1E-3)
})


