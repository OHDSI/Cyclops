library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("gradient", {

    data <- Cyclops::createCyclopsData(Surv(stop, event) ~ (rx - 1) + size, data = bladder, modelType = "cox")

    fit <- Cyclops::fitCyclopsModel(data)

    gradientAtMode <- gradient(fit)

    expect_equivalent(gradientAtMode, rep(0, length(coef(fit))))
})


