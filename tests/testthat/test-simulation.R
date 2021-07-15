library("testthat")

context("test-simulation.R")

#
# Simulation
#

test_that("Check logistic output from other programs", {
    set.seed(123)
    tolerance <- 1E-2
    data <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 2, zeroEffectSizeProp = 0.0, eCovarsPerRow = 0.5,
                                model = "logistic")
    suppressWarnings(cyclopsFit <- fitCyclopsSimulation(data, model = "logistic", useCyclops = TRUE))
    otherFit <- fitCyclopsSimulation(data, model = "logistic", useCyclops = FALSE)
    expect_equal(cyclopsFit$coef, otherFit$coef, tolerance = tolerance)
})

test_that("Check survival output from other programs", {
    set.seed(123)
    tolerance <- 1E-2
    data <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 2, zeroEffectSizeProp = 0.0, eCovarsPerRow = 0.5,
                                model = "survival")
    suppressWarnings(cyclopsFit <- fitCyclopsSimulation(data, model = "survival", useCyclops = TRUE))
    otherFit <- fitCyclopsSimulation(data, model = "survival", useCyclops = FALSE)
    expect_equal(cyclopsFit$coef, otherFit$coef, tolerance = tolerance)

    plot <- Cyclops:::plotCyclopsSimulationFit(cyclopsFit, log(data$effectSizes$rr), label = "Cyclops")
    expect_false(is.null(plot))
})

test_that("Check simulation edge cases", {
    expect_error(simulateCyclopsData(model = "not_known"),
                 "Unknown model")

    expect_equal(mse(goldStandard = c(0,2), estimates = c(0,0)), 2)
})
