library("testthat")

#
# Small Logistic regression
#

test_that("Small logistic dense regression", {
    set.seed(123)
    counts <- c(1, 0, 0, 0, 1, 0, 1, 1, 1)
    treatment <- rnorm(9,1,.1) + c(0,1,1,1,1,0,0,0,0)
    treatment2 <- rnorm(9,1,0.5) + c(1,0,0,0,1,1,0,1,1)
    control <- createControl(gpu = "testing")
    cyclopsData <- createCyclopsData(
        counts ~ treatment + treatment2,
        modelType = "lr")
    cyclopsFit <- fitCyclopsModel(cyclopsData, forceNewObject = TRUE, control = control)
    coefficients(cyclopsFit)
})

