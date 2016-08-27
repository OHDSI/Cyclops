library("testthat")

#
# ABRIDGE regression
#

test_that("Cyclops simulation routines", {
    # Generate data
    set.seed(666)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 10000, ncovars = 100,
                        effectSizeSd = 1, zeroEffectSizeProp = 0.9, eCovarsPerRow = 10,
                        model = "survival")
    data <- convertToCyclopsData(sim$outcomes,
                                 sim$covariates,
                                 modelType = "cox")

    # Fit model
    abridge <- fitCyclopsModel(data,
                               prior = createAbridgePrior())

    # Examine estimates vis-a-vis simulation conditions
    which(log(sim$effectSizes$rr) != 0)
    which(coef(abridge) != 0)
})

test_that("ABRIDGE simulated logistic regression", {
    set.seed(666)
    p <- 20
    n <- 1000

    beta1 <- c(0.5, 0, 0, -1, 1.2)
    beta2 <- seq(0, 0, length = p - length(beta1))
    beta <- c(beta1,beta2)

    x <- matrix(rnorm(p * n, mean = 0, sd = 1), ncol = p)

    exb <- exp(x %*% beta)
    prob <- exb / (1 + exb)
    y <- rbinom(n, 1, prob)

    cyclopsData <- createCyclopsData(y ~ x - 1,modelType = "lr")
    abridge <- fitCyclopsModel(cyclopsData, prior = createAbridgePrior("bic"),
                               control = createControl(noiseLevel = "silent"))

    expect_equivalent(which(coef(abridge) != 0.0), which(beta != 0.0))

    # Determine MLE
    non_zero <- which(beta != 0.0)
    glm <- glm(y ~ x[,non_zero] - 1, family = binomial())
    #expect_equivalent(coef(abridge)[which(coef(abridge) != 0.0)], coef(glm)) # ERROR; this should be true
})
