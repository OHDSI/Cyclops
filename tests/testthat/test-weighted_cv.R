library(testthat)
suppressWarnings(RNGversion("3.5.0"))

test_that("Logistic regression with cross-validation", {
    skip_on_cran()
    set.seed(123)

    sim <- simulateCyclopsData(nstrata=1,
                               ncovars=100,
                               nrows=10000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="logistic")

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes, covariates = sim$covariates, modelType = "lr")
    fit <- fitCyclopsModel(cyclopsData = cyclopsData,
                           prior = createPrior("laplace", useCrossValidation = TRUE),
                           control = createControl(fold = 10, cvRepetitions = 1,
                                                   startingVariance = 0.1, noiseLevel = "quiet", seed = 123))
    coef(fit)[1:10]

    # (Intercept)            1            2            3            4            5
    # -0.99228229  0.00000000  0.13961894  0.01131845  0.00000000  0.26817152
    # 6           7           8           9
    # -0.02982060  0.00000000  0.09693092  0.00000000

    expect_equal(coef(fit)[1], -0.99, tolerance = 0.01, check.attributes = FALSE)
})
