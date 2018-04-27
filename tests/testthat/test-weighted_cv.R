library(testthat)

test_that("Logistic regression with cross-validation", {

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
                           control = createControl(fold = 10, cvRepetitions = 1, startingVariance = 0.1, noiseLevel = "quiet"))
    coef(fit)[1:10]

    # (Intercept)            1            2            3            4            5
    # -0.999163139  0.000000000  0.168884281  0.039489886  0.003542555  0.296874842
    # 6            7            8            9
    # -0.058098905  0.000000000  0.126797050  0.000000000

    expect_equal(sum(coef(fit) != 0), 46)
})
