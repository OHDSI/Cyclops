library("testthat")

suppressWarnings(RNGversion("3.5.0"))

test_that("kkt convergence and active-set cycling find the tight optimum", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 2000, ncovars = 200, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    fitWith <- function(prior, ...) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(noiseLevel = "silent", ...))
    }
    for (prior in list(createPrior("laplace", 0.05, exclude = 0), createPrior("normal", 0.5, exclude = 0))) {
        tight <- fitWith(prior, convergenceType = "lange", tolerance = 1E-13, maxIterations = 10000)
        kkt <- fitWith(prior, convergenceType = "kkt", tolerance = 1E-6)
        expect_equal(kkt$return_flag, "SUCCESS")
        expect_equal(coef(kkt), coef(tight), tolerance = 1E-5)
        if (prior$priorType[1] == "laplace") {
            for (threshold in c(0, 0.01)) {
                active <- fitWith(prior, convergenceType = "kkt", tolerance = 1E-6, activeSetEvery = 10,
                                  activeSetThreshold = threshold)
                expect_equal(active$return_flag, "SUCCESS")
                expect_equal(coef(active), coef(tight), tolerance = 1E-5)
                expect_equal(coef(active) != 0, coef(tight) != 0)
            }
        }
    }
    expect_error(fitWith(createPrior("laplace", 0.05, exclude = 0), activeSetEvery = 10), "convergenceType")
    expect_error(createControl(convergenceType = "kkt", activeSetEvery = 1))
    expect_error(createControl(convergenceType = "kkt", activeSetEvery = -1))
    expect_error(createControl(activeSetThreshold = 1))
})
