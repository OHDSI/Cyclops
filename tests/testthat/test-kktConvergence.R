library("testthat")

suppressWarnings(RNGversion("3.5.0"))

test_that("kkt convergence finds the tight optimum", {
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
    }
    expect_error(createControl(convergenceType = "newton"))
})
