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
    expect_error(createControl(cvSearchDirection = "sideways"))
    expect_error(createControl(cvConvergenceType = "newton"))
})

test_that("A tolerance of 0 runs maxIterations cycles", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 500, ncovars = 20, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    for (convergenceType in c("gradient", "kkt")) {
        fit <- fitCyclopsModel(data, prior = createPrior("laplace", 0.1, exclude = 0), forceNewObject = TRUE,
                               warnings = FALSE, control = createControl(convergenceType = convergenceType,
                                                                         tolerance = 0, maxIterations = 7))
        expect_equal(fit$return_flag, "MAX_ITERATIONS")
        expect_equal(fit$iterations, 7)
    }
    expect_error(createControl(tolerance = -1E-6))
})

test_that("Adaptive intercept steps reach the tight optimum", {
    set.seed(321)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 3000, ncovars = 300, eCovarsPerRow = 30, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    prior <- createPrior("laplace", 0.05, exclude = 0)
    fitWith <- function(...) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(...))
    }
    tight <- fitWith(convergenceType = "lange", tolerance = 1E-13, maxIterations = 10000)
    log <- capture.output(adaptive <- fitWith(convergenceType = "kkt", tolerance = 1E-6, adaptiveIntercept = TRUE,
                                              noiseLevel = "noisy"))
    steps <- as.numeric(sub(".*: ", "", grep("Extra intercept steps", log, value = TRUE)))
    expect_gt(steps, 0)
    expect_equal(adaptive$return_flag, "SUCCESS")
    expect_equal(coef(adaptive), coef(tight), tolerance = 1E-5)
    expect_equal(coef(adaptive) != 0, coef(tight) != 0)
    expect_error(fitWith(adaptiveIntercept = TRUE), "convergenceType")
    expect_error(createControl(adaptiveIntercept = "yes"))
})

test_that("The super fitter changes only logistic regression, and createControl() arguments override it", {
    resolve <- function(modelType, nCovariates, ...) {
        Cyclops:::.resolveControl(createControl(fitter = "super", ...), modelType, nCovariates)
    }
    lr <- resolve("lr", 999)
    expect_equal(lr[c("convergenceType", "tolerance", "maxIterations", "cvConvergenceType", "cvTolerance",
                      "cvSearchDirection", "cvRefine", "cvNullModelBound", "cvWarmStart", "activeSetEvery",
                      "adaptiveIntercept")],
                 list(convergenceType = "kkt", tolerance = 1E-4, maxIterations = 5000, cvConvergenceType = "kkt",
                      cvTolerance = 3E-3, cvSearchDirection = "down", cvRefine = TRUE, cvNullModelBound = TRUE,
                      cvWarmStart = TRUE, activeSetEvery = 0, adaptiveIntercept = TRUE))
    expect_false(Cyclops:::.resolveControl(createControl(), "lr", 999)$cvNullModelBound)
    expect_equal(resolve("lr", 1000)$activeSetEvery, 10)
    for (modelType in c("cpr", "clr", "cox")) {
        expect_equal(resolve(modelType, 5000)[c("convergenceType", "tolerance", "maxIterations", "activeSetEvery",
                                                "adaptiveIntercept")],
                     list(convergenceType = "gradient", tolerance = 1E-6, maxIterations = 1000,
                          activeSetEvery = 0, adaptiveIntercept = FALSE))
    }
    expect_equal(resolve("lr", 10, convergenceType = "lange", maxIterations = 50)[c("convergenceType",
                                                                                 "tolerance", "maxIterations")],
                 list(convergenceType = "lange", tolerance = 1E-6, maxIterations = 50))
    expect_equal(resolve("cpr", 10, convergenceType = "kkt")[c("tolerance", "maxIterations")],
                 list(tolerance = 1E-4, maxIterations = 5000))

    # Arguments win over the fitter, and fold fits with their own rule get its defaults
    explicit <- resolve("lr", 5000, cvWarmStart = FALSE, cvRefine = FALSE, cvNullModelBound = FALSE,
                        cvSearchDirection = "auto", activeSetEvery = 0, cvConvergenceType = "lange")
    expect_equal(explicit[c("cvWarmStart", "cvRefine", "cvNullModelBound", "cvSearchDirection", "activeSetEvery",
                            "cvConvergenceType", "cvTolerance", "cvMaxIterations")],
                 list(cvWarmStart = FALSE, cvRefine = FALSE, cvNullModelBound = FALSE, cvSearchDirection = "auto",
                      activeSetEvery = 0, cvConvergenceType = "lange", cvTolerance = 1E-6, cvMaxIterations = 1000))
    expect_error(createControl(cvNullModelBound = "yes"))
    default <- Cyclops:::.resolveControl(createControl(cvConvergenceType = "kkt", activeSetEvery = 20), "lr", 10)
    expect_equal(default[c("convergenceType", "tolerance", "cvTolerance", "cvMaxIterations", "activeSetEvery",
                           "cvWarmStart")],
                 list(convergenceType = "gradient", tolerance = 1E-6, cvTolerance = 3E-3, cvMaxIterations = 5000,
                      activeSetEvery = 20, cvWarmStart = TRUE)) # resetCoefficients = FALSE
    expect_null(Cyclops:::.resolveControl(createControl(cvTolerance = 1E-8), "lr", 10)$cvConvergenceType)

    # Without "kkt", the fitter's settings that need it lapse
    lapsed <- resolve("lr", 5000, convergenceType = "lange", cvConvergenceType = "lange")
    expect_equal(lapsed[c("activeSetEvery", "adaptiveIntercept")], list(activeSetEvery = 0, adaptiveIntercept = FALSE))
    expect_error(resolve("lr", 5000, convergenceType = "lange", cvConvergenceType = "lange", activeSetEvery = 10),
                 "activeSetEvery")
    expect_error(resolve("lr", 5000, convergenceType = "lange", cvConvergenceType = "lange", adaptiveIntercept = TRUE),
                 "adaptiveIntercept")

    # A tolerance without its convergence type would be read in the units of "kkt"
    expect_warning(control <- resolve("lr", 10, tolerance = 1E-8), "tolerance is ignored")
    expect_equal(control$tolerance, 1E-4)
    expect_warning(resolve("lr", 10, tolerance = 1E-8, convergenceType = "gradient"), NA)
    expect_warning(resolve("cpr", 10, tolerance = 1E-8), NA)
})

test_that("The super fitter reaches the optimum of logistic regression", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 2000, ncovars = 1200, eCovarsPerRow = 4, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    prior <- createPrior("laplace", 0.05, exclude = 0)
    tight <- fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                             control = createControl(convergenceType = "lange", tolerance = 1E-13,
                                                     maxIterations = 10000))
    super <- fitCyclopsModel(data, prior = prior, forceNewObject = TRUE,
                             control = createControl(fitter = "super"))
    expect_equal(super$return_flag, "SUCCESS")
    expect_equal(coef(super), coef(tight), tolerance = 1E-3)
    expect_equal(super$log_likelihood + super$log_prior, tight$log_likelihood + tight$log_prior,
                 tolerance = 1E-7)
})
