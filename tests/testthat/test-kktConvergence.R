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

    # Least squares: "kkt", active sets, adaptive intercept steps and a search up from the null model under both
    # fitters; super adds looser fold fits; legacy keeps the earlier settings
    lsNames <- c("convergenceType", "tolerance", "cvConvergenceType", "cvTolerance", "cvSearchDirection", "cvRefine",
                 "cvNullModelBound", "activeSetEvery", "adaptiveIntercept")
    expect_equal(Cyclops:::.resolveControl(createControl(), "ls", 5000)[lsNames],
                 list(convergenceType = "kkt", tolerance = 1E-4, cvConvergenceType = NULL, cvTolerance = NULL,
                      cvSearchDirection = "up", cvRefine = TRUE, cvNullModelBound = TRUE, activeSetEvery = 10,
                      adaptiveIntercept = TRUE))
    expect_equal(Cyclops:::.resolveControl(createControl(), "ls", 999)$activeSetEvery, 0)
    expect_equal(resolve("ls", 5000)[lsNames],
                 list(convergenceType = "kkt", tolerance = 1E-4, cvConvergenceType = "kkt", cvTolerance = 0.1,
                      cvSearchDirection = "up", cvRefine = TRUE, cvNullModelBound = TRUE, activeSetEvery = 10,
                      adaptiveIntercept = TRUE))
    expect_equal(resolve("ls", 999)$activeSetEvery, 0)
    expect_equal(Cyclops:::.resolveControl(createControl(fitter = "legacy"), "ls", 5000)[c("convergenceType",
                                                                                           "cvSearchDirection")],
                 list(convergenceType = "gradient", cvSearchDirection = "auto"))
    # a given tolerance applies to the "kkt" criterion of least squares (in standard-error units)
    expect_silent(ls <- Cyclops:::.resolveControl(createControl(tolerance = 1E-8), "ls", 10))
    expect_equal(ls[c("convergenceType", "tolerance")], list(convergenceType = "kkt", tolerance = 1E-8))
    expect_silent(ls <- resolve("ls", 10, tolerance = 1E-8))
    expect_equal(ls$tolerance, 1E-8)
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

test_that("Active-set Newton steps resolve near-duplicate covariates", {
    # Coordinate descent moves weight between two covariates with nearly the same rows by a small fraction
    # per cycle; FeatureExtraction's group eras give such pairs. Covariate 41 is covariate 1 less two rows,
    # covariate 42 is covariate 2 exactly
    set.seed(42)
    n <- 4000
    x <- sapply(runif(40, 0.05, 0.3), function(p) rbinom(n, 1, p))
    x <- cbind(x, x[, 1], x[, 2])
    x[which(x[, 1] == 1)[1:2], 41] <- 0
    y <- rbinom(n, 1, plogis(-2 + 1.2 * x[, 1] - 0.8 * x[, 2] + 0.5 * x[, 3]))
    rows <- which(x != 0, arr.ind = TRUE)
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y),
                                 data.frame(rowId = rows[, 1], covariateId = rows[, 2], covariateValue = 1),
                                 modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    prior <- createPrior("laplace", 0.5, exclude = 0)
    fitWith <- function(...) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(noiseLevel = "silent", ...))
    }
    logPosterior <- function(fit) {
        b <- coef(fit)[-1]
        fit$log_likelihood - sqrt(2 / 0.5) * sum(abs(b))
    }
    tight <- fitWith(convergenceType = "lange", tolerance = 1E-14, maxIterations = 100000)
    newton <- fitWith(fitter = "super", convergenceType = "kkt", tolerance = 1E-6)
    expect_equal(newton$return_flag, "SUCCESS")
    expect_true(logPosterior(newton) >= logPosterior(tight) - 1E-7)
    # the pairs are identified only through their sums
    expect_equal(sum(coef(newton)[c("1", "41")]), sum(coef(tight)[c("1", "41")]), tolerance = 1E-4)
    expect_equal(sum(coef(newton)[c("2", "42")]), sum(coef(tight)[c("2", "42")]), tolerance = 1E-4)
    expect_equal(coef(newton)[-c(2, 3, 42, 43)], coef(tight)[-c(2, 3, 42, 43)], tolerance = 1E-4)
    without <- fitWith(fitter = "super", convergenceType = "kkt", tolerance = 1E-6, activeSetNewton = FALSE,
                       maxIterations = 20000)
    expect_lt(newton$iterations, without$iterations)

    expect_error(fitWith(fitter = "default", activeSetNewton = TRUE), "convergenceType")
    expect_error(createControl(activeSetNewton = 1))
})

test_that("The super fitter reaches the optimum with an exact null direction", {
    # An intercept and a full set of category columns: intercept - sum of the categories is an exact null
    # direction, along which only the L1 penalty changes; its Hessian is singular. Case 2 scales one category
    # by 1e9; case 3 leaves one row without a category (the identity is broken there)
    set.seed(7)
    n <- 4000
    level <- sample.int(5, n, replace = TRUE)
    x <- sapply(runif(20, 0.05, 0.3), function(p) rbinom(n, 1, p))
    y <- rbinom(n, 1, plogis(-1 + c(-1, -0.5, 0, 0.5, 1)[level] + x %*% rnorm(20, 0, 0.4)))
    rows <- which(x != 0, arr.ind = TRUE)
    makeData <- function(scale, broken) {
        cats <- data.frame(rowId = 1:n, covariateId = 100 + level, covariateValue = scale[level])
        if (broken) cats <- cats[-1, ]
        covariates <- rbind(cats, data.frame(rowId = rows[, 1], covariateId = rows[, 2], covariateValue = 1))
        covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
        convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                             quiet = TRUE)
    }
    prior <- createPrior("laplace", 0.05, exclude = 0)
    for (case in list(list(rep(1, 5), FALSE), list(c(1, 1, 1, 1, 1e9), FALSE), list(rep(1, 5), TRUE))) {
        data <- makeData(case[[1]], case[[2]])
        fitWith <- function(...) {
            fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(noiseLevel = "silent", ...))
        }
        logPosterior <- function(fit) fit$log_likelihood - sqrt(2 / 0.05) * sum(abs(coef(fit)[-1]))
        tight <- fitWith(convergenceType = "lange", tolerance = 1E-14, maxIterations = 100000)
        super <- fitWith(fitter = "super", convergenceType = "kkt", tolerance = 1E-6)
        expect_equal(super$return_flag, "SUCCESS")
        expect_equal(logPosterior(super), logPosterior(tight), tolerance = 1E-9)
        expect_equal(predict(super), predict(tight), tolerance = 1E-5)
    }
})

nearDuplicateData <- function(n = 4000, seed = 42) {
    # Covariates 41 and 42 nearly or exactly repeat covariates 1 and 2 (see the test above)
    set.seed(seed)
    x <- sapply(runif(40, 0.05, 0.3), function(p) rbinom(n, 1, p))
    x <- cbind(x, x[, 1], x[, 2])
    x[which(x[, 1] == 1)[1:2], 41] <- 0
    y <- rbinom(n, 1, plogis(-2 + 1.2 * x[, 1] - 0.8 * x[, 2] + 0.5 * x[, 3]))
    rows <- which(x != 0, arr.ind = TRUE)
    list(outcomes = data.frame(rowId = 1:n, y = y),
         covariates = data.frame(rowId = rows[, 1], covariateId = rows[, 2], covariateValue = 1))
}

logPosteriorOf <- function(fit, variance, exclude = "(Intercept)") {
    b <- coef(fit)
    fit$log_likelihood - sqrt(2 / variance) * sum(abs(b[!(names(b) %in% exclude)]))
}

test_that("Active-set Newton steps keep the optimum with weights, unpenalized covariates and no follow-up", {
    d <- nearDuplicateData()
    data <- convertToCyclopsData(d$outcomes, d$covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    set.seed(3)
    weights <- runif(nrow(d$outcomes), 0.2, 2)
    weights[sample.int(length(weights), 400)] <- 0 # held-out rows, as in cross-validation
    for (exclude in list(0, c(0, 3))) {
        prior <- createPrior("laplace", 0.5, exclude = exclude)
        fitWith <- function(...) {
            fitCyclopsModel(data, prior = prior, weights = weights, forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(noiseLevel = "silent", ...))
        }
        excluded <- c("(Intercept)", if (3 %in% exclude) "3")
        tight <- fitWith(convergenceType = "lange", tolerance = 1E-14, maxIterations = 100000)
        for (followUp in c(TRUE, FALSE)) {
            control <- createControl(noiseLevel = "silent", fitter = "super", convergenceType = "kkt", tolerance = 1E-6)
            control$activeSetNewtonFollowUp <- followUp # internal switch
            newton <- fitCyclopsModel(data, prior = prior, weights = weights, forceNewObject = TRUE, warnings = FALSE,
                                      control = control)
            expect_equal(newton$return_flag, "SUCCESS")
            expect_true(logPosteriorOf(newton, 0.5, excluded) >= logPosteriorOf(tight, 0.5, excluded) - 1E-7)
            expect_equal(predict(newton), predict(tight), tolerance = 1E-4)
        }
    }
})

test_that("Cross-validation with active-set Newton steps does not depend on the threads", {
    d <- nearDuplicateData(n = 3000)
    data <- convertToCyclopsData(d$outcomes, d$covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    prior <- createPrior("laplace", exclude = 0, useCrossValidation = TRUE)
    fitWith <- function(threads) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(noiseLevel = "silent", fitter = "super", cvType = "auto", fold = 4,
                                                cvRepetitions = 1, seed = 5, startingVariance = 0.1, threads = threads))
    }
    one <- fitWith(1)
    four <- fitWith(4)
    expect_equal(one$return_flag, "SUCCESS")
    expect_identical(one$variance, four$variance)
    expect_identical(coef(one), coef(four))
})

test_that("Active-set Newton steps are skipped above their size limit", {
    # More than 2000 non-zero coefficients: the fit proceeds by coordinate descent alone
    set.seed(11)
    n <- 8000
    p <- 3000
    x <- matrix(rbinom(n * p, 1, 0.05), n, p)
    y <- rbinom(n, 1, plogis(-1 + x %*% rnorm(p, 0, 0.3)))
    rows <- which(x != 0, arr.ind = TRUE)
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y),
                                 data.frame(rowId = rows[, 1], covariateId = rows[, 2], covariateValue = 1),
                                 modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    prior <- createPrior("laplace", 10, exclude = 0)
    fitWith <- function(...) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(noiseLevel = "silent", ...))
    }
    super <- fitWith(fitter = "super", convergenceType = "kkt", tolerance = 1E-6)
    without <- fitWith(fitter = "super", convergenceType = "kkt", tolerance = 1E-6, activeSetNewton = FALSE)
    expect_gt(sum(coef(super)[-1] != 0), 2000)
    expect_equal(super$return_flag, "SUCCESS")
    expect_identical(coef(super), coef(without))
})
