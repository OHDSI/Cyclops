library("testthat")
library("survival")

#
# createControl(fitter = ): "legacy" runs the Cyclops 3.7 code, "default" the current code
#

fitterControl <- function(fitter, ...) {
    createControl(noiseLevel = "silent", fitter = fitter, ...)
}

fitterModels <- function() {
    # The stored values below come from the default generators; other test files set RNGversion("3.5.0")
    oldKind <- RNGkind()
    on.exit(suppressWarnings(RNGkind(oldKind[1], oldKind[2], oldKind[3])))
    suppressWarnings(RNGkind("Mersenne-Twister", "Inversion", "Rejection"))
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 15, effectSizeSd = 0.5, eCovarsPerRow = 2,
                               model = "logistic")
    lr <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    set.seed(11)
    n <- 1500
    stratum <- rep(1:300, each = 5)
    x1 <- rbinom(n, 1, 0.3)
    x2 <- rbinom(n, 1, 0.4) * sample(1:3, n, TRUE)
    x3 <- rbinom(n, 1, 0.3) * round(rexp(n), 2)
    time <- round(runif(n, 1, 30))
    y <- rpois(n, time / 30 * exp(0.3 * x1 - 0.2 * x2 + 0.4 * x3))
    list(lr = lr, grouped = data.frame(y, stratum, x1, x2, x3, time))
}

fitAll <- function(fitter, models) {
    out <- list()
    lr <- models$lr
    f <- fitCyclopsModel(lr, prior = createPrior("laplace", 0.1, exclude = 0),
                         control = fitterControl(fitter, tolerance = 1E-10), forceNewObject = TRUE)
    out$lr_laplace <- list(coef = coef(f), iterations = f$iterations)
    f <- fitCyclopsModel(lr, prior = createPrior("normal", 1, exclude = c(0, 1)),
                         control = fitterControl(fitter, tolerance = 1E-10), forceNewObject = TRUE)
    out$lr_normal <- list(coef = coef(f), iterations = f$iterations)
    out$lr_confint <- confint(f, "1")[, 2:3]
    out$lr_profile <- getCyclopsProfileLogLikelihood(f, "1", x = coef(f)["1"] + c(-0.3, -0.1, 0.1, 0.3))$value
    for (m in c("cpr", "sccs")) {
        data <- createCyclopsData(y ~ strata(stratum), indicatorFormula = ~ x1, sparseFormula = ~ x2 + x3,
                                  time = if (m == "sccs") models$grouped$time else NULL,
                                  data = models$grouped, modelType = m)
        f <- fitCyclopsModel(data, prior = createPrior("none"), control = fitterControl(fitter, tolerance = 1E-10))
        out[[m]] <- list(coef = coef(f), iterations = f$iterations)
        if (m == "cpr") {
            out$cpr_confint <- confint(f, "x1")[, 2:3]
        }
    }
    for (m in c("clr", "clr_efron", "clr_exact")) {
        data <- createCyclopsData(case ~ strata(stratum), sparseFormula = ~ spontaneous + induced,
                                  data = infert, modelType = m)
        f <- fitCyclopsModel(data, prior = createPrior("none"), control = fitterControl(fitter, tolerance = 1E-10))
        out[[m]] <- list(coef = coef(f), iterations = f$iterations)
    }
    out
}

test_that("The legacy fitter reproduces Cyclops 3.7.1", {
    skip_on_cran()
    legacy <- fitAll("legacy", fitterModels())

    # Computed with Cyclops 3.7.1 from CRAN
    expect_equal(unname(legacy$lr_laplace$coef),
                 c(-2.29594490730625, 0, 0, 0, 0.141108612051048, 0.470685641372349, 0.0552144599053789, 0,
                   0.246347848880378, 0, 0.0999935821688652, -0.0419964847942916, 0, 0, 0, 0), tolerance = 1E-8)
    expect_equal(unname(legacy$lr_normal$coef),
                 c(-2.37700014644654, -0.291287719628063, 0.0376111952838865, -0.190804000749507,
                   0.438416334459589, 0.712800109322313, 0.321990016087341, 0.0990505513064242,
                   0.605781039182984, 0.0591694986027172, 0.391719662896247, -0.491999763655372,
                   0.158911255861975, -0.328544481756087, -0.344824772199549, -0.118184882810829),
                 tolerance = 1E-8)
    expect_equal(unname(unlist(legacy$lr_confint)), c(-1.00804579471714, 0.334230489771452), tolerance = 1E-8)
    expect_equal(legacy$lr_profile, c(-334.27288048185, -333.948209972762, -333.949986945961, -334.320769553447),
                 tolerance = 1E-8)
    expect_equal(unname(legacy$cpr$coef), c(-0.239920655902568, 0.422114985932726, 0.377480894532604),
                 tolerance = 1E-8)
    expect_equal(unname(unlist(legacy$cpr_confint)), c(0.215197636055514, 0.539106743281617), tolerance = 1E-8)
    expect_equal(unname(legacy$sccs$coef), c(-0.237241892644747, 0.38265812686454, 0.348188807480366),
                 tolerance = 1E-8)
    for (m in c("clr", "clr_efron", "clr_exact")) {
        expect_equal(unname(legacy[[m]]$coef), c(1.98587551648332, 1.40901163173074), tolerance = 1E-8)
    }
})

test_that("The default and legacy fitters agree on converged fits", {
    skip_on_cran()
    models <- fitterModels()
    legacy <- fitAll("legacy", models)
    default <- fitAll("default", models)
    for (m in c("lr_laplace", "lr_normal", "cpr", "sccs", "clr", "clr_efron", "clr_exact")) {
        expect_equal(default[[m]]$coef, legacy[[m]]$coef, tolerance = 1E-10)
        expect_equal(default[[m]]$iterations, legacy[[m]]$iterations)
    }
    expect_equal(default$lr_profile, legacy$lr_profile, tolerance = 1E-8)
    # legacy keeps the bracketing search of Cyclops 3.7 (tolerance 1e-3); default takes Newton steps
    expect_lt(max(abs(unlist(default$lr_confint) - unlist(legacy$lr_confint))), 1E-3)
    expect_lt(max(abs(unlist(default$cpr_confint) - unlist(legacy$cpr_confint))), 1E-3)
})

test_that("The default fitter scores held-out logistic rows by the Bernoulli log-likelihood", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 2000, ncovars = 40, effectSizeSd = 0.5,
                               eCovarsPerRow = 2, model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", addIntercept = TRUE,
                                        quiet = TRUE)
    y <- sim$outcomes$y[order(sim$outcomes$rowId)]
    set.seed(1)
    w <- rbinom(length(y), 1, 0.7)
    variance <- 0.05
    heldOut <- function(fitter) { # the single held-out score reported by the weight-based selector
        control <- createControl(cvType = "grid", lowerLimit = variance, upperLimit = variance, gridSteps = 1,
                                 fold = -1, noiseLevel = "quiet", convergenceType = "lange", tolerance = 1E-10,
                                 fitter = fitter)
        log <- capture.output(fitCyclopsModel(cyclopsData, weights = w, forceNewObject = TRUE,
                                              prior = createPrior("laplace", exclude = c(0),
                                                                  useCrossValidation = TRUE),
                                              control = control))
        as.numeric(sub(".*pred log like = ", "", grep("pred log like = ", log, value = TRUE)))
    }
    fit <- fitCyclopsModel(cyclopsData, weights = w, forceNewObject = TRUE,
                           prior = createPrior("laplace", variance = variance, exclude = c(0)),
                           control = createControl(convergenceType = "lange", tolerance = 1E-10))
    p <- predict(fit)
    expect_equal(heldOut("default"), sum((1 - w) * (y * log(p) + (1 - y) * log(1 - p))), tolerance = 1E-5)
    # legacy keeps the cases-only score of Cyclops 3.7
    expect_equal(heldOut("legacy"), sum((1 - w) * y * log(p)), tolerance = 1E-5)
})

test_that("Cross-validation threads run the fitter of the original model", {
    skip_on_cran()
    set.seed(42)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 1500, ncovars = 30, effectSizeSd = 0.5,
                               eCovarsPerRow = 3, model = "logistic")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    cv <- function(fitter, threads) {
        fitCyclopsModel(cyclopsData, prior = createPrior("laplace", exclude = 0, useCrossValidation = TRUE),
                        forceNewObject = TRUE,
                        control = createControl(cvType = "grid", lowerLimit = 0.01, upperLimit = 1, gridSteps = 3,
                                                fold = 5, seed = 1, resetCoefficients = TRUE, noiseLevel = "silent",
                                                threads = threads, fitter = fitter))
    }
    for (fitter in c("legacy", "default")) {
        one <- cv(fitter, 1)
        three <- cv(fitter, 3)
        expect_identical(coef(three), coef(one))
        expect_identical(three$cross_validation, one$cross_validation)
    }
    # The two scores differ, so a clone running the other fitter would show above
    expect_false(identical(cv("legacy", 1)$cross_validation, cv("default", 1)$cross_validation))
})

test_that("Changing the fitter rebuilds a logistic model", {
    models <- fitterModels()
    prior <- createPrior("laplace", 0.1, exclude = 0)
    default <- fitCyclopsModel(models$lr, prior = prior, control = fitterControl("default", tolerance = 1E-10))
    expect_equal(models$lr$fitter, "default")
    legacy <- fitCyclopsModel(models$lr, prior = prior, control = fitterControl("legacy", tolerance = 1E-10))
    expect_equal(models$lr$fitter, "legacy")
    fresh <- fitCyclopsModel(models$lr, prior = prior, control = fitterControl("legacy", tolerance = 1E-10),
                             forceNewObject = TRUE)
    expect_identical(coef(legacy), coef(fresh))
    expect_equal(createControl()$fitter, "default")
    expect_equal(createControl(fitter = "super")$fitter, "super")
})
