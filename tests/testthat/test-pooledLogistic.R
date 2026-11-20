library("testthat")

context("test-pooledLogistic.R")

# 40 subjects (two per person) with weights and offsets, one per subject (implicit rows), a time basis (linear and
# an indicator of interval 1) and an interaction of covariate 1 with the linear basis
simulatePooled <- function(n = 40, seed = 1) {
    set.seed(seed)
    subjects <- data.frame(rowId = 1:n, entry = sample(1:2, n, TRUE))
    subjects$exit <- subjects$entry + sample(0:4, n, TRUE)
    subjects$y <- rbinom(n, 1, 0.4)
    subjects$weights <- runif(n, 0.5, 2)
    subjects$offset <- rnorm(n, 0, 0.2)
    subjects$subjectId <- rep(seq_len(n / 2), each = 2)
    x1 <- rnorm(n)
    x2 <- rbinom(n, 1, 0.5)
    covariates <- rbind(data.frame(rowId = 1:n, covariateId = 1, covariateValue = x1),
                        data.frame(rowId = which(x2 == 1), covariateId = 2, covariateValue = 1))
    timeBasis <- rbind(data.frame(interval = 1:6, covariateId = 10, covariateValue = (1:6) / 6),
                       data.frame(interval = 1, covariateId = 11, covariateValue = 1))
    map <- data.frame(covariateId = 1, timeCovariateId = 10, interactionId = 20)
    long <- convertToLongOutcome(subjects)
    long$person <- subjects$subjectId[long$rowId]
    long$x1 <- x1[long$rowId]
    long$x2 <- x2[long$rowId]
    long$lin <- long$interval / 6
    long$day1 <- as.numeric(long$interval == 1)
    long$x1lin <- long$x1 * long$lin
    explicit <- convertToCyclopsData(long[, c("rowId", "interval", "y", "weights", "offset")], covariates,
                                     modelType = "plr", timeBasis = timeBasis, timeEffectMap = map)
    implicit <- convertToCyclopsData(subjects, covariates, modelType = "plr", timeBasis = timeBasis,
                                     timeEffectMap = map)
    cloglog <- convertToCyclopsData(subjects, covariates, modelType = "plr", timeBasis = timeBasis,
                                    timeEffectMap = map, link = "cloglog")
    list(long = long, explicit = explicit, implicit = implicit, cloglog = cloglog)
}

data <- simulatePooled()
control <- createControl(tolerance = 1E-11, noiseLevel = "silent")

test_that("plr equals glm(binomial) on the expanded rows", {
    gold <- suppressWarnings(glm(y ~ x1 + x2 + lin + day1 + x1lin, data = data$long, family = binomial(),
                                 weights = weights, offset = offset, control = glm.control(epsilon = 1E-12)))
    fit <- fitCyclopsModel(data$explicit, prior = createPrior("none"), control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(-2 * fit$log_likelihood, deviance(gold), tolerance = 1E-6)
    expect_equal(unname(sqrt(diag(vcov(fit)))), unname(sqrt(diag(vcov(gold)))), tolerance = 1E-5)
})

test_that("plr implicit and explicit rows give the same fit", {
    explicit <- fitCyclopsModel(data$explicit, prior = createPrior("none"), control = control)
    implicit <- fitCyclopsModel(data$implicit, prior = createPrior("none"), control = control)
    expect_equal(coef(implicit), coef(explicit), tolerance = 1E-10)
    expect_equal(implicit$log_likelihood, explicit$log_likelihood, tolerance = 1E-10)
})

test_that("plr equals lr on the expanded rows at a fixed prior variance", {
    lr <- createCyclopsData(y ~ x1 + x2 + lin + day1 + x1lin, data = data$long, modelType = "lr",
                            weights = data$long$weights, offset = data$long$offset)
    fitLr <- fitCyclopsModel(lr, prior = createPrior("normal", variance = 0.5, exclude = "(Intercept)"),
                             control = control)
    fitPlr <- fitCyclopsModel(data$implicit, prior = createPrior("normal", variance = 0.5, exclude = 0),
                              control = control)
    expect_equal(unname(coef(fitPlr)), unname(coef(fitLr)), tolerance = 1E-6)
})

test_that("plr Newton steps reach glm's fit, with its vcov, and the robust vcov clustered by person", {
    gold <- suppressWarnings(glm(y ~ x1 + x2 + lin + day1 + x1lin, data = data$long, family = binomial(),
                                 weights = weights, offset = offset, control = glm.control(epsilon = 1E-12)))
    fit <- fitCyclopsModel(data$implicit, prior = createPrior("none"),
                           control = createControl(algorithm = "newton", noiseLevel = "silent"))
    expect_true(fit$convergence$certified)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(unname(vcov(fit)), unname(vcov(gold)), tolerance = 1E-6)
    skip_if_not_installed("sandwich")
    robust <- sandwich::vcovCL(gold, cluster = ~person, type = "HC0", cadjust = FALSE)
    expect_equal(unname(vcov(fit, type = "robust")), unname(robust), tolerance = 1E-6)
})

test_that("plr with the cloglog link equals glm(binomial(\"cloglog\"))", {
    gold <- suppressWarnings(glm(y ~ x1 + x2 + lin + day1 + x1lin, data = data$long,
                                 family = binomial("cloglog"), weights = weights, offset = offset,
                                 control = glm.control(epsilon = 1E-12)))
    fit <- fitCyclopsModel(data$cloglog, prior = createPrior("none"), control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(-2 * fit$log_likelihood, deviance(gold), tolerance = 1E-6)
})

test_that("plr cross-validation keeps a person's rows in one fold, as lr on the expanded rows", {
    set.seed(2)
    n <- 120 # 60 persons with two clones each
    x <- rnorm(n)
    subjects <- data.frame(rowId = 1:n, subjectId = rep(1:60, each = 2), entry = 1,
                           exit = sample(1:5, n, TRUE), y = rbinom(n, 1, plogis(-1 + 1.5 * x)))
    covariates <- data.frame(rowId = 1:n, covariateId = 1, covariateValue = x)
    plr <- convertToCyclopsData(subjects, covariates, modelType = "plr")
    long <- convertToLongOutcome(subjects)
    long$subjectId <- subjects$subjectId[long$rowId]
    long$x <- x[long$rowId]
    long$rowId <- seq_len(nrow(long))
    lr <- convertToCyclopsData(long[, c("rowId", "y", "subjectId")],
                               data.frame(rowId = long$rowId, covariateId = 1, covariateValue = long$x),
                               modelType = "lr", quiet = TRUE)
    prior <- createPrior("laplace", useCrossValidation = TRUE, exclude = 0)
    cv <- function(threads) createControl(cvType = "grid", gridSteps = 5, lowerLimit = 0.01, upperLimit = 10,
                                          fold = 5, cvRepetitions = 1, minCVData = 10, seed = 7,
                                          threads = threads, noiseLevel = "silent")
    fitPlr <- fitCyclopsModel(plr, prior = prior, control = cv(1))
    fitLr <- fitCyclopsModel(lr, prior = prior, control = cv(1))
    expect_equal(fitPlr$cross_validation, fitLr$cross_validation)
    expect_equal(coef(fitPlr), coef(fitLr), tolerance = 1E-6)
    expect_equal(fitCyclopsModel(plr, prior = prior, control = cv(2))$variance, fitPlr$variance)
    # the auto search starts from the default variance of the person-interval rows, as lr's
    auto <- createControl(fold = 5, cvRepetitions = 1, minCVData = 10, seed = 7, noiseLevel = "silent")
    expect_equal(fitCyclopsModel(plr, prior = prior, control = auto)$variance,
                 fitCyclopsModel(lr, prior = prior, control = auto)$variance, tolerance = 1E-6)
    # and the super fitter's search (null-model bound, step down, refine) selects lr's variance
    super <- createControl(fold = 5, cvRepetitions = 1, minCVData = 10, seed = 7, noiseLevel = "silent",
                           fitter = "super", startingVariance = 0.01)
    superPlr <- fitCyclopsModel(plr, prior = prior, control = super)
    superLr <- fitCyclopsModel(lr, prior = prior, control = super)
    expect_equal(superPlr$variance, superLr$variance, tolerance = 1E-6)
    expect_equal(coef(superPlr), coef(superLr), tolerance = 1E-5)
    # also with coordinate descent, where super uses the "kkt" criterion
    ccd <- modifyList(super, list(algorithm = "ccd"))
    expect_equal(fitCyclopsModel(plr, prior = prior, control = ccd)$variance,
                 fitCyclopsModel(lr, prior = prior, control = ccd)$variance, tolerance = 1E-4)
})

test_that("plr flags separation in a time-basis column, as lr on the expanded rows", {
    set.seed(4)
    long <- data.frame(rowId = rep(1:30, each = 3), interval = rep(1:3, 30), x = rep(rnorm(30), each = 3))
    long$y <- rbinom(90, 1, ifelse(long$interval == 3, 0, 0.3)) # no event in interval 3
    long$day2 <- as.numeric(long$interval == 2)
    long$day3 <- as.numeric(long$interval == 3)
    plr <- convertToCyclopsData(long[, c("rowId", "interval", "y")],
                                data.frame(rowId = 1:30, covariateId = 1, covariateValue = long$x[long$interval == 1]),
                                modelType = "plr",
                                timeBasis = data.frame(interval = 2:3, covariateId = c(2, 3), covariateValue = 1))
    # (a prior on x: with "none" on every coefficient, a separated fit is ILLCONDITIONED)
    fitPlr <- suppressWarnings(fitCyclopsModel(plr, prior = createPrior("normal", variance = 10, exclude = c(0, 2, 3))))
    fitLr <- suppressWarnings(fitCyclopsModel(createCyclopsData(y ~ x + day2 + day3, data = long, modelType = "lr"),
                                              prior = createPrior("normal", variance = 10,
                                                                  exclude = c("(Intercept)", "day2", "day3"))))
    expect_equal(fitPlr$divergentCoefficients, "3")
    expect_equal(unname(coef(fitPlr)), unname(coef(fitLr)), tolerance = 1E-6)
})

test_that("plr takes adaptive intercept steps under the super fitter's coordinate descent", {
    prior <- createPrior("laplace", variance = 0.5, exclude = 0)
    fit <- function(adaptive, noise = "silent") {
        fitCyclopsModel(data$implicit, prior = prior,
                        control = createControl(fitter = "super", algorithm = "ccd", tolerance = 1E-8,
                                                convergenceType = "kkt", adaptiveIntercept = adaptive,
                                                noiseLevel = noise))
    }
    steps <- capture.output(adaptive <- fit(TRUE, "noisy"))
    expect_true(any(grepl("Extra intercept steps", steps)))
    expect_equal(coef(adaptive), coef(fit(FALSE)), tolerance = 1E-6)
})

test_that("plr fits with threads equal those without", {
    set.seed(5) # large enough for several column ranges and subject blocks in the Newton Hessian
    n <- 600
    covariates <- data.frame(rowId = rep(1:n, 150), covariateId = rep(1:150, each = n),
                             covariateValue = rbinom(150 * n, 1, 0.3))
    covariates <- covariates[covariates$covariateValue != 0, ]
    subjects <- data.frame(rowId = 1:n, entry = 1, exit = sample(1:4, n, TRUE), y = rbinom(n, 1, 0.3))
    plr <- convertToCyclopsData(subjects, covariates, modelType = "plr",
                                timeBasis = data.frame(interval = 1:4, covariateId = 1000, covariateValue = (1:4) / 4),
                                timeEffectMap = data.frame(covariateId = 1:20, timeCovariateId = 1000,
                                                           interactionId = 2001:2020))
    fit <- function(threads) fitCyclopsModel(plr, prior = createPrior("laplace", variance = 5, exclude = c(0, 1000)),
                                             control = createControl(threads = threads, noiseLevel = "silent"))
    fit4 <- fit(4)
    expect_equal(fit4$algorithm, "newton")
    expect_identical(coef(fit4), coef(fit(1)))
    # and by the super fitter's coordinate descent, whose cycles skip coordinates (active set) and take extra
    # intercept steps on threads as on one
    ccd <- function(threads) fitCyclopsModel(plr, prior = createPrior("laplace", variance = 0.01, exclude = c(0, 1000)),
                                             control = createControl(threads = threads, noiseLevel = "silent",
                                                                     fitter = "super", algorithm = "ccd",
                                                                     activeSetEvery = 3))
    ccd2 <- ccd(2)
    ccd1 <- ccd(1)
    expect_identical(coef(ccd2), coef(ccd1))
    expect_identical(ccd2$iterations, ccd1$iterations)
})

test_that("plr gradient and Hessian block equal lr's on the expanded rows, in cache tiles and on threads", {
    set.seed(6) # 300 columns of increasing density: the Hessian block reorders them and runs two tiles on one thread
    n <- 150
    p <- 300
    X <- matrix(rbinom(n * p, 1, rep(seq(0.02, 0.5, length.out = p), each = n)), n, p)
    entries <- which(X == 1, arr.ind = TRUE)
    covariates <- data.frame(rowId = entries[, 1], covariateId = entries[, 2], covariateValue = 1)
    subjects <- data.frame(rowId = 1:n, entry = 1, exit = sample(1:3, n, TRUE), y = rbinom(n, 1, 0.4))
    timeBasis <- data.frame(interval = 2:3, covariateId = c(1002, 1003), covariateValue = 1)
    long <- convertToLongOutcome(subjects)
    long$row <- seq_len(nrow(long))
    expanded <- merge(covariates, long[, c("rowId", "row", "interval")], by = "rowId")
    interactions <- expanded[expanded$covariateId %in% 1:20 & expanded$interval == 2, ]
    interacting <- sort(unique(interactions$covariateId)) # the interactions with rows in interval 2
    map <- data.frame(covariateId = interacting, timeCovariateId = 1002, interactionId = 2000 + interacting)
    plr <- convertToCyclopsData(subjects, covariates, modelType = "plr", timeBasis = timeBasis, timeEffectMap = map)
    interactions$covariateId <- interactions$covariateId + 2000
    basis <- data.frame(row = long$row[long$interval > 1], covariateId = 1000 + long$interval[long$interval > 1],
                        covariateValue = 1)
    lrCovariates <- rbind(expanded[, c("row", "covariateId", "covariateValue")],
                          interactions[, c("row", "covariateId", "covariateValue")], basis)
    names(lrCovariates)[1] <- "rowId"
    lr <- convertToCyclopsData(data.frame(rowId = long$row, y = long$y), lrCovariates[order(lrCovariates$rowId), ],
                               modelType = "lr")
    secondOrderAt <- function(data, threads) { # at the same coefficients, by covariate id
        ids <- as.numeric(getCovariateIds(data))
        set.seed(7)
        beta <- setNames(rnorm(length(ids), 0, 0.1), ids)[order(ids)]
        fitCyclopsModel(data, startingCoefficients = unname(beta[as.character(ids)]),
                        fixedCoefficients = rep(TRUE, length(ids)), control = createControl(threads = threads))
        so <- Cyclops:::.cyclopsSecondOrder(data$cyclopsInterfacePtr, NULL, rep(0, length(ids)), 0, 1)
        o <- order(ids)
        list(gradient = so$gradient[o], hessian = so$hessian[o, o])
    }
    one <- secondOrderAt(plr, 1)
    expected <- secondOrderAt(lr, 1)
    expect_equal(one$gradient, expected$gradient, tolerance = 1E-10)
    expect_equal(one$hessian, expected$hessian, tolerance = 1E-10)
    expect_identical(secondOrderAt(plr, 3)$hessian, one$hessian)
})

test_that("plr edge cases: separation, an overflowing row offset, timeBasis checks, arguments of other models", {
    set.seed(3)
    rows <- data.frame(rowId = rep(1:30, each = 2), interval = rep(1:2, 30))
    rows$y <- as.numeric(rows$rowId <= 10 | (rows$rowId %% 3 == 0 & rows$interval == 2))
    covariates <- data.frame(rowId = 1:30, covariateId = 1, covariateValue = rnorm(30))
    separated <- rbind(covariates, data.frame(rowId = 1:10, covariateId = 2, covariateValue = 1)) # events only
    fit <- expect_warning(fitCyclopsModel(convertToCyclopsData(rows, separated, modelType = "plr"),
                                          prior = createPrior("none")), "Separation")
    expect_equal(fit$divergentCoefficients, "2")
    rows$offset <- 0
    rows$offset[60] <- 800
    fit <- fitCyclopsModel(convertToCyclopsData(rows, covariates, modelType = "plr"), prior = createPrior("none"))
    expect_true(is.finite(fit$log_likelihood))
    expect_error(convertToCyclopsData(rows, covariates, modelType = "plr",
                                      timeBasis = data.frame(interval = 0:2, covariateId = 10, covariateValue = 1)),
                 "positive integers")
    expect_warning(convertToCyclopsData(rows[, c("rowId", "y")], covariates, modelType = "lr", link = "cloglog",
                                        quiet = TRUE), "plr models only")
})
