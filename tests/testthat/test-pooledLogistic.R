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
})
