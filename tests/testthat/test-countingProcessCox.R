library("testthat")
library("survival")

tolerance <- 1E-6
control <- createControl(tolerance = 1E-10, noiseLevel = "silent")

fitCounting <- function(formula, data, ...) {
    cyclopsData <- createCyclopsData(formula, data = data, modelType = "cox_counting")
    fitCyclopsModel(cyclopsData, control = control, ...)
}

expectMatchesCoxph <- function(cyclopsFit, goldFit) {
    expect_equal(unname(coef(cyclopsFit)), unname(coef(goldFit)), tolerance = tolerance)
    expect_equal(as.numeric(logLik(cyclopsFit)), goldFit$loglik[2], tolerance = tolerance)
}

delayedEntry <- read.table(header = TRUE, sep = ",", text = "
start, stop, event, x1, x2
2.0, 4.0, 1, 0, 0
1.0, 3.5, 1, 2, 0
0.5, 3.0, 0, 0, 1
0.0, 2.5, 1, 0, 1
0.0, 2.0, 1, 1, 1
1.0, 1.5, 0, 1, 0
0.0, 1.0, 1, 1, 0
")

tied <- read.table(header = TRUE, sep = ",", text = "
start, stop, event, x1, x2
0.0, 4.0, 1, 0, 0
1.0, 3.0, 1, 2, 0
0.0, 3.0, 0, 0, 1
2.0, 3.0, 1, 1, 1
0.0, 2.0, 1, 0, 1
2.0, 2.5, 0, 1, 0
0.0, 2.0, 1, 1, 1
0.0, 1.0, 0, 1, 0
0.5, 1.0, 1, 1, 0
0.0, 3.0, 1, 1, 0
")
tied$stratum <- rep(1:2, length.out = nrow(tied))

simulateLeftTruncated <- function(n, seed) {
    set.seed(seed)
    x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.3); x3 <- runif(n)
    entry <- runif(n, 0, 2)
    eventTime <- rexp(n, exp(0.4 * x1 - 0.6 * x2 + 0.3 * x3))
    censorTime <- entry + rexp(n, 0.2)
    data <- data.frame(start = entry, stop = pmin(eventTime, censorTime),
                       event = as.numeric(eventTime <= censorTime),
                       x1 = x1, x2 = x2, x3 = x3,
                       stratum = sample(1:4, n, replace = TRUE),
                       weight = rgamma(n, 2, 2))
    data[eventTime > entry, ] # subjects who had the event before entry are never observed
}

test_that("Small cox_counting example with delayed entry", {
    # A start equal to an event time excludes the row from that risk set; the censored row
    # (1, 1.5] is never at risk at an event time
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2, delayedEntry, ties = "breslow")
    expectMatchesCoxph(fitCounting(Surv(start, stop, event) ~ x1 + x2, delayedEntry), gold)
})

test_that("cox_counting with failure ties, censoring ties and starts tied to event times", {
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2, tied, ties = "breslow")
    expectMatchesCoxph(fitCounting(Surv(start, stop, event) ~ x1 + x2, tied), gold)
})

test_that("cox_counting with strata", {
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), tied, ties = "breslow")
    expectMatchesCoxph(fitCounting(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), tied), gold)
})

test_that("cox_counting with case weights, including a held-out row", {
    weights <- c(1, 2, 0.5, 1, 1.5, 1, 0, 1, 2, 0.7)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), tied,
                  weights = weights, subset = weights > 0, ties = "breslow")

    fit <- fitCounting(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), tied,
                       weights = weights)
    expectMatchesCoxph(fit, gold)
})

test_that("Splitting right-censored rows into counting-process rows leaves the fit unchanged", {
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                       start = "start", end = "stop", event = "event")

    rightCensored <- fitCyclopsModel(
        createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog, data = lung2, modelType = "cox"),
        control = control)
    counting <- fitCounting(Surv(start, stop, event) ~ age + sex + ph.ecog, split)

    expect_equal(coef(counting), coef(rightCensored), tolerance = tolerance)
    expect_equal(as.numeric(logLik(counting)), as.numeric(logLik(rightCensored)),
                 tolerance = tolerance)

    gold <- coxph(Surv(start, stop, event) ~ age + sex + ph.ecog, split, ties = "breslow")
    expectMatchesCoxph(counting, gold)
})

test_that("cox_counting with left truncation, strata and weights matches coxph", {
    data <- simulateLeftTruncated(1500, 456)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data,
                  weights = weight, ties = "breslow")

    fit <- fitCounting(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data,
                       weights = data$weight)
    expectMatchesCoxph(fit, gold)
})

test_that("cox_counting with all start times at 0 reproduces the cox engine", {
    data <- simulateLeftTruncated(1500, 789)
    data$start <- 0

    cox <- fitCyclopsModel(createCyclopsData(Surv(stop, event) ~ x1 + x2 + x3 + strata(stratum),
                                             data = data, modelType = "cox"),
                           control = control)
    counting <- fitCounting(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data)
    expect_equal(coef(counting), coef(cox), tolerance = tolerance)
    expect_equal(as.numeric(logLik(counting)), as.numeric(logLik(cox)), tolerance = tolerance)

    # Two-column Surv() means every row is at risk from the time origin
    twoColumn <- fitCounting(Surv(stop, event) ~ x1 + x2 + x3 + strata(stratum), data)
    expect_equal(coef(twoColumn), coef(cox), tolerance = tolerance)
})

test_that("cox_counting with sparse and indicator covariates", {
    data <- simulateLeftTruncated(1000, 101)
    data$x4 <- ifelse(data$x1 > 1, data$x1, 0) # sparse

    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + x4 + strata(stratum), data, ties = "breslow")

    cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ x1 + strata(stratum),
                                     sparseFormula = ~ x4, indicatorFormula = ~ x2,
                                     data = data, modelType = "cox_counting")
    fit <- fitCyclopsModel(cyclopsData, control = control)
    expect_equal(unname(coef(fit)[c("x1", "x2", "x4")]),
                 unname(coef(gold)[c("x1", "x2", "x4")]), tolerance = tolerance)
})

test_that("convertToCyclopsData with a time-varying exposure matches coxph", {
    set.seed(123)
    n <- 400
    x <- rnorm(n)
    switchTime <- rexp(n, 0.5)
    eventTime <- sapply(seq_len(n), function(i) {
        t1 <- rexp(1, exp(0.5 * x[i]))
        if (t1 < switchTime[i]) t1 else switchTime[i] + rexp(1, exp(0.5 * x[i] + 0.8))
    })
    censorTime <- rexp(n, 0.3)
    stopTime <- pmin(eventTime, censorTime)
    status <- as.numeric(eventTime <= censorTime)

    before <- data.frame(subject = 1:n, startTime = 0, time = pmin(stopTime, switchTime),
                         y = ifelse(stopTime <= switchTime, status, 0), x = x, exposed = 0)
    after <- data.frame(subject = 1:n, startTime = switchTime, time = stopTime, y = status,
                        x = x, exposed = 1)[stopTime > switchTime, ]
    rows <- rbind(before, after)
    rows$rowId <- seq_len(nrow(rows))

    gold <- coxph(Surv(startTime, time, y) ~ x + exposed, rows, ties = "breslow")

    outcomes <- rows[, c("rowId", "y", "time", "startTime")]
    covariates <- rbind(data.frame(rowId = rows$rowId, covariateId = 1, covariateValue = rows$x),
                        data.frame(rowId = rows$rowId, covariateId = 2, covariateValue = rows$exposed))
    covariates <- covariates[covariates$covariateValue != 0, ]

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "cox_counting",
                                        quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, control = control)
    expectMatchesCoxph(fit, gold)

    # Andromeda input follows the same path
    andromeda <- Andromeda::andromeda(outcomes = outcomes, covariates = covariates)
    on.exit(Andromeda::close(andromeda))
    fitAndromeda <- fitCyclopsModel(convertToCyclopsData(andromeda$outcomes, andromeda$covariates,
                                                         modelType = "cox_counting", quiet = TRUE),
                                    control = control)
    expect_equal(coef(fitAndromeda), coef(fit), tolerance = tolerance)
})

test_that("cox_counting predictive log-likelihood uses held-out (start, stop] risk sets", {
    # Fit on the first copy of the data, then score the second copy
    data <- rbind(data.frame(tied, copy = 0), data.frame(tied, copy = 1))
    fit <- fitCounting(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), data,
                       weights = 1 - data$copy)

    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), tied, ties = "breslow",
                  init = coef(fit), control = coxph.control(iter.max = 0))
    expect_equal(Cyclops:::getCyclopsPredictiveLogLikelihood(fit, data$copy), gold$loglik[1],
                 tolerance = tolerance)
})

test_that("cox_counting cross-validation with start times at 0 reproduces the cox engine", {
    data <- simulateLeftTruncated(1000, 202)
    data$start <- 0
    prior <- createPrior("laplace", useCrossValidation = TRUE)
    cvControl <- createControl(noiseLevel = "silent", seed = 666, fold = 5, cvRepetitions = 1,
                               tolerance = 1E-8)

    set.seed(1)
    cox <- fitCyclopsModel(createCyclopsData(Surv(stop, event) ~ x1 + x2 + x3, data = data,
                                             modelType = "cox"),
                           prior = prior, control = cvControl)
    set.seed(1)
    counting <- fitCyclopsModel(createCyclopsData(Surv(start, stop, event) ~ x1 + x2 + x3,
                                                  data = data, modelType = "cox_counting"),
                                prior = prior, control = cvControl)
    expect_equal(getHyperParameter(counting), getHyperParameter(cox), tolerance = 1E-4)
    expect_equal(coef(counting), coef(cox), tolerance = 1E-4)
})

test_that("cox_counting profile-likelihood confidence interval", {
    gold <- coxph(Surv(start, stop, event) ~ x1, tied, ties = "breslow")
    fit <- fitCounting(Surv(start, stop, event) ~ x1, tied)
    interval <- confint(fit, "x1")

    # log-likelihood drops by qchisq(0.95, 1) / 2 at both limits
    for (limit in interval[1, c("2.5 %", "97.5 %")]) {
        atLimit <- coxph(Surv(start, stop, event) ~ x1, tied, ties = "breslow", init = limit,
                         control = coxph.control(iter.max = 0))
        expect_equal(gold$loglik[2] - atLimit$loglik[1], qchisq(0.95, 1) / 2, tolerance = 1E-4)
    }
})

test_that("cox_counting in single precision", {
    data <- simulateLeftTruncated(1000, 303)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3, data, ties = "breslow")
    cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ x1 + x2 + x3, data = data,
                                     modelType = "cox_counting", floatingPoint = 32)
    fit <- fitCyclopsModel(cyclopsData, control = createControl(noiseLevel = "silent"))
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-3)
})

test_that("cox_counting input checks and unsupported methods", {
    outcomes <- data.frame(rowId = 1:3, y = c(1, 0, 1), time = c(2, 3, 4), startTime = c(0, 3, 1))
    covariates <- data.frame(rowId = 1:3, covariateId = 1, covariateValue = c(1, 0.5, 2))
    expect_error(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                 "Start times must be less than stop times")

    fit <- fitCounting(Surv(start, stop, event) ~ x1, delayedEntry)
    expect_error(residuals(fit), "only Cox models")
    expect_error(survfit(fit), "not yet implemented for cox_counting")
})
