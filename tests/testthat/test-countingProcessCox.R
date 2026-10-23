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

test_that("Sparse and indicator gradients and Hessians match coxph away from the mode", {
    # Sparse and indicator columns take the prefix-sum (cumulative hazard) path, dense columns the scan
    data <- simulateLeftTruncated(800, 707)
    data$x4 <- ifelse(data$x1 > 0.5, data$x1, 0)
    beta <- c(x1 = 0.3, x4 = -0.2, x2 = 0.5)
    for (modelType in c("cox_counting", "cox")) {
        d <- if (modelType == "cox") transform(data, start = 0) else data
        cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ x1 + strata(stratum),
                                         sparseFormula = ~ x4, indicatorFormula = ~ x2,
                                         data = d, modelType = modelType)
        fit <- fitCyclopsModel(cyclopsData, control = control)
        Cyclops:::.cyclopsSetBeta(fit$interface, unname(beta[names(coef(fit))]))

        gold <- coxph.detail(coxph(Surv(start, stop, event) ~ x1 + x4 + x2 + strata(stratum), d,
                                   ties = "breslow", init = beta, control = coxph.control(iter.max = 0)))
        score <- setNames(colSums(gold$score), names(beta))
        information <- setNames(diag(apply(gold$imat, c(1, 2), sum)), names(beta))

        expect_equal(gradient(fit)[names(beta)], score, tolerance = 1E-10)
        # The accessor returns the log-likelihood Hessian, the negative of the information
        hessian <- setNames(Cyclops:::.cyclopsGetLogLikelihoodHessianDiagonal(fit$interface,
                                                                                getCovariateIds(cyclopsData)),
                            names(coef(fit)))
        expect_equal(-hessian[names(beta)], information, tolerance = 1E-10)
    }
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

test_that("convertToCyclopsData keeps startTime with its row, with strata and weights", {
    data <- simulateLeftTruncated(1500, 818)
    data$weight[1:40] <- 0 # held out
    data$rowId <- seq_len(nrow(data))
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data, weights = weight,
                  subset = weight > 0, ties = "breslow", model = TRUE)

    # Rows in no particular order, so conversion has to sort them with their start times
    set.seed(818)
    outcomes <- data.frame(rowId = data$rowId, y = data$event, time = data$stop, startTime = data$start,
                           stratumId = data$stratum, weights = data$weight)[sample(nrow(data)), ]
    covariates <- rbind(data.frame(rowId = data$rowId, covariateId = 1, covariateValue = data$x1),
                        data.frame(rowId = data$rowId, covariateId = 2, covariateValue = data$x2),
                        data.frame(rowId = data$rowId, covariateId = 3, covariateValue = data$x3))
    covariates <- covariates[covariates$covariateValue != 0, ]
    covariates <- covariates[sample(nrow(covariates)), ]

    fit <- fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                           control = control)
    expectMatchesCoxph(fit, gold)
    # Residuals depend on each row's own risk interval
    expect_equivalent(residuals(fit), residuals(gold, "schoenfeld"), tolerance = tolerance)

    # Andromeda input; that path needs stratumId on the covariates, which CohortMethod provides
    andromeda <- Andromeda::andromeda(outcomes = outcomes,
                                      covariates = merge(covariates, outcomes[, c("rowId", "stratumId")]))
    on.exit(Andromeda::close(andromeda))
    fitAndromeda <- fitCyclopsModel(convertToCyclopsData(andromeda$outcomes, andromeda$covariates,
                                                         modelType = "cox_counting", quiet = TRUE),
                                    control = control)
    expect_equal(coef(fitAndromeda), coef(fit), tolerance = tolerance)
    expect_equal(as.numeric(logLik(fitAndromeda)), as.numeric(logLik(fit)), tolerance = tolerance)
})

test_that("convertToCyclopsData start-time checks, and starts that leave risk sets unchanged", {
    data <- simulateLeftTruncated(500, 919)
    outcomes <- data.frame(rowId = seq_len(nrow(data)), y = data$event, time = data$stop - data$start)
    covariates <- data.frame(rowId = outcomes$rowId, covariateId = 1, covariateValue = data$x1)
    fitFrom <- function(outcomes, modelType) {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = modelType, quiet = TRUE),
                        control = control)
    }
    cox <- fitFrom(outcomes, "cox")

    # A start-time column of zeros is accepted by cox and changes nothing
    expect_identical(coef(fitFrom(transform(outcomes, startTime = 0), "cox")), coef(cox))
    # cox_counting without start times is cox
    expect_equal(coef(fitFrom(outcomes, "cox_counting")), coef(cox), tolerance = tolerance)
    # A risk window opening at the same time for every row, e.g. 30 days after cohort start, has the
    # same risk sets as time measured from the start of the window
    shifted <- transform(outcomes, startTime = 30, time = time + 30)
    expect_equal(coef(fitFrom(shifted, "cox_counting")), coef(cox), tolerance = tolerance)

    # The input checks also hold for Andromeda input
    andromeda <- Andromeda::andromeda(nonZeroStarts = transform(outcomes, startTime = 1:nrow(outcomes) %% 2),
                                      startAfterStop = transform(outcomes, startTime = time),
                                      covariates = covariates)
    on.exit(Andromeda::close(andromeda))
    expect_error(convertToCyclopsData(andromeda$nonZeroStarts, andromeda$covariates, modelType = "cox",
                                      quiet = TRUE), "cox_counting")
    expect_error(convertToCyclopsData(andromeda$startAfterStop, andromeda$covariates,
                                      modelType = "cox_counting", quiet = TRUE),
                 "Start times must be less than stop times")
})

test_that("convertToCyclopsData takes the end of each interval as time or endTime", {
    data <- simulateLeftTruncated(500, 717)
    outcomes <- data.frame(rowId = seq_len(nrow(data)), y = data$event, time = data$stop, startTime = data$start)
    covariates <- data.frame(rowId = outcomes$rowId, covariateId = 1, covariateValue = data$x1)
    withEndTime <- data.frame(rowId = outcomes$rowId, y = outcomes$y, startTime = outcomes$startTime,
                              endTime = outcomes$time)
    fitFrom <- function(outcomes, covariates) {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                        control = control)
    }
    expected <- fitFrom(outcomes, covariates)
    expect_identical(coef(fitFrom(withEndTime, covariates)), coef(expected))
    expect_error(convertToCyclopsData(transform(outcomes, endTime = time), covariates,
                                      modelType = "cox_counting", quiet = TRUE), "not both")

    andromeda <- Andromeda::andromeda(withEndTime = withEndTime, both = transform(outcomes, endTime = time),
                                      covariates = covariates)
    on.exit(Andromeda::close(andromeda))
    expect_equal(coef(fitFrom(andromeda$withEndTime, andromeda$covariates)), coef(expected), tolerance = tolerance)
    expect_error(convertToCyclopsData(andromeda$both, andromeda$covariates, modelType = "cox_counting",
                                      quiet = TRUE), "not both")
})

test_that("convertToCyclopsData subjectId changes folds and draws, not the fit", {
    # survSplit rows tie at each cut, and subject order is the reverse of row order
    lung2 <- na.omit(lung[, c("time", "status", "age")])
    lung2$status <- lung2$status - 1
    lung2$id <- rev(seq_len(nrow(lung2)))
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365),
                       start = "start", end = "stop", event = "event")
    outcomes <- data.frame(rowId = seq_len(nrow(split)), y = split$event, time = split$stop,
                           startTime = split$start)
    covariates <- data.frame(rowId = outcomes$rowId, covariateId = 1, covariateValue = split$age)
    withSubjects <- transform(outcomes, subjectId = split$id)
    fitFrom <- function(outcomes, covariates) {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                        control = control)
    }
    without <- fitFrom(outcomes, covariates)
    expect_equal(coef(fitFrom(withSubjects, covariates)), coef(without), tolerance = tolerance)

    andromeda <- Andromeda::andromeda(outcomes = withSubjects, covariates = covariates)
    on.exit(Andromeda::close(andromeda))
    expect_equal(coef(fitFrom(andromeda$outcomes, andromeda$covariates)), coef(without), tolerance = tolerance)
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

test_that("Threads within a fit give identical results", {
    # Gradients of coefficients at zero are evaluated together; the fit must not change
    set.seed(909)
    n <- 3000
    J <- 200
    X <- matrix(rbinom(n * J, 1, 0.03), n)
    entry <- runif(n, 0, 2)
    eventTime <- entry + rexp(n, 0.3 * exp(drop(X[, 1:5] %*% rep(0.7, 5))))
    censorTime <- entry + rexp(n, 0.3)
    outcomes <- data.frame(rowId = 1:n, y = as.numeric(eventTime <= censorTime),
                           time = pmin(eventTime, censorTime), startTime = entry)
    nonZero <- which(X != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nonZero[, 1], covariateId = nonZero[, 2], covariateValue = 1)

    fitWith <- function(modelType, threads) {
        data <- outcomes
        if (modelType == "cox") {
            data$startTime <- NULL
        }
        fitCyclopsModel(convertToCyclopsData(data, covariates, modelType = modelType, quiet = TRUE),
                        prior = createPrior("laplace", 0.05),
                        control = createControl(noiseLevel = "silent", threads = threads, seed = 1))
    }
    for (modelType in c("cox", "cox_counting")) {
        one <- fitWith(modelType, 1)
        four <- fitWith(modelType, 4)
        expect_true(sum(coef(one) != 0) < J / 2)
        expect_identical(coef(four), coef(one))
        expect_identical(logLik(four), logLik(one))
        expect_identical(runBootstrap(four, 5)$samples, runBootstrap(one, 5)$samples)
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

test_that("cox_counting input checks", {
    outcomes <- data.frame(rowId = 1:3, y = c(1, 0, 1), time = c(2, 3, 4), startTime = c(0, 3, 1))
    covariates <- data.frame(rowId = 1:3, covariateId = 1, covariateValue = c(1, 0.5, 2))
    expect_error(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                 "Start times must be less than stop times")

    cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ x1, data = delayedEntry,
                                     modelType = "cox_counting")
    expect_equal(Cyclops:::getStartTimeVector(cyclopsData), delayedEntry$start[cyclopsData$sortOrder])
    expect_error(fitCyclopsModel(cyclopsData,
                                 control = createControl(selectorType = "bySubject", noiseLevel = "silent")),
                 "bySubject")
})

test_that("Baseline survival for cox_counting matches survfit.coxph", {
    data <- simulateLeftTruncated(300, 606)
    for (d in list(delayedEntry, data)) {
        gold <- survfit(coxph(Surv(start, stop, event) ~ x1 + x2, d, ties = "breslow", nocenter = NULL))
        cyclops <- survfit(fitCounting(Surv(start, stop, event) ~ x1 + x2, d), type = "aalen")
        expect_equal(cyclops$time, gold$time)
        expect_equal(cyclops$surv, gold$surv, tolerance = tolerance)
    }
})

test_that("Stratified baseline survival for cox_counting matches survfit.coxph", {
    data <- simulateLeftTruncated(400, 808)
    gold <- survfit(coxph(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), data, ties = "breslow",
                          nocenter = NULL))
    cyclops <- survfit(fitCounting(Surv(start, stop, event) ~ x1 + x2 + strata(stratum), data),
                       type = "aalen")
    expect_equal(cyclops$time, gold$time)
    expect_equal(cyclops$surv, gold$surv, tolerance = tolerance)
    expect_equal(unname(cyclops$strata), unname(gold$strata))
})

test_that("Proportional hazards test for cox_counting matches cox.zph", {
    data <- simulateLeftTruncated(500, 505)
    gold <- coxph(Surv(start, stop, event) ~ x1, data, ties = "breslow")
    fit <- fitCounting(Surv(start, stop, event) ~ x1, data)
    for (transform in c("km", "identity")) {
        expect_equal(unname(testProportionality(fit, transform = transform)$table),
                     unname(cox.zph(gold, transform = transform, global = FALSE)$table),
                     tolerance = 1E-5)
    }
})

test_that("Weighted proportional hazards test matches cox.zph on the rows in the fit", {
    data <- simulateLeftTruncated(1500, 505)
    data$weight[c(which(data$event == 1)[1:10], which(data$event == 0)[1:10])] <- 0 # held out
    data$unit <- as.numeric(data$weight > 0)

    expectMatchesZph <- function(formula, modelType, weight) {
        data$w <- data[[weight]]
        gold <- coxph(formula, data, weights = w, subset = w > 0, ties = "breslow", model = TRUE)
        fit <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = modelType),
                               weights = data$w, control = control)
        for (transform in c("km", "rank", "identity")) {
            expect_equal(unname(testProportionality(fit, transform = transform)$table),
                         unname(cox.zph(gold, transform = transform, global = FALSE)$table),
                         tolerance = 1E-5)
        }
    }
    for (weight in c("weight", "unit")) {
        expectMatchesZph(Surv(start, stop, event) ~ x1, "cox_counting", weight)
        expectMatchesZph(Surv(stop, event) ~ x1, "cox", weight)
    }
})

test_that("The default and legacy fitters give the same Cox fits", {
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    fitWith <- function(fitter, formula, ...) {
        fitCyclopsModel(createCyclopsData(formula, data = lung2, modelType = "cox"),
                        control = createControl(noiseLevel = "silent", tolerance = 1E-10, fitter = fitter),
                        ...)
    }
    for (formula in c(Surv(time, status) ~ age + ph.ecog, Surv(time, status) ~ age + ph.ecog + strata(sex))) {
        default <- fitWith("default", formula)
        legacy <- fitWith("legacy", formula)
        expect_equal(coef(legacy), coef(default), tolerance = 1E-8)
        expect_equal(as.numeric(logLik(legacy)), as.numeric(logLik(default)), tolerance = 1E-8)
        # legacy keeps the bracketing search of Cyclops 3.7 (tolerance 1e-3); default takes Newton steps
        expect_lt(max(abs(confint(legacy, "age")[, 2:3] - confint(default, "age")[, 2:3])), 1E-3)
    }
    weights <- rep(c(1, 2, 0, 0.5), length.out = nrow(lung2))
    expect_equal(coef(fitWith("legacy", Surv(time, status) ~ age + ph.ecog, weights = weights)),
                 coef(fitWith("default", Surv(time, status) ~ age + ph.ecog, weights = weights)), tolerance = 1E-8)

    # L1 with threads: the legacy engine evaluates gradients one at a time
    set.seed(42)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 2000, ncovars = 40, model = "survival")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "cox", quiet = TRUE)
    fits <- lapply(c("default", "legacy"), function(fitter) {
        fitCyclopsModel(data, prior = createPrior("laplace", 0.1), forceNewObject = TRUE,
                        control = createControl(noiseLevel = "silent", tolerance = 1E-10, threads = 2,
                                                fitter = fitter))
    })
    expect_equal(coef(fits[[2]]), coef(fits[[1]]), tolerance = 1E-8)
})

test_that("Changing the fitter rebuilds the engine", {
    data <- createCyclopsData(Surv(stop, event) ~ x1 + x2, data = tied, modelType = "cox")
    default <- fitCyclopsModel(data, control = control)
    expect_equal(data$fitter, "default")
    legacy <- fitCyclopsModel(data, control = createControl(noiseLevel = "silent", tolerance = 1E-10,
                                                            fitter = "legacy"))
    expect_equal(data$fitter, "legacy")
    expect_equal(coef(legacy), coef(default), tolerance = tolerance)

    # A control from before fitter existed means "default"
    older <- control
    older$fitter <- NULL
    fitCyclopsModel(data, control = older)
    expect_equal(data$fitter, "default")

    counting <- createCyclopsData(Surv(start, stop, event) ~ x1, data = tied, modelType = "cox_counting")
    expect_error(fitCyclopsModel(counting, control = createControl(fitter = "legacy")), "start times")
    expect_error(createControl(fitter = "other"))
})

test_that("cox_counting takes negative start times", {
    # e.g. CohortMethod's index scale puts a risk window that opens on day 0 at (-1, end]
    shifted <- transform(delayedEntry, start = start - 1.5, stop = stop - 1.5)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2, shifted, ties = "breslow")
    expectMatchesCoxph(fitCounting(Surv(start, stop, event) ~ x1 + x2, shifted), gold)
})

test_that("cox requires start times of 0", {
    expect_error(createCyclopsData(Surv(start, stop, event) ~ x1 + x2, data = delayedEntry,
                                   modelType = "cox"), "cox_counting")

    outcomes <- data.frame(rowId = 1:3, y = c(1, 0, 1), time = c(2, 3, 4), startTime = c(0, 1, 0))
    covariates <- data.frame(rowId = 1:3, covariateId = 1, covariateValue = c(1, 0.5, 2))
    expect_error(convertToCyclopsData(outcomes, covariates, modelType = "cox", quiet = TRUE),
                 "cox_counting")

    zero <- transform(delayedEntry, start = 0)
    gold <- coxph(Surv(stop, event) ~ x1 + x2, zero, ties = "breslow")
    fit <- fitCyclopsModel(createCyclopsData(Surv(start, stop, event) ~ x1 + x2, data = zero,
                                             modelType = "cox"), control = control)
    expectMatchesCoxph(fit, gold)
})

test_that("vcov for cox and cox_counting matches coxph", {
    data <- simulateLeftTruncated(500, 404)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data, ties = "breslow")
    fit <- fitCounting(Surv(start, stop, event) ~ x1 + x2 + x3 + strata(stratum), data)
    expect_equal(unname(vcov(fit)), unname(vcov(gold)), tolerance = tolerance)

    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    goldCox <- coxph(Surv(time, status) ~ age + sex + ph.ecog + strata(ph.ecog > 1), lung2,
                     ties = "breslow")
    cox <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog +
                                                 strata(ph.ecog > 1), data = lung2, modelType = "cox"),
                           control = control)
    expect_equal(unname(vcov(cox)), unname(vcov(goldCox)), tolerance = tolerance)
})

test_that("Schoenfeld residuals for cox_counting match coxph", {
    goldResiduals <- function(formula, data, weights, type = "schoenfeld") {
        data$w <- weights
        residuals(coxph(formula, data, weights = w, subset = w > 0, ties = "breslow", model = TRUE),
                  type)
    }
    expectResiduals <- function(formula, data, weights = rep(1, nrow(data))) {
        fit <- fitCounting(formula, data, weights = weights)
        expect_equivalent(residuals(fit), goldResiduals(formula, data, weights), tolerance = tolerance)
        expect_equal(names(residuals(fit)), names(goldResiduals(formula, data, weights)))
    }

    expectResiduals(Surv(start, stop, event) ~ x1, delayedEntry)
    expectResiduals(Surv(start, stop, event) ~ x1, tied) # failure ties
    expectResiduals(Surv(start, stop, event) ~ x1 + strata(stratum), tied)
    expectResiduals(Surv(start, stop, event) ~ x1, tied, weights = c(1, 2, 0.5, 1, 1.5, 1, 0.3, 1, 2, 0.7))
    # a held-out event has no residual
    expectResiduals(Surv(start, stop, event) ~ x1, tied, weights = c(1, 2, 0.5, 1, 1.5, 1, 0, 1, 2, 0.7))

    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                       start = "start", end = "stop", event = "event")
    expectResiduals(Surv(start, stop, event) ~ age, split)
    expectResiduals(Surv(start, stop, event) ~ x1 + strata(stratum), simulateLeftTruncated(800, 456))

    # several covariates: one column each, or a subset through parm
    formula <- Surv(start, stop, event) ~ age + sex + ph.ecog
    fit <- fitCounting(formula, split)
    gold <- goldResiduals(formula, split, rep(1, nrow(split)))
    expect_equal(residuals(fit), gold, tolerance = tolerance)
    expect_equal(residuals(fit, parm = c("sex", "age")), gold[, c("sex", "age")], tolerance = tolerance)
    expect_equal(residuals(fit, parm = "ph.ecog"), gold[, "ph.ecog"], tolerance = tolerance)

    # scaled residuals use the variance, so they need every parameter
    expect_equal(residuals(fit, type = "scaledsch"),
                 goldResiduals(formula, split, rep(1, nrow(split)), "scaledsch"), tolerance = tolerance)
    expect_error(residuals(fit, parm = "age", type = "scaledsch"), "all parameters")
})

test_that("Cross-validation keeps all rows of a subject in the same fold", {
    # Subjects are the fold units, so cross-validation on survSplit rows reproduces
    # cross-validation on the original one-row-per-subject data
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    lung2$id <- seq_len(nrow(lung2))
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                       start = "start", end = "stop", event = "event")

    # The default starting variance depends on the number of rows, so fix it
    prior <- createPrior("laplace", useCrossValidation = TRUE)
    cvControl <- createControl(noiseLevel = "silent", seed = 123, fold = 5, cvRepetitions = 1,
                               tolerance = 1E-10, startingVariance = 1)
    fitCV <- function(cyclopsData) fitCyclopsModel(cyclopsData, prior = prior, control = cvControl)

    unsplit <- fitCV(createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog, data = lung2,
                                       modelType = "cox"))
    bySubject <- fitCV(createCyclopsData(Surv(start, stop, event) ~ age + sex + ph.ecog + cluster(id),
                                         data = split, modelType = "cox_counting"))
    expect_equal(getHyperParameter(bySubject), getHyperParameter(unsplit), tolerance = tolerance)
    expect_equal(coef(bySubject), coef(unsplit), tolerance = tolerance)

    # convertToCyclopsData takes subjects from outcomes$subjectId
    outcomes <- data.frame(rowId = seq_len(nrow(split)), subjectId = split$id, y = split$event,
                           time = split$stop, startTime = split$start)
    covariates <- data.frame(rowId = rep(outcomes$rowId, 3), covariateId = rep(1:3, each = nrow(split)),
                             covariateValue = c(split$age, split$sex, split$ph.ecog))
    fromOutcomes <- fitCV(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting",
                                               quiet = TRUE))
    expect_equal(getHyperParameter(fromOutcomes), getHyperParameter(unsplit), tolerance = tolerance)

    # Row-level folds separate the rows of a subject
    byRow <- fitCV(createCyclopsData(Surv(start, stop, event) ~ age + sex + ph.ecog, data = split,
                                     modelType = "cox_counting"))
    expect_false(isTRUE(all.equal(getHyperParameter(byRow), getHyperParameter(unsplit))))
})

test_that("Bootstrap resamples all rows of a subject together", {
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    lung2$id <- seq_len(nrow(lung2))
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                       start = "start", end = "stop", event = "event")
    bootControl <- function(selectorType = "auto") {
        createControl(noiseLevel = "silent", seed = 321, tolerance = 1E-10, selectorType = selectorType)
    }
    replicates <- 20

    unsplit <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog,
                                                 data = lung2, modelType = "cox"),
                               control = bootControl())
    bootUnsplit <- runBootstrap(unsplit, replicates)$samples
    fitSplit <- function(selectorType) {
        cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ age + sex + ph.ecog + cluster(id),
                                         data = split, modelType = "cox_counting")
        fitCyclopsModel(cyclopsData, control = bootControl(selectorType))
    }

    # With subjects, "auto" resamples subjects, which reproduces the one-row-per-subject bootstrap
    expect_equal(runBootstrap(fitSplit("auto"), replicates)$samples, bootUnsplit, tolerance = tolerance)
    expect_equal(runBootstrap(fitSplit("bySubject"), replicates)$samples, bootUnsplit, tolerance = tolerance)
    expect_false(isTRUE(all.equal(runBootstrap(fitSplit("byRow"), replicates)$samples, bootUnsplit)))
})

test_that("Bootstrap by subject leaves out subjects with zero weight", {
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    lung2$id <- seq_len(nrow(lung2))
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                       start = "start", end = "stop", event = "event")
    set.seed(7)
    heldOut <- split$id %in% sample(lung2$id, 40) # every row of 40 subjects
    fitSplit <- function(data, weights = NULL) {
        cyclopsData <- createCyclopsData(Surv(start, stop, event) ~ age + sex + ph.ecog + cluster(id),
                                         data = data, modelType = "cox_counting")
        fitCyclopsModel(cyclopsData, weights = weights,
                        control = createControl(noiseLevel = "silent", seed = 321, tolerance = 1E-10,
                                                selectorType = "bySubject"))
    }

    # The same draws as a bootstrap of the subjects in the fit alone
    expect_equal(runBootstrap(fitSplit(split, as.numeric(!heldOut)), 20)$samples,
                 runBootstrap(fitSplit(split[!heldOut, ]), 20)$samples, tolerance = tolerance)
})
