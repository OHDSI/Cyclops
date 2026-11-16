library("testthat")
library("survival")

context("test-coxPredict.R")

# Data with tied times, strata, case weights and (start, time] intervals, in convertToCyclopsData format
simulateCoxPredictData <- function(n = 300, seed = 11) {
    set.seed(seed)
    data <- data.frame(rowId = 1:n, stratumId = sample(c(7, 3, 11), n, replace = TRUE),
                       time = sample(1:30, n, replace = TRUE), y = rbinom(n, 1, 0.5),
                       x1 = rnorm(n), x2 = rbinom(n, 1, 0.4), w = runif(n, 0.2, 3))
    data$startTime <- pmax(0, data$time - sample(1:20, n, replace = TRUE))
    data
}

toCovariates <- function(data) {
    rbind(data.frame(rowId = data$rowId, covariateId = 1, covariateValue = data$x1),
          data.frame(rowId = data$rowId[data$x2 == 1], covariateId = 2, covariateValue = 1))
}

newData <- data.frame(rowId = c(5, 1, 3), stratumId = c(3, 7, 11), x1 = c(-1, 0, 2), x2 = c(1, 0, 1))
predictTimes <- c(0.5, 1, 4, 10, 10.5, 29, 35) # before the first event, at and between event times, after the last

fitCoxPredict <- function(data, modelType, weighted, stratified) {
    outcomes <- data[, c("rowId", "time", "y",
                         if (modelType == "cox_counting") "startTime",
                         if (stratified) "stratumId")]
    if (weighted) {
        outcomes$weights <- data$w
    }
    cyclopsData <- convertToCyclopsData(outcomes, toCovariates(data), modelType = modelType, quiet = TRUE)
    fitCyclopsModel(cyclopsData, control = createControl(convergenceType = "gradient", tolerance = 1E-10,
                                                         noiseLevel = "silent"))
}

goldCoxPredict <- function(data, modelType, weighted, stratified) {
    formula <- if (modelType == "cox_counting") Surv(startTime, time, y) ~ x1 + x2 else Surv(time, y) ~ x1 + x2
    if (stratified) {
        formula <- update(formula, . ~ . + strata(stratumId))
    }
    data$caseWeight <- if (weighted) data$w else 1
    coxph(formula, data, weights = caseWeight, ties = "breslow")
}

# S(t | x) of each row of newData (by sorted rowId) at predictTimes, from survfit.coxph
goldSurvival <- function(gold, newdata) {
    curves <- survfit(gold, newdata = newdata, ctype = 1, stype = 2)
    survival <- t(sapply(seq_len(nrow(newdata)), function(i) {
        summary(curves[i], times = predictTimes, extend = TRUE)$surv
    }))
    survival[order(newdata$rowId), , drop = FALSE]
}

test_that("Cox survival prediction matches survfit.coxph with weights, strata and counting process", {
    data <- simulateCoxPredictData()
    newCovariates <- toCovariates(newData)
    for (modelType in c("cox", "cox_counting")) {
        for (weighted in c(FALSE, TRUE)) {
            for (stratified in c(FALSE, TRUE)) {
                fit <- fitCoxPredict(data, modelType, weighted, stratified)
                gold <- goldCoxPredict(data, modelType, weighted, stratified)
                expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-8)

                prediction <- predict(fit, newData[, c("rowId", "stratumId")], newCovariates, times = predictTimes)
                expect_equal(dim(prediction), c(nrow(newData), length(predictTimes)))
                expect_equal(rownames(prediction), as.character(sort(newData$rowId)))
                expect_equal(unname(prediction), goldSurvival(gold, newData), tolerance = 1E-6)
            }
        }
    }
})

test_that("Weighted Breslow survfit matches survfit.coxph at the mean covariates", {
    data <- simulateCoxPredictData(400, 12)
    means <- data.frame(x1 = mean(data$x1), x2 = mean(data$x2)) # survfit.cyclopsFit: unweighted means
    for (modelType in c("cox", "cox_counting")) {
        for (stratified in c(FALSE, TRUE)) {
            fit <- fitCoxPredict(data, modelType, weighted = TRUE, stratified)
            gold <- survfit(goldCoxPredict(data, modelType, weighted = TRUE, stratified), newdata = means,
                            ctype = 1, stype = 2)
            cyclops <- survfit(fit, type = "aalen")
            expect_equal(cyclops$time, gold$time)
            expect_equal(cyclops$surv, as.vector(gold$surv), tolerance = 1E-6)
            if (stratified) {
                expect_equal(unname(cyclops$strata), unname(gold$strata))
            }
        }
    }
})

test_that("Rows with weight 0 drop out of the Breslow baseline", {
    data <- simulateCoxPredictData(300, 13)
    data$w <- rep(c(1, 0, 1), length.out = nrow(data))
    newCovariates <- toCovariates(newData)
    for (modelType in c("cox", "cox_counting")) {
        fit <- fitCoxPredict(data, modelType, weighted = TRUE, stratified = TRUE)
        subset <- data[data$w > 0, ]
        gold <- goldCoxPredict(subset, modelType, weighted = FALSE, stratified = TRUE)
        prediction <- predict(fit, newData[, c("rowId", "stratumId")], newCovariates, times = predictTimes)
        expect_equal(unname(prediction), goldSurvival(gold, newData), tolerance = 1E-6)
        expect_equal(survfit(fit)$time,
                     survfit(fitCoxPredict(subset, modelType, weighted = FALSE, stratified = TRUE))$time)
    }
})

test_that("Cox prediction for a row without covariates is the baseline at a linear predictor of 0", {
    data <- simulateCoxPredictData(200, 14)
    fit <- fitCoxPredict(data, "cox_counting", weighted = TRUE, stratified = FALSE)
    gold <- goldCoxPredict(data, "cox_counting", weighted = TRUE, stratified = FALSE)
    baseline <- basehaz(gold, centered = FALSE)
    noCovariates <- data.frame(rowId = 1, covariateId = 1, covariateValue = 0)[0, ]
    prediction <- predict(fit, data.frame(rowId = 1), noCovariates, times = baseline$time)
    expect_equal(as.vector(prediction), exp(-baseline$hazard), tolerance = 1E-6)
})

test_that("Cox prediction from Andromeda tables equals prediction from data frames", {
    skip_if_not_installed("Andromeda")
    data <- simulateCoxPredictData(300, 15)
    newCovariates <- toCovariates(newData)
    outcomes <- data[, c("rowId", "stratumId", "startTime", "time", "y")]
    outcomes$weights <- data$w
    covariates <- merge(toCovariates(data), data[, c("rowId", "stratumId")]) # the tbl_dbi path sorts by stratumId
    andromeda <- Andromeda::andromeda(outcomes = outcomes, covariates = covariates,
                                      newOutcomes = newData[, c("rowId", "stratumId")],
                                      newCovariates = newCovariates)
    on.exit(Andromeda::close(andromeda))
    cyclopsData <- convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = "cox_counting",
                                        quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, control = createControl(convergenceType = "gradient", tolerance = 1E-10,
                                                                noiseLevel = "silent"))
    fromAndromeda <- predict(fit, andromeda$newOutcomes, andromeda$newCovariates, times = predictTimes)
    fromDataFrames <- predict(fit, newData[, c("rowId", "stratumId")], newCovariates, times = predictTimes)
    expect_equal(fromAndromeda, fromDataFrames)
    expect_equal(unname(fromAndromeda),
                 goldSurvival(goldCoxPredict(data, "cox_counting", weighted = TRUE, stratified = TRUE), newData),
                 tolerance = 1E-6)
})

test_that("Cox prediction stops once another model is fitted to the same data", {
    data <- simulateCoxPredictData(300, 17)
    newCovariates <- toCovariates(newData)
    newOutcomes <- newData[, c("rowId", "stratumId")]
    control <- createControl(convergenceType = "gradient", tolerance = 1E-10, noiseLevel = "silent")
    data <- data[order(-data$time, data$y, data$rowId), ] # data order, for the weights of fitCyclopsModel()
    weights <- data$w
    cyclopsData <- convertToCyclopsData(data[, c("rowId", "time", "y")], toCovariates(data), modelType = "cox",
                                        quiet = TRUE)
    first <- fitCyclopsModel(cyclopsData, weights = weights, control = control)
    gold <- goldSurvival(goldCoxPredict(data, "cox", weighted = TRUE, stratified = FALSE), newData)
    expect_equal(unname(predict(first, newOutcomes, newCovariates, times = predictTimes)), gold, tolerance = 1E-6)
    firstSurvfit <- survfit(first)

    for (later in list(list(weights = rev(weights)), list(weights = weights, prior = createPrior("normal", 0.01)),
                       list())) { # other weights, another prior, no weights
        latest <- do.call(fitCyclopsModel, c(list(cyclopsData, control = control), later))
        expect_error(predict(first, newOutcomes, newCovariates, times = predictTimes), "Another model")
        expect_error(survfit(first), "Another model")
        expect_silent(predict(latest, newOutcomes, newCovariates, times = predictTimes))
    }
    unweighted <- goldSurvival(goldCoxPredict(data, "cox", weighted = FALSE, stratified = FALSE), newData)
    expect_equal(unname(predict(latest, newOutcomes, newCovariates, times = predictTimes)), unweighted,
                 tolerance = 1E-6)

    # A later fit with its own interface leaves the first fit alone
    first <- fitCyclopsModel(cyclopsData, weights = weights, control = control)
    other <- fitCyclopsModel(cyclopsData, control = control, forceNewObject = TRUE)
    expect_equal(unname(predict(first, newOutcomes, newCovariates, times = predictTimes)), gold, tolerance = 1E-6)
    expect_equal(survfit(first), firstSurvfit)
    expect_equal(unname(predict(other, newOutcomes, newCovariates, times = predictTimes)), unweighted,
                 tolerance = 1E-6)

    # confint() and runBootstrap() put the fit back
    ci <- confint(first, c(1, 2))
    bootstrap <- runBootstrap(first, replicates = 5)
    expect_equal(unname(predict(first, newOutcomes, newCovariates, times = predictTimes)), gold, tolerance = 1E-6)
})

test_that("Cox prediction of a fit with all coefficients 0, from data frames and Andromeda tables", {
    data <- simulateCoxPredictData(300, 18)
    newCovariates <- toCovariates(newData)
    newOutcomes <- newData[, c("rowId", "stratumId")]
    outcomes <- data[, c("rowId", "stratumId", "time", "y")]
    outcomes$weights <- data$w
    cyclopsData <- convertToCyclopsData(outcomes, toCovariates(data), modelType = "cox", quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("laplace", variance = 1E-8))
    expect_true(all(coef(fit) == 0))
    gold <- coxph(Surv(time, y) ~ x1 + x2 + strata(stratumId), data, weights = w, ties = "breslow", init = c(0, 0),
                  control = coxph.control(iter.max = 0))
    fromDataFrames <- predict(fit, newOutcomes, newCovariates, times = predictTimes)
    expect_equal(rownames(fromDataFrames), as.character(sort(newData$rowId)))
    expect_equal(unname(fromDataFrames), goldSurvival(gold, newData), tolerance = 1E-6)
    skip_if_not_installed("Andromeda")
    andromeda <- Andromeda::andromeda(newOutcomes = newOutcomes, newCovariates = newCovariates)
    on.exit(Andromeda::close(andromeda))
    expect_equal(predict(fit, andromeda$newOutcomes, andromeda$newCovariates, times = predictTimes), fromDataFrames)
})

test_that("Cox prediction input checks", {
    data <- simulateCoxPredictData(100, 16)
    newCovariates <- toCovariates(newData)
    fit <- fitCoxPredict(data, "cox", weighted = FALSE, stratified = TRUE)
    expect_error(predict(fit, newData[, c("rowId", "stratumId")], newCovariates), "times")
    expect_error(predict(fit, times = 1), "times")
    expect_error(predict(fit, newData[, "rowId", drop = FALSE], newCovariates, times = 1), "stratumId")
    unknownStratum <- newData[, c("rowId", "stratumId")]
    unknownStratum$stratumId[1] <- 99
    expect_error(predict(fit, unknownStratum, newCovariates, times = 1), "stratumIds")
    rm("stratumIds", envir = fit$cyclopsData) # as data from another constructor
    expect_error(predict(fit, newData[, c("rowId", "stratumId")], newCovariates, times = 1), "convertToCyclopsData")

    formulaData <- createCyclopsData(Surv(time, y) ~ x1 + x2, data = data, modelType = "cox")
    formulaFit <- fitCyclopsModel(formulaData)
    newCovariates$covariateId <- c("x1", "x2")[newCovariates$covariateId]
    expect_error(predict(formulaFit, newData[, "rowId", drop = FALSE], newCovariates, times = 1), "covariateIds")
})
