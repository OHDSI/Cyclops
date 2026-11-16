library("testthat")
library("Cyclops")
library("survival")

simulateRefitData <- function(modelType, n = 400, p = 3, nstrata = 40) {
    x <- matrix(rnorm(n * p), n, p)
    eta <- drop(x %*% c(0.5, -0.3, 0.2))
    outcomes <- data.frame(rowId = 1:n, stratumId = sort(sample(1:nstrata, n, TRUE)), time = rexp(n) * 10 + 1)
    if (modelType %in% c("lr", "clr")) outcomes$y <- rbinom(n, 1, plogis(eta - 0.5))
    if (modelType %in% c("pr", "cpr")) outcomes$y <- rpois(n, outcomes$time * exp(eta - 2))
    if (modelType == "ls") outcomes$y <- eta + rnorm(n)
    if (modelType %in% c("cox", "cox_counting")) {
        eventTime <- ceiling(rexp(n, exp(eta) / 20))
        censoringTime <- ceiling(runif(n, 1, 40))
        outcomes$time <- pmin(eventTime, censoringTime)
        outcomes$y <- as.numeric(eventTime <= censoringTime)
        outcomes$stratumId <- 0
        if (modelType == "cox_counting") {
            outcomes$startTime <- pmax(0, outcomes$time - ceiling(runif(n, 1, 10)))
        }
    }
    if (modelType %in% c("lr", "pr", "ls")) outcomes$stratumId <- NULL
    covariates <- data.frame(rowId = rep(1:n, p), covariateId = rep(1:p, each = n), covariateValue = c(x))
    if (!is.null(outcomes$stratumId)) covariates$stratumId <- rep(outcomes$stratumId, p)
    list(outcomes = outcomes, covariates = covariates)
}

# Weights with zeros: in conditional models, whole strata
simulateRefitWeights <- function(modelType, outcomes) {
    if (modelType %in% c("clr", "cpr")) {
        strata <- unique(outcomes$stratumId)
        return(ifelse(outcomes$stratumId %in% sample(strata, length(strata) %/% 3), 0, rpois(nrow(outcomes), 1) + 1))
    }
    rpois(nrow(outcomes), 1) # about a third zero
}

tightControl <- createControl(noiseLevel = "silent", tolerance = 1E-12, convergenceType = "gradient")

fitData <- function(outcomes, covariates, modelType, prior = createPrior("none")) {
    fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = modelType, quiet = TRUE),
                    prior = prior, control = tightControl, warnings = FALSE)
}

test_that("A refit after updateWeights equals a fit of rebuilt data, and zero weights equal dropping the rows", {
    set.seed(123)
    for (modelType in c("lr", "pr", "ls", "cox", "cox_counting", "clr", "cpr")) {
        for (prior in list(createPrior("none"), createPrior("normal", variance = 0.5))) {
            sim <- simulateRefitData(modelType)
            cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = modelType, quiet = TRUE)
            fitCyclopsModel(cyclopsData, prior = prior, control = tightControl, warnings = FALSE)

            weights <- simulateRefitWeights(modelType, sim$outcomes)
            updateWeights(cyclopsData, data.frame(rowId = sim$outcomes$rowId, weights = weights))
            refit <- fitCyclopsModel(cyclopsData, prior = prior, control = tightControl, warnings = FALSE)

            weighted <- sim$outcomes
            weighted$weights <- weights
            rebuilt <- fitData(weighted, sim$covariates, modelType, prior)
            expect_equal(coef(refit), coef(rebuilt), tolerance = 1E-8, info = modelType)
            expect_equal(refit$log_likelihood, rebuilt$log_likelihood, tolerance = 1E-8, info = modelType)

            kept <- weighted[weights > 0, ]
            dropped <- fitData(kept, sim$covariates[sim$covariates$rowId %in% kept$rowId, ], modelType, prior)
            expect_equal(coef(refit), coef(dropped), tolerance = 1E-8, info = modelType)
            expect_equal(refit$log_likelihood, dropped$log_likelihood, tolerance = 1E-8, info = modelType)
            if (modelType %in% c("lr", "cox", "cox_counting") && prior$priorType == "none") {
                expect_equal(vcov(refit), vcov(dropped), tolerance = 1E-6, info = modelType)
            }
        }
    }
})

test_that("updateWeights takes a vector in the order of fitCyclopsModel weights", {
    set.seed(123)
    sim <- simulateRefitData("cox", n = 200)
    data <- data.frame(time = sim$outcomes$time, y = sim$outcomes$y,
                       x1 = sim$covariates$covariateValue[sim$covariates$covariateId == 1],
                       x2 = sim$covariates$covariateValue[sim$covariates$covariateId == 2])
    weights <- rpois(nrow(data), 1)

    cyclopsData <- createCyclopsData(Surv(time, y) ~ x1 + x2, data = data, modelType = "cox")
    updateWeights(cyclopsData, weights)
    refit <- fitCyclopsModel(cyclopsData, control = tightControl)
    expect_warning(passed <- fitCyclopsModel(cyclopsData, weights = weights, control = tightControl),
                   "Using weights passed")
    expect_equal(coef(refit), coef(passed), tolerance = 1E-10)
    expect_equal(coef(refit), coef(fitCyclopsModel(createCyclopsData(Surv(time, y) ~ x1 + x2,
                                                                      data = data[weights > 0, ], modelType = "cox"),
                                                   weights = weights[weights > 0], control = tightControl)),
                 tolerance = 1E-8)

    # NULL: back to unweighted
    updateWeights(cyclopsData, NULL)
    expect_equal(coef(fitCyclopsModel(cyclopsData, control = tightControl)),
                 coef(fitCyclopsModel(createCyclopsData(Surv(time, y) ~ x1 + x2, data = data, modelType = "cox"),
                                      control = tightControl)), tolerance = 1E-8)
})

test_that("Refits with frequency weights are reproducible with a seed", {
    set.seed(123)
    sim <- simulateRefitData("lr")
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    replicates <- function(seed) {
        set.seed(seed)
        t(sapply(1:5, function(i) {
            updateWeights(cyclopsData, as.vector(stats::rmultinom(1, nrow(sim$outcomes), rep(1, nrow(sim$outcomes)))))
            coef(fitCyclopsModel(cyclopsData, control = tightControl))
        }))
    }
    first <- replicates(1)
    expect_equal(replicates(1), first, tolerance = 0)
    expect_false(isTRUE(all.equal(replicates(2), first)))
})

test_that("updateWeights checks its input", {
    set.seed(123)
    sim <- simulateRefitData("lr", n = 50)
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    expect_error(updateWeights(cyclopsData, rep(1, 49)), "for each of the 50")
    expect_error(updateWeights(cyclopsData, c(-1, rep(1, 49))), "non-negative")
    expect_error(updateWeights(cyclopsData, c(NA, rep(1, 49))), "finite")
    expect_error(updateWeights(cyclopsData, data.frame(rowId = 1:50, weight = 1)), "columns rowId and weights")
    expect_error(updateWeights(cyclopsData, data.frame(rowId = 1:49, weights = 1)), "for each of the 50")
    expect_error(updateWeights(cyclopsData, data.frame(rowId = c(1:50, 51), weights = 1)), "no other rowIds")
    expect_error(updateWeights(cyclopsData, data.frame(rowId = c(1:49, 1), weights = 1)), "Duplicate")

    formulaData <- createCyclopsData(y ~ 1, data = data.frame(y = sim$outcomes$y), modelType = "lr")
    expect_error(updateWeights(formulaData, data.frame(rowId = 1:50, weights = 1)), "no row ids")

    sim <- simulateRefitData("clr", n = 100, nstrata = 10)
    cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "clr", quiet = TRUE)
    weights <- rep(1, 100)
    weights[sim$outcomes$stratumId == sim$outcomes$stratumId[1]] <- 0 # a whole stratum
    expect_silent(updateWeights(cyclopsData, weights))
    weights[which(sim$outcomes$stratumId == sim$outcomes$stratumId[100])[1]] <- 0 # one row of a stratum
    expect_error(updateWeights(cyclopsData, weights), "1 strata have rows of weight 0")
})

test_that("A refit after updateOutcomes equals a fit of rebuilt data", {
    set.seed(123)
    for (modelType in c("lr", "pr", "ls", "cox", "cox_counting", "clr", "cpr")) {
        sim <- simulateRefitData(modelType)
        n <- nrow(sim$outcomes)
        cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = modelType, quiet = TRUE)
        fitCyclopsModel(cyclopsData, control = tightControl, warnings = FALSE)

        y <- switch(modelType,
                    lr = , clr = rbinom(n, 1, 0.4),
                    pr = , cpr = rpois(n, 1),
                    ls = rnorm(n),
                    { # Cox: new events where no other row has the same time, which keeps the sort order
                        y <- sim$outcomes$y
                        unique <- !(duplicated(sim$outcomes$time) | duplicated(sim$outcomes$time, fromLast = TRUE))
                        y[unique] <- rbinom(sum(unique), 1, 0.5)
                        y
                    })
        updateOutcomes(cyclopsData, data.frame(rowId = sim$outcomes$rowId, y = y))
        refit <- fitCyclopsModel(cyclopsData, control = tightControl, warnings = FALSE)

        newOutcomes <- sim$outcomes
        newOutcomes$y <- y
        rebuilt <- fitData(newOutcomes, sim$covariates, modelType)
        expect_equal(coef(refit), coef(rebuilt), tolerance = 1E-8, info = modelType)
        expect_equal(refit$log_likelihood, rebuilt$log_likelihood, tolerance = 1E-8, info = modelType)
    }
})

test_that("updateOutcomes keeps the sort order of Cox data", {
    data <- data.frame(time = c(4, 3, 3, 2, 2, 1, 1, 5), y = c(1, 0, 1, 1, 1, 0, 1, 0), x = c(0, 2, 0, 0, 1, 1, 1, 1))
    cyclopsData <- createCyclopsData(Surv(time, y) ~ x, data = data, modelType = "cox")
    fit <- fitCyclopsModel(cyclopsData)

    expect_error(updateOutcomes(cyclopsData, c(1, 1, 0, 1, 1, 0, 1, 0)), "change the order of the rows")
    expect_error(updateOutcomes(cyclopsData, c(1, 2, 0, 1, 1, 0, 1, 0)), "0 or 1")

    y <- c(0, 0, 1, 1, 1, 0, 1, 1) # data order
    updateOutcomes(cyclopsData, y)
    newData <- data
    newData$y <- y
    expect_equal(coef(fitCyclopsModel(cyclopsData)),
                 coef(fitCyclopsModel(createCyclopsData(Surv(time, y) ~ x, data = newData, modelType = "cox"))),
                 tolerance = 1E-8)
})

test_that("updateOutcomes is not available for other model types", {
    set.seed(123)
    outcomes <- data.frame(rowId = 1:20, time = 1:20, y = rep(c(0, 1, 2, 1), 5))
    covariates <- data.frame(rowId = 1:20, covariateId = 1, covariateValue = rnorm(20))
    capture.output(cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "fgr", quiet = TRUE))
    expect_error(updateOutcomes(cyclopsData, outcomes[, c("rowId", "y")]), "not available for modelType = \"fgr\"")
})
