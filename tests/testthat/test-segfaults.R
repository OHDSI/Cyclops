library("testthat")

context("test-segfaults.R")

test_that("sumByStratum and getNumberOfStrata do not segfault on unstratified convertToCyclopsData", {
    outcomes <- data.frame(rowId = 1:6, y = c(0, 1, 0, 1, 1, 0))
    covariates <- data.frame(
        rowId = c(1, 2, 4, 5, 2, 3, 6),
        covariateId = c(1, 1, 1, 1, 2, 2, 2),
        covariateValue = c(1, 1, 1, 1, 2.5, -1.5, 3.0)
    )
    dataPtr <- convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE)

    expect_equal(getNumberOfStrata(dataPtr), 6)

    s1 <- reduce(dataPtr, 1, groupBy = "stratum")
    expect_equal(dim(s1), c(6, 1))
    expect_equal(as.numeric(s1[, 1]), c(1, 1, 0, 1, 1, 0))

    s2 <- reduce(dataPtr, 2, groupBy = "stratum", power = 2)
    expect_equal(as.numeric(s2[, 1]), c(0, 6.25, 2.25, 0, 0, 9.0))

    s0 <- reduce(dataPtr, 2, groupBy = "stratum", power = 0)
    expect_equal(as.numeric(s0[, 1]), c(0, 1, 1, 0, 0, 1))
})

test_that("median and q95 normalization handle empty sparse columns without segfault", {
    outcomes <- data.frame(rowId = 1:5, y = c(1, 2, 1, 3, 2))
    xEmpty <- rep(0, 5)
    xActive <- c(0, 2, -4, 0, 6)

    for (normType in c("median", "q95")) {
        dataPtr <- createCyclopsData(
            y ~ 1,
            sparseFormula = ~ xEmpty + xActive,
            data = data.frame(y = outcomes$y, xEmpty = xEmpty, xActive = xActive),
            modelType = "pr",
            normalize = normType
        )
        expect_true(is.finite(summary(dataPtr)$scale[3]))
    }
})

test_that("cox_raw handles tied event times and zero weights without heap overflow", {
    test <- data.frame(
        length = c(4.0, 3.0, 3.0, 2.0, 2.0, 1.5, 1.0),
        event  = c(1,   1,   1,   1,   1,   0,   1),
        x1     = c(0,   2,   0,   0,   1,   1,   1),
        x2     = c(0,   0,   1,   1,   1,   0,   0)
    )

    ctrl <- createControl(convergenceType = "kkt", tolerance = 1e-10, noiseLevel = "silent")
    rawData <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox_raw")
    rawFit <- fitCyclopsModel(rawData, prior = createPrior("normal", variance = 2.0), control = ctrl)
    expect_equal(rawFit$return_flag, "SUCCESS")

    tiedData <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test, modelType = "cox")
    tiedFit <- fitCyclopsModel(tiedData, prior = createPrior("normal", variance = 2.0), control = ctrl)
    expect_equal(coef(rawFit), coef(tiedFit), tolerance = 1e-8)

    # Zero-weight rows (as in cross-validation folds)
    w <- c(0, 1, 1, 0, 1, 1, 1)
    weightedFit <- fitCyclopsModel(rawData, prior = createPrior("normal", variance = 2.0), control = ctrl, weights = w)
    expect_equal(weightedFit$return_flag, "SUCCESS")

    # All-zero weights must not read past weights[K]
    zeroFit <- fitCyclopsModel(rawData, prior = createPrior("normal", variance = 2.0), control = ctrl, weights = rep(0, nrow(test)))
    expect_true(is.list(zeroFit))

    # Unstratified convertToCyclopsData with cox_raw (leaves pid empty -> hPidInternal.size() < K)
    outcomes <- data.frame(rowId = seq_len(nrow(test)), y = test$event, time = test$length)
    covariates <- data.frame(
        rowId = c(2, 5, 6, 7, 3, 4, 5),
        covariateId = c(1, 1, 1, 1, 2, 2, 2),
        covariateValue = c(2, 1, 1, 1, 1, 1, 1)
    )
    rawConv <- convertToCyclopsData(
        outcomes, covariates, modelType = "cox_raw", addIntercept = FALSE, quiet = TRUE
    )
    convFit <- fitCyclopsModel(rawConv, prior = createPrior("normal", variance = 2.0), control = ctrl)
    expect_equal(convFit$return_flag, "SUCCESS")
    expect_equal(as.numeric(coef(convFit)), as.numeric(coef(tiedFit)), tolerance = 1e-8)
})
