library("testthat")
library("survival")

tolerance <- 1E-6

# Logistic data with each subject's row twice (as a person contributes several rows to a pooled hazard design),
# and the same data with one row of weight 2 per subject. Subjects are the units in the first, rows in the second,
# so cross-validation and the bootstrap should draw the same units from both.
subjectData <- function() {
    # Explicit kinds: other test files change the generator, and the data need a clear signal (a cross-validated
    # variance well away from the null model)
    set.seed(11, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    n <- 400
    x <- matrix(rbinom(n * 10, 1, 0.3), n, 10)
    y <- rbinom(n, 1, plogis(-1 + as.vector(x %*% c(1, -1, 0.7, rep(0, 7)))))
    nonZero <- which(x != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nonZero[, 1], covariateId = nonZero[, 2], covariateValue = 1)
    list(copies = data.frame(rowId = seq_len(2 * n), subjectId = rep(seq_len(n), each = 2) * 10, y = rep(y, each = 2)),
         copyCovariates = rbind(transform(covariates, rowId = 2 * rowId - 1),
                                transform(covariates, rowId = 2 * rowId)),
         weighted = data.frame(rowId = seq_len(n), y = y, weights = 2),
         covariates = covariates)
}

cvPrior <- createPrior("laplace", useCrossValidation = TRUE, exclude = 0)
cvControl <- function(selectorType = "auto") {
    createControl(noiseLevel = "silent", seed = 123, fold = 5, cvRepetitions = 1, tolerance = 1E-10,
                  startingVariance = 0.1, selectorType = selectorType)
}

test_that("convertToCyclopsData loads subjects for every model type", {
    d <- subjectData()
    nSubjects <- length(unique(d$copies$subjectId))
    for (modelType in c("lr", "pr", "ls")) {
        outcomes <- transform(d$copies, time = 1)
        cyclopsData <- convertToCyclopsData(outcomes, d$copyCovariates, modelType = modelType, quiet = TRUE)
        expect_equal(Cyclops:::.cyclopsGetNumberOfSubjects(cyclopsData), nSubjects)
        cyclopsData <- convertToCyclopsData(outcomes[, names(outcomes) != "subjectId"], d$copyCovariates,
                                            modelType = modelType, quiet = TRUE)
        expect_equal(Cyclops:::.cyclopsGetNumberOfSubjects(cyclopsData), 0)
    }

    andromeda <- Andromeda::andromeda(outcomes = d$copies, covariates = d$copyCovariates)
    on.exit(Andromeda::close(andromeda))
    cyclopsData <- convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = "lr", quiet = TRUE)
    expect_equal(Cyclops:::.cyclopsGetNumberOfSubjects(cyclopsData), nSubjects)
})

test_that("lr cross-validation keeps all rows of a subject in one fold", {
    d <- subjectData()
    fitCV <- function(outcomes, covariates, selectorType = "auto", returnCvPredictions = FALSE) {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE),
                        prior = cvPrior, control = cvControl(selectorType),
                        returnCvPredictions = returnCvPredictions)
    }
    weighted <- fitCV(d$weighted, d$covariates)

    # With subjects, "auto" folds by subject, which reproduces the folds of one row per subject
    for (selectorType in c("auto", "bySubject")) {
        bySubject <- fitCV(d$copies, d$copyCovariates, selectorType)
        expect_equal(getHyperParameter(bySubject), getHyperParameter(weighted), tolerance = tolerance)
        expect_equal(coef(bySubject), coef(weighted), tolerance = tolerance)
    }
    folds <- fitCV(d$copies, d$copyCovariates, returnCvPredictions = TRUE)$cvPredictions$fold
    expect_true(all(tapply(folds, d$copies$subjectId, function(f) length(unique(f))) == 1))

    # Row-level folds separate the rows of a subject
    byRow <- fitCV(d$copies, d$copyCovariates, "byRow")
    expect_false(isTRUE(all.equal(getHyperParameter(byRow), getHyperParameter(weighted))))
})

test_that("lr bootstrap resamples all rows of a subject together", {
    d <- subjectData()
    fitBoot <- function(outcomes, covariates, selectorType = "auto") {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE),
                        prior = createPrior("normal", variance = 1, exclude = 0),
                        control = createControl(noiseLevel = "silent", seed = 321, tolerance = 1E-10,
                                                selectorType = selectorType))
    }
    replicates <- 20
    bootWeighted <- runBootstrap(fitBoot(d$weighted, d$covariates), replicates)$samples

    expect_equal(runBootstrap(fitBoot(d$copies, d$copyCovariates), replicates)$samples, bootWeighted,
                 tolerance = tolerance)
    expect_equal(runBootstrap(fitBoot(d$copies, d$copyCovariates, "bySubject"), replicates)$samples, bootWeighted,
                 tolerance = tolerance)
    expect_false(isTRUE(all.equal(runBootstrap(fitBoot(d$copies, d$copyCovariates, "byRow"), replicates)$samples,
                                  bootWeighted)))
})

test_that("cluster() gives subjects to lr formula data", {
    set.seed(5, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    n <- 300
    one <- data.frame(id = seq_len(n), matrix(rnorm(n * 8), n, 8, dimnames = list(NULL, paste0("x", 1:8))))
    one$y <- rbinom(n, 1, plogis(-0.5 + 0.5 * one$x1 - 0.5 * one$x2))
    two <- one[rep(seq_len(n), each = 2), ]
    fitCV <- function(cyclopsData) {
        fitCyclopsModel(cyclopsData, prior = createPrior("laplace", useCrossValidation = TRUE, exclude = "(Intercept)"),
                        control = cvControl())
    }
    formula <- y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

    clustered <- createCyclopsData(update(formula, . ~ . + cluster(id)), data = two, modelType = "lr")
    expect_equal(Cyclops:::.cyclopsGetNumberOfSubjects(clustered), n)
    weighted <- fitCV(createCyclopsData(formula, data = one, weights = rep(2, n), modelType = "lr"))
    expect_equal(getHyperParameter(fitCV(clustered)), getHyperParameter(weighted), tolerance = tolerance)
})
