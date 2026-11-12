library("testthat")

# Logistic data with large, shuffled row ids, and PLP-style folds (stratified by outcome, labels "a" to "e")
foldData <- function() {
    set.seed(7)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 3000, ncovars = 200, eCovarsPerRow = 8, model = "logistic")
    ids <- sample(5e9, 3000) # beyond 32 bits
    sim$outcomes$rowId <- ids[sim$outcomes$rowId]
    sim$covariates$rowId <- ids[sim$covariates$rowId]
    o <- sim$outcomes
    folds <- data.frame(rowId = o$rowId, fold = 0L)
    for (y in 0:1) {
        rows <- which(o$y == y)
        folds$fold[rows] <- sample(rep(1:5, length.out = length(rows)))
    }
    folds <- folds[sample(nrow(folds)), ] # any order
    folds$fold <- c("a", "b", "c", "d", "e")[folds$fold] # any labels
    list(sim = sim, folds = folds,
         data = function() convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE))
}

# out-of-fold predictions by refitting each fold at the given variance, as PLP's getCV() does
refitPredictions <- function(data, variance, foldOfRow) {
    prediction <- rep(NA_real_, length(foldOfRow))
    for (f in unique(foldOfRow[!is.na(foldOfRow)])) {
        fit <- fitCyclopsModel(data, createPrior("laplace", variance = variance),
                               createControl(tolerance = 1e-8, noiseLevel = "silent"),
                               weights = as.numeric(!is.na(foldOfRow) & foldOfRow != f), forceNewObject = TRUE,
                               warnings = FALSE)
        p <- predict(fit)
        prediction[which(foldOfRow == f)] <- p[which(foldOfRow == f)]
    }
    prediction
}

prior <- createPrior("laplace", useCrossValidation = TRUE)
control <- createControl(cvType = "auto", seed = 1, noiseLevel = "silent", tolerance = 1e-8)

test_that("given folds are used, and out-of-fold predictions match refits of each fold", {
    d <- foldData()
    fit <- fitCyclopsModel(d$data(), prior, control, cvFolds = d$folds, returnCvPredictions = TRUE)
    cp <- fit$cvPredictions
    expect_equal(nrow(cp), 3000)
    expect_false(anyNA(cp$prediction))
    expect_equal(cp$fold, d$folds$fold[match(as.character(cp$rowId), as.character(d$folds$rowId))])
    expect_true(setequal(cp$rowId, d$sim$outcomes$rowId))
    expect_equal(cp$prediction, refitPredictions(d$data(), fit$variance, cp$fold), tolerance = 1e-5)

    # the seed does not change given folds; predictions only when asked
    other <- fitCyclopsModel(d$data(), prior, createControl(cvType = "auto", seed = 99, noiseLevel = "silent",
                                                            tolerance = 1e-8), cvFolds = d$folds)
    expect_equal(other$variance, fit$variance)
    expect_null(other$cvPredictions)
})

test_that("random folds report the folds they used", {
    d <- foldData()
    fit <- fitCyclopsModel(d$data(), prior, control, returnCvPredictions = TRUE)
    cp <- fit$cvPredictions
    expect_equal(length(unique(cp$fold)), 10)
    expect_equal(cp$prediction, refitPredictions(d$data(), fit$variance, cp$fold), tolerance = 1e-5)
    plain <- fitCyclopsModel(d$data(), prior, control)
    expect_equal(fit$variance, plain$variance)
    expect_equal(coef(fit), coef(plain))
})

test_that("rows with weight 0 may lack a fold and get no prediction", {
    d <- foldData()
    weights <- rep(1, 3000)
    weights[1:100] <- 0 # rows in data order: the outcomes are sorted by rowId
    data <- d$data()
    rowIds <- sort(d$sim$outcomes$rowId)
    folds <- d$folds[!(as.character(d$folds$rowId) %in% as.character(rowIds[1:50])), ] # 50 of them without a fold
    expect_no_warning(fit <- fitCyclopsModel(data, prior, control, weights = weights, cvFolds = folds,
                                             returnCvPredictions = TRUE),
                      message = "refits") # NA by design is no failed refit
    cp <- fit$cvPredictions
    expect_true(all(is.na(cp$prediction[1:100])))
    expect_false(anyNA(cp$prediction[-(1:100)]))
    expect_true(all(is.na(cp$fold[1:50])))
    # a row with positive weight needs a fold
    expect_error(fitCyclopsModel(d$data(), prior, control, cvFolds = folds), "positive weight have no fold")
})

test_that("cvFolds and returnCvPredictions check their arguments", {
    d <- foldData()
    folds <- d$folds
    expect_error(fitCyclopsModel(d$data(), prior, createControl(fold = 10, noiseLevel = "silent"), cvFolds = folds),
                 "control\\$fold is 10 but cvFolds has 5 folds")
    expect_silent(fitCyclopsModel(d$data(), prior, createControl(fold = 5, noiseLevel = "silent"), cvFolds = folds,
                                  warnings = FALSE))
    expect_error(fitCyclopsModel(d$data(), prior, createControl(cvRepetitions = 2, noiseLevel = "silent"),
                                 cvFolds = folds), "cvRepetitions")
    expect_error(fitCyclopsModel(d$data(), prior, control, cvFolds = rbind(folds, folds[1, ])), "duplicate")
    expect_error(fitCyclopsModel(d$data(), prior, control, cvFolds = rbind(folds, data.frame(rowId = 1, fold = "a"))),
                 "not in the data")
    expect_error(fitCyclopsModel(d$data(), prior, control, cvFolds = transform(folds, fold = "a")), "two folds")
    expect_error(fitCyclopsModel(d$data(), prior, control, cvFolds = rep(1:2, 1500)), "data frame")
    expect_error(fitCyclopsModel(d$data(), createPrior("laplace", 0.1), control, cvFolds = folds), "useCrossValidation")
})

test_that("conditional models keep strata in one fold; predictions are for lr, pr and ls", {
    set.seed(8)
    sim <- simulateCyclopsData(nstrata = 300, nrows = 1500, ncovars = 50, eCovarsPerRow = 5, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "clr", quiet = TRUE)
    strata <- unique(sim$outcomes$stratumId)
    foldOfStratum <- setNames(sample(rep(1:4, length.out = length(strata))), strata)
    folds <- data.frame(rowId = sim$outcomes$rowId, fold = foldOfStratum[as.character(sim$outcomes$stratumId)])
    controlC <- createControl(cvType = "auto", selectorType = "byPid", seed = 1, noiseLevel = "silent")
    fit <- fitCyclopsModel(data, prior, controlC, cvFolds = folds)
    expect_equal(fit$return_flag, "SUCCESS")
    bad <- folds
    r <- which(duplicated(sim$outcomes$stratumId))[1] # a row of a stratum with several rows
    bad$fold[r] <- bad$fold[r] %% 4 + 1
    expect_error(fitCyclopsModel(data, prior, controlC, cvFolds = bad), "same cross-validation fold")
    expect_error(fitCyclopsModel(data, prior, controlC, cvFolds = folds, returnCvPredictions = TRUE),
                 "returnCvPredictions is available")
    # a row of weight 0 without a fold takes its stratum's
    weights <- as.numeric(.cyclopsGetRowLabels(data) != as.character(sim$outcomes$rowId[r])) # data order
    fit0 <- fitCyclopsModel(data, prior, controlC, weights = weights, cvFolds = folds[-r, ], warnings = FALSE)
    expect_equal(fit0$return_flag, "SUCCESS")
})

test_that("out-of-fold predictions keep row ids beyond 53 bits", {
    set.seed(10)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 400, ncovars = 20, eCovarsPerRow = 4, model = "logistic")
    ids <- bit64::as.integer64("9007199254740993") + bit64::as.integer64(sample(1e6, 400)) # above 2^53
    sim$outcomes$rowId <- ids[sim$outcomes$rowId]
    sim$covariates$rowId <- ids[sim$covariates$rowId]
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    folds <- data.frame(rowId = sim$outcomes$rowId, fold = rep(1:4, 100))
    fit <- fitCyclopsModel(data, prior, createControl(cvType = "auto", seed = 1, noiseLevel = "silent", minCVData = 10),
                           cvFolds = folds, returnCvPredictions = TRUE, warnings = FALSE)
    cp <- fit$cvPredictions
    expect_true(bit64::is.integer64(cp$rowId))
    expect_true(all(as.character(cp$rowId) %in% as.character(folds$rowId)))
    expect_equal(length(unique(as.character(cp$rowId))), 400)
})

test_that("rowIds given as a factor or as text select the same folds as numbers", {
    set.seed(11)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 400, ncovars = 20, eCovarsPerRow = 4, model = "logistic")
    data <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "lr", quiet = TRUE)
    folds <- data.frame(rowId = sim$outcomes$rowId, fold = sample(rep(1:4, 100))) # ids 1 to 400
    folds <- folds[sample(nrow(folds)), ]
    fitWith <- function(rowId) {
        fitCyclopsModel(data, prior, createControl(cvType = "auto", seed = 1, noiseLevel = "silent", minCVData = 10),
                        cvFolds = data.frame(rowId = rowId, fold = folds$fold), returnCvPredictions = TRUE,
                        warnings = FALSE)
    }
    byNumber <- fitWith(folds$rowId)
    # the levels of a factor of text sort as text ("1", "10", "100", ...), so its codes are not the ids
    for (rowId in list(factor(as.character(folds$rowId)), as.character(folds$rowId))) {
        fit <- fitWith(rowId)
        cp <- fit$cvPredictions
        expect_true(is.character(cp$rowId))
        expect_equal(cp$fold, folds$fold[match(cp$rowId, as.character(folds$rowId))])
        expect_equal(fit$variance, byNumber$variance)
        expect_equal(cp$prediction, byNumber$cvPredictions$prediction)
    }
})

test_that("formula data take a vector of folds in data order", {
    set.seed(9)
    n <- 600
    df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rbinom(n, 1, 0.3))
    df$y <- rpois(n, exp(0.2 + 0.3 * df$x1 - 0.2 * df$x3))
    data <- createCyclopsData(y ~ x1 + x2 + x3, data = df, modelType = "pr")
    fold <- rep(1:3, length.out = n)
    fit <- fitCyclopsModel(data, createPrior("normal", useCrossValidation = TRUE, exclude = 1),
                           createControl(cvType = "auto", seed = 1, noiseLevel = "silent"), cvFolds = fold,
                           returnCvPredictions = TRUE, warnings = FALSE)
    expect_equal(fit$cvPredictions$fold, if (is.null(data$sortOrder)) fold else fold[data$sortOrder])
    expect_false(anyNA(fit$cvPredictions$prediction))
})

test_that("out-of-fold predictions are the same on 1 and 4 threads", {
    skip_on_cran() # threads
    d <- foldData()
    one <- fitCyclopsModel(d$data(), prior, control, cvFolds = d$folds, returnCvPredictions = TRUE)
    four <- fitCyclopsModel(d$data(), prior, createControl(cvType = "auto", seed = 1, noiseLevel = "silent",
                                                           tolerance = 1e-8, threads = 4),
                            cvFolds = d$folds, returnCvPredictions = TRUE)
    expect_equal(four$variance, one$variance)
    expect_equal(four$cvPredictions$prediction, one$cvPredictions$prediction, tolerance = 1e-10)
})
