library("testthat")

# convertToCyclopsData() reads the covariates in place and sorts lr and pr covariates through a permutation: the data
# must equal those of covariates sorted beforehand, whatever the types of the ids and the order of the rows

makeTriplets <- function(n = 200, p = 25, entries = 1500, seed = 1) {
    set.seed(seed)
    key <- unique(data.frame(rowId = sample.int(n, entries, TRUE), covariateId = sample.int(p, entries, TRUE)))
    key$covariateValue <- ifelse(runif(nrow(key)) < 0.3, round(rnorm(nrow(key)), 2), 1)
    list(outcomes = data.frame(rowId = 1:n, y = rbinom(n, 1, 0.3), time = rexp(n) + 0.1), covariates = key)
}

fingerprint <- function(data) {
    ids <- getCovariateIds(data)
    beta <- seq(-1, 1, length.out = length(ids))
    fit <- fitCyclopsModel(data, prior = createPrior("none"), startingCoefficients = beta,
                           fixedCoefficients = rep(TRUE, length(ids)), warnings = FALSE)
    list(ids = as.character(ids), names = data$coefficientNames, types = getCovariateTypes(data, ids),
         prediction = predict(fit))
}

test_that("unsorted covariates give the data of sorted ones, for any id type", {
    s <- makeTriplets()
    sorted <- s$covariates[order(s$covariates$covariateId, s$covariates$rowId), ]
    for (modelType in c("lr", "pr")) {
        outcomes <- s$outcomes[, c("rowId", "y", if (modelType == "pr") "time")]
        reference <- fingerprint(convertToCyclopsData(outcomes, sorted, modelType = modelType, quiet = TRUE))
        for (ordering in c("byRow", "random")) {
            covariates <- if (ordering == "byRow") s$covariates[order(s$covariates$rowId, s$covariates$covariateId), ] else
                s$covariates[sample.int(nrow(s$covariates)), ]
            for (idType in c("double", "integer", "integer64")) {
                typed <- covariates
                typed$covariateId <- switch(idType, double = as.numeric(typed$covariateId),
                                            integer = as.integer(typed$covariateId),
                                            integer64 = bit64::as.integer64(typed$covariateId))
                expect_equal(fingerprint(convertToCyclopsData(outcomes, typed, modelType = modelType, quiet = TRUE)),
                             reference, info = paste(modelType, ordering, idType))
            }
        }
    }
})

test_that("unsorted covariates with rowIds missing from the outcomes are dropped as before", {
    s <- makeTriplets(seed = 2)
    covariates <- s$covariates[sample.int(nrow(s$covariates)), ]
    covariates$rowId[1:4] <- 1000 + 1:4
    kept <- covariates[covariates$rowId %in% s$outcomes$rowId, ]
    kept <- kept[order(kept$covariateId, kept$rowId), ]
    expect_equal(fingerprint(convertToCyclopsData(s$outcomes[, c("rowId", "y")], covariates, quiet = TRUE)),
                 fingerprint(convertToCyclopsData(s$outcomes[, c("rowId", "y")], kept, quiet = TRUE)))
})

test_that("the loader reads ids of any numeric type in place and checks a given order", {
    s <- makeTriplets(seed = 3)
    sorted <- s$covariates[order(s$covariates$covariateId, s$covariates$rowId), ]
    load <- function(covariateId, rowId, order = NULL, values = sorted$covariateValue) {
        data <- Cyclops:::createSqlCyclopsData(modelType = "lr")
        Cyclops:::loadNewSqlCyclopsDataY(data, NULL, s$outcomes$rowId, s$outcomes$y, NULL)
        Cyclops:::loadNewSqlCyclopsDataMultipleX(data, covariateId, rowId, values,
                                                 name = unique(sorted$covariateId), order = order)
        data
    }
    reference <- fingerprint(load(bit64::as.integer64(sorted$covariateId), bit64::as.integer64(sorted$rowId)))
    expect_equal(fingerprint(load(as.numeric(sorted$covariateId), as.integer(sorted$rowId))), reference)
    expect_equal(fingerprint(load(as.character(sorted$covariateId), as.character(sorted$rowId))), reference)
    shuffle <- sample.int(nrow(sorted))
    shuffled <- sorted[shuffle, ]
    expect_equal(fingerprint(load(shuffled$covariateId, shuffled$rowId, order = order(shuffle),
                                  values = shuffled$covariateValue)), reference)
    expect_error(load(shuffled$covariateId, shuffled$rowId, order = c(0L, seq_len(nrow(sorted) - 1)),
                      values = shuffled$covariateValue), "permutation")
})

test_that("the copy-free helpers agree with unique(), match() and order()", {
    x <- c(3, 3, 5, 7, 7, 7, 10)
    expect_equal(x[Cyclops:::.runStarts(x)], unique(x))
    expect_null(Cyclops:::.runStarts(c(2, 1)))
    expect_null(Cyclops:::.runStarts(c(1, NA)))
    expect_true(Cyclops:::.allIn(c(1L, 3L, 3L), c(3, 2, 1)))
    expect_false(Cyclops:::.allIn(c(1, 4), 1:3))
    expect_true(is.na(Cyclops:::.allIn(c(1, NA), 1:3)))
    set.seed(4)
    a <- sample(c(-0, 0, 1:5), 500, TRUE); b <- sample(c(2.5, -1, 7L), 500, TRUE)
    expect_identical(Cyclops:::.sortOrder(a, b), order(a, b))
    expect_identical(Cyclops:::.sortOrder(as.integer(a), b), order(as.integer(a), b))
    expect_null(Cyclops:::.sortOrder(c(1, NaN), c(1, 2)))
    expect_null(Cyclops:::.sortOrder(bit64::as.integer64(1:2), c(1, 2)))
})

test_that("lr and pr Andromeda covariates in any order give the data of sorted data frames", {
    skip_if_not_installed("Andromeda")
    s <- makeTriplets(n = 30000, p = 30, entries = 300000, seed = 5) # several batches of 100,000
    sorted <- s$covariates[order(s$covariates$covariateId, s$covariates$rowId), ]
    for (modelType in c("lr", "pr")) {
        outcomes <- s$outcomes[, c("rowId", "y", if (modelType == "pr") "time")]
        reference <- fingerprint(convertToCyclopsData(outcomes, sorted, modelType = modelType, quiet = TRUE))
        for (ordering in c("byRow", "random")) {
            covariates <- if (ordering == "byRow") s$covariates[order(s$covariates$rowId, s$covariates$covariateId), ] else
                s$covariates[sample.int(nrow(s$covariates)), ]
            andromeda <- Andromeda::andromeda(outcomes = outcomes, covariates = covariates)
            expect_equal(fingerprint(convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = modelType,
                                                          quiet = TRUE)),
                         reference, info = paste(modelType, ordering))
            Andromeda::close(andromeda)
        }
    }
})

test_that("unsorted Andromeda covariates keep indicator, value and error behaviour", {
    skip_if_not_installed("Andromeda")
    s <- makeTriplets(seed = 6)
    covariates <- s$covariates[sample.int(nrow(s$covariates)), ]
    covariates$covariateValue[covariates$covariateId == 3] <- 1 # an indicator column among columns with values
    convert <- function(covariates, ...) {
        andromeda <- Andromeda::andromeda(outcomes = s$outcomes[, c("rowId", "y")], covariates = covariates)
        on.exit(Andromeda::close(andromeda))
        convertToCyclopsData(andromeda$outcomes, andromeda$covariates, quiet = TRUE, ...)
    }
    sorted <- covariates[order(covariates$covariateId, covariates$rowId), ]
    expect_equal(fingerprint(convert(covariates)),
                 fingerprint(convertToCyclopsData(s$outcomes[, c("rowId", "y")], sorted, quiet = TRUE)))
    expect_equal(unname(getCovariateTypes(convert(covariates), 3)), "indicator")
    stray <- covariates
    inFirst <- which(stray$covariateId == min(stray$covariateId))[1:2]
    stray$rowId[inFirst] <- c(5001, 5000)
    expect_error(convert(stray, checkRowIds = FALSE), "rowId 5000 is not among") # the first in covariateId, rowId order
    duplicated <- rbind(covariates, covariates[10, ])
    expect_error(convert(duplicated), "Repeated row-column entry")
})

test_that("clr, cpr and Cox-family Andromeda covariates in any order give the data of sorted data frames", {
    skip_if_not_installed("Andromeda")
    set.seed(7)
    n <- 600
    outcomes <- data.frame(rowId = 1:n, stratumId = rep(1:100, each = 6), time = round(rexp(n) * 10, 1) + 0.1)
    outcomes$time[seq(1, n, 7)] <- outcomes$time[seq(2, n, 7)] # ties, so that y and rowId order rows too
    key <- unique(data.frame(rowId = sample.int(n, 5000, TRUE), covariateId = sample.int(30, 5000, TRUE)))
    key$covariateValue <- ifelse(runif(nrow(key)) < 0.3, round(rnorm(nrow(key)), 2), 1)
    key$stratumId <- outcomes$stratumId[key$rowId]
    shuffled <- key[sample.int(nrow(key)), ]
    for (modelType in c("clr", "cpr", "cox", "fgr", "cox_counting")) {
        o <- outcomes
        o$y <- switch(modelType, clr = as.numeric(!duplicated(o$stratumId)), cpr = rpois(n, 1),
                      fgr = sample(0:2, n, TRUE), rbinom(n, 1, 0.4))
        if (modelType == "cox_counting") o$startTime <- floor(o$time * runif(n, 0.1, 0.9) * 10) / 10
        andromeda <- Andromeda::andromeda(outcomes = o, covariates = shuffled)
        expect_equal(fingerprint(convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = modelType,
                                                      quiet = TRUE)),
                     fingerprint(convertToCyclopsData(o, key, modelType = modelType, quiet = TRUE)), info = modelType)
        Andromeda::close(andromeda)
    }
})

test_that("Andromeda inputs with unusual ids load as before (review findings)", {
    skip_if_not_installed("Andromeda")
    outcomes <- data.frame(rowId = 1:6, y = c(0, 1, 0, 1, 0, 1))
    viaAndromeda <- function(covariates, outcomes = data.frame(rowId = 1:6, y = c(0, 1, 0, 1, 0, 1)), modelType = "lr") {
        andromeda <- Andromeda::andromeda(outcomes = outcomes, covariates = covariates)
        on.exit(Andromeda::close(andromeda))
        convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = modelType, quiet = TRUE)
    }
    names <- function(data) list(data$coefficientNames, as.character(getCovariateIds(data)))
    # character ids: database (lexicographic) order, names on their own columns
    data <- viaAndromeda(data.frame(rowId = c(1, 2, 3), covariateId = c("2", "10", "10"), covariateValue = 1))
    expect_equal(names(data), list(c("(Intercept)", "10", "2"), c("0", "10", "2")))
    # a covariate id that is the intercept's, not first in its batch: loaded as before, no crash
    data <- viaAndromeda(data.frame(rowId = c(1, 2), covariateId = c(-1, 0), covariateValue = 1))
    expect_equal(names(data), list(c("(Intercept)", "-1", "0"), c("0", "-1", "0")))
    # NA and colliding fractional ids load as before
    expect_equal(names(viaAndromeda(data.frame(rowId = 1:3, covariateId = c(1, NA, 2), covariateValue = 1)))[[2]],
                 c("0", "1", "2", NA))
    expect_equal(names(viaAndromeda(data.frame(rowId = 1:2, covariateId = c(1.2, 1.7), covariateValue = 1)))[[2]],
                 c("0", "1"))
    # duplicated outcome rowIds in a Cox model: still the duplicate error
    expect_error(viaAndromeda(data.frame(rowId = 1, covariateId = 5, covariateValue = 1),
                              outcomes = data.frame(rowId = c(1, 1, 2), y = c(0, 1, 0), time = c(10, 5, 8)),
                              modelType = "cox"), "Repeated row-column entry")
})

test_that("wide Andromeda designs load in time linear in the covariates", {
    skip_if_not_installed("Andromeda")
    set.seed(8)
    n <- 20000
    covariates <- data.frame(rowId = sample.int(n, 100000, TRUE), covariateId = as.numeric(1:100000), covariateValue = 1)
    andromeda <- Andromeda::andromeda(outcomes = data.frame(rowId = 1:n, y = rbinom(n, 1, 0.1)), covariates = covariates)
    on.exit(Andromeda::close(andromeda))
    time <- system.time(data <- convertToCyclopsData(andromeda$outcomes, andromeda$covariates, quiet = TRUE))[3]
    expect_equal(getNumberOfCovariates(data), 100001)
    expect_lt(time, 20) # one linear search per covariate took minutes
})
