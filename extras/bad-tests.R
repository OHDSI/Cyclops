library(testthat)
library(boot)

test_that("Small conditional logistic regression bootstrap", {
    set.seed(123)
    dataPtr <- createCyclopsData(case ~ spontaneous + induced + strata(stratum),
                                 data = infert,
                                 modelType = "clr")

    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"))

    bs <- runBootstrap(cyclopsFit, replicates = 4999)

    execute <- function(d, f) {
        coef(clogit(case ~ spontaneous + induced + strata(stratum),
                 data = d[f,]))
    }

    bb <- boot(infert, execute, R = 4999)


    expect_lt(abs(mean(bs$summary$bias)), 0.001)
    # Error: abs(mean(bs$summary$bias)) is not strictly less than 0.001. Difference: 0.0808
})

convertToDf <- function(sim, ncovars) {

    if (ncovars < max(sim$covariates$covariateId)) {
        stop("Too few covariates")
    }

    mat <- matrix(0, nrow = nrow(sim$outcomes), ncol = ncovars)

    invisible(mapply(function(rowId, covariateId, covariateValue) {
            mat[rowId, covariateId] <<- covariateValue
        },
        sim$covariates$rowId, sim$covariates$covariateId, sim$covariates$covariateValue))

    df <- as.data.frame(mat)
    outcomes <- sim$outcomes[order(sim$outcomes$rowId),]
    df$y <- outcomes$y
    df$stratumId <- outcomes$stratumId

    if (!is.null(outcomes$weights)) {
        df$weights <- outcomes$weights
    }

    if (!is.null(outcomes$time)) {
        df$time <- outcomes$time
    }

    return(df)
}

getAllPercentileIntervals <- function(bb) {
    as.data.frame(
    t(sapply(1:length(bb$t0), function(idx) {
        tmp <- boot.ci(bb, index = idx, type = "perc")
        tmp$percent[4:5]
    })))
}

test_that("Large logistic Poisson bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="logistic")
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "clr",
                                        addIntercept = FALSE)
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "clr",
                                        addIntercept = FALSE)
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bs <- runBootstrap(fitCyclops, replicates = 1999)
    expect_lt(abs(mean(bs$summary$bias)), 0.002)
})

test_that("Large conditional Poisson bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="poisson")
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cpr",
                                        addIntercept = FALSE)
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cpr",
                                        addIntercept = FALSE)
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bs <- runBootstrap(fitCyclops, replicates = 1999)
    expect_lt(abs(mean(bs$summary$bias)), 0.002)
})

test_that("Large Cox bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="survival")
    sim$outcomes$stratumId <- NULL
    sim$covariates$stratumId <- NULL
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bs <- runBootstrap(fitCyclops, replicates = 1999)
    expect_lt(abs(mean(bs$summary$bias)), 0.002)
})

test_that("Large conditional Cox bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="survival")
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bs <- runBootstrap(fitCyclops, replicates = 1999)
    expect_lt(abs(mean(bs$summary$bias)), 0.002)
})
