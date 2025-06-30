library("testthat")
library("boot")
library("Cyclops")
library("survival")

test_that("Small conditional logistic regression bootstrap", {
    set.seed(123)
    dataPtr <- createCyclopsData(case ~ spontaneous + induced + strata(stratum),
                                 data = infert,
                                 modelType = "clr")

    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"))

    bs <- runBootstrap(cyclopsFit, replicates = 4999)
    expect_lt(abs(mean(bs$summary$bias)), 0.001)
    # Error: abs(mean(bs$summary$bias)) is not strictly less than 0.001. Difference: 0.0808
})

test_that("Small Poisson bootstrap examples with and without weights", {

    set.seed(123)

    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome2 = c(0,1,0,0,1,0,0,1,0),
        outcome3 = c(0,0,1,0,0,1,0,0,1),
        treatment2 = c(0,0,0,1,1,1,0,0,0),
        treatment3 = c(0,0,0,0,0,0,1,1,1),
        weights = c(0.17,0.03,0.93,0.64,0.94,0.19,0.12,0.58,0.01)
    )

    execute <- function(d, f) {
        cd <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3, data = d[f,], modelType = "pr")
        fit <- fitCyclopsModel(cd, prior = createPrior("normal", exclude = "(Intercept)"))
        coef(fit)
    }

    bb <- boot(dobson, execute, R = 4999)
    bBias <- apply(bb$t, 2L, mean) - bb$t0

    cd <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3, data = dobson, modelType = "pr")
    cf <- fitCyclopsModel(cd, prior = createPrior("normal", exclude = "(Intercept)"),
                          control = createControl(seed = 123))
    cb <- runBootstrap(cf, replicates = 4999)

    expect_equivalent(bBias, cb$summary$bias, tolerance = 1E-2) # Check unweighted bias

    wexecute <- function(d, f) {
        cd <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3, data = d[f,], modelType = "pr")
        fit <- fitCyclopsModel(cd, weights = d[f,]$weights,
                               prior = createPrior("normal", exclude = "(Intercept)"))
        coef(fit)
    }

    wbb <- boot(dobson, wexecute, R = 4999)
    wbbBias <- apply(wbb$t, 2L, mean) - wbb$t0

    wcd <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3, data = dobson, modelType = "pr")
    wcf <- fitCyclopsModel(wcd, weights = dobson$weights,
                          prior = createPrior("normal", exclude = "(Intercept)"),
                          control = createControl(seed = 123))
    wcb <- runBootstrap(wcf, replicates = 4999)

    expect_equivalent(wbbBias, wcb$summary$bias, tolerance = 1E-2) # Check weighted bias

    expect_gt(mean(abs(cb$summary$bias - wcb$summary$bias)), 0.01) # Difference estimates with and without weights
})

test_that("Small Poisson bootstrap examples with an offset", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome2 = c(0,1,0,0,1,0,0,1,0),
        outcome3 = c(0,0,1,0,0,1,0,0,1),
        treatment2 = c(0,0,0,1,1,1,0,0,0),
        treatment3 = c(0,0,0,0,0,0,1,1,1),
        offset = c(1,1,1,1,1,1,1,1,1)
    )

    set.seed(123)
    cdo <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3 + offset(offset),
                            data = dobson, modelType = "pr")
    cfo <- fitCyclopsModel(cdo, prior = createPrior("normal", exclude = "(Intercept)"),
                          control = createControl(seed = 123))
    cbo <- runBootstrap(cfo, replicates = 4999)

    set.seed(123)
    cdn <- createCyclopsData(counts ~ outcome2 + outcome3 + treatment2 + treatment3,
                             data = dobson, modelType = "pr")
    cfn <- fitCyclopsModel(cdn, prior = createPrior("normal", exclude = "(Intercept)"),
                           control = createControl(seed = 123))
    cbn <- runBootstrap(cfn, replicates = 4999)

    expect_equal(cbo$summary$original + c(1,0,0,0,0), cbn$summary$original, tolerance = 1E-4)

    expect_equal(cbo$summary$bpi_lower + c(1,0,0,0,0), cbn$summary$bpi_lower, tolerance = 1E-4)
})

test_that("Cox example bootstrap with failure ties, strata and data weights", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")
    test <- rbind(data.frame(test, index = 1:7), data.frame(test, index = 1:7))

    cyclopsData <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2),
                                     data = test,
                                     modelType = "cox")

    fit <- fitCyclopsModel(cyclopsData, weights = rep(c(0,1), 7))
    bootstrap <- runBootstrap(fit, replicates = 4999)
    # Error in quantile.default(X[[i]], ...) :
    #     missing values and NaN's not allowed if 'na.rm' is FALSE
})

test_that("Large logistic bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="logistic")
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "lr")
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 4999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)
    # Error: abs(mean(bsNoWeights$summary$bias)) is not strictly less than 0.002. Difference: 0.000326

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "lr")
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bs <- runBootstrap(fitCyclops, replicates = 4999)
    expect_lt(abs(mean(bs$summary$bias)), 0.002)
    # Error: abs(mean(bsNoWeights$summary$bias)) is not strictly less than 0.002. Difference: 0.000326
})

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

test_that("Large Poisson bootstrap with and without weights", {
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=100,
                               ncovars=4,
                               nrows=1000,
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="poisson")
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "pr")
    fitCyclopsNoWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)
    expect_lt(abs(mean(bsNoWeights$summary$bias)), 0.002)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "pr")
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
