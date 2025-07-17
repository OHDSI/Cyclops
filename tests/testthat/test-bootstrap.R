library("testthat")
library("boot")
library("rsample")
library("Cyclops")
library("survival")

## helper functions

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
    result <- t(apply(bb, 2L, quantile, c(0.025, 0.975)))
    colnames(result) <- c("bpi_lower", "bpi_upper")
    as.data.frame(result)
}

bootstrap <- function(df, func, R) {

    replicates <- rsample::bootstraps(df, times = R)

    betas <- lapply(replicates$splits, func) %>%
        bind_rows()

    return(betas)
}

## End helper functions

test_that("Small Poisson bootstrap examples with and without weights", {

    skip_on_cran()

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

    skip_on_cran()

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


# boot.out <- list(
#     t = as.matrix(cb$samples),
#     sim = "ordinary",
#     stype = "i",
#     R = nrow(cb$samples),
#     t0 = coef(cf),
#     data = matrix(nrow = Cyclops::getNumberOfRows(cd)), # bb$data
#     strata = rep(1, Cyclops::getNumberOfRows(cd)) # bb$strata
# )
#
# boot.ci(boot.out, index = 1L, type = "bca")

test_that("bootstrap option for na.rm", {
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

    set.seed(123)

    cyclopsData <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2),
                                     data = test,
                                     modelType = "cox")

    fit <- fitCyclopsModel(cyclopsData, weights = rep(c(0,1), 7))
    expect_error(bootstrap <- runBootstrap(fit, replicates = 4999), "missing values and NaN")

    fit <- fitCyclopsModel(cyclopsData, weights = rep(c(0,1), 7))
    bootstrap <- runBootstrap(fit, replicates = 4999, na.rm = TRUE)
    expect_lt(nrow(bootstrap$samples), 4999)
})

test_that("Large logistic bootstrap with and without weights", {

    skip_on_cran()

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
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 1999)

    bbNoWeights <- bootstrap(
        convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            fit <- glm(y ~ V1 + V2 + V3 + V4,
                       family = "binomial", data = data)
            coef(fit)
        }, R = 1999)

    bbNoStdError <- sqrt(apply(bbNoWeights, 2L, var))
    expect_equivalent(bsNoWeights$summary$std_err, bbNoStdError, tolerance = 0.02)

    expect_equivalent(bsNoWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbNoWeights),
                      tolerance = 0.075)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "lr")
    fitCyclopsYesWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsYesWeights <- runBootstrap(fitCyclopsYesWeights, replicates = 1999)

    bbYesWeights <- bootstrap(convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            suppressWarnings(
                fit <- glm(y ~ V1 + V2 + V3 + V4, weights = data$weights,
                           family = "binomial", data = data))
            coef(fit)
        }, R = 1999)

    bbYesStdError <- sqrt(apply(bbYesWeights, 2L, var))
    expect_equivalent(bsYesWeights$summary$std_err, bbYesStdError, tolerance = 0.02)

    expect_equivalent(bsYesWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbYesWeights),
                      tolerance = 0.075)
})

test_that("Large Poisson bootstrap with and without weights", {

    skip_on_cran()

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

    bbNoWeights <- bootstrap(
        convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            fit <- glm(y ~ V1 + V2 + V3 + V4, offset = log(time),
                       family = "poisson", data = data)
            coef(fit)
        }, R = 1999)

    bbNoStdError <- sqrt(apply(bbNoWeights, 2L, var))
    expect_equivalent(bsNoWeights$summary$std_err, bbNoStdError, tolerance = 0.02)

    expect_equivalent(bsNoWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbNoWeights),
                      tolerance = 0.075)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "pr")
    fitCyclopsYesWeights <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsYesWeights <- runBootstrap(fitCyclopsYesWeights, replicates = 1999)

    bbYesWeights <- bootstrap(
        convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            fit <- glm(y ~ V1 + V2 + V3 + V4, offset = log(time),
                       family = "poisson", weights = data$weights, data = data)
            coef(fit)
        }, R = 1999)

    bbYesStdError <- sqrt(apply(bbYesWeights, 2L, var))
    expect_equivalent(bsYesWeights$summary$std_err, bbYesStdError, tolerance = 0.02)

    expect_equivalent(bsYesWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbYesWeights),
                      tolerance = 0.075)
})

test_that("Large Cox bootstrap with and without weights", {

    skip_on_cran()

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
    bsNoWeights <- runBootstrap(fitCyclopsNoWeights, replicates = 3999)

    bbNoWeights <- bootstrap(
        convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            fit <- coxph(Surv(time, y) ~ V1 + V2 + V3 + V4,
                         data = data)
            coef(fit)
        }, R = 3999)

    bbNoStdError <- sqrt(apply(bbNoWeights, 2L, var))
    expect_equivalent(bsNoWeights$summary$std_err, bbNoStdError, tolerance = 0.02)

    expect_equivalent(bsNoWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbNoWeights),
                      tolerance = 0.075)

    sim$outcomes$weights <- rep(c(0.1,0.9), nrow(sim$outcomes) / 2)

    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
    bsYesWeights <- runBootstrap(fitCyclops, replicates = 3999)

    bbYesWeights <- bootstrap(
        convertToDf(sim, 4),
        function(split) {
            data <- rsample::analysis(split)
            fit <- coxph(Surv(time, y) ~ V1 + V2 + V3 + V4,
                         weights = data$weights, data = data)
            coef(fit)
        }, R = 3999)

    bbYesStdError <- sqrt(apply(bbYesWeights, 2L, var))
    expect_equivalent(bsYesWeights$summary$std_err, bbYesStdError, tolerance = 0.02)

    expect_equivalent(bsYesWeights$summary[,c("bpi_lower", "bpi_upper")],
                      getAllPercentileIntervals(bbYesWeights),
                      tolerance = 0.075)
})
