library("testthat")
library("boot")

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

# test_that("Large Cox with weights and bootstrapping", {
#     set.seed(123)
#     sim <- simulateCyclopsData(nstrata=1000,
#                                ncovars=10,
#                                nrows=10000,
#                                effectSizeSd=0.5,
#                                eCovarsPerRow=2,
#                                model="survival")
#     sim$outcomes$weights <- 1/sim$outcomes$rr
#
#     # Cyclops
#     cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
#                                         covariates = sim$covariates,
#                                         modelType = "cox")
#     fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)
#
#     bs <- runBootstrap(fitCyclops, outFileName = "out.txt", treatmentId = "1", replicates = 100)
#     result <- read.csv("out.txt")
#     result
# })
