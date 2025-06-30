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