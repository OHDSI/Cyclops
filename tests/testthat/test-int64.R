library("testthat")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check int64 handling", {
    set.seed(123)

    data <- simulateCyclopsData(nstrata = 1,
                                nrows = 1000,
                                ncovars = 5,
                                model="logistic")

    # using int32
    cyclopsData <- convertToCyclopsData(data$outcomes,data$covariates, modelType = "lr", addIntercept = TRUE)
    fit <- fitCyclopsModel(cyclopsData)
    int32Result <- coef(fit)

    # using int64
    data$outcomes$rowId <- bit64::as.integer64(data$outcomes$rowId)
    data$covariates$rowId <- bit64::as.integer64(data$covariates$rowId)
    data$covariates$covariateId <- bit64::as.integer64(data$covariates$covariateId)

    cyclopsData64 <- convertToCyclopsData(data$outcomes,data$covariates, modelType = "lr", addIntercept = TRUE)
    fit64 <- fitCyclopsModel(cyclopsData64)
    int64Result <- coef(fit64)

    expect_equal(int64Result, int32Result)
})

test_that("Check int64 reductions", {
    set.seed(123)

    data <- simulateCyclopsData(nstrata = 2,
                                nrows = 1000,
                                ncovars = 5,
                                model="logistic")

    # using int64
    data$outcomes$rowId <- bit64::as.integer64(data$outcomes$rowId)
    data$covariates$rowId <- bit64::as.integer64(data$covariates$rowId)
    data$covariates$covariateId <- bit64::as.integer64(data$covariates$covariateId)
    data$covariates$covariateId[1] <- bit64::as.integer64("9223372036854775807")
    data$covariates$covariateId[2] <- bit64::as.integer64("9223372036854775806")

    cyclopsData64 <- convertToCyclopsData(data$outcomes,data$covariates, modelType = "lr", addIntercept = TRUE)
    covariates <- getCovariateIds(cyclopsData64)

    expect_equal(length(Cyclops:::reduce(cyclopsData64, covariates, power = 0)), 8)
    #expect_equal(ncol(Cyclops:::reduce(cyclopsData64, covariates, groupBy = "stratum", power = 0)), 8) # Causes crash
    expect_equal(nrow(summary(cyclopsData64)), 8)

    expect_equal(
        getUnivariableCorrelation(cyclopsData64)[7],
        getUnivariableCorrelation(cyclopsData64, covariates = bit64::as.integer64("9223372036854775807"))
    )

    expect_equal(
        getUnivariableSeparability(cyclopsData64)[7],
        getUnivariableSeparability(cyclopsData64, covariates = bit64::as.integer64("9223372036854775807"))
    )
})
