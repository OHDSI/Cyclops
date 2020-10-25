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
