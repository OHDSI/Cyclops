
test_that("data with no covariates (an intercept-only design) convert and fit", {
    # convertToCyclopsData() crashed R when the covariates had no rows (also in Cyclops 3.7.1)
    set.seed(12)
    outcomes <- data.frame(rowId = 1:200, y = rbinom(200, 1, 0.3))
    covariates <- data.frame(rowId = numeric(0), covariateId = numeric(0), covariateValue = numeric(0))
    data <- convertToCyclopsData(outcomes, covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    expect_equal(getNumberOfCovariates(data), 1)
    weights <- rbinom(200, 1, 0.8)
    fit <- fitCyclopsModel(data, prior = createPrior("none"), weights = weights, warnings = FALSE)
    expect_equal(fit$return_flag, "SUCCESS")
    p <- mean(outcomes$y[weights == 1])
    expect_equal(unname(coef(fit)), log(p / (1 - p)), tolerance = 1e-6)
})
