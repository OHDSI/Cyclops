library("testthat")

# Outcomes with consecutive row IDs keep no row map or labels (one ID per row by arithmetic); fits, names and errors
# are those of any other row IDs
test_that("consecutive row IDs fit, name and check rows as other row IDs", {
    set.seed(8)
    outcomes <- data.frame(rowId = 5:12, y = c(1, 0, 1, 1, 0, 0, 1, 0))
    covariates <- data.frame(rowId = c(5:12, 6, 9, 11), covariateId = c(rep(1, 8), 2, 2, 2),
                             covariateValue = c(rnorm(8), 1, 1, 1))
    weights <- c(1, 2, 1, 1, 3, 1, 1, 2)
    fitOf <- function(outcomes, covariates) {
        fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "lr", quiet = TRUE),
                        prior = createPrior("laplace", variance = 1, exclude = 0), weights = weights)
    }
    consecutive <- fitOf(outcomes, covariates)
    gap <- outcomes
    gap$rowId[8] <- 40 # not consecutive: the row map
    covariatesGap <- covariates
    covariatesGap$rowId[covariatesGap$rowId == 12] <- 40
    expect_identical(coef(consecutive), coef(fitOf(gap, covariatesGap)))
    expect_equal(names(predict(consecutive)), as.character(5:12))
    # a covariate row outside the outcomes is an error without checkRowIds, and dropped with it
    stray <- rbind(covariates, data.frame(rowId = 13, covariateId = 1, covariateValue = 1))
    expect_error(convertToCyclopsData(outcomes, stray, modelType = "lr", checkRowIds = FALSE, quiet = TRUE),
                 "not among the outcome rowIds")
    expect_identical(coef(fitOf(outcomes, stray)), coef(consecutive))
    # negative IDs
    shifted <- transform(outcomes, rowId = rowId - 9)
    expect_identical(coef(fitOf(shifted, transform(covariates, rowId = rowId - 9))), coef(consecutive))
})
