library("testthat")
library("survival")

#
# Small Poisson MLE regression
#

test_that("univariable correlation", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                  modelType = "pr")

    allCorrelations <- getUnivariableCorrelation(dataPtrD)
    expect_equal(length(allCorrelations), 5)

    gold <- c(NA, sapply(2:5, function(i) {
        with(dobson, cor(counts, model.matrix(~outcome + treatment)[,i]))
    }))
    expect_equivalent(gold, allCorrelations, tolerance)

    someCorrelations <- getUnivariableCorrelation(dataPtrD, c("outcome2","outcome3"))
    expect_equal(length(someCorrelations), 2)

    someCorrelations <- getUnivariableCorrelation(dataPtrD, c("outcome2","outcome3"),
                                               threshold = 0.5)
    expect_equal(length(someCorrelations), 1)

    # Try SQL data constructor
    covariates <- data.frame(stratumId = rep(infert$stratum, 2),
                             rowId = rep(1:nrow(infert), 2),
                             covariateId = rep(4:5, each = nrow(infert)),
                             covariateValue = c(infert$spontaneous, infert$induced))
    outcomes <- data.frame(stratumId = infert$stratum,
                           rowId = 1:nrow(infert),
                           y = infert$case)
    covariates <- covariates[covariates$covariateValue != 0, ]

    cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "clr",
                                        addIntercept = FALSE)

    allCorrelations <- getUnivariableCorrelation(cyclopsData, threshold = 0.3)
    expect_equal(names(allCorrelations), c("4"))
})
