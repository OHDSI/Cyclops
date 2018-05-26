library("testthat")
library("survival")

context("test-believedBroken.R")

#
# These tests are believed to be broken; they need confirmation and fixes
#

# test_that("Returns correct counts" ,{
#   Number of rows
#   Number of unique denominators
#   Number of strata
# })

# test_that("Returns cross-validated variance estimate" ,{})

# test_that("Data validity when loaded with (y,dx,sx,ix)" ,{})

# test_that("Dimension checking on objects in createCyclopsModelData" ,{})

# test_that("Approximations for ties in CLR" ,{})

# test_that("getSEs() throws error when all covariates are not included", {})

# test_that("Extract Y from data object", {})

# test_that("Extract X from data object", {})

# test_that("Predict CLR model", {})

# test_that("Predict SCCS model", {})

# test_that("Predict Cox model", {})

# test_that("Throw error with more than one case in CLR" ,{})

# test_that("Check SCCS model via SQL", {})

# test_that("Check default regularization variance", {})

# test_that("Check starting regularization with cross validation", {})

# test_that("Standardize covariates", {})

# test_that("Check correct dimensions in matrices in createCyclopsData", {})

# test_that("Fail to convergence", {})

# test_that("Make intercept dense in SQL input", {})

# test_that("SCCS as conditional Poisson regression likelihoods" ,{
#     expect_equal(logLik(cyclopsFit), logLik(gold.cp)) # TODO Why are these different?
#})

# test_that("SCCS as SCCS likelihoods" ,{
#     expect_equal(logLik(cyclopsFit), MJS values) # TODO Why are these different?
#})

# test_that("Reuse data object", {
#
#     dataPtr <- createCyclopsData(case ~ spontaneous + induced + strata(stratum),
#                                       data = infert,
#                                       modelType = "clr")
#
#     cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"))
#
#     cyclopsFitR <- fitCyclopsModel(dataPtr,
#                                    prior = createPrior("laplace", 1, exclude = 1))
#
#     # Error: both cyclopsFit and cyclopsFitR share the same interface ptr
#     confint(cyclopsFit, c(1:2), includePenalty = TRUE) # Should not throw error
# })

test_that("Check asymptotic variance in Cox example with failure ties and strata", {
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

    # We get the correct answer when last entry is censored
    gold <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")

    dataPtr <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test,
                                      modelType = "cox")

    cyclopsFit <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-4
    #     expect_equal(vcov(cyclopsFit), vcov(gold), tolerance = tolerance)

})
