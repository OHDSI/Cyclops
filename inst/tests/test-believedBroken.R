library("testthat")

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

# test_that("Dimension checking on objects in createCyclopsModelDataFrame" ,{})

# test_that("Approximations for ties in CLR" ,{})

# test_that("Preclude SEs for regularized covariates", {})

# test_that("getSEs() throws error when all covariates are not included", {})

# test_that("Extract Y from data object", {})

# test_that("Extract X from data object", {})

# test_that("Predict CLR model", {})

# test_that("Predict SCCS model", {})

# test_that("Predict Cox model", {})

# test_that("Throw error with more than one case in CLR" ,{})

# test_that("Compute confint under CLR model", {})

# test_that("Check SCCS model via SQL", {})

# test_that("Check profile conditional posterior vs likelihood", {})

# test_that("Check default regularization variance", {})

# test_that("Check starting regularization with cross validation", {})

# test_that("Standardize covariates", {})

# test_that("Check correct dimensions in matrices in createCyclopsDataFrame", {})

# test_that("Fail to convergence", {})

# test_that("Make intercept dense in SQL input", {})

# test_that("Make logLike object" , {
#expect_equal(logLik(cyclopsFit), logLik(gold))
#})

# test_that("SCCS as conditional Poisson regression likelihoods" ,{
#     expect_equal(logLik(cyclopsFit), logLik(gold.cp)[1]) # TODO Why are these different?
#})

# test_that("SCCS as SCCS likelihoods" ,{
#     expect_equal(logLik(cyclopsFit), MJS values) # TODO Why are these different?
#})

# test_that("Reuse data object", {
#     
#     dataPtr <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
#                                       data = infert,
#                                       modelType = "clr")
#     
#     cyclopsFit <- fitCyclopsModel(dataPtr, prior = prior("none"))
#     
#     cyclopsFitR <- fitCyclopsModel(dataPtr, 
#                                    prior = prior("laplace", 1, exclude = 1))
#     
#     # Error: both cyclopsFit and cyclopsFitR share the same interface ptr
#     confint(cyclopsFit, c(1:2), includePenalty = TRUE) # Should not throw error       
# })

test_that("Set seed for cross-validation", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)   
    
    dataPtr <- createCyclopsDataFrame(counts ~ outcome + treatment, 
                                  modelType = "pr") 
    
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = prior("laplace",    																		
                                        exclude = c("(Intercept)")),
                          control = control(seed = 666))
    # How to check seed?
})
