library("testthat")

#
# These tests are believed to be broken; they need confirmation and fixes
#

# test_that("Extract Y from data object", {})

# test_that("Extract X from data object", {})

# test_that("Check Cox model", {})

# test_that("Predict CLR model", {})

# test_that("Throw error with more than one case in CLR" ,{})

# test_that("Compute confint under CLR model", {})

# test_that("Check SCCS model via SQL", {})

# test_that("Check profile conditional posterior vs likelihood", {})

# test_that("Check default regularization variance", {})

# test_that("Check starting regularization with cross validation", {})

# test_that("Standardize covariates", {})

# test_that("Check correct dimensions in matrices in createCcdDataFrame", {})

# test_that("Fail to convergence", {})

# test_that("Use shared_ptr to handle most data", {})

# test_that("Return data summary statistics", {})

# test_that("Make intercept dense in SQL input", {})

# test_that("Make logLike object" , {
#expect_equal(logLik(ccdFit), logLik(gold))
#})

# test_that("SCCS as conditional Poisson regression likelihoods" ,{
#     expect_equal(logLik(ccdFit), logLik(gold.cp)[1]) # TODO Why are these different?
#})

# test_that("SCCS as SCCS likelihoods" ,{
#     expect_equal(logLik(ccdFit), MJS values) # TODO Why are these different?
#})

test_that("Set seed for cross-validation", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)   
    
    dataPtr <- createCcdDataFrame(counts ~ outcome + treatment, 
                                  modelType = "pr") 
    
    ccdFit <- fitCcdModel(dataPtr,
                          prior = prior("laplace",    																		
                                        exclude = c("(Intercept)")),
                          control = control(seed = 666))
    # How to check seed?
})
