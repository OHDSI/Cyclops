library("testthat")

test_that("Find covariate by name and number", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard	
    
    dataPtr <- createCyclopsData(counts ~ outcome + treatment, 
                                  modelType = "pr")
    
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("laplace",																			
                                        exclude = c("(Intercept)", "outcome2", "outcome3")),
                          control = createControl(noiseLevel = "silent"))
    
    # Shrinkage on treatment-effects
    expect_less_than(coef(cyclopsFit)[4], coef(glmFit)[4])
    expect_less_than(coef(cyclopsFit)[5], coef(glmFit)[5])	
    
    dataPtr2 <- createCyclopsData(counts ~ outcome + treatment, 
                                   modelType = "pr")
    
    cyclopsFit2 <- fitCyclopsModel(dataPtr2,
                           prior = createPrior("laplace", 
                                         exclude = c(1:3)),
                           control = createControl(noiseLevel = "silent"))	
    # Check c(i:j) notation
    expect_equal(coef(cyclopsFit), coef(cyclopsFit2))
})

test_that("Error when covariate not found", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    dataPtr <- createCyclopsData(counts ~ outcome + treatment, 
                                  modelType = "pr")
    
    expect_error(
        fitCyclopsModel(dataPtr,
                    prior = createPrior("laplace", 
                                  exclude = c("BAD", "outcome2", "outcome3")),
                    control = createControl(noiseLevel = "silent")))
    
    dataPtr2 <- createCyclopsData(counts ~ outcome + treatment, 
                                   modelType = "pr")
    
    expect_error(
        fitCyclopsModel(dataPtr2,
                    prior = createPrior("laplace", 
                                  exclude = c(10,1:3)),
                    control = createControl(noiseLevel = "silent")))
})

test_that("Preclude profiling regularized coefficients", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    dataPtr <- createCyclopsData(counts ~ outcome + treatment, 
                                  modelType = "pr")
    
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("laplace", exclude = "(Intercept)"),
                          control = createControl(noiseLevel = "silent"))
    
    expect_true(
        !is.null(confint(cyclopsFit, "(Intercept)")) # not regularized
    )
    expect_error(
        confint(cyclopsFit, "outcome2") # regularized
    )
})

test_that("Preclude intercept regularization by default", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    dataPtr <- createCyclopsData(counts ~ outcome + treatment, 
                                  modelType = "pr")
    
#     expect_error(fitCyclopsModel(dataPtr,
#                 prior = createPrior("laplace", 0.1)))
    
    c1 <- fitCyclopsModel(dataPtr,
                      forceNewObject = TRUE,
                      prior = createPrior("laplace", 0.1, forceIntercept = TRUE))    
    
    c2 <- fitCyclopsModel(dataPtr,
                      forceNewObject = TRUE,
                      prior = createPrior("laplace", 0.1, exclude = "(Intercept)"))
    
    c3 <- fitCyclopsModel(dataPtr,
                      forceNewObject = TRUE,
                      prior = createPrior("laplace", 0.1, exclude = 1))   
    
    expect_equal(coef(c2),
                 coef(c3))
    
    expect_less_than(coef(c1)[1],  # Intercept is regularized
                     coef(c2)[1])
})

test_that("Mixture report should show full details of components", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)   
    
    dataPtr <- createCyclopsData(counts ~ outcome + treatment, 
                                  modelType = "pr")    
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("laplace",    																		
                                        exclude = c("(Intercept)", "outcome2", "outcome3")))
    expect_equal(length(strsplit(cyclopsFit$prior_info, ' ')[[1]]),
                 4) # 4 different prior assignments    
})
