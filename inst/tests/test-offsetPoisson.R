library("testthat")
library(MASS)

test_that("Check offset in model formula", {
    
    tolerance <- 1E-4
    Insurance$logHolders <- log(Insurance$Holders)
    
    glmFit <- glm(Claims ~ District + Group + Age + offset(logHolders), 
                  data = Insurance,
                  family = poisson()) # gold standard    
            
    dataPtr <- createCyclopsDataFrame(Claims ~ District + Group + Age + logHolders,
                                  data = Insurance,
                                  modelType = "pr")	
    finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = "logHolders", offsetAlreadyOnLogScale = TRUE)    

    ## Test new number of covariates

    cyclopsFit <- fitCyclopsModel(dataPtr, 
                          prior = createPrior("none"),
                          control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFit), coef(glmFit), tolerance = tolerance)
        
    dataPtr2 <- createCyclopsDataFrame(Claims ~ District + Group + Age + offset(logHolders),
                                   data = Insurance,
                                   modelType = "pr")   

    cyclopsFit2 <- fitCyclopsModel(dataPtr2, prior = createPrior("none"))
    expect_equal(coef(cyclopsFit2), coef(glmFit), tolerance = tolerance)
    
    # Need to test now using finalize to (1) add intercept and (2) log-transform        
})
