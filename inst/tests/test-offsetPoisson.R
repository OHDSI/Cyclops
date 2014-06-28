library("testthat")
library(MASS)

test_that("Check offset in model formula", {
    
    tolerance <- 1E-4
    Insurance$logHolders <- log(Insurance$Holders)
    
    glmFit <- glm(Claims ~ District + Group + Age + offset(logHolders), 
                  data = Insurance,
                  family = poisson()) # gold standard    
            
    dataPtr <- createCcdDataFrame(Claims ~ District + Group + Age + logHolders,
                                  data = Insurance,
                                  modelType = "pr")	
    finalizeSqlCcdData(dataPtr, useOffsetCovariate = "logHolders", offsetAlreadyOnLogScale = TRUE)    

    ## Test new number of covariates

    ccdFit <- fitCcdModel(dataPtr, 
                          prior = prior("none"),
                          control = control(noiseLevel = "silent"))
    expect_equal(coef(ccdFit), coef(glmFit), tolerance = tolerance)
        
    dataPtr2 <- createCcdDataFrame(Claims ~ District + Group + Age + offset(logHolders),
                                   data = Insurance,
                                   modelType = "pr")   

    ccdFit2 <- fitCcdModel(dataPtr2, prior = prior("none"))
    expect_equal(coef(ccdFit2), coef(glmFit), tolerance = tolerance)
    
    # Need to test now using finalize to (1) add intercept and (2) log-transform        
})
