library(testthat)
library(survival)

test_that("Check CV seeds", {
    cyclopsData <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum), data=infert,modelType="clr")    
    myControl <- createControl(lowerLimit = 0.01, 
                  upperLimit = 10,
                  fold = 5,
                  noiseLevel = "quiet",
                  seed = 0)
    fit1 <- fitCyclopsModel(cyclopsData,
                            prior = createPrior("laplace", 
                                                useCrossValidation = TRUE),
#                             control = createControl(lowerLimit = 0.01, 
#                                                     upperLimit = 10,
#                                                     fold = 5,
#                                                     noiseLevel = "quiet",
#                                                     seed = 0)
                            control = myControl
                            )  
    f <- function(){
        cyclopsData <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum), data=infert,modelType="clr")    
        fit <- fitCyclopsModel(cyclopsData,prior = createPrior("laplace", useCrossValidation = TRUE),
                               control = createControl(lowerLimit=0.01, upperLimit=10, fold=5, noiseLevel = "quiet",seed = 0))  
        fit
    }
    fit2 <- f()
    
    expect_equal(fit1$variance, fit2$variance)
    
    expect_equal(coef(fit1), coef(fit2))
})