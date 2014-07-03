?library("testthat")

#
# Small Poisson MLE regression
#

test_that("Small Poisson dense regression", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, data = dobson, family = poisson()) # gold standard
    
    dataPtrD <- createCcdDataFrame(counts ~ outcome + treatment, data = dobson,
                                   modelType = "pr")														
    ccdFitD <- fitCcdModel(dataPtrD, 
                           prior = prior("none"),
                           control = control(noiseLevel = "silent"))
    expect_equal(coef(ccdFitD), coef(glmFit), tolerance = tolerance)
    expect_equal(ccdFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(ccdFitD, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(ccdFitD), predict(glmFit, type = "response"), tolerance = tolerance)
    expect_equal(confint(ccdFitD, c("(Intercept)","outcome3")), confint(ccdFitD, c(1,3)))
})

test_that("Specify CI level", {
###function(object, parm, level, ...)
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard    
    
    dataPtr <- createCcdDataFrame(counts ~ outcome + treatment,
                                   modelType = "pr")    													
    ccdFit <- fitCcdModel(dataPtr, 
                           prior = prior("none"),
                           control = control(noiseLevel = "silent"))

    expect_equal(
        confint(ccdFit, c(1:3), level = 0.99)[,2:3], 
        confint(glmFit, c(1:3), level = 0.99), 
        tolerance = tolerance)
    
    
})

test_that("Small Poisson indicator regression", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard	
    
    dataPtrI <- createCcdDataFrame(counts ~ outcome, indicatorFormula =  ~ treatment, 
                                   modelType = "pr")
    
    ccdFitI <- fitCcdModel(dataPtrI, 
                           prior = prior("none"),
                           control = control(noiseLevel = "silent"))
    expect_equal(coef(ccdFitI), coef(glmFit), tolerance = tolerance)
    expect_equal(ccdFitI$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(ccdFitI, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(ccdFitI), predict(glmFit, type = "response"), tolerance = tolerance)
})

test_that("Small Poisson sparse regression", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard		
    
    dataPtrS <- createCcdDataFrame(counts ~ outcome, sparseFormula =  ~ treatment, 
                                   modelType = "pr")
    ccdFitS <- fitCcdModel(dataPtrS, 
                           prior = prior("none"),
                           control = control(noiseLevel = "silent"))
    expect_equal(coef(ccdFitS), coef(glmFit), tolerance = tolerance)
    expect_equal(ccdFitS$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(ccdFitS, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(ccdFitS), predict(glmFit, type = "response"), tolerance = tolerance)
})
