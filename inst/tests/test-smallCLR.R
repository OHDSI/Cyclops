library("testthat")
library("survival")

#
# Small conditional logistic regression
#

test_that("Small conditional logistic regression", {
    
    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)  
    
    dataPtr <- createCcdDataFrame(case ~ spontaneous + induced + strata(stratum),
                                  data = infert,
                                  modelType = "clr")
    
    ccdFit <- fitCcdModel(dataPtr, prior = prior("none"))
    
	tolerance <- 1E-4

	expect_equal(coef(ccdFit), coef(gold), tolerance = tolerance)
	expect_equal(ccdFit$log_likelihood, logLik(gold)[[1]], tolerance = tolerance)
    # The following are broken:
#	expect_equal(confint(ccdFit, c(1:2))[,2:3], confint(gold, c(1:2)), tolerance = tolerance)
#	expect_equal(predict(ccdFit), predict(gold, type = "expected"), tolerance = tolerance)
})

test_that("Add intercept via finalize", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)
    
    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
    
    tolerance <- 1E-4
    
    glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard
    
    dataPtrD <- createCcdDataFrame(y ~ log_bid - 1, modelType = "lr")
    finalizeSqlCcdData(dataPtrD, addIntercept = TRUE)
    ccdFitN <- fitCcdModel(dataPtrD, prior = prior("none"), forceColdStart = TRUE,
                           control = control(noiseLevel = "silent"))
    expect_equal(coef(ccdFitN), coef(glmFit), tolerance = tolerance) 
    
    expect_error(finalizeSqlCcdData(dataPtrD, addIntercept = TRUE))
    
    dataPtrI <- createCcdDataFrame(y ~ log_bid, modelType = "lr")
    expect_error(finalizeSqlCcdData(dataPtrI, addIntercept = TRUE))
})

test_that("Small Bernoulli sparse regression", {
	binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
	binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
	binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)
	
	log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
	y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
	
	tolerance <- 1E-4
	
	glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard
	
	dataPtrS <- createCcdDataFrame(y ~ 1, sparseFormula = ~ log_bid, modelType = "lr")														
	ccdFitS <- fitCcdModel(dataPtrS, prior = prior("none"),
												 control = control(noiseLevel = "silent"))
	expect_equal(coef(ccdFitS), coef(glmFit), tolerance = tolerance)
	expect_equal(ccdFitS$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
	expect_equal(confint(ccdFitS, c(1:2))[,2:3], confint(glmFit, c(1:2)), tolerance = tolerance)
	expect_equal(predict(ccdFitS), predict(glmFit, type = "response"), tolerance = tolerance)
})
