library("testthat")

#
# Small Bernoulli MLE regression
#

test_that("Small Bernoulli dense regression", {
	binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
	binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
	binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

	log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
	y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

	tolerance <- 1E-4
        	
	glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard
	
	dataPtrD <- createCyclopsData(y ~ log_bid, modelType = "lr")														
	cyclopsFitD <- fitCyclopsModel(dataPtrD, prior = createPrior("none"),
	                       control = createControl(noiseLevel = "silent"))
	expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
	expect_equal(cyclopsFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
	expect_equal(confint(cyclopsFitD, c(1:2))[,2:3], confint(glmFit, c(1:2)), tolerance = tolerance)
	expect_equal(predict(cyclopsFitD), predict(glmFit, type = "response"), tolerance = tolerance)
})

test_that("Add intercept via finalize", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)
    
    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
    
    tolerance <- 1E-4
    
    glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard
    
    dataPtrD <- createCyclopsData(y ~ log_bid - 1, modelType = "lr")
    finalizeSqlCyclopsData(dataPtrD, addIntercept = TRUE)
    cyclopsFitN <- fitCyclopsModel(dataPtrD, prior = createPrior("none"), forceNewObject = TRUE,
                           control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitN), coef(glmFit), tolerance = tolerance) 
    
    expect_error(finalizeSqlCyclopsData(dataPtrD, addIntercept = TRUE))
    
    dataPtrI <- createCyclopsData(y ~ log_bid, modelType = "lr")
    expect_error(finalizeSqlCyclopsData(dataPtrI, addIntercept = TRUE))
})

test_that("Small Bernoulli sparse regression", {
	binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
	binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
	binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)
	
	log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
	y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
	
	tolerance <- 1E-4
	
	glmFit <- glm(y ~ log_bid, family = binomial()) # gold standard
	
	dataPtrS <- createCyclopsData(y ~ 1, sparseFormula = ~ log_bid, modelType = "lr")														
	cyclopsFitS <- fitCyclopsModel(dataPtrS, prior = createPrior("none"),
												 control = createControl(noiseLevel = "silent"))
	expect_equal(coef(cyclopsFitS), coef(glmFit), tolerance = tolerance)
	expect_equal(cyclopsFitS$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
	expect_equal(confint(cyclopsFitS, c(1:2))[,2:3], confint(glmFit, c(1:2)), tolerance = tolerance)
	expect_equal(predict(cyclopsFitS), predict(glmFit, type = "response"), tolerance = tolerance)
})
