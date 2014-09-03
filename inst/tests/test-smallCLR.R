library("testthat")
library("survival")

#
# Small conditional logistic regression
#

test_that("Small conditional logistic regression", {
    
    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)  
    
    dataPtr <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
                                  data = infert,
                                  modelType = "clr")
    
    cyclopsFit <- fitCyclopsModel(dataPtr, prior = prior("none"))
    
	tolerance <- 1E-4

	expect_equal(coef(cyclopsFit), coef(gold), tolerance = tolerance)
	expect_equal(cyclopsFit$log_likelihood, logLik(gold)[[1]], tolerance = tolerance)
    
    expect_equal(vcov(cyclopsFit), vcov(gold), tolerance = tolerance)
    
    expect_equal(aconfint(cyclopsFit), confint(gold), tolerance = tolerance)

# This is broken:
# 	expect_equal(predict(cyclopsFit), predict(gold, type = "risk"), tolerance = tolerance)
})
