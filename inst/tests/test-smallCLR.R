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
    
# confint(gold) uses +/- 1.96 * sd, which yields different estimates
#     expect_equal(confint(cyclopsFit, c(1:2))[,2:3], confint(gold, c(1:2)), tolerance = tolerance)

# This is broken:
# 	expect_equal(predict(cyclopsFit), predict(gold, type = "risk"), tolerance = tolerance)
})
