library("testthat")

#
# Large Bernoulli regression
#

test_that("Large Bernoulli CCD data file read", {
	tolerance <- 1E-4
	dataPtr <- readCcdData(system.file("extdata/CCD_LOGISTIC_TEST_17var.txt", 
																		 package="CCD"), "lr")
	expect_equal(getNumberOfRows(dataPtr), 22296) # Reads rows
	expect_equal(getNumberOfStrata(dataPtr), 22296) # Generates unique ids
	expect_equal(getNumberOfCovariates(dataPtr), 18) # Adds intercept
		
	ccdFit <- fitCcdModel(dataPtr, prior = prior("none"), 
												control = control(noiseLevel = "silent"))	
	expect_equal(ccdFit$log_likelihood * 2, -1578.046, tolerance = tolerance) # SAS fit
	expect_named(coef(ccdFit)) # Reads covariate names from file
	expect_named(predict(ccdFit)) # Reads row names from file	
})