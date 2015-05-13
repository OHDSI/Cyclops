library("testthat")

#
# Large Bernoulli regression
#

# test_that("Large Bernoulli CCD data file read", {
# 	tolerance <- 1E-4
# 	dataPtr <- readCyclopsData(system.file("extdata/CCD_LOGISTIC_TEST_17var.txt",
# 																		 package="Cyclops"), "lr")
# 	expect_equal(getNumberOfRows(dataPtr), 22296) # Reads rows
# 	expect_equal(getNumberOfStrata(dataPtr), 22296) # Generates unique ids
# 	expect_equal(getNumberOfCovariates(dataPtr), 18) # Adds intercept
#
# 	cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"),
# 												control = createControl(noiseLevel = "silent"))
# 	expect_equal(cyclopsFit$log_likelihood * 2, -1578.046, tolerance = tolerance) # SAS fit
# 	expect_named(coef(cyclopsFit)) # Reads covariate names from file
# 	expect_named(predict(cyclopsFit)) # Reads row names from file
# })
