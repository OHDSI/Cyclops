library("testthat")

context("test-halfStep.R")

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

	dataPtr <- createCyclopsData(y ~ log_bid, modelType = "lr")
	output <- capture.output(
	    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"),
	                                  control = createControl(noiseLevel = "noisy")))
	expect_equal(output[2], "Using step-size multiplier 1")

	output <- capture.output(
	    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"),
	                                  control = createControl(noiseLevel = "noisy",
	                                                          stepSizeMultiplier = 0.5)))

	expect_equal(output[2], "Using step-size multiplier 0.5")
})


