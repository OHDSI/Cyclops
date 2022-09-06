library("testthat")

context("test-fastBarUpdate.R")

test_that("Small Bernoulli dense regression", {
	binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
	binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
	binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

	log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
	y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

	tolerance <- 1E-4

	data <- createCyclopsData(y ~ log_bid, modelType = "lr")

	expect_error(prior <- createPrior(priorType = "barupdate",
	                                  useCrossValidation = TRUE),
	             "Cannot perform")

	prior <- createPrior(priorType = "barupdate", variance = 1)
	control <- createControl(convergenceType = "onestep")

	expect_warning(fit <- fitCyclopsModel(data, prior, control),
	               "Excluding intercept")

	expect_equal(fit$iterations, 1)
	expect_equal(fit$return_flag, "SUCCESS")
})

test_that("Small Poisson dense regression; unpenalized regression", {
    y <- c(1, 6, 16, 23, 27, 39, 31, 30, 43, 51, 63, 70, 88, 97, 91, 104, 110, 113, 149, 159)
    x <- 1:20

    data <- createCyclopsData(y ~ x, modelType = "pr")
    #Run "unpenalized regression"
    tolerance <- 1E-8
    delta <- 1
    b.old <- c(0, 0)
    while(delta > tolerance) {
        prior <- createPrior(priorType = "barupdate", variance = Inf)
        control <- createControl(convergenceType = "onestep")
        fit <- fitCyclopsModel(data, prior = prior, control = control,
                               startingCoefficients = b.old, warnings = FALSE)
        delta <- max(abs((coef(fit) - b.old)))
        b.old <- coef(fit)
    }
    glm.fit <- glm(y ~ x, family = "poisson")
    expect_equal(b.old, glm.fit$coefficients, tolerance = 1E-6)
})
