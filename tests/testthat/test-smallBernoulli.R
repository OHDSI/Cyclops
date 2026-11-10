library("testthat")

context("test-smallBernoulli.R")

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

	# Try with Zhang-Oles criterion
	zoFit <- fitCyclopsModel(dataPtrD, prior = createPrior("none"), forceNewObject = TRUE,
	                         control = createControl(convergenceType = "zhang", noiseLevel = "noisy"))
	expect_equal(coef(cyclopsFitD), coef(zoFit), tolerance = tolerance)

	# Try with KKT swindle
	gold <- fitCyclopsModel(dataPtrD, prior = createPrior("laplace"), forceNewObject = TRUE, warnings = FALSE)

	kktFit <- fitCyclopsModel(dataPtrD, prior = createPrior("laplace"), forceNewObject = TRUE, warnings = FALSE,
	                          control = createControl(useKKTSwindle = TRUE, noiseLevel = "quiet"))
	expect_equal(coef(gold), coef(kktFit))

	# Test old logging routines
	file <- tempfile()
	Cyclops:::.cyclopsLogResults(cyclopsFitD$interface, file, TRUE)
	result <- read.csv(file)
	expect_equivalent(result$estimate, coef(cyclopsFitD), tolerance = tolerance)

	cap <- capture.output(print(dataPtrD))
	expect_false(is.null(cap))
})

test_that("Old BLR format", {
    data <- readCyclopsData(system.file("extdata/infert_ccd.txt", package = "Cyclops"), "clr")
    expect_equal(getNumberOfRows(data), 248)
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
    expect_equal(summary(dataPtrD)$type[1], "intercept") # stored as convertToCyclopsData() stores it, not as dense ones
    cyclopsFitN <- fitCyclopsModel(dataPtrD, prior = createPrior("none"), forceNewObject = TRUE,
                           control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitN), coef(glmFit), tolerance = tolerance)

    expect_error(finalizeSqlCyclopsData(dataPtrD, addIntercept = TRUE))

    dataPtrI <- createCyclopsData(y ~ log_bid, modelType = "lr")
    expect_error(finalizeSqlCyclopsData(dataPtrI, addIntercept = TRUE))
})

test_that("An intercept added at finalize is fit as convertToCyclopsData's intercept", {
    # Both paths must give the column fitter = "super" recognises for its adaptive intercept steps; a dense column
    # of ones fit the same optimum but without them (39 coordinate-descent iterations instead of 21 here)
    set.seed(11)
    n <- 5000
    p <- 400
    rows <- lapply(1:p, function(j) sort(sample.int(n, rbinom(1, n, 10^runif(1, -2.5, -0.7)))))
    cov <- data.frame(rowId = unlist(rows), covariateId = rep(1:p, lengths(rows)), covariateValue = 1)
    cov <- cov[order(cov$rowId, cov$covariateId), ]
    out <- data.frame(rowId = 1:n, y = rbinom(n, 1, plogis(-2.5 + 0.3 * tabulate(cov$rowId[cov$covariateId <= 40], n))))
    converted <- convertToCyclopsData(out, cov, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    finalized <- createSqlCyclopsData(modelType = "lr")
    invisible(appendSqlCyclopsData(finalized, oStratumId = out$rowId, oRowId = out$rowId, oY = out$y, oTime = rep(0, n),
                                   cRowId = cov$rowId, cCovariateId = cov$covariateId, cCovariateValue = cov$covariateValue))
    finalizeSqlCyclopsData(finalized, addIntercept = TRUE)
    expect_equal(summary(finalized)$type[1], "intercept")
    control <- createControl(fitter = "super", algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-6,
                             noiseLevel = "silent")
    fit <- function(data) fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.05), control = control,
                                          forceNewObject = TRUE, warnings = FALSE)
    a <- fit(converted)
    b <- fit(finalized)
    expect_equal(b$iterations, a$iterations)
    expect_equal(coef(b)[names(coef(a))], coef(a), tolerance = 1e-6)

    # Asking for the intercept to be dense (as the documentation of makeCovariatesDense used to suggest) keeps it as
    # it is; converting an intercept column crashed R before
    again <- convertToCyclopsData(out, cov, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    finalizeSqlCyclopsData(again, makeCovariatesDense = 0)
    expect_equal(summary(again)$type[1], "intercept")
    expect_equal(coef(fit(again)), coef(a))
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
