library("testthat")

simulateL1 <- function(n = 3000, p = 200, seed = 11) {
    set.seed(seed)
    X <- matrix(rbinom(n * p, 1, runif(p, 0.01, 0.3)[col(matrix(0, n, p))]), n, p)
    X[, 1:50] <- X[, 1:50] * runif(n * 50, 0.5, 2)
    y <- rbinom(n, 1, plogis(-1.5 + drop(X[, 1:20] %*% rnorm(20, 0, 0.6))))
    nz <- which(X != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
    covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
    convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                         quiet = TRUE)
}

test_that("L1 logistic fits are certified by their duality gap, and the bound holds", {
    data <- simulateL1()
    variance <- 0.5
    lambda <- sqrt(2 / variance)
    objective <- function(fit) {
        b <- coef(fit)
        -fit$log_likelihood + lambda * sum(abs(b[-1]))
    }
    reference <- fitCyclopsModel(data, prior = createPrior("laplace", variance = variance, exclude = 0),
                                 control = createControl(algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-10,
                                                         maxIterations = 100000, noiseLevel = "silent"),
                                 forceNewObject = TRUE, warnings = FALSE)
    for (algorithm in c("newton", "ccd", "auto")) {
        fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = variance, exclude = 0),
                               control = createControl(algorithm = algorithm, convergenceType = "gap",
                                                       noiseLevel = "silent"),
                               forceNewObject = TRUE, warnings = FALSE)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$convergence$certificate, "dual")
        expect_true(fit$convergence$certified)
        expect_lte(fit$convergence$gapEstimate, fit$convergence$tauEff)
        # the duality gap bounds the distance to the optimum
        expect_lte(objective(fit) - objective(reference), fit$convergence$gapEstimate + 1e-9)
    }
})

test_that("fits the duality gap does not apply to keep their earlier stops", {
    data <- simulateL1()
    # a second free unpenalized column, and an elastic net (a quadratic part)
    for (prior in list(createPrior("laplace", variance = 0.5, exclude = c(0, 1)),
                       createPrior("elasticnet", variance = 0.5, laplaceFraction = 0.5, exclude = 0))) {
        fit <- fitCyclopsModel(data, prior = prior, control = createControl(convergenceType = "gap",
                                                                            noiseLevel = "silent"),
                               forceNewObject = TRUE, warnings = FALSE)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_false(identical(fit$convergence$certificate, "dual"))
        expect_equal(fit$convergence$dualChecks, 0)
    }
})

test_that("a gap below what the duality gap can certify stops uncertified, not at the iteration limit", {
    # at the optimum rounding leaves the dual point short of it (about 40 eps |f|): a gap of 1e-13 cannot be
    # certified, and the fit stops on its estimate as before once the duality gap no longer halves
    data <- simulateL1()
    for (algorithm in c("newton", "ccd")) {
        fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.5, exclude = 0),
                               control = createControl(algorithm = algorithm, convergenceType = "gap",
                                                       tolerance = 1e-13, noiseLevel = "silent"),
                               forceNewObject = TRUE, warnings = FALSE)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$convergence$certificate, "none")
        expect_lte(fit$convergence$dualChecks, 4)
        expect_lt(fit$convergence$gapEstimate, 1e-8) # the duality gap reached, an upper bound
    }
})

test_that("the duality gap allows for fixed coefficients and fits without an intercept", {
    # a fixed coefficient is part of the linear predictor's fixed part o: the dual gains -theta' o (without it the
    # "gap" could be negative); without an intercept there is no refit and no constraint sum(theta) = 0
    set.seed(5)
    n <- 2000
    X <- matrix(rbinom(n * 40, 1, 0.2), n, 40)
    y <- rbinom(n, 1, plogis(-1 + 1.5 * X[, 1] + drop(X[, 2:6] %*% rep(0.4, 5))))
    nz <- which(X != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = 1)
    covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
    variance <- 0.5
    lambda <- sqrt(2 / variance)
    for (intercept in c(TRUE, FALSE)) {
        data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr",
                                     addIntercept = intercept, quiet = TRUE)
        p <- getNumberOfCovariates(data)
        fixed <- rep(FALSE, p)
        fixed[if (intercept) 2 else 1] <- TRUE # covariate 1, fixed at 2
        start <- rep(0, p)
        start[fixed] <- 2
        prior <- createPrior("laplace", variance = variance, exclude = if (intercept) 0 else c())
        fitWith <- function(control) fitCyclopsModel(data, prior = prior, control = control, fixedCoefficients = fixed,
                                                     startingCoefficients = start, forceNewObject = TRUE,
                                                     warnings = FALSE)
        fit <- fitWith(createControl(algorithm = "auto", convergenceType = "gap", noiseLevel = "silent"))
        reference <- fitWith(createControl(algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-10,
                                           maxIterations = 100000, noiseLevel = "silent"))
        penalized <- !fixed & (if (intercept) seq_len(p) > 1 else TRUE)
        objective <- function(f) -f$log_likelihood + lambda * sum(abs(coef(f)[penalized]))
        expect_equal(unname(coef(fit)[fixed]), 2)
        expect_equal(fit$convergence$certificate, "dual")
        expect_gte(fit$convergence$gapEstimate, 0)
        expect_lte(objective(fit) - objective(reference), fit$convergence$gapEstimate + 1e-9)
    }
})

test_that("outcomes other than 0 and 1 get no duality-gap certificate", {
    set.seed(6)
    n <- 1000
    X <- matrix(rbinom(n * 20, 1, 0.3), n, 20)
    y <- ifelse(rbinom(n, 1, 0.4) == 1, 0.8, 0)
    nz <- which(X != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = 1)
    covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                                 quiet = TRUE)
    fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.5, exclude = 0),
                           control = createControl(convergenceType = "gap", noiseLevel = "silent"),
                           forceNewObject = TRUE, warnings = FALSE)
    expect_equal(fit$return_flag, "SUCCESS")
    expect_false(identical(fit$convergence$certificate, "dual"))
})
