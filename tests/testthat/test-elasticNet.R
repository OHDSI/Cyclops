library("testthat")

suppressWarnings(RNGversion("3.5.0"))

simulateElasticNet <- function(modelType, n = 1000, p = 60, seed = 123) {
    set.seed(seed)
    x <- matrix(rbinom(n * p, 1, 0.1), n, p)
    x[, 1:5] <- rnorm(n * 5)
    colnames(x) <- paste0("x", 1:p)
    eta <- drop(x %*% c(rnorm(10) / 2, rep(0, p - 10))) - 1
    y <- if (modelType == "lr") rbinom(n, 1, plogis(eta)) else eta + rnorm(n)
    list(x = x, y = y, data = createCyclopsData(y ~ x, modelType = modelType))
}

# The optimality conditions, from the data: the gradient g of -log likelihood plus the penalty's subgradient
kktResidual <- function(sim, fit, modelType, variance, laplaceFraction, weights = rep(1, length(sim$y))) {
    beta <- coef(fit)
    eta <- drop(beta[1] + sim$x %*% beta[-1])
    g <- if (modelType == "lr") {
        drop(crossprod(sim$x, weights * (plogis(eta) - sim$y)))
    } else {
        -2 * drop(crossprod(sim$x, weights * (sim$y - eta))) # the log likelihood of "ls" is -RSS
    }
    lambda <- sqrt(2 / variance)
    l1 <- laplaceFraction * lambda
    l2 <- (1 - laplaceFraction) * lambda
    b <- beta[-1]
    residual <- ifelse(b == 0, pmax(abs(g) - l1, 0), abs(g + l2 * b + l1 * sign(b)))
    max(residual) / max(abs(g), l1)
}

fitWith <- function(sim, prior, ...) {
    fitCyclopsModel(sim$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                    control = createControl(noiseLevel = "silent", ...))
}

test_that("createPrior checks laplaceFraction", {
    expect_error(createPrior("elasticnet", 1), "laplaceFraction")
    expect_error(createPrior("elasticnet", 1, laplaceFraction = 1.5), "between 0 and 1")
    expect_error(createPrior("elasticnet", 1, laplaceFraction = -0.1), "between 0 and 1")
    expect_error(createPrior("elasticnet", 1, laplaceFraction = NA_real_), "between 0 and 1")
    expect_error(createPrior("elasticnet", 1, laplaceFraction = c(0.5, 0.5)), "between 0 and 1")
    expect_error(createPrior("laplace", 1, laplaceFraction = 0.5), "only to elastic net")
    prior <- createPrior("elasticnet", 1, laplaceFraction = 0.5)
    expect_equal(prior$laplaceFraction, 0.5)
    expect_null(createPrior("laplace", 1)$laplaceFraction)
})

test_that("An elastic net with laplaceFraction 1 is the Laplace prior", {
    for (modelType in c("lr", "ls")) {
        sim <- simulateElasticNet(modelType)
        for (threads in c(1, 2)) {
            laplace <- fitWith(sim, createPrior("laplace", 0.05), threads = threads)
            enet <- fitWith(sim, createPrior("elasticnet", 0.05, laplaceFraction = 1), threads = threads)
            expect_identical(coef(enet), coef(laplace))
            expect_identical(logLik(enet), logLik(laplace))
        }
        laplace <- fitWith(sim, createPrior("laplace", useCrossValidation = TRUE),
                           cvType = "auto", fold = 5, seed = 1)
        enet <- fitWith(sim, createPrior("elasticnet", laplaceFraction = 1, useCrossValidation = TRUE),
                        cvType = "auto", fold = 5, seed = 1)
        expect_identical(coef(enet), coef(laplace))
        expect_identical(getHyperParameter(enet), getHyperParameter(laplace))
    }
})

test_that("An elastic net with laplaceFraction 0 is a normal prior with variance 1 / lambda", {
    for (modelType in c("lr", "ls")) {
        sim <- simulateElasticNet(modelType)
        normal <- fitWith(sim, createPrior("normal", sqrt(0.05 / 2)), tolerance = 1E-10)
        enet <- fitWith(sim, createPrior("elasticnet", 0.05, laplaceFraction = 0), tolerance = 1E-10)
        expect_equal(coef(enet), coef(normal), tolerance = 1E-8)
    }
})

test_that("Elastic net fits meet the optimality conditions", {
    for (modelType in c("lr", "ls")) {
        sim <- simulateElasticNet(modelType)
        for (laplaceFraction in c(0.2, 0.5, 0.9)) {
            for (fitter in c("default", "super")) {
                fit <- fitWith(sim, createPrior("elasticnet", 0.002, laplaceFraction = laplaceFraction),
                               convergenceType = "kkt", tolerance = 1E-8, fitter = fitter)
                expect_equal(fit$return_flag, "SUCCESS")
                expect_lt(kktResidual(sim, fit, modelType, 0.002, laplaceFraction), 1E-5)
                expect_true(any(coef(fit)[-1] == 0)) # the Laplace part sets some coefficients to zero
            }
        }
    }
})

test_that("Elastic net cross-validation with weights", {
    sim <- simulateElasticNet("lr", n = 2000)
    weights <- rep(c(1, 0, 1, 1), length.out = length(sim$y))
    prior <- createPrior("elasticnet", laplaceFraction = 0.5, useCrossValidation = TRUE)
    fit <- fitCyclopsModel(sim$data, prior = prior, weights = weights, forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(cvType = "auto", fold = 5, seed = 1, noiseLevel = "silent",
                                                   convergenceType = "kkt", tolerance = 1E-8))
    expect_equal(fit$return_flag, "SUCCESS")
    variance <- getHyperParameter(fit)
    expect_true(is.finite(variance) && variance > 0)
    expect_lt(kktResidual(sim, fit, "lr", variance, 0.5, weights), 1E-5)
})

test_that("The elastic net log density is normalized", {
    set.seed(4)
    x <- rnorm(500)
    y <- rbinom(500, 1, plogis(0.5 * x))
    data <- createCyclopsData(y ~ x, modelType = "lr")
    values <- c(-0.7, 0, 0.3, 1.2)
    for (setting in list(c(0.5, 0.9), c(8E-5, 0.9), c(0.05, 0))) {
        fit <- fitCyclopsModel(data, createPrior("elasticnet", setting[1], laplaceFraction = setting[2]),
                               warnings = FALSE)
        withPenalty <- getCyclopsProfileLogLikelihood(fit, "x", x = values, includePenalty = TRUE)$value
        withoutPenalty <- getCyclopsProfileLogLikelihood(fit, "x", x = values, includePenalty = FALSE)$value
        lambda <- sqrt(2 / setting[1])
        l1 <- setting[2] * lambda
        l2 <- (1 - setting[2]) * lambda
        density <- function(b) exp(-l1 * abs(b) - l2 * b^2 / 2)
        upper <- 40 / max(l1, sqrt(l2)) # the density is below exp(-40) beyond
        logNormalizer <- log(2 * integrate(density, 0, upper, rel.tol = 1E-12)$value)
        expect_equal(withPenalty - withoutPenalty, -l1 * abs(values) - l2 * values^2 / 2 - logNormalizer,
                     tolerance = 1E-8)
    }
})

test_that("Elastic net priors are not combined with hierarchies or neighborhoods", {
    sim <- simulateElasticNet("lr", n = 200, p = 10)
    prior <- createPrior(c("elasticnet", "elasticnet"), c(1, 1), laplaceFraction = 0.5,
                         neighborhood = list(list("xx1", c("xx2"))))
    expect_error(fitCyclopsModel(sim$data, prior = prior, warnings = FALSE), "Laplace-Laplace")
})

