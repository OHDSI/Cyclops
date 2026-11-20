library("testthat")

suppressWarnings(RNGversion("3.5.0"))

# Logistic data whose unconstrained L1 fit has negative coefficients
simulateBounded <- function(n = 500, p = 12, seed = 2) {
    set.seed(seed)
    x <- matrix(rnorm(n * p), n, p)
    x[, 3] <- x[, 2] + 0.2 * rnorm(n)
    beta <- c(1, -1, 0.8, -0.5, 0.5, -1.5, rep(0, p - 6))
    y <- rbinom(n, 1, plogis(0.2 + x %*% beta))
    list(x = x, y = y, n = n, p = p, data = createCyclopsData(y ~ x, modelType = "lr"))
}

boundedFit <- function(sim, prior, ...) {
    fitCyclopsModel(sim$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                    control = createControl(noiseLevel = "silent", ...))
}

# The one-sided optimality conditions, from the data: at its bound a coefficient may have a gradient of the penalized
# objective pointing upward; elsewhere the usual conditions
boundedKktResidual <- function(sim, fit, variance, lowerBound, laplaceFraction = 1) {
    beta <- coef(fit)
    eta <- drop(beta[1] + sim$x %*% beta[-1])
    g <- drop(crossprod(sim$x, plogis(eta) - sim$y))
    lambda <- sqrt(2 / variance)
    l1 <- laplaceFraction * lambda
    l2 <- (1 - laplaceFraction) * lambda
    b <- beta[-1]
    lb <- rep_len(lowerBound, length(b))
    right <- g + l2 * b + ifelse(b >= 0, l1, -l1)
    residual <- ifelse(b <= lb, pmax(-right, 0),
                       ifelse(b == 0, pmax(abs(g) - l1, 0), abs(g + l2 * b + l1 * sign(b))))
    c(residual = max(residual) / max(abs(g), l1), intercept = abs(sum(plogis(eta) - sim$y)))
}

test_that("a lower bound of 0 holds the coefficients that would be negative at 0", {
    sim <- simulateBounded()
    variance <- 2 / (sim$n * 0.005)^2
    free <- coef(boundedFit(sim, createPrior("laplace", variance = variance), tolerance = 1e-10))
    expect_true(sum(free[-1] < 0) >= 3)

    for (algorithm in c("auto", "ccd")) {
        fit <- boundedFit(sim, createPrior("laplace", variance = variance, lowerBound = 0),
                          algorithm = algorithm, convergenceType = "gap", tolerance = 1e-10)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$convergence$certificate, "dual")
        beta <- coef(fit)
        expect_true(all(beta[-1] >= 0))
        expect_true(all(beta[-1][free[-1] < 0] == 0))
        kkt <- boundedKktResidual(sim, fit, variance, 0)
        expect_lt(kkt[["residual"]], 1e-6)
        expect_lt(kkt[["intercept"]], 1e-6)
    }
})

test_that("lower bounds agree across the coordinate-descent paths and thread counts", {
    sim <- simulateBounded(seed = 5)
    variance <- 2 / (sim$n * 0.004)^2
    prior <- createPrior("laplace", variance = variance, lowerBound = 0)
    reference <- coef(boundedFit(sim, prior, algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-10))
    expect_equal(coef(boundedFit(sim, prior, tolerance = 1e-10)), reference, tolerance = 1e-7)
    expect_equal(coef(boundedFit(sim, prior, fitter = "super", convergenceType = "kkt", tolerance = 1e-10)),
                 reference, tolerance = 1e-7)
    expect_equal(coef(boundedFit(sim, prior, algorithm = "ccd", convergenceType = "gradient", tolerance = 1e-12)),
                 reference, tolerance = 1e-7)
    expect_equal(coef(boundedFit(sim, prior, algorithm = "mm", convergenceType = "lange", tolerance = 1e-13,
                                 maxIterations = 100000)),
                 reference, tolerance = 1e-5)
    expect_equal(coef(boundedFit(sim, prior, algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-10, threads = 3)),
                 reference, tolerance = 1e-10)
})

test_that("negative and per-coefficient lower bounds, elastic net and normal priors", {
    sim <- simulateBounded(p = 10, seed = 7)
    variance <- 2 / (sim$n * 0.005)^2
    lowerBound <- c(-Inf, 0, -0.3, 0, -0.2, 0, -Inf, -0.1, -0.1, -0.1, -0.1) # one per covariate, the intercept first
    fit <- boundedFit(sim, createPrior("laplace", variance = variance, lowerBound = lowerBound), tolerance = 1e-10)
    beta <- coef(fit)
    expect_true(all(beta >= lowerBound))
    expect_true(any(beta[-1] == lowerBound[-1] & lowerBound[-1] < 0))
    expect_lt(boundedKktResidual(sim, fit, variance, lowerBound[-1])[["residual"]], 1e-6)

    for (laplaceFraction in c(0.5, 0)) {
        fit <- boundedFit(sim, createPrior("elasticnet", variance = variance, laplaceFraction = laplaceFraction,
                                           lowerBound = 0), tolerance = 1e-10)
        expect_true(all(coef(fit)[-1] >= 0))
        expect_lt(boundedKktResidual(sim, fit, variance, 0, laplaceFraction)[["residual"]], 1e-6)
    }
    # a normal prior with variance 1 / lambda is the elastic net with laplaceFraction 0
    normal <- boundedFit(sim, createPrior("normal", variance = 1 / sqrt(2 / variance), lowerBound = 0),
                         tolerance = 1e-10)
    expect_equal(coef(normal), coef(fit), tolerance = 1e-7)
})

test_that("a start below a bound starts at it, and the next fit without bounds is unconstrained", {
    sim <- simulateBounded(seed = 11)
    variance <- 2 / (sim$n * 0.005)^2
    prior <- createPrior("laplace", variance = variance, lowerBound = 0)
    reference <- coef(boundedFit(sim, prior, tolerance = 1e-10))
    fit <- fitCyclopsModel(sim$data, prior = prior, warnings = FALSE,
                           startingCoefficients = rep(-1, sim$p + 1),
                           control = createControl(noiseLevel = "silent", tolerance = 1e-10))
    expect_equal(coef(fit), reference, tolerance = 1e-7)

    free <- fitCyclopsModel(sim$data, prior = createPrior("laplace", variance = variance), warnings = FALSE,
                            control = createControl(noiseLevel = "silent", tolerance = 1e-10))
    expect_true(any(coef(free)[-1] < 0))
})

test_that("lower bounds on a sparse indicator design (active set, zero certificates)", {
    set.seed(3)
    n <- 2000
    p <- 300
    rows <- rep(1:n, each = 6)
    columns <- sample.int(p, length(rows), replace = TRUE)
    keep <- !duplicated(cbind(rows, columns))
    covariates <- data.frame(rowId = rows[keep], covariateId = columns[keep], covariateValue = 1)
    effect <- c(rnorm(30), rep(0, p - 30))
    eta <- -1 + tapply(effect[covariates$covariateId], covariates$rowId, sum)
    outcomes <- data.frame(rowId = 1:n, y = rbinom(n, 1, plogis(eta)))
    data <- convertToCyclopsData(outcomes, covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    variance <- 2 / (n * 0.002)^2
    prior <- createPrior("laplace", variance = variance, lowerBound = 0)

    reference <- coef(fitCyclopsModel(data, prior = prior, warnings = FALSE,
                                      control = createControl(noiseLevel = "silent", algorithm = "ccd",
                                                              convergenceType = "kkt", fitter = "legacy",
                                                              tolerance = 1e-10)))
    expect_true(all(reference[-1] >= 0))
    for (control in list(createControl(noiseLevel = "silent", tolerance = 1e-9),
                         createControl(noiseLevel = "silent", fitter = "super", threads = 2))) {
        fit <- fitCyclopsModel(data, prior = prior, control = control, warnings = FALSE)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(coef(fit), reference, tolerance = 1e-6)
    }
})

test_that("the duality gap under lower bounds bounds the distance to the optimum", {
    sim <- simulateBounded(p = 20, seed = 4)
    variance <- 2 / (sim$n * 0.004)^2
    lambda <- sqrt(2 / variance)
    lowerBound <- c(-Inf, rep(c(0, -0.1), 10))
    objective <- function(beta) {
        eta <- drop(beta[1] + sim$x %*% beta[-1])
        sum(log1p(exp(eta)) - sim$y * eta) + lambda * sum(abs(beta[-1]))
    }
    prior <- createPrior("laplace", variance = variance, lowerBound = lowerBound)
    optimum <- objective(coef(boundedFit(sim, prior, algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-12)))
    for (iterations in c(2, 4, 8)) {
        fit <- boundedFit(sim, prior, algorithm = "ccd", convergenceType = "gap", maxIterations = iterations,
                          tolerance = 1e-12)
        expect_gte(fit$convergence$gapEstimate, objective(coef(fit, ignoreConvergence = TRUE)) - optimum - 1e-9)
    }
})

test_that("glmnet lower.limits agree (standardize = FALSE)", {
    skip_if_not_installed("glmnet")
    sim <- simulateBounded(seed = 2)
    lambda <- 0.005 # glmnet's lambda: Cyclops' variance is 2 / (n lambda)^2 for "lr"
    variance <- 2 / (sim$n * lambda)^2
    lowerBound <- c(0, -0.3, 0, -0.2, 0, -Inf, rep(-0.1, sim$p - 6))
    control <- list(thresh = 1e-14, maxit = 1e7)
    for (bounds in list(0, lowerBound)) {
        reference <- as.numeric(coef(glmnet::glmnet(sim$x, sim$y, family = "binomial", lambda = lambda,
                                                    lower.limits = bounds, standardize = FALSE, control = control)))
        free <- as.numeric(coef(glmnet::glmnet(sim$x, sim$y, family = "binomial", lambda = lambda,
                                               standardize = FALSE, control = control)))
        expect_true(any(free[-1] < rep_len(bounds, sim$p)))
        prior <- createPrior("laplace", variance = variance,
                             lowerBound = if (length(bounds) == 1) bounds else c(-Inf, bounds))
        for (algorithm in c("auto", "ccd")) {
            fit <- boundedFit(sim, prior, algorithm = algorithm, tolerance = 1e-10)
            expect_equal(as.numeric(coef(fit)), reference, tolerance = 1e-6)
        }
    }
})

test_that("lower bounds are validated", {
    sim <- simulateBounded(seed = 2)
    expect_error(createPrior("laplace", lowerBound = 0.5), "0 or negative")
    expect_error(createPrior("laplace", lowerBound = NA), "0 or negative")
    expect_error(boundedFit(sim, createPrior("laplace", lowerBound = c(0, 0))), "one bound per covariate")
    expect_error(boundedFit(sim, createPrior("laplace", lowerBound = 0), algorithm = "newton"),
                 "Newton steps are not available with lower bounds")
    expect_error(boundedFit(sim, createPrior("laplace", lowerBound = 0), useKKTSwindle = TRUE),
                 "kkt swindle")
})
