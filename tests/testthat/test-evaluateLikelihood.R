library("testthat")
library("survival")

# evaluateLikelihood() against finite differences of itself: the gradient of its log-likelihood, and its information
# as minus the derivative of its gradient, for every model type, with and without weights
set.seed(9)
n <- 300
X <- matrix(rnorm(n * 3), n, 3)
eta <- as.vector(X %*% c(0.5, -0.4, 0.3))
strata <- rep(1:60, each = 5)
designs <- list(
    lr = function() createCyclopsData(rbinom(n, 1, plogis(eta)) ~ X, modelType = "lr"),
    pr = function() createCyclopsData(rpois(n, exp(eta)) ~ X, modelType = "pr"),
    ls = function() createCyclopsData(eta + rnorm(n) ~ X, modelType = "ls"),
    cox = function() createCyclopsData(Surv(rexp(n, exp(eta)), rbinom(n, 1, 0.7)) ~ X, modelType = "cox"),
    scox = function() createCyclopsData(Surv(rexp(n, exp(eta)), rbinom(n, 1, 0.7)) ~ X + strata(strata),
                                        modelType = "cox"),
    clr = function() createCyclopsData(rbinom(n, 1, plogis(eta)) ~ X + strata(strata), modelType = "clr"))

finiteDifferences <- function(data, beta, weights) {
    k <- length(beta)
    h <- 1e-5
    at <- function(b) evaluateLikelihood(data, b, weights = weights)
    gradient <- sapply(seq_len(k), function(j) {
        e <- replace(numeric(k), j, h)
        (at(beta + e)$logLikelihood - at(beta - e)$logLikelihood) / (2 * h)
    })
    hessian <- sapply(seq_len(k), function(j) {
        e <- replace(numeric(k), j, h)
        (at(beta + e)$gradient - at(beta - e)$gradient) / (2 * h)
    })
    list(gradient = gradient, information = -(hessian + t(hessian)) / 2)
}

test_that("evaluateLikelihood matches finite differences for every model, with and without weights", {
    for (model in names(designs)) {
        for (weights in list(NULL, as.numeric(seq_len(n) %% 3 != 0))) {
            data <- designs[[model]]() # never fitted
            names <- Cyclops:::.coefficientNames(data)
            beta <- rnorm(length(names), 0, 0.3)
            value <- evaluateLikelihood(data, beta, weights = weights, information = names, informationDiagonal = names)
            reference <- finiteDifferences(data, beta, weights)
            expect_equal(unname(value$gradient), reference$gradient, tolerance = 1e-6, info = model)
            expect_equal(unname(value$information), unname(reference$information), tolerance = 1e-6, info = model)
            expect_equal(value$informationDiagonal, diag(value$information), info = model)
            expect_equal(names(value$gradient), names)
            expect_equal(dimnames(value$information), list(names, names))
        }
    }
})

test_that("Poisson and least-squares information take weights that are not whole numbers", {
    weights <- runif(n, 0.2, 2)
    for (model in c("pr", "ls", "lr")) {
        data <- designs[[model]]()
        names <- Cyclops:::.coefficientNames(data)
        beta <- rnorm(length(names), 0, 0.3)
        value <- evaluateLikelihood(data, beta, weights = weights, information = names)
        expect_equal(unname(value$information), unname(finiteDifferences(data, beta, weights)$information),
                     tolerance = 1e-6, info = model)
    }
})

test_that("weights count rows, held-out rows add up, and an earlier fit's weights are not used", {
    y <- rbinom(n, 1, plogis(eta))
    train <- as.numeric(seq_len(n) %% 3 != 0)
    data <- createCyclopsData(y ~ X, modelType = "lr")
    subset <- createCyclopsData(y[train == 1] ~ X[train == 1, ], modelType = "lr")
    beta <- c(-0.2, 0.5, -0.4, 0.3)
    weighted <- evaluateLikelihood(data, beta, weights = train, information = 1:4)
    dropped <- evaluateLikelihood(subset, beta, information = 1:4)
    expect_equal(weighted$logLikelihood, dropped$logLikelihood)
    expect_equal(unname(weighted$gradient), unname(dropped$gradient))
    expect_equal(unname(weighted$information), unname(dropped$information))

    all <- evaluateLikelihood(data, beta)$logLikelihood
    expect_equal(weighted$logLikelihood + evaluateLikelihood(data, beta, weights = 1 - train)$logLikelihood, all)

    fit <- fitCyclopsModel(data, prior = createPrior("none"), weights = train)
    expect_equal(evaluateLikelihood(data, beta)$logLikelihood, all)
})

test_that("covariates by id, integer64 or name; threads do not change the information", {
    data <- designs$lr()
    beta <- c(-0.2, 0.5, -0.4, 0.3)
    byId <- evaluateLikelihood(data, beta, information = c(1, 3))$information
    expect_equal(unname(evaluateLikelihood(data, beta, information = bit64::as.integer64(c(1, 3)))$information),
                 unname(byId))
    expect_equal(unname(evaluateLikelihood(data, beta, information = c("(Intercept)", "X2"))$information),
                 unname(byId))
    expect_error(evaluateLikelihood(data, beta, information = 99), "Unknown covariates")
    expect_error(evaluateLikelihood(data, beta[-1]), "a value for each")
    expect_equal(evaluateLikelihood(data, setNames(rev(beta), rev(Cyclops:::.coefficientNames(data))))$logLikelihood,
                 evaluateLikelihood(data, beta)$logLikelihood)

    wide <- createCyclopsData(rbinom(n, 1, 0.3) ~ matrix(rbinom(n * 40, 1, 0.3), n, 40), modelType = "lr")
    names <- Cyclops:::.coefficientNames(wide)
    b <- rnorm(length(names), 0, 0.1)
    expect_identical(evaluateLikelihood(wide, b, information = names, threads = 1)$information,
                     evaluateLikelihood(wide, b, information = names, threads = 2)$information)
})

test_that("an offset covariate enters the likelihood and is not a coefficient", {
    y <- rbinom(n, 1, plogis(eta))
    offset <- rnorm(n)
    covariates <- rbind(data.frame(rowId = rep(1:n, 3), covariateId = rep(1:3, each = n), covariateValue = as.numeric(X)),
                        data.frame(rowId = 1:n, covariateId = 99, covariateValue = offset))
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                                 quiet = TRUE)
    finalizeSqlCyclopsData(data, useOffsetCovariate = 99, offsetAlreadyOnLogScale = TRUE)
    beta <- c(-0.2, 0.5, -0.4, 0.3)
    value <- evaluateLikelihood(data, beta)
    expect_length(value$gradient, 4)
    expect_equal(value$logLikelihood, sum(dbinom(y, 1, plogis(cbind(1, X) %*% beta + offset), log = TRUE)))
})

test_that("the internal Fisher information is computed at the coefficients just set", {
    data <- designs$pr()
    fit <- fitCyclopsModel(data, prior = createPrior("none"))
    ptr <- data$cyclopsInterfacePtr
    ids <- getCovariateIds(data)
    beta <- coef(fit)
    Cyclops:::.cyclopsSetBeta(ptr, beta + 1)
    Cyclops:::.cyclopsGetLogLikelihood(ptr)
    Cyclops:::.cyclopsSetBeta(ptr, beta)
    expect_equal(unname(Cyclops:::.cyclopsGetFisherInformation(ptr, ids)),
                 unname(evaluateLikelihood(data, beta, information = ids)$information))
})
