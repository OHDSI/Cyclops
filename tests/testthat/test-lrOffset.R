library("testthat")

context("test-lrOffset.R")

# Logistic data with an offset (log of a sampling fraction), case weights and a dense covariate,
# in convertToCyclopsData format
simulateOffsetData <- function(n = 2000, seed = 3) {
    set.seed(seed)
    x <- cbind(rbinom(n, 1, 0.3), rbinom(n, 1, 0.3), rnorm(n), rbinom(n, 1, 0.1))
    offset <- log(runif(n, 0.05, 1))
    y <- rbinom(n, 1, plogis(-1 + drop(x %*% c(0.5, -0.5, 0.3, 0.05)) + offset))
    nz <- which(x != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = x[nz])
    list(x = x, y = y, offset = offset, weights = runif(n, 0.3, 2),
         outcomes = data.frame(rowId = 1:n, y = y, offset = offset),
         covariates = covariates[order(covariates$covariateId, covariates$rowId), ])
}

tightControl <- function(algorithm = "auto") {
    createControl(algorithm = algorithm, convergenceType = "gradient", tolerance = 1E-11, noiseLevel = "silent")
}

test_that("lr offset matches glm(offset =), with and without weights and intercept", {
    data <- simulateOffsetData()
    gold <- glm(data$y ~ data$x, family = binomial(), offset = data$offset, control = glm.control(epsilon = 1E-14))
    goldWeighted <- glm(data$y ~ data$x, family = quasibinomial(), offset = data$offset, weights = data$weights,
                        control = glm.control(epsilon = 1E-14))
    goldNoIntercept <- glm(data$y ~ data$x - 1, family = binomial(), offset = data$offset,
                           control = glm.control(epsilon = 1E-14))
    for (algorithm in c("ccd", "auto", "newton")) {
        cyclopsData <- convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", quiet = TRUE)
        fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"), control = tightControl(algorithm))
        expect_equal(names(coef(fit)), c("(Intercept)", "1", "2", "3", "4"))
        expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
        expect_equal(unname(predict(fit)), unname(fitted(gold)), tolerance = 1E-6)

        fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"), weights = data$weights,
                               control = tightControl(algorithm))
        expect_equal(unname(coef(fit)), unname(coef(goldWeighted)), tolerance = 1E-6)

        cyclopsData <- convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", addIntercept = FALSE,
                                            quiet = TRUE)
        fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"), control = tightControl(algorithm))
        expect_equal(unname(coef(fit)), unname(coef(goldNoIntercept)), tolerance = 1E-6)
    }
})

test_that("lr offset from Andromeda tables matches glm(offset =)", {
    skip_if_not_installed("Andromeda")
    data <- simulateOffsetData(1000, 4)
    gold <- glm(data$y ~ data$x, family = binomial(), offset = data$offset, control = glm.control(epsilon = 1E-14))
    andromeda <- Andromeda::andromeda(outcomes = data$outcomes, covariates = data$covariates)
    on.exit(Andromeda::close(andromeda))
    cyclopsData <- convertToCyclopsData(andromeda$outcomes, andromeda$covariates, modelType = "lr", quiet = TRUE)
    fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"), control = tightControl())
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(predict(fit, andromeda$outcomes, andromeda$covariates), predict(fit), tolerance = 1E-10)
})

# Gradient of the weighted log-likelihood with the offset, for the intercept and each covariate
offsetGradient <- function(data, beta, weights) {
    design <- cbind(1, data$x)
    p <- plogis(drop(design %*% beta) + data$offset)
    drop(crossprod(design, weights * (data$y - p)))
}

test_that("lr offset with a normal prior at a fixed variance matches the penalized likelihood", {
    data <- simulateOffsetData(2000, 5)
    variance <- 0.05
    for (weights in list(rep(1, length(data$y)), data$weights)) {
        # Reference: Newton on the weighted log-likelihood minus beta_j^2 / (2 variance), intercept unpenalized
        design <- cbind(1, data$x)
        penalty <- diag(c(0, rep(1 / variance, ncol(data$x))))
        beta <- rep(0, ncol(design))
        for (i in 1:50) {
            p <- plogis(drop(design %*% beta) + data$offset)
            gradient <- drop(crossprod(design, weights * (data$y - p))) - penalty %*% beta
            hessian <- crossprod(design, weights * p * (1 - p) * design) + penalty
            beta <- beta + drop(solve(hessian, gradient))
        }
        for (algorithm in c("ccd", "auto")) {
            fit <- fitCyclopsModel(convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", quiet = TRUE),
                                   prior = createPrior("normal", variance = variance, exclude = 0),
                                   weights = weights, control = tightControl(algorithm))
            expect_equal(unname(coef(fit)), beta, tolerance = 1E-6)
        }
    }
})

test_that("lr offset with a Laplace prior at a fixed variance meets the optimality (KKT) conditions", {
    data <- simulateOffsetData(2000, 6)
    variance <- 0.002
    lambda <- sqrt(2 / variance)
    for (weights in list(rep(1, length(data$y)), data$weights)) {
        for (algorithm in c("ccd", "auto")) {
            fit <- fitCyclopsModel(convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", quiet = TRUE),
                                   prior = createPrior("laplace", variance = variance, exclude = 0),
                                   weights = weights, control = tightControl(algorithm))
            beta <- unname(coef(fit))
            gradient <- offsetGradient(data, beta, weights)
            expect_true(any(beta[-1] == 0) && any(beta[-1] != 0)) # both conditions are tested
            expect_equal(gradient[1], 0, tolerance = 1E-6) # unpenalized intercept
            active <- which(beta != 0)[-1]
            expect_equal(gradient[active], lambda * sign(beta[active]), tolerance = 1E-6)
            expect_true(all(abs(gradient[beta == 0]) <= lambda * (1 + 1E-6)))
        }
    }
})

test_that("Cross-validated lr fit with an offset", {
    data <- simulateOffsetData(1000, 7)
    fit <- fitCyclopsModel(convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", quiet = TRUE),
                           prior = createPrior("laplace", useCrossValidation = TRUE, exclude = 0),
                           control = createControl(fold = 5, cvRepetitions = 1, seed = 1, noiseLevel = "silent",
                                                   convergenceType = "gradient", tolerance = 1E-11))
    beta <- unname(coef(fit))
    gradient <- offsetGradient(data, beta, rep(1, length(data$y)))
    lambda <- sqrt(2 / fit$variance)
    active <- which(beta != 0)[-1]
    expect_equal(gradient[active], lambda * sign(beta[active]), tolerance = 1E-6) # the final fit used the offset
})

test_that("lr predictions on new data add the offset of newOutcomes", {
    data <- simulateOffsetData(500, 8)
    fit <- fitCyclopsModel(convertToCyclopsData(data$outcomes, data$covariates, modelType = "lr", quiet = TRUE),
                           prior = createPrior("none"), control = tightControl())
    expect_equal(predict(fit, data$outcomes, data$covariates), predict(fit), tolerance = 1E-10)
    linearPredictor <- drop(cbind(1, data$x) %*% coef(fit))
    withoutOffset <- predict(fit, data$outcomes[, c("rowId", "y")], data$covariates)
    expect_equal(unname(withoutOffset), plogis(linearPredictor), tolerance = 1E-10)
})
