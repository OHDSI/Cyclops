library("testthat")
library("survival")

# Separation contract (Newton plan 2.7, D5): with a prior "none" on every free coefficient a separated fit is
# ILLCONDITIONED with NaN in its divergent coefficients; otherwise it is SUCCESS with the divergent coefficients
# reported at their limit. Either way fit$divergentCoefficients names them and the fit warns.

separatedCox <- function(seed = 1, K = 400) {
    set.seed(seed)
    frame <- data.frame(time = ceiling(rexp(K, 0.1)), y = rbinom(K, 1, 0.6), x1 = rnorm(K),
                        x2 = rbinom(K, 1, 0.3), arm = rbinom(K, 1, 0.5))
    frame$y[frame$arm == 1] <- 0 # no events in one arm: the arm's MLE is -Inf
    frame
}

test_that("a separated Cox fit without priors is ILLCONDITIONED with NaN only in the divergent coefficient", {
    frame <- separatedCox()
    data <- createCyclopsData(Surv(time, y) ~ x1 + x2 + arm, data = frame, modelType = "cox")
    expect_warning(fit <- fitCyclopsModel(data), "coefficient may be infinite")
    expect_equal(fit$return_flag, "ILLCONDITIONED")
    expect_equal(fit$divergentCoefficients, "arm")
    beta <- coef(fit, ignoreConvergence = TRUE)
    expect_true(is.nan(beta[["arm"]]))
    expect_true(all(is.finite(beta[c("x1", "x2")])))

    # the other coefficients are those of the limit: the arm's rows never at the risk-set minimum drop out
    limit <- coxph(Surv(time, y) ~ x1 + x2, data = frame[frame$arm == 0, ], ties = "breslow")
    expect_equal(unname(beta[c("x1", "x2")]), unname(coef(limit)), tolerance = 1e-5)

    ci <- confint(fit, parm = "arm")
    expect_true(all(is.na(ci[, 2:3])))
})

test_that("with a proper prior elsewhere, the fit is SUCCESS and the divergent coefficient sits at its limit", {
    frame <- separatedCox(2)
    data <- createCyclopsData(Surv(time, y) ~ x1 + x2 + arm, data = frame, modelType = "cox")
    prior <- createPrior("normal", variance = 1, exclude = "arm")
    expect_warning(fit <- fitCyclopsModel(data, prior = prior), "coefficient may be infinite")
    expect_equal(fit$return_flag, "SUCCESS")
    expect_equal(fit$divergentCoefficients, "arm")
    beta <- coef(fit)
    expect_equal(beta[["arm"]], -40)
    expect_true(all(is.finite(beta)))
})

test_that("logistic regression: one-signed columns whose rows all have the same outcome diverge", {
    set.seed(3)
    frame <- data.frame(y = rbinom(300, 1, 0.4), x1 = rnorm(300), rare = 0)
    frame$rare[sample(which(frame$y == 1), 5)] <- 1 # every row with rare = 1 is an event: +Inf
    data <- createCyclopsData(y ~ x1 + rare, data = frame, modelType = "lr")
    expect_warning(fit <- fitCyclopsModel(data), "coefficient may be infinite")
    expect_equal(fit$return_flag, "ILLCONDITIONED")
    expect_equal(fit$divergentCoefficients, "rare")
    expect_true(all(is.finite(coef(fit, ignoreConvergence = TRUE)[c("(Intercept)", "x1")])))

    frame$rare[which(frame$y == 0)[1]] <- 1 # one non-event: no longer separated
    data <- createCyclopsData(y ~ x1 + rare, data = frame, modelType = "lr")
    fit <- fitCyclopsModel(data)
    expect_null(fit$divergentCoefficients)
    expect_equal(fit$return_flag, "SUCCESS")
})
