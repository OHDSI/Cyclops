library("testthat")
library("survival")

# Model-based variance with non-integer weights and the cluster-robust (sandwich) variance, against
# coxph(weights, cluster, ties = "breslow") and sandwich::vcovCL(glm(weights), type = "HC0", cadjust = FALSE)

tolerance <- 1E-6
control <- createControl(tolerance = 1E-12, noiseLevel = "silent")
glmControl <- glm.control(epsilon = 1E-14, maxit = 100)

# Three rows per subject: counting-process intervals, two strata, non-integer weights
robustData <- function(seed = 1) {
    set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    n <- 300
    d <- data.frame(id = rep(1:100, each = 3), x1 = rnorm(n), x2 = rbinom(n, 1, 0.4), s = rep(1:2, length.out = n))
    d$stop <- ceiling(rexp(n, 0.1 * exp(0.5 * d$x1 - 0.3 * d$x2)))
    d$start <- pmax(0, d$stop - ceiling(runif(n, 1, 10)))
    d$event <- rbinom(n, 1, 0.7)
    d$w <- runif(n, 0.2, 3)
    d$y <- rbinom(n, 1, plogis(-0.5 + d$x1))
    d$count <- rpois(n, exp(0.2 + 0.3 * d$x1))
    d$exposure <- runif(n, 0.5, 2)
    d
}

expectVcov <- function(observed, expected) {
    expect_equal(unname(observed), unname(expected), tolerance = tolerance)
}

test_that("cox_counting: weighted model-based and robust variances match coxph with strata and clusters", {
    d <- robustData()
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(s), data = d, weights = w, cluster = id,
                  ties = "breslow")
    fit <- fitCyclopsModel(createCyclopsData(Surv(start, stop, event) ~ x1 + x2 + strata(s) + cluster(id), data = d,
                                             modelType = "cox_counting"),
                           weights = d$w, control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit), gold$naive.var)
    expect_equal(vcov(fit, type = "modelBased"), vcov(fit))
    robust <- vcov(fit, type = "robust")
    expectVcov(robust, gold$var)
    expect_equal(dimnames(robust), list(c("x1", "x2"), c("x1", "x2")))

    # without subjects each row is its own cluster
    noCluster <- fitCyclopsModel(createCyclopsData(Surv(start, stop, event) ~ x1 + x2 + strata(s), data = d,
                                                   modelType = "cox_counting"),
                                 weights = d$w, control = control)
    d$row <- seq_len(nrow(d))
    goldRows <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(s), data = d, weights = w, cluster = row,
                      ties = "breslow")
    expectVcov(vcov(noCluster, type = "robust"), goldRows$var)
})

test_that("cox: weighted model-based and robust variances match coxph with strata and clusters", {
    d <- robustData(2)
    gold <- coxph(Surv(stop, event) ~ x1 + x2 + strata(s), data = d, weights = w, cluster = id, ties = "breslow")
    fit <- fitCyclopsModel(createCyclopsData(Surv(stop, event) ~ x1 + x2 + strata(s) + cluster(id), data = d,
                                             modelType = "cox"),
                           weights = d$w, control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit), gold$naive.var)
    expectVcov(vcov(fit, type = "robust"), gold$var)
})

test_that("cox_counting robust variance from convertToCyclopsData subjects (and weight-0 rows) matches coxph", {
    d <- robustData(3)
    d$w[1:20] <- 0
    outcomes <- data.frame(rowId = seq_len(nrow(d)), stratumId = d$s, subjectId = d$id * 7, startTime = d$start,
                           time = d$stop, y = d$event, weights = d$w)
    covariates <- data.frame(rowId = rep(seq_len(nrow(d)), 2), covariateId = rep(1:2, each = nrow(d)),
                             covariateValue = c(d$x1, d$x2))
    fit <- fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "cox_counting", quiet = TRUE),
                           control = control)
    gold <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(s), data = d[d$w > 0, ], weights = w, cluster = id,
                  ties = "breslow")
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit), gold$naive.var)
    expectVcov(vcov(fit, type = "robust"), gold$var)
})

test_that("lr and pr: weighted model-based and robust variances match glm and sandwich::vcovCL", {
    skip_if_not_installed("sandwich")
    d <- robustData(4)

    gold <- suppressWarnings(glm(y ~ x1 + x2, family = binomial, data = d, weights = w, control = glmControl))
    fit <- fitCyclopsModel(createCyclopsData(y ~ x1 + x2 + cluster(id), data = d, modelType = "lr"),
                           weights = d$w, control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit), vcov(gold))
    expectVcov(vcov(fit, type = "robust"), sandwich::vcovCL(gold, cluster = ~id, type = "HC0", cadjust = FALSE))

    gold <- glm(count ~ x1 + x2 + offset(log(exposure)), family = poisson, data = d, weights = w,
                control = glmControl)
    fit <- fitCyclopsModel(createCyclopsData(count ~ x1 + x2 + offset(log(exposure)) + cluster(id), data = d,
                                             modelType = "pr"),
                           weights = d$w, control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit), vcov(gold))
    expectVcov(vcov(fit, type = "robust"), sandwich::vcovCL(gold, cluster = ~id, type = "HC0", cadjust = FALSE))

    # subjects from convertToCyclopsData
    outcomes <- data.frame(rowId = seq_len(nrow(d)), subjectId = d$id, y = d$count, time = d$exposure,
                           weights = d$w)
    covariates <- data.frame(rowId = rep(seq_len(nrow(d)), 2), covariateId = rep(1:2, each = nrow(d)),
                             covariateValue = c(d$x1, d$x2))
    fit <- fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "pr", quiet = TRUE),
                           control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = tolerance)
    expectVcov(vcov(fit, type = "robust"), sandwich::vcovCL(gold, cluster = ~id, type = "HC0", cadjust = FALSE))
})

test_that("robust variance drops fixed coefficients and excludes the prior", {
    skip_if_not_installed("sandwich")
    d <- robustData(5)

    # a fixed coefficient acts as an offset
    fixed <- fitCyclopsModel(createCyclopsData(y ~ x1 + x2 + cluster(id), data = d, modelType = "lr"),
                             weights = d$w, control = control, startingCoefficients = c(0, 0, 0.5),
                             fixedCoefficients = c(FALSE, FALSE, TRUE))
    gold <- suppressWarnings(glm(y ~ x1 + offset(0.5 * x2), family = binomial, data = d, weights = w,
                                 control = glmControl))
    robust <- vcov(fixed, type = "robust")
    expect_equal(rownames(robust), c("(Intercept)", "x1"))
    expectVcov(robust, sandwich::vcovCL(gold, cluster = ~id, type = "HC0", cadjust = FALSE))

    # under a prior: the information and scores of the likelihood at the posterior mode
    ridge <- fitCyclopsModel(createCyclopsData(y ~ x1 + x2 + cluster(id), data = d, modelType = "lr"),
                             weights = d$w, prior = createPrior("normal", variance = 0.1, exclude = "(Intercept)"),
                             control = control)
    X <- cbind(1, d$x1, d$x2)
    p <- plogis(as.vector(X %*% coef(ridge)))
    bread <- solve(crossprod(X, d$w * p * (1 - p) * X))
    meat <- crossprod(rowsum(d$w * (d$y - p) * X, d$id))
    expectVcov(vcov(ridge, type = "robust"), bread %*% meat %*% bread)
})

test_that("confint(method = \"robust\") gives Wald intervals from the robust variance", {
    d <- robustData(6)
    fit <- fitCyclopsModel(createCyclopsData(Surv(start, stop, event) ~ x1 + x2 + cluster(id), data = d,
                                             modelType = "cox_counting"),
                           weights = d$w, control = control)
    se <- sqrt(diag(vcov(fit, type = "robust")))
    ci <- confint(fit, c("x1", "x2"), level = 0.9, method = "robust")
    expect_equal(rownames(ci), c("x1", "x2"))
    expect_equal(colnames(ci), c("covariate", "5.0 %", "95.0 %", "evaluations"))
    expect_equal(unname(ci[, 2]), unname(coef(fit) - qnorm(0.95) * se), tolerance = 1E-12)
    expect_equal(unname(ci[, 3]), unname(coef(fit) + qnorm(0.95) * se), tolerance = 1E-12)
    expect_error(confint(fit, "x1", method = "wald"))
})

test_that("robust variance: NA for aliased coefficients, the reduced model's elsewhere; errors for other models", {
    d <- robustData(7)
    d$x1copy <- d$x1
    full <- fitCyclopsModel(createCyclopsData(Surv(stop, event) ~ x1 + x2 + x1copy + cluster(id), data = d,
                                              modelType = "cox"),
                            weights = d$w, control = control)
    reduced <- coxph(Surv(stop, event) ~ x1 + x2, data = d, weights = w, cluster = id, ties = "breslow")
    expect_warning(robust <- vcov(full, type = "robust"), "aliased")
    expect_true(all(is.na(robust[c("x1", "x1copy"), ])))
    expect_equal(robust["x2", "x2"], reduced$var[2, 2], tolerance = tolerance)

    cpr <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                             data = Cyclops::oxford, modelType = "cpr"))
    expect_error(vcov(cpr, type = "robust"), "not implemented")
})
