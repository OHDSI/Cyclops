library("testthat")
library("survival")

# Newton plan Phase 6: prox-Newton for Cox models with Laplace priors

lassoProblem <- function(seed, K = 3000, p = 40) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.25), K, p)
    X[, 1:3] <- rnorm(3 * K)
    beta <- c(0.5, -0.4, 0.3, rnorm(5, 0, 0.4), rep(0, p - 8))
    frame <- data.frame(start = round(runif(K, 0, 3)), y = rbinom(K, 1, 0.7),
                        stratum = sort(sample.int(3, K, replace = TRUE)), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(drop(X %*% beta))))
    list(frame = frame,
         formula = as.formula(paste("Surv(start, time, y) ~", paste0("X", 1:p, collapse = " + "), "+ strata(stratum)")))
}

lassoFit <- function(problem, prior, ...) {
    fitCyclopsModel(createCyclopsData(problem$formula, data = problem$frame, modelType = "cox_counting"),
                    prior = prior, control = createControl(noiseLevel = "silent", ...), warnings = FALSE)
}
objective <- function(fit) -(fit$log_likelihood + fit$log_prior)

test_that("prox-Newton reaches the Laplace optimum and its zero set", {
    problem <- lassoProblem(1)
    for (variance in c(0.01, 0.1)) {
        prior <- createPrior("laplace", variance = variance)
        reference <- lassoFit(problem, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5)
        for (algorithm in c("newton", "auto")) {
            fit <- lassoFit(problem, prior, algorithm = algorithm, tolerance = 1e-9, convergenceType = "gap")
            expect_equal(fit$return_flag, "SUCCESS")
            expect_false(fit$convergence$certified) # the L1 decrease is an estimate (plan 3.2)
            expect_lte(objective(fit) - objective(reference), 1e-8 + 1e3 * .Machine$double.eps * abs(objective(fit)))
            expect_equal(coef(fit) != 0, coef(reference) != 0)
            expect_equal(coef(fit), coef(reference), tolerance = 1e-5)
        }
    }
})

test_that("prox-Newton with an unpenalized coefficient, and threads", {
    problem <- lassoProblem(2)
    prior <- createPrior("laplace", variance = 0.05, exclude = "X1")
    reference <- lassoFit(problem, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5)
    one <- lassoFit(problem, prior, algorithm = "newton", threads = 1)
    two <- lassoFit(problem, prior, algorithm = "newton", threads = 2)
    expect_identical(coef(two), coef(one))
    expect_equal(coef(one), coef(reference), tolerance = 1e-5)
})

test_that("Laplace cross-validation under auto selects the same variance", {
    problem <- lassoProblem(3)
    cv <- function(...) lassoFit(problem, createPrior("laplace", useCrossValidation = TRUE), cvType = "auto", fold = 5,
                                 seed = 3, ...)
    expect_equal(cv(algorithm = "auto")$variance, cv()$variance, tolerance = 1e-2)
})
