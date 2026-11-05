library("testthat")
library("survival")

# Newton plan Phase 3: algorithm = "auto" / "newton" and convergenceType = "gap" for cox and cox_counting

newtonProblem <- function(seed, K = 3000, p = 25, delayed = TRUE) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1:2] <- rnorm(2 * K)
    X[, 4] <- X[, 3] + rbinom(K, 1, 0.05) # a correlated pair slows coordinate descent
    eta <- drop(X %*% rnorm(p, 0, 0.3))
    frame <- data.frame(start = if (delayed) round(runif(K, 0, 3)) else 0, y = rbinom(K, 1, 0.7),
                        stratum = sort(sample.int(3, K, replace = TRUE)), w = sample(1:3, K, replace = TRUE), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(eta)))
    rhs <- paste(paste0("X", 1:p), collapse = " + ")
    formula <- as.formula(paste(if (delayed) "Surv(start, time, y)" else "Surv(time, y)", "~", rhs, "+ strata(stratum)"))
    list(frame = frame, formula = formula, model = if (delayed) "cox_counting" else "cox")
}

newtonFit <- function(problem, prior, ..., weights = NULL) {
    fitCyclopsModel(createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model),
                    prior = prior, control = createControl(noiseLevel = "silent", ...), weights = weights,
                    warnings = FALSE)
}

objective <- function(fit) -(fit$log_likelihood + fit$log_prior)

test_that("auto, newton and gap reach the optimum within the certified gap", {
    for (problem in list(newtonProblem(1), newtonProblem(2, delayed = FALSE))) {
        for (prior in list(createPrior("normal", variance = 0.5), createPrior("none"))) {
            reference <- newtonFit(problem, prior, convergenceType = "kkt", tolerance = 1e-12, maxIterations = 1e5)
            for (control in list(list(algorithm = "auto"), list(algorithm = "newton"), list(convergenceType = "gap"),
                                 list(fitter = "super"))) {
                fit <- do.call(newtonFit, c(list(problem, prior), control))
                expect_equal(fit$return_flag, "SUCCESS")
                expect_true(fit$convergence$certified)
                expect_lte(fit$convergence$gapEstimate, 1e-6) # the default tolerance
                expect_lte(objective(fit) - objective(reference), 1e-6 + 1e3 * .Machine$double.eps * abs(objective(fit)))
                if (identical(control, list(algorithm = "newton")) || identical(fit$algorithm, "newton")) {
                    expect_equal(fit$algorithm, "newton") # Newton finishes at the optimum
                    expect_equal(coef(fit), coef(reference), tolerance = 1e-6)
                }
            }
        }
    }
})

test_that("explicit criteria under auto, and default controls, leave fits as they were", {
    problem <- newtonProblem(3)
    prior <- createPrior("normal", variance = 0.5)
    for (control in list(list(), list(convergenceType = "kkt", tolerance = 1e-6), list(convergenceType = "gradient"))) {
        before <- do.call(newtonFit, c(list(problem, prior), control))
        under <- do.call(newtonFit, c(list(problem, prior), control,
                                      list(algorithm = if (length(control)) "auto" else "ccd")))
        expect_identical(coef(under), coef(before))
        expect_identical(under$iterations, before$iterations)
        expect_null(before$convergence)
    }
})

test_that("ineligible fits error under newton or an explicit gap, and fall back under auto", {
    lr <- createCyclopsData(case ~ spontaneous + induced, data = infert, modelType = "lr")
    expect_error(fitCyclopsModel(lr, control = createControl(algorithm = "newton")), "not available")
    expect_error(fitCyclopsModel(lr, control = createControl(convergenceType = "gap")), "not available")

    problem <- newtonProblem(4)
    laplace <- createPrior("laplace", variance = 0.1)
    default <- newtonFit(problem, laplace)
    auto <- newtonFit(problem, laplace, algorithm = "auto")
    expect_identical(coef(auto), coef(default))
    expect_identical(auto$iterations, default$iterations)
    expect_equal(auto$algorithm, "ccd")
    expect_match(auto$convergence$ineligible, "prior other than normal or none")
    expect_error(newtonFit(problem, laplace, algorithm = "newton"), "not available")

    # a tolerance given without a criterion applies to the fallback only
    looser <- newtonFit(problem, laplace, algorithm = "auto", tolerance = 1e-3)
    expect_identical(coef(looser), coef(newtonFit(problem, laplace, tolerance = 1e-3)))
})

test_that("explicit criteria map onto Newton iterations under newton", {
    problem <- newtonProblem(5)
    prior <- createPrior("normal", variance = 0.5)
    one <- newtonFit(problem, prior, algorithm = "newton", convergenceType = "onestep")
    expect_equal(one$convergence$newtonIterations, 1)
    expect_false(one$convergence$certified)
    kkt <- newtonFit(problem, prior, algorithm = "newton", convergenceType = "kkt", tolerance = 1e-6)
    expect_equal(kkt$return_flag, "SUCCESS")
    expect_gte(kkt$convergence$newtonIterations, 1)
})

test_that("the gap stop is invariant to rescaling weights, and Newton results do not depend on threads", {
    problem <- newtonProblem(6, delayed = FALSE)
    none <- createPrior("none")
    w <- problem$frame$w
    unit <- newtonFit(problem, none, algorithm = "auto", weights = w)
    tiny <- newtonFit(problem, none, algorithm = "auto", weights = w * 1e-6)
    expect_equal(coef(tiny), coef(unit), tolerance = 1e-8)

    prior <- createPrior("normal", variance = 0.5)
    one <- newtonFit(problem, prior, algorithm = "auto", threads = 1)
    two <- newtonFit(problem, prior, algorithm = "auto", threads = 2)
    expect_identical(coef(two), coef(one))
    expect_identical(two$iterations, one$iterations)
})

test_that("maxIterations under gap warns with the estimated gap", {
    problem <- newtonProblem(7)
    data <- createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model)
    expect_warning(fit <- fitCyclopsModel(data, prior = createPrior("normal", variance = 0.5),
                                          control = createControl(convergenceType = "gap", maxIterations = 3)),
                   "estimated gap")
    expect_equal(fit$return_flag, "MAX_ITERATIONS")
    expect_false(fit$convergence$certified)
})
