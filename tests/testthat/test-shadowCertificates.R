library("testthat")
library("survival")

# Observe-only stopping statistics (CYCLOPS_SHADOW): fits are unchanged, and the certified bounds sit above
# the true gap along the CCD path

shadowProblem <- function(seed, K = 2000, p = 15, delayed = FALSE) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1] <- rnorm(K)
    eta <- drop(X %*% rnorm(p, 0, 0.3))
    frame <- data.frame(start = if (delayed) round(runif(K, 0, 3)) else 0, y = rbinom(K, 1, 0.7),
                        stratum = sort(sample.int(4, K, replace = TRUE)), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.1 * exp(eta)))
    rhs <- paste(paste0("X", 1:p), collapse = " + ")
    formula <- as.formula(paste(if (delayed) "Surv(start, time, y)" else "Surv(time, y)", "~", rhs, "+ strata(stratum)"))
    list(frame = frame, formula = formula, model = if (delayed) "cox_counting" else "cox")
}

shadowFit <- function(problem, prior, shadow, control = list()) {
    old <- Sys.getenv("CYCLOPS_SHADOW", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("CYCLOPS_SHADOW") else Sys.setenv(CYCLOPS_SHADOW = old))
    Sys.setenv(CYCLOPS_SHADOW = shadow)
    fitCyclopsModel(createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model),
                    prior = prior, control = do.call(createControl, c(list(noiseLevel = "silent"), control)))
}

test_that("the shadow log does not change fits", {
    for (problem in list(shadowProblem(1), shadowProblem(2, delayed = TRUE))) {
        for (prior in list(createPrior("normal", variance = 0.1), createPrior("none"),
                           createPrior("laplace", variance = 0.05))) {
            for (control in list(list(), list(convergenceType = "kkt", tolerance = 1e-6))) {
                off <- shadowFit(problem, prior, "0", control)
                on <- shadowFit(problem, prior, "2", control)
                expect_null(off$shadow)
                expect_identical(coef(on), coef(off))
                expect_identical(on$log_likelihood, off$log_likelihood)
                expect_identical(on$iterations, off$iterations)
                expect_equal(nrow(on$shadow), on$iterations)
            }
        }
    }
})

test_that("R, gsc and the U-form gsc bound the gap along the path, and the bracket holds Lambda^2 / 2", {
    for (problem in list(shadowProblem(3), shadowProblem(4, delayed = TRUE))) {
        for (prior in list(createPrior("normal", variance = 0.1), createPrior("none"))) {
            path <- shadowFit(problem, prior, "2")$shadow
            optimum <- shadowFit(problem, prior, "0", list(convergenceType = "kkt", tolerance = 1e-12,
                                                           maxIterations = 10000))
            fStar <- -(optimum$log_likelihood + optimum$log_prior)
            gap <- path$f - fStar
            floor <- 100 * .Machine$double.eps * abs(fStar)
            above <- gap > floor
            expect_true(any(above))
            for (bound in c("R", "gsc", "gscU")) {
                b <- path[[bound]]
                ok <- above & is.finite(b)
                expect_true(all(b[ok] >= gap[ok] - floor), info = bound)
            }
            if (prior$priorType == "normal") {
                expect_true(all(is.finite(path$R)))
                expect_true(all(path$bracketLow <= path$lambda2Half * (1 + 1e-10) + 1e-300))
                expect_true(all(path$bracketHigh >= path$lambda2Half * (1 - 1e-10)))
            } else {
                expect_true(any(is.finite(path$gscU)))
            }
            expect_true(all(path$consistency >= 1 - 1e-8, na.rm = TRUE))
        }
    }
})

test_that("fit$convergence reports the smallest certified bound at the returned point", {
    fit <- shadowFit(shadowProblem(5), createPrior("normal", variance = 0.1), "1")
    last <- fit$shadow[nrow(fit$shadow), ]
    expect_true(last$curvature)
    expect_true(fit$convergence$certified)
    expect_equal(fit$convergence$gapEstimate, min(last$R, last$gsc, na.rm = TRUE))
    expect_equal(fit$convergence$cycles, fit$iterations)

    lasso <- shadowFit(shadowProblem(5), createPrior("laplace", variance = 0.05), "1")
    expect_false(lasso$convergence$certified)
    expect_equal(lasso$convergence$gapEstimate, lasso$shadow$lambda2Half[nrow(lasso$shadow)])
})
