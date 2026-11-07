library("testthat")

# Newton plan Phase 7: second-order kernels and Newton fits for logistic regression

logisticProblem <- function(seed, K = 800, p = 5) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.2), K, p)
    X[, 1] <- rnorm(K)
    colnames(X) <- paste0("x", 1:p)
    y <- rbinom(K, 1, plogis(-0.5 + drop(X %*% c(0.5, -0.3, 0.4, 0.2, -0.6)[seq_len(p)])))
    frame <- data.frame(y = y, X)
    formula <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
    list(frame = frame, X = cbind(1, X), y = y, formula = formula,
         data = createCyclopsData(formula, data = frame, modelType = "lr"))
}

logisticReference <- function(problem, beta, w) {
    eta <- drop(problem$X %*% beta)
    p <- plogis(eta)
    list(ll = sum(w * (problem$y * eta - log1p(exp(eta)))), g = drop(crossprod(problem$X, w * (p - problem$y))),
         H = crossprod(problem$X, w * p * (1 - p) * problem$X))
}

test_that("T1-lr: g, H, Hv and trial steps, with 0/1 and continuous weights", {
    problem <- logisticProblem(1)
    beta <- c(-0.4, 0.3, -0.2, 0.5, 0.1, -0.3)
    v <- c(0.2, 1, -0.5, 0.25, 0.8, -1)
    K <- nrow(problem$frame)
    for (w in list(NULL, rbinom(K, 1, 0.8), runif(K, 0.2, 3))) {
        fitCyclopsModel(problem$data, startingCoefficients = beta, fixedCoefficients = rep(TRUE, 6), weights = w,
                        control = createControl(noiseLevel = "silent"))
        so <- Cyclops:::.cyclopsSecondOrder(problem$data$cyclopsInterfacePtr, NULL, v, 0.37, 1)
        weights <- if (is.null(w)) rep(1, K) else w
        ref <- logisticReference(problem, beta, weights)
        expect_true(so$supported)
        expect_equal(so$gradient, unname(ref$g), tolerance = 1e-12)
        expect_equal(unname(so$hessian), unname(ref$H), tolerance = 1e-12)
        expect_equal(so$hv, unname(drop(ref$H %*% v)), tolerance = 1e-12)
        expect_equal(so$trial, logisticReference(problem, beta + 0.37 * v, weights)$ll, tolerance = 1e-12)
        expect_equal(so$committed, so$trial, tolerance = 1e-12)
    }
})

test_that("logistic regression reaches the optimum under auto and newton", {
    problem <- logisticProblem(2)
    objective <- function(fit) -(fit$log_likelihood + fit$log_prior)
    for (prior in list(createPrior("none"), createPrior("normal", variance = 0.5),
                       createPrior("laplace", variance = 0.05))) {
        reference <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                     control = createControl(convergenceType = "kkt", tolerance = 1e-10,
                                                             maxIterations = 1e5, noiseLevel = "silent"))
        auto <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                control = createControl(algorithm = "auto", noiseLevel = "silent"))
        expect_equal(auto$return_flag, "SUCCESS")
        expect_lte(objective(auto) - objective(reference), 1e-6)
        newton <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                  control = createControl(algorithm = "newton", convergenceType = "gap",
                                                          tolerance = 1e-12, noiseLevel = "silent"))
        expect_equal(newton$return_flag, "SUCCESS")
        expect_equal(coef(newton), coef(reference), tolerance = 1e-6)
        for (solver in c("dense", "cg")) { # both solvers on unpenalized and ridge fits
            if (prior$priorType == "laplace") next
            forced <- local({
                old <- Sys.getenv("CYCLOPS_NEWTON_SOLVER", unset = NA)
                on.exit(if (is.na(old)) Sys.unsetenv("CYCLOPS_NEWTON_SOLVER") else Sys.setenv(CYCLOPS_NEWTON_SOLVER = old))
                Sys.setenv(CYCLOPS_NEWTON_SOLVER = solver)
                fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                control = createControl(algorithm = "newton", noiseLevel = "silent"))
            })
            expect_equal(forced$convergence$solver, solver)
            expect_lte(objective(forced) - objective(reference), 1e-6)
        }
    }
})

test_that("T2-lr: a certified stop is within tau of the optimum, with a rare indicator and the intercept unpenalized", {
    set.seed(3)
    K <- 5000
    frame <- data.frame(y = rbinom(K, 1, 0.3), x1 = rnorm(K), rare = 0)
    frame$rare[sample.int(K, 5)] <- 1 # 99.9% prevalence of 0
    frame$y[frame$rare == 1][1] <- 1 - frame$y[frame$rare == 1][1] # not separated
    data <- createCyclopsData(y ~ x1 + rare, data = frame, modelType = "lr")
    prior <- createPrior("normal", variance = 1) # excludes the intercept
    objective <- function(fit) -(fit$log_likelihood + fit$log_prior)
    reference <- suppressWarnings(fitCyclopsModel(data, prior = prior, forceNewObject = TRUE,
        control = createControl(algorithm = "newton", convergenceType = "gap", tolerance = 1e-13, noiseLevel = "silent")))
    for (tau in c(1e-4, 1e-6)) {
        fit <- suppressWarnings(fitCyclopsModel(data, prior = prior, forceNewObject = TRUE,
            control = createControl(algorithm = "auto", convergenceType = "gap", tolerance = tau, noiseLevel = "silent")))
        expect_equal(fit$return_flag, "SUCCESS")
        expect_lte(objective(fit) - objective(reference), tau)
    }
})

test_that("T8-lr: fold fits under a fold partition match tight coordinate descent fold by fold", {
    problem <- logisticProblem(4, K = 1500)
    folds <- function(...) {
        out <- capture.output(fitCyclopsModel(problem$data, forceNewObject = TRUE, warnings = FALSE,
            prior = createPrior("normal", useCrossValidation = TRUE, exclude = "(Intercept)"),
            control = createControl(cvType = "grid", gridSteps = 3, lowerLimit = 0.1, upperLimit = 10, fold = 5,
                                    seed = 1, noiseLevel = "noisy", threads = 1, ...)))
        as.numeric(sub(".*pred log like = ", "", grep("pred log like", out, value = TRUE)))
    }
    newton <- folds(algorithm = "newton", convergenceType = "gap", tolerance = 1e-12)
    ccd <- folds(convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5)
    expect_length(newton, 15)
    expect_equal(newton, ccd, tolerance = 1e-8)
})
