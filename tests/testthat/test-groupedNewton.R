library("testthat")
library("survival")

# Newton plan Phase 7: second-order kernels and Newton fits for the grouped models cpr, sccs and clr

groupedProblem <- function(model, seed, nStrata = 60, p = 6) {
    set.seed(seed)
    sizes <- sample(2:8, nStrata, TRUE)
    id <- rep(seq_len(nStrata), sizes)
    K <- length(id)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1] <- rnorm(K)
    colnames(X) <- paste0("x", 1:p)
    len <- runif(K, 0.5, 3)
    eta <- drop(X %*% rnorm(p, 0, 0.4))
    y <- if (model == "clr") {
        y <- integer(K)
        for (s in unique(id)) {
            r <- which(id == s)
            y[r[which.max(eta[r] + rlogis(length(r)))]] <- 1L
        }
        y
    } else {
        rpois(K, len * exp(eta) * 0.5)
    }
    frame <- data.frame(id = id, y = y, len = len, X)
    formula <- as.formula(paste("y ~", paste(colnames(X), collapse = " + "), "+ strata(id)"))
    data <- if (model == "sccs") {
        createCyclopsData(formula, data = frame, time = frame$len, modelType = model)
    } else {
        createCyclopsData(formula, data = frame, modelType = model)
    }
    list(frame = frame, X = X, model = model, data = data)
}

# g and H of the negative log likelihood sum_k w_k y_k eta_k - sum_s n_s log sum_{k in s} u_k
groupedReference <- function(problem, beta, w = rep(1, nrow(problem$X))) {
    X <- problem$X
    eta <- drop(X %*% beta)
    u <- if (problem$model == "sccs") problem$frame$len * exp(eta) else exp(eta)
    g <- -colSums(w * problem$frame$y * X)
    H <- matrix(0, ncol(X), ncol(X))
    for (s in unique(problem$frame$id)) {
        r <- which(problem$frame$id == s)
        n <- sum(w[r] * problem$frame$y[r])
        if (n == 0) next
        pi <- u[r] / sum(u[r])
        xbar <- colSums(pi * X[r, , drop = FALSE])
        g <- g + n * xbar
        H <- H + n * (crossprod(X[r, , drop = FALSE], pi * X[r, , drop = FALSE]) - tcrossprod(xbar))
    }
    list(g = g, H = H)
}

secondOrderAt <- function(problem, beta, v, t = 0.37, weights = NULL) {
    fitCyclopsModel(problem$data, startingCoefficients = beta, fixedCoefficients = rep(TRUE, length(beta)),
                    weights = weights, control = createControl(noiseLevel = "silent"))
    Cyclops:::.cyclopsSecondOrder(problem$data$cyclopsInterfacePtr, NULL, v, t, 1)
}

test_that("T1-grouped: g, H, Hv and trial steps of cpr, sccs and clr", {
    for (model in c("cpr", "sccs", "clr")) {
        problem <- groupedProblem(model, 1)
        beta <- c(0.3, -0.2, 0.1, 0.4, -0.3, 0.2)
        v <- c(1, -0.5, 0.25, 0.8, -1, 0.3)
        so <- secondOrderAt(problem, beta, v)
        expect_true(so$supported)
        ref <- groupedReference(problem, beta)
        expect_equal(so$gradient, unname(ref$g), tolerance = 1e-12)
        expect_equal(unname(so$hessian), unname(ref$H), tolerance = 1e-12)
        expect_equal(so$hv, unname(drop(ref$H %*% v)), tolerance = 1e-12)
        expect_equal(so$committed, so$trial, tolerance = 1e-12)
        h <- 1e-6 # the trial log likelihood moves along -g'v
        expect_equal((secondOrderAt(problem, beta, v, h)$trial - secondOrderAt(problem, beta, v, -h)$trial) / (2 * h),
                     -sum(so$gradient * v), tolerance = 1e-6)

        w <- ifelse(problem$frame$id %% 2 == 0, 0, 1) # half the strata held out
        weighted <- secondOrderAt(problem, beta, v, weights = w)
        refW <- groupedReference(problem, beta, w)
        expect_equal(weighted$gradient, unname(refW$g), tolerance = 1e-12)
        expect_equal(unname(weighted$hessian), unname(refW$H), tolerance = 1e-12)
    }
})

test_that("grouped models reach the optimum under auto and newton", {
    for (model in c("cpr", "sccs", "clr")) {
        problem <- groupedProblem(model, 2)
        for (prior in list(createPrior("none"), createPrior("normal", variance = 0.5))) {
            reference <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE,
                                         control = createControl(convergenceType = "kkt", tolerance = 1e-10,
                                                                 maxIterations = 1e5, noiseLevel = "silent"))
            objective <- function(fit) -(fit$log_likelihood + fit$log_prior)
            auto <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE,
                                    control = createControl(algorithm = "auto", noiseLevel = "silent"))
            expect_equal(auto$return_flag, "SUCCESS")
            expect_lte(objective(auto) - objective(reference), 1e-6) # the default gap tolerance
            newton <- fitCyclopsModel(problem$data, prior = prior, forceNewObject = TRUE,
                                      control = createControl(algorithm = "newton", noiseLevel = "silent"))
            expect_equal(newton$return_flag, "SUCCESS")
            expect_gte(newton$convergence$newtonIterations, 1)
            expect_equal(coef(newton), coef(reference), tolerance = 1e-8)
        }
    }
})

test_that("a zero-event SCCS exposure is reported as divergent (D5: ILLCONDITIONED at the limit without priors)", {
    problem <- groupedProblem("sccs", 3)
    frame <- problem$frame
    frame$x6 <- as.numeric(frame$y == 0 & frame$x6 == 1) # x6 = 1 only on rows without events
    data <- createCyclopsData(y ~ x1 + x2 + x3 + x4 + x5 + x6 + strata(id), data = frame, time = frame$len,
                              modelType = "sccs")
    for (algorithm in c("ccd", "auto")) {
        fit <- suppressWarnings(fitCyclopsModel(data, forceNewObject = TRUE,
                                                control = createControl(algorithm = algorithm, noiseLevel = "silent")))
        expect_equal(fit$return_flag, "ILLCONDITIONED")
        expect_equal(fit$divergentCoefficients, "x6")
        beta <- coef(fit, ignoreConvergence = TRUE)
        expect_equal(beta[["x6"]], -40)
        expect_true(all(is.finite(beta[paste0("x", 1:5)])))
        expect_true(is.finite(fit$log_likelihood))
        ci <- confint(fit, parm = c("x1", "x6"))
        expect_true(all(is.na(ci["x6", 2:3])) && all(is.finite(ci["x1", 2:3])))
        expect_true(all(is.finite(getCyclopsProfileLogLikelihood(fit, parm = "x6", x = c(-4, -2))$value)))
    }
})
