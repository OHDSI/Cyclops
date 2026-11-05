library("testthat")
library("survival")

# Second-order kernels of the Cox engine (cox and cox_counting on the CPU): gradient, Hessian,
# Hessian-vector product, trial log likelihood and commit, through the internal .cyclopsSecondOrder()

coxProblem <- function(seed, K = 400, p = 6, nstrata = 3, delayed = TRUE, weighted = TRUE, ties = TRUE,
                       sparse = TRUE) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    if (!sparse) X[, 1] <- rnorm(K)
    X[, p] <- round(rnorm(K), 1)
    strata <- sort(sample.int(nstrata, K, replace = TRUE))
    start <- if (delayed) round(runif(K, 0, 5), if (ties) 0 else 6) else rep(0, K)
    stop <- start + if (ties) ceiling(rexp(K, 0.3)) else rexp(K, 0.3) + 1e-3
    event <- rbinom(K, 1, 0.6)
    frame <- data.frame(start, stop, event, strata, X)
    names(frame)[-(1:4)] <- paste0("x", 1:p)
    formula <- as.formula(paste(if (delayed) "Surv(start, stop, event)" else "Surv(stop, event)",
                                "~", paste0("x", 1:p, collapse = " + "), "+ strata(strata)"))
    list(frame = frame, formula = formula, model = if (delayed) "cox_counting" else "cox",
         weights = if (weighted) sample(c(0, 1, 2, 3), K, replace = TRUE) else NULL,
         beta = rnorm(p, 0, 0.3), v = rnorm(p), p = p)
}

# The engine at fixed coefficients beta
fixedAt <- function(problem, beta = problem$beta, threads = 1) {
    data <- createCyclopsData(problem$formula, data = problem$frame, modelType = problem$model)
    fitCyclopsModel(data, startingCoefficients = beta, fixedCoefficients = rep(TRUE, length(beta)),
                    weights = problem$weights, control = createControl(threads = threads))
    data
}

secondOrder <- function(data, v, t = 0) {
    Cyclops:::.cyclopsSecondOrder(data$cyclopsInterfacePtr, NULL, v, t, 1)
}

test_that("Cox gradient, Hessian, Hv and log likelihood match coxph and finite differences", {
    problems <- list(coxProblem(1), coxProblem(2, nstrata = 1), coxProblem(3, delayed = FALSE),
                     coxProblem(4, weighted = FALSE), coxProblem(5, ties = FALSE),
                     coxProblem(6, K = 1000, p = 10, nstrata = 7, sparse = FALSE),
                     coxProblem(7, K = 1000, p = 8, nstrata = 1))
    for (problem in problems) {
        data <- fixedAt(problem)
        so <- secondOrder(data, problem$v)
        expect_true(so$supported)

        frame <- problem$frame
        frame$w <- if (is.null(problem$weights)) 1 else problem$weights
        gold <- suppressWarnings(coxph(problem$formula, data = frame[frame$w != 0, ], weights = w,
                                       ties = "breslow", init = problem$beta, model = TRUE, x = TRUE,
                                       control = coxph.control(iter.max = 0)))
        information <- solve(if (is.null(gold$naive.var)) gold$var else gold$naive.var)
        score <- colSums(residuals(gold, type = "score", weighted = TRUE))

        # derivatives of -log likelihood
        expect_lt(abs(so$trial0 - gold$loglik[1]), 1e-12 * abs(gold$loglik[1]))
        expect_lt(max(abs(-so$gradient - score)), 1e-12 * max(abs(score)))
        expect_lt(max(abs(so$hessian - information)), 1e-12 * max(abs(information)))
        expect_lt(max(abs(so$hv - so$hessian %*% problem$v)), 1e-12 * max(abs(so$hv)))
        expect_equal(so$hessian, t(so$hessian))

        # along v: first and second directional derivatives of the trial log likelihood
        ll <- function(t) secondOrder(data, problem$v, t)$trial
        h <- 1e-5
        first <- (ll(h) - ll(-h)) / (2 * h)
        h <- 1e-3
        second <- (-ll(2 * h) + 16 * ll(h) - 30 * so$trial0 + 16 * ll(-h) - ll(-2 * h)) / (12 * h^2)
        expect_lt(abs(first + sum(so$gradient * problem$v)), 1e-6 * abs(first))
        expect_lt(abs(second + sum(problem$v * so$hv)), 1e-6 * abs(second))

        # a committed step reaches the trial log likelihood, and the state is restored from beta after
        moved <- secondOrder(data, problem$v, 0.3)
        expect_lt(abs(moved$committed - moved$trial), 1e-12 * abs(moved$trial))
        expect_identical(secondOrder(data, problem$v)$trial0, so$trial0)
    }
})

test_that("Cox second-order kernels do not depend on threads", {
    problem <- coxProblem(11, K = 1000, p = 8, nstrata = 4)
    one <- secondOrder(fixedAt(problem, threads = 1), problem$v, 0.2)
    two <- secondOrder(fixedAt(problem, threads = 2), problem$v, 0.2)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(two[[name]], one[[name]])
    }
})

test_that("Fisher information from the Hessian block matches the per-pair legacy engine", {
    set.seed(21)
    K <- 600
    frame <- data.frame(time = ceiling(rexp(K, 0.2)), status = rbinom(K, 1, 0.7),
                        x1 = rnorm(K), x2 = rbinom(K, 1, 0.4), x3 = rbinom(K, 1, 0.1),
                        x4 = round(runif(K), 1),
                        stratum = sort(sample.int(5, K, replace = TRUE)))
    formula <- Surv(time, status) ~ x1 + x2 + x3 + x4 + strata(stratum)
    data <- createCyclopsData(formula, data = frame, modelType = "cox")
    fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.02),
                           control = createControl(noiseLevel = "silent"))
    beta <- coef(fit)
    expect_true(any(beta == 0)) # Laplace zeros are part of the information

    legacyData <- createCyclopsData(formula, data = frame, modelType = "cox")
    legacy <- fitCyclopsModel(legacyData, startingCoefficients = beta, fixedCoefficients = rep(TRUE, length(beta)),
                              control = createControl(fitter = "legacy", noiseLevel = "silent"))
    expect_identical(coef(legacy), beta)

    information <- Cyclops:::.cyclopsGetFisherInformation(data$cyclopsInterfacePtr, NULL)
    legacyInformation <- Cyclops:::.cyclopsGetFisherInformation(legacyData$cyclopsInterfacePtr, NULL)
    expect_lt(max(abs(information - legacyInformation)), 1e-10 * max(abs(legacyInformation)))

    # with a fixed coefficient, on a subset of covariates
    fixed <- fitCyclopsModel(data, startingCoefficients = beta, fixedCoefficients = c(TRUE, FALSE, TRUE, FALSE),
                             control = createControl(noiseLevel = "silent"))
    legacyFixed <- fitCyclopsModel(legacyData, startingCoefficients = beta, fixedCoefficients = c(TRUE, FALSE, TRUE, FALSE),
                                   control = createControl(fitter = "legacy", noiseLevel = "silent"))
    subset <- c("x2", "x4")
    ses <- suppressWarnings(getSEs(fixed, subset))
    legacySes <- suppressWarnings(getSEs(legacyFixed, subset))
    expect_lt(max(abs(ses - legacySes)), 1e-10 * max(abs(legacySes)))
})

test_that("Fisher information of cox_counting with integer weights matches coxph", {
    problem <- coxProblem(31, K = 800, p = 5)
    data <- fixedAt(problem)
    information <- Cyclops:::.cyclopsGetFisherInformation(data$cyclopsInterfacePtr, NULL)
    frame <- problem$frame
    frame$w <- problem$weights
    gold <- coxph(problem$formula, data = frame[frame$w != 0, ], weights = w,
                  ties = "breslow", init = problem$beta, control = coxph.control(iter.max = 0))
    expect_lt(max(abs(information - solve(gold$var))), 1e-10 * max(abs(information)))
})

test_that("vcov and getSEs give NA for aliased coefficients and match the reduced model elsewhere", {
    set.seed(41)
    K <- 800
    frame <- data.frame(time = ceiling(rexp(K, 0.2)), status = rbinom(K, 1, 0.7),
                        x1 = rnorm(K), x2 = rbinom(K, 1, 0.4), x3 = round(runif(K), 1))
    frame$x1copy <- frame$x1
    control <- createControl(tolerance = 1e-10, noiseLevel = "silent")
    full <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ x1 + x2 + x3 + x1copy, data = frame,
                                              modelType = "cox"), control = control)
    reduced <- coxph(Surv(time, status) ~ x1 + x2 + x3, data = frame, ties = "breslow")

    expect_warning(v <- vcov(full), "2 aliased coefficients")
    expect_true(all(is.na(v[c("x1", "x1copy"), ])))
    expect_true(all(is.na(v[, c("x1", "x1copy")])))
    expect_equal(v[c("x2", "x3"), c("x2", "x3")], vcov(reduced)[c("x2", "x3"), c("x2", "x3")], tolerance = 1e-6)

    expect_warning(ses <- getSEs(full, c("x1", "x2", "x3", "x1copy")), "aliased")
    expect_true(all(is.na(ses[c("x1", "x1copy")])))
    expect_equal(unname(ses[c("x2", "x3")]), unname(sqrt(diag(vcov(reduced)))[c("x2", "x3")]), tolerance = 1e-6)
})

test_that("vcov of an identified model equals the inverse information", {
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    fit <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog, data = lung2,
                                             modelType = "cox"),
                           control = createControl(tolerance = 1e-10, noiseLevel = "silent"))
    information <- Cyclops:::.cyclopsGetFisherInformation(fit$cyclopsData$cyclopsInterfacePtr, NULL)
    kappa <- kappa(information, exact = TRUE)
    expect_silent(v <- vcov(fit))
    expect_lt(max(abs(unname(v) - solve(information))), 1e-12 * kappa * max(abs(v)))
})
