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
    # the legacy engine at the same coefficients: each refit stops at its own tolerance, and from some
    # starting points the default fit ends ~5e-9 from the optimum, which moves the standard errors more than 1e-10
    legacyFixed <- fitCyclopsModel(legacyData, startingCoefficients = coef(fixed),
                                   fixedCoefficients = rep(TRUE, length(beta)),
                                   control = createControl(fitter = "legacy", noiseLevel = "silent"))
    expect_identical(coef(legacyFixed), coef(fixed))
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

test_that("Cox parallel hessianBlock (n >= 32, dense-at-end) and fullGradient paths are bitwise thread-invariant", {
    skip_on_cran() # threads
    set.seed(51)
    K <- 3000
    p <- 80
    # Mix sparse binary indicators, an all-zero column, and dense continuous columns at the right end of F
    # (columns 71:80 sit in the last 15 positions where rangeStart must clamp splits to avoid false sharing)
    X <- matrix(rbinom(K * p, 1, 0.08), K, p)
    X[, 40] <- 0
    X[, 71:80] <- matrix(rnorm(K * 10), K, 10)
    strata <- sort(sample.int(5, K, replace = TRUE))
    start <- round(runif(K, 0, 3))
    stop <- start + ceiling(rexp(K, 0.25))
    event <- rbinom(K, 1, 0.55)
    weights <- sample(c(0, 1, 2), K, replace = TRUE, prob = c(0.15, 0.65, 0.2))
    beta <- rnorm(p, 0, 0.05)
    beta[40] <- 0
    v <- rnorm(p)

    # Sparse data frame so columns 1:70 are stored sparsely/indicator and 71:80 densely
    nz <- which(X != 0, arr.ind = TRUE)
    covDf <- data.frame(stratumId = strata[nz[, 1]], rowId = nz[, 1],
                        covariateId = nz[, 2], covariateValue = X[nz])
    covDf <- covDf[order(covDf$stratumId, covDf$rowId, covDf$covariateId), ]
    outDf <- data.frame(stratumId = strata, rowId = seq_len(K), y = event, time = stop,
                        startTime = start, weights = weights)
    makeFixed <- function(threads) {
        cd <- convertToCyclopsData(outDf, covDf, modelType = "cox_counting", quiet = TRUE)
        ids <- as.integer(getCovariateIds(cd))
        fitCyclopsModel(cd, startingCoefficients = beta[ids], fixedCoefficients = rep(TRUE, length(ids)),
                        control = createControl(threads = threads, noiseLevel = "silent"))
        list(data = cd, ids = ids)
    }
    f1 <- makeFixed(1)
    f4 <- makeFixed(4)
    vOrd <- v[f1$ids]

    # Full p = 79 non-empty set: exercises parallel colRangeS2 (n >= 32) with dense-at-end columns
    so1 <- secondOrder(f1$data, vOrd, 0.15)
    so4 <- secondOrder(f4$data, vOrd, 0.15)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(so4[[name]], so1[[name]])
    }

    # Explicit K >= 16384 & totalEntries >= 65536 (plus an explicit all-zero dense column via formula):
    # exercises parallel fillR, parallel dotCols, and parallel colRangeS2 together
    set.seed(53)
    Kbig <- 17000
    Xbig <- matrix(rbinom(Kbig * 26, 1, 0.12), Kbig, 26)
    Xbig <- cbind(Xbig, rep(0, Kbig), matrix(rnorm(Kbig * 10), Kbig, 10)) # p = 37, col 27 is all-zero, cols 28:37 dense
    dfBig <- data.frame(time = ceiling(rexp(Kbig, 0.2)), event = rbinom(Kbig, 1, 0.5), Xbig)
    names(dfBig)[-(1:2)] <- paste0("z", seq_len(ncol(Xbig)))
    formBig <- as.formula(paste("Surv(time, event) ~", paste0("z", seq_len(ncol(Xbig)), collapse = " + ")))
    betaBig <- rnorm(ncol(Xbig), 0, 0.05)
    betaBig[27] <- 0
    vBig <- rnorm(ncol(Xbig))
    makeBig <- function(threads) {
        cd <- createCyclopsData(formBig, data = dfBig, modelType = "cox")
        fitCyclopsModel(cd, startingCoefficients = betaBig, fixedCoefficients = rep(TRUE, length(betaBig)),
                        control = createControl(threads = threads, noiseLevel = "silent"))
        cd
    }
    big1 <- secondOrder(makeBig(1), vBig, 0.1)
    big4 <- secondOrder(makeBig(4), vBig, 0.1)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(big4[[name]], big1[[name]])
    }
    expect_equal(big1$gradient[27], 0)
    expect_equal(big1$hessian[27, ], rep(0, ncol(Xbig)))

    # 2-column sparse subset F: exercises fullGradient's low-nnz fast path (totalEntries < K / 2)
    subIds <- bit64::as.integer64(f1$ids[1:2])
    sub1 <- Cyclops:::.cyclopsSecondOrder(f1$data$cyclopsInterfacePtr, subIds, vOrd[1:2], 0.15, 1)
    sub4 <- Cyclops:::.cyclopsSecondOrder(f4$data$cyclopsInterfacePtr, subIds, vOrd[1:2], 0.15, 1)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(sub4[[name]], sub1[[name]])
    }
    expect_equal(sub1$gradient, so1$gradient[1:2], tolerance = 1e-12)
    expect_equal(sub1$hessian, so1$hessian[1:2, 1:2], tolerance = 1e-12)

    # Exact n = 32 boundary subset (triggers parallel colRangeS2 at n == 32 with maxSplit = 17)
    sub32Ids <- bit64::as.integer64(c(f1$ids[1:22], tail(f1$ids, 10)))
    sub32_1 <- Cyclops:::.cyclopsSecondOrder(f1$data$cyclopsInterfacePtr, sub32Ids, rep(0.1, 32), 0.1, 1)
    sub32_4 <- Cyclops:::.cyclopsSecondOrder(f4$data$cyclopsInterfacePtr, sub32Ids, rep(0.1, 32), 0.1, 1)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(sub32_4[[name]], sub32_1[[name]])
    }

    # Verify full gradient & Hessian against survival::coxph
    used <- weights != 0
    dfGold <- data.frame(start = start[used], stop = stop[used], event = event[used],
                         strata = strata[used], w = weights[used], X[used, f1$ids, drop = FALSE])
    formGold <- as.formula(paste("Surv(start, stop, event) ~",
                                 paste0("X", seq_along(f1$ids), collapse = " + "), "+ strata(strata)"))
    gold <- suppressWarnings(coxph(formGold, data = dfGold, weights = w, ties = "breslow",
                                   init = beta[f1$ids], model = TRUE, x = TRUE,
                                   control = coxph.control(iter.max = 0)))
    information <- solve(if (is.null(gold$naive.var)) gold$var else gold$naive.var)
    expect_lt(max(abs(-so1$gradient - colSums(residuals(gold, type = "score", weighted = TRUE)))),
              1e-11 * max(abs(so1$gradient)))
    expect_lt(max(abs(so1$hessian - information)), 1e-11 * max(abs(so1$hessian)))

    # Fine-Gray (hasTails = TRUE) with p = 70: exercises parallel colRangeS2 + rowHazard tails
    statusFg <- sample(c(0L, 1L, 2L), K, replace = TRUE, prob = c(0.45, 0.3, 0.25))
    tFg <- ceiling(rexp(K, 0.1))
    censT <- sort(unique(tFg[statusFg == 0L]))
    if (length(censT) > 40L) censT <- censT[round(seq(1, length(censT), length.out = 40L))]
    censDf <- data.frame(fineGrayGroup = 1L, time = censT, survival = exp(-0.4 * seq_along(censT) / length(censT)))
    Xfg <- X[, 1:70, drop = FALSE]
    Xfg[, 65:70] <- matrix(rnorm(K * 6), K, 6)
    nzFg <- which(Xfg != 0, arr.ind = TRUE)
    covFg <- data.frame(stratumId = 1L, rowId = nzFg[, 1], covariateId = nzFg[, 2], covariateValue = Xfg[nzFg])
    covFg <- covFg[order(covFg$rowId, covFg$covariateId), ]
    outFg <- data.frame(stratumId = 1L, rowId = seq_len(K), startTime = 0, time = tFg,
                        y = as.numeric(statusFg == 1L), fineGrayGroup = ifelse(statusFg == 2L, 1L, 0L))
    makeFg <- function(threads) {
        cd <- convertToCyclopsData(outFg, covFg, modelType = "cox_counting", fineGrayCensoring = censDf, quiet = TRUE)
        ids <- as.integer(getCovariateIds(cd))
        fitCyclopsModel(cd, startingCoefficients = beta[ids], fixedCoefficients = rep(TRUE, length(ids)),
                        control = createControl(threads = threads, noiseLevel = "silent"))
        list(data = cd, ids = ids)
    }
    fg1 <- makeFg(1)
    fg4 <- makeFg(4)
    fgSo1 <- secondOrder(fg1$data, v[fg1$ids], 0.1)
    fgSo4 <- secondOrder(fg4$data, v[fg4$ids], 0.1)
    for (name in c("gradient", "hessian", "hv", "trial0", "trial", "committed")) {
        expect_identical(fgSo4[[name]], fgSo1[[name]])
    }
})

test_that("Cox profile likelihood and confint with threads restore fit state cleanly", {
    skip_on_cran() # threads
    problem <- coxProblem(52, K = 1000, p = 20, nstrata = 3, delayed = FALSE, weighted = FALSE)
    cd1 <- createCyclopsData(problem$formula, data = problem$frame, modelType = "cox")
    cd4 <- createCyclopsData(problem$formula, data = problem$frame, modelType = "cox")
    fit1 <- fitCyclopsModel(cd1, prior = createPrior("none"),
                            control = createControl(threads = 1, noiseLevel = "silent"))
    fit4 <- fitCyclopsModel(cd4, prior = createPrior("none"),
                            control = createControl(threads = 4, noiseLevel = "silent"))
    expect_identical(coef(fit4), coef(fit1))

    ci1 <- confint(fit1, parm = c("x1", "x2"), threads = 1)
    ci4 <- confint(fit4, parm = c("x1", "x2"), threads = 2)
    expect_equal(ci4, ci1, tolerance = 1e-6)

    prof1 <- getCyclopsProfileLogLikelihood(fit1, parm = "x1", bounds = c(-0.5, 0.5))
    prof4 <- getCyclopsProfileLogLikelihood(fit4, parm = "x1", bounds = c(-0.5, 0.5))
    expect_equal(prof4$value, prof1$value, tolerance = 1e-6)

    # Subsequent refit on the same multi-threaded data object still matches serial exactly
    refit1 <- fitCyclopsModel(cd1, prior = createPrior("laplace", 0.1),
                              control = createControl(threads = 1, noiseLevel = "silent"))
    refit4 <- fitCyclopsModel(cd4, prior = createPrior("laplace", 0.1),
                              control = createControl(threads = 4, noiseLevel = "silent"))
    expect_identical(coef(refit4), coef(refit1))
})
