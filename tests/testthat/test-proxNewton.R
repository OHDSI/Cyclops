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

sparseLogisticData <- function(X, y) {
    entries <- which(X != 0, arr.ind = TRUE)
    convertToCyclopsData(data.frame(rowId = seq_along(y), y = y),
                         data.frame(rowId = entries[, 1], covariateId = entries[, 2], covariateValue = 1),
                         modelType = "lr", addIntercept = TRUE, quiet = TRUE)
}

test_that("the threads of one fit do not take prox-Newton away", {
    skip_on_cran()
    set.seed(21)
    n <- 4000
    p <- 1400
    X <- matrix(rbinom(n * p, 1, 0.08), n, p)
    y <- rbinom(n, 1, plogis(-1.5 + 0.3 * rowSums(X[, 1:40]) - 0.3 * rowSums(X[, 41:80])))
    data <- sparseLogisticData(X, y)
    prior <- createPrior("laplace", variance = 5, exclude = 0)
    # The memory check reads the value of threads, not the cores available, so 8 needs no 8-core machine. The working set
    # (over 1,200 columns) fits one 256 MB budget, but not one divided among 8 threads
    fit <- function(threads) {
        fitCyclopsModel(data, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                        control = createControl(threads = threads, maxIterations = 300, noiseLevel = "silent",
                                                newtonMemoryBudget = 256))
    }
    one <- fit(1)
    eight <- fit(8)
    expect_equal(eight$return_flag, "SUCCESS")
    expect_gt(eight$convergence$newtonIterations, 0)
    expect_equal(eight$convergence$fallbacks, 0) # prox-Newton did not give up as the working set grew
    expect_gt(sum(coef(eight)[-1] != 0), 1182) # over the limit that 8 threads would have imposed
    expect_equal(eight$log_likelihood + eight$log_prior, one$log_likelihood + one$log_prior, tolerance = 1e-8)
})

test_that("a face with columns equal on the rows in use reaches the optimum without an eigendecomposition", {
    set.seed(31)
    n <- 1500
    p <- 30
    X <- matrix(rbinom(n * p, 1, 0.2), n, p)
    X[, 2] <- X[, 1]
    row <- which(X[, 1] == 1)[1]
    X[row, 2] <- 0L # the two columns differ only on this row ...
    weights <- rep(1, n)
    weights[row] <- 0 # ... which the fit does not use, as in a cross-validation fold
    y <- rbinom(n, 1, plogis(-1.2 + 0.9 * X[, 1] - 0.6 * X[, 3] + 0.5 * X[, 4]))
    prior <- createPrior("laplace", variance = 100, exclude = 0)
    control <- createControl(algorithm = "newton", tolerance = 1e-9, maxIterations = 500, mergeDuplicates = FALSE,
                             noiseLevel = "silent")

    used <- weights > 0
    single <- X[used, ]
    single[, 2] <- 0L # one column of the pair: the identified fit
    reference <- coef(fitCyclopsModel(sparseLogisticData(single, y[used]), prior = prior, control = control,
                                      forceNewObject = TRUE, warnings = FALSE))

    # Same-sign start stays on the singular face; a small opposite-sign start (0.7, -0.3) can have
    # inner CD eliminate col 2 to exact 0 before the face solve (platform-dependent at the 1-ULP
    # level), whereas (5.0, -4.5) keeps both opposite-sign columns active into the face solve
    cases <- list(
        list(start = c(0.5, 0.3), expectSingular = TRUE),
        list(start = c(0.7, -0.3), expectSingular = FALSE),
        list(start = c(5.0, -4.5), expectSingular = TRUE)
    )
    for (case in cases) {
        fit <- fitCyclopsModel(sparseLogisticData(X, y), prior = prior, control = control, weights = weights,
                               startingCoefficients = c(0, case$start, rep(0, p - 2)), forceNewObject = TRUE,
                               warnings = FALSE)
        b <- coef(fit)
        others <- setdiff(names(b), c("1", "2"))
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(b[["1"]] + b[["2"]], reference[["1"]], tolerance = 1e-8)
        expect_equal(b[others], reference[others], tolerance = 1e-8)
        if (case$expectSingular) {
            expect_gt(fit$convergence$rankDeficientFaceRounds, 0) # both columns reached the face
        }
        expect_equal(fit$convergence$eigenFaceRounds, 0)
    }
})

test_that("a singular face follows its null direction to the sign boundary, however far that is", {
    # Identical columns of opposite sign under a negligible penalty: the model is almost flat along the null direction, but
    # the optimum is (0, 0), a long way from the start
    covariates <- data.frame(rowId = rep(1:4, 2), covariateId = rep(1:2, each = 4), covariateValue = rep(c(1, 1, -1, -1), 2))
    data <- convertToCyclopsData(data.frame(rowId = 1:4, y = c(0, 1, 0, 1)), covariates, modelType = "lr",
                                 addIntercept = FALSE, quiet = TRUE)
    fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 2e18), startingCoefficients = c(1e6, -1e6),
                           forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(algorithm = "newton", tolerance = 1e-9, maxIterations = 200,
                                                   mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(fit$return_flag, "SUCCESS")
    expect_equal(unname(coef(fit)), c(0, 0))
    expect_lte(fit$convergence$newtonIterations, 5)
})

test_that("a face holding a column and a scaled copy of it is treated as singular", {
    # Column 2 is 1e4 times column 1 (not merged: only identical columns are): plain Cholesky of the face succeeds, with
    # a last pivot of rounding size that min/max pivot ratio alone accepts
    x1 <- c(1, 2, -1, -2)
    covariates <- data.frame(rowId = rep(1:4, 2), covariateId = rep(1:2, each = 4), covariateValue = c(x1, 1e4 * x1))
    data <- convertToCyclopsData(data.frame(rowId = 1:4, y = c(0, 1, 0, 1)), covariates, modelType = "lr",
                                 addIntercept = FALSE, quiet = TRUE)
    fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 2e18), startingCoefficients = c(1e8, -1e4),
                           forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(algorithm = "newton", tolerance = 1e-9, maxIterations = 200,
                                                   mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(fit$return_flag, "SUCCESS")
    expect_equal(unname(coef(fit)), c(0, 0))
    expect_gt(fit$convergence$rankDeficientFaceRounds, 0)
    expect_lte(fit$convergence$newtonIterations, 5)
})

test_that("prox-Newton reaches the optimum when a column is the sum of others", {
    # 30 disjoint indicators and their class, the union, as a column: faces holding all of them are singular, the
    # inner descent's signs settle while mass still moves to the class, and many violators enter at once
    set.seed(21)
    n <- 3000
    member <- sample(0:30, n, replace = TRUE) # 0: no member of the class
    rows <- lapply(1:30, function(j) which(member == j))
    other <- matrix(rbinom(n * 20, 1, 0.2), n, 20)
    y <- rbinom(n, 1, plogis(-1.5 + 0.8 * (member > 0) + drop(other[, 1:3] %*% c(0.5, -0.4, 0.3))))
    covariates <- rbind(
        data.frame(rowId = unlist(rows), covariateId = rep(1:30, lengths(rows)), covariateValue = 1),
        data.frame(rowId = which(member > 0), covariateId = 100, covariateValue = 1),
        data.frame(rowId = row(other)[other == 1], covariateId = 200 + col(other)[other == 1], covariateValue = 1))
    covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                                 quiet = TRUE)
    fit <- function(control) fitCyclopsModel(data, prior = createPrior("laplace", variance = 1, exclude = 0),
                                             control = control, forceNewObject = TRUE, warnings = FALSE)
    newton <- fit(createControl(algorithm = "newton", noiseLevel = "silent"))
    ccd <- fit(createControl(algorithm = "ccd", convergenceType = "kkt", tolerance = 1e-8, maxIterations = 100000,
                             noiseLevel = "silent"))
    expect_equal(newton$return_flag, "SUCCESS")
    expect_gt(newton$convergence$faceRounds, 0)
    expect_equal(newton$log_likelihood + newton$log_prior, ccd$log_likelihood + ccd$log_prior, tolerance = 1e-6)
})

test_that("Newton fits and cross-validation are the same bit for bit on 1 and 4 threads", {
    # The logistic Hessian block splits its columns into ranges, one owner per entry, each a sum over rows in increasing
    # order: its result does not depend on the threads. 600 columns (a third with values other than 1) and a weak
    # penalty give blocks of several ranges; 8 threads over 3 folds give each fold's block 2
    skip_on_cran() # threads
    set.seed(31)
    n <- 4000
    p <- 600
    X <- matrix(rbinom(n * p, 1, runif(p, 0.01, 0.4)[col(matrix(0, n, p))]), n, p)
    X[, 1:200] <- X[, 1:200] * runif(n * 200, 0.5, 2)
    y <- rbinom(n, 1, plogis(-1 + drop(X[, 1:20] %*% rnorm(20, 0, 0.5))))
    nz <- which(X != 0, arr.ind = TRUE)
    covariates <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
    covariates <- covariates[order(covariates$rowId, covariates$covariateId), ]
    data <- convertToCyclopsData(data.frame(rowId = 1:n, y = y), covariates, modelType = "lr", addIntercept = TRUE,
                                 quiet = TRUE)
    fit <- function(data, prior, threads, cv = FALSE) {
        control <- if (cv) createControl(algorithm = "newton", threads = threads, cvType = "auto", fold = 3, seed = 1,
                                         noiseLevel = "silent") else
            createControl(algorithm = "newton", threads = threads, noiseLevel = "silent")
        fitCyclopsModel(data, prior = prior, control = control, forceNewObject = TRUE, warnings = FALSE)
    }
    for (prior in list(createPrior("laplace", variance = 5, exclude = 0),
                       createPrior("normal", variance = 0.5, exclude = 0))) {
        one <- fit(data, prior, 1)
        four <- fit(data, prior, 4)
        expect_identical(coef(one), coef(four))
        expect_identical(one$log_likelihood, four$log_likelihood)
    }
    prior <- createPrior("laplace", useCrossValidation = TRUE, exclude = 0)
    one <- fit(data, prior, 1, cv = TRUE)
    eight <- fit(data, prior, 8, cv = TRUE)
    expect_identical(one$variance, eight$variance)
    expect_identical(coef(one), coef(eight))

    # dense continuous columns (the formula interface stores them densely)
    Z <- as.data.frame(matrix(rnorm(2000 * 100), 2000, 100))
    Z$y <- rbinom(2000, 1, plogis(drop(as.matrix(Z[, 1:5]) %*% rep(0.3, 5))))
    dense <- createCyclopsData(as.formula(paste("y ~", paste(names(Z)[1:100], collapse = " + "))), data = Z,
                               modelType = "lr")
    one <- fit(dense, createPrior("normal", variance = 0.5, exclude = 1), 1)
    four <- fit(dense, createPrior("normal", variance = 0.5, exclude = 1), 4)
    expect_identical(coef(one), coef(four))
})

test_that("newtonMemoryBudget bounds prox-Newton; a tiny budget leaves the fit on coordinate descent", {
    expect_equal(createControl()$newtonMemoryBudget, 1024)
    expect_equal(createControl(newtonMemoryBudget = 2)$newtonMemoryBudget, 2)
    expect_error(createControl(newtonMemoryBudget = 0))
    set.seed(1)
    X <- matrix(rnorm(300 * 5), 300, 5)
    frame <- data.frame(y = rbinom(300, 1, plogis(X[, 1] - X[, 2])), X)
    data <- createCyclopsData(y ~ X1 + X2 + X3 + X4 + X5, data = frame, modelType = "lr")
    fit <- function(...) fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.1, exclude = 1),
                                         control = createControl(algorithm = "auto", noiseLevel = "silent", ...),
                                         forceNewObject = TRUE)
    default <- expect_no_warning(fit())
    # 10 bytes: below 24 |W|^2 for any active set
    expect_warning(tiny <- fit(newtonMemoryBudget = 1e-5), "more than newtonMemoryBudget = 1e-05 MB")
    expect_equal(default$algorithm, "newton")
    expect_gt(default$convergence$newtonIterations, 0)
    expect_true(is.na(default$convergence$budgetBytes))
    expect_equal(tiny$algorithm, "ccd")
    expect_equal(tiny$convergence$newtonIterations, 0)
    expect_gt(tiny$convergence$budgetBytes, 10)
    expect_no_warning(fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.1, exclude = 1), warnings = FALSE,
                                      control = createControl(algorithm = "auto", noiseLevel = "silent", newtonMemoryBudget = 1e-5),
                                      forceNewObject = TRUE))
    expect_equal(coef(tiny), coef(default), tolerance = 1e-6)
})

test_that("coordinate descent under \"gap\" does not warn about newtonMemoryBudget; logistic \"auto\" with a Laplace prior does", {
    design <- function(rho) {
        set.seed(1)
        x <- sqrt(rho) * rnorm(1000) + sqrt(1 - rho) * matrix(rnorm(1000 * 60), 1000)
        createCyclopsData(rbinom(1000, 1, plogis(x[, 1:5] %*% rep(0.4, 5))) ~ x, modelType = "lr")
    }
    fit <- function(data, ...) fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.1, exclude = 1),
                                               control = createControl(noiseLevel = "silent", ...), forceNewObject = TRUE)
    # an explicit "ccd" with "gap" runs the kkt-guided coordinate descent, which takes no Newton steps: a budget too small
    # for any changes nothing and records nothing
    for (rho in c(0, 0.99)) {
        data <- design(rho)
        ample <- expect_no_warning(fit(data, algorithm = "ccd", convergenceType = "gap"))
        tiny <- expect_no_warning(fit(data, algorithm = "ccd", convergenceType = "gap", newtonMemoryBudget = 1e-5))
        expect_equal(tiny$convergence$newtonIterations, 0)
        expect_true(is.na(tiny$convergence$budgetBytes))
        expect_identical(coef(tiny), coef(ample))
    }
    # logistic "auto" with a Laplace prior keeps the Newton-first path: a budget that refuses its Newton steps warns
    correlated <- design(0.99)
    expect_gt(fit(correlated)$convergence$newtonIterations, 0)
    expect_warning(refused <- fit(correlated, newtonMemoryBudget = 1e-5), "more than newtonMemoryBudget")
    expect_equal(refused$convergence$newtonIterations, 0)
    expect_gt(refused$convergence$budgetBytes, 10)
})

test_that("a duality gap that Newton steps no longer reduce ends the fit uncertified", {
    # Dense columns with a few values in (0, 1] beside binary ones (a CCW censoring model's design): at the optimum the
    # dual point's gap stays near 3e-9, above tau_eff = 1e-9, while prox-Newton is stationary. The fit used to run
    # 50-step Newton phases between coordinate descent until maxIterations (3,000 iterations, 50 s)
    set.seed(1)
    n <- 20000
    p <- 300
    prevalence <- runif(p, 0.05, 0.95)
    rows <- lapply(seq_len(p), function(j) which(runif(n) < prevalence[j]))
    dense <- sapply(c(11, 4, 7), function(k) sample(seq_len(k), n, replace = TRUE) / k)
    covariates <- data.frame(rowId = c(unlist(rows), rep(seq_len(n), 3)),
                             covariateId = c(rep(seq_len(p), lengths(rows)), rep(p + 1:3, each = n)),
                             covariateValue = c(rep(1, sum(lengths(rows))), as.vector(dense)))
    beta <- c(rnorm(p, 0, 0.15), 1.5, -1, 0.8)
    lp <- 0.8 + as.vector(tapply(covariates$covariateValue * beta[covariates$covariateId], covariates$rowId, sum)) -
        0.5 * sum(prevalence * beta[1:p])
    data <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = rbinom(n, 1, plogis(lp))), covariates,
                                 modelType = "lr", quiet = TRUE)
    fit <- function(tolerance) {
        fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.01, exclude = 0), forceNewObject = TRUE,
                        control = createControl(convergenceType = "gap", tolerance = tolerance, noiseLevel = "silent",
                                                maxIterations = 3000))
    }
    certified <- fit(1e-8)
    expect_equal(certified$convergence$certificate, "dual")
    stalled <- fit(1e-9)
    expect_equal(stalled$return_flag, "SUCCESS")
    expect_equal(stalled$convergence$certificate, "none")
    expect_equal(stalled$convergence$fallbacks, 0)
    expect_lt(stalled$convergence$newtonIterations, 50)
    expect_gt(stalled$convergence$gapEstimate, 1e-9)
    expect_equal(coef(stalled), coef(certified), tolerance = 1e-5)
})

test_that("blocked solveSemidefinite crosses nb = 64 panel boundaries accurately on LR and Cox", {
    skip_on_cran()
    for (pBase in c(75L, 145L)) {
        set.seed(100 + pBase)
        n <- 2500
        X <- matrix(rbinom(n * pBase, 1, 0.25), n, pBase)
        # Add 2 exact duplicates + 1 linear sum -> active face strictly crosses nb = 64 and 2*nb = 128
        X <- cbind(X, X[, 1], X[, 2], X[, 3] + X[, 4])
        eta <- -0.8 + drop(X[, 1:12] %*% rep(c(0.35, -0.3), 6))
        yLr <- rbinom(n, 1, plogis(eta))
        nz <- which(X != 0, arr.ind = TRUE)
        covDf <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
        covDf <- covDf[order(covDf$rowId, covDf$covariateId), ]
        cdLr <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = yLr), covDf,
                                     modelType = "lr", addIntercept = TRUE, quiet = TRUE)
        priorLr <- createPrior("laplace", variance = 5, exclude = 0)
        newtLr <- fitCyclopsModel(cdLr, prior = priorLr, forceNewObject = TRUE, warnings = FALSE,
                                  control = createControl(algorithm = "newton", convergenceType = "gap",
                                                          tolerance = 1e-8, mergeDuplicates = FALSE,
                                                          noiseLevel = "silent"))
        ccdLr <- fitCyclopsModel(cdLr, prior = priorLr, forceNewObject = TRUE, warnings = FALSE,
                                 control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                         tolerance = 1e-8, maxIterations = 50000,
                                                         mergeDuplicates = FALSE, noiseLevel = "silent"))
        expect_equal(newtLr$return_flag, "SUCCESS")
        expect_gt(sum(coef(newtLr) != 0), if (pBase == 75L) 64L else 128L)
        expect_gt(newtLr$convergence$rankDeficientFaceRounds, 0)
        expect_equal(newtLr$convergence$eigenFaceRounds, 0)
        expect_lte(objective(newtLr) - objective(ccdLr), 1e-6 + 1e-9 * abs(objective(ccdLr)))
    }

    # Cox across nb = 64 boundary with duplicates + sum (active face > 64)
    set.seed(265)
    n <- 2000
    X <- matrix(rbinom(n * 75, 1, 0.25), n, 75)
    X <- cbind(X, X[, 1], X[, 2], X[, 3] + X[, 4])
    eta <- drop(X[, 1:10] %*% rep(c(0.3, -0.25), 5))
    tObs <- ceiling(rexp(n, 0.05 * exp(eta)))
    yCox <- rbinom(n, 1, 0.7)
    nz <- which(X != 0, arr.ind = TRUE)
    covDf <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = X[nz])
    covDf <- covDf[order(covDf$rowId, covDf$covariateId), ]
    cdCox <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = yCox, time = tObs), covDf,
                                  modelType = "cox", quiet = TRUE)
    priorCox <- createPrior("laplace", variance = 5)
    newtCox <- fitCyclopsModel(cdCox, prior = priorCox, forceNewObject = TRUE, warnings = FALSE,
                               control = createControl(algorithm = "newton", convergenceType = "gap",
                                                       tolerance = 1e-8, mergeDuplicates = FALSE,
                                                       noiseLevel = "silent"))
    ccdCox <- fitCyclopsModel(cdCox, prior = priorCox, forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                      tolerance = 1e-8, maxIterations = 50000,
                                                      mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(newtCox$return_flag, "SUCCESS")
    expect_gt(sum(coef(newtCox) != 0), 64L)
    expect_gt(newtCox$convergence$rankDeficientFaceRounds, 0)
    expect_equal(newtCox$convergence$eigenFaceRounds, 0)
    expect_lte(objective(newtCox) - objective(ccdCox), 1e-6 + 1e-9 * abs(objective(ccdCox)))
})

test_that("kept rank-revealing FaceFactor handles promotion, tiny-curvature columns, scaling, rounding floor, and maxDependent", {
    # 1) Collinear sum + duplicates + tiny-curvature column + scaled copy: exclude col 2, xScaled, and xTiny so they
    # stay active in FaceFactor (guards M1 tiny-curvature thrash and M2 >1e4 scaling ping-pong)
    set.seed(71)
    n <- 1500
    X <- matrix(rbinom(n * 20, 1, 0.25), n, 20)
    xSum <- X[, 1] + X[, 2] + X[, 3]
    xDup <- X[, 1]
    xScaled <- 1e5 * X[, 2]
    xTiny <- rep(0, n); xTiny[1] <- 1e-7 # H(a,a) ~ 1e-14 << 1e-10 * maxDiag
    Xfull <- cbind(X, xSum, xDup, xScaled, xTiny)
    y <- rbinom(n, 1, plogis(-1 + 0.7 * X[, 1] - 0.6 * X[, 2] + 0.5 * X[, 3] - 0.4 * X[, 4]))
    nz <- which(Xfull != 0, arr.ind = TRUE)
    covDf <- data.frame(rowId = nz[, 1], covariateId = nz[, 2], covariateValue = Xfull[nz])
    covDf <- covDf[order(covDf$rowId, covDf$covariateId), ]
    cd <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = y), covDf,
                               modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    prior <- createPrior("laplace", variance = 2, exclude = c(0, 2, 23, 24))
    start <- c(-1, -0.5, 0.5, -0.4, rep(0, 17), 0.4, 0.3, -1e-5, 0.2)
    newt <- fitCyclopsModel(cd, prior = prior, startingCoefficients = start, forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(algorithm = "newton", convergenceType = "gap",
                                                    tolerance = 1e-8, mergeDuplicates = FALSE, noiseLevel = "silent"))
    ccd <- fitCyclopsModel(cd, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                   tolerance = 1e-8, maxIterations = 50000,
                                                   mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(newt$return_flag, "SUCCESS")
    expect_gt(newt$convergence$rankDeficientFaceRounds, 0)
    expect_equal(newt$convergence$eigenFaceRounds, 0)
    expect_equal(newt$convergence$fallbacks, 0)
    expect_lte(objective(newt) - objective(ccd), 1e-6 + 1e-9 * abs(objective(ccd)))

    # 2) Round 0 full-rank, duplicate violator enters later via addRankRevealing: rankDeficientFaceRounds must be > 0
    X2 <- cbind(X[, 1:10], X[, 1])
    y2 <- rbinom(n, 1, plogis(-0.5 + 1.2 * X[, 1] - 0.5 * X[, 2]))
    nz2 <- which(X2 != 0, arr.ind = TRUE)
    cov2 <- data.frame(rowId = nz2[, 1], covariateId = nz2[, 2], covariateValue = X2[nz2])
    cov2 <- cov2[order(cov2$rowId, cov2$covariateId), ]
    cd2 <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = y2), cov2,
                                modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    fit2 <- fitCyclopsModel(cd2, prior = createPrior("laplace", variance = 5, exclude = 0),
                            startingCoefficients = c(-0.5, 0.1, rep(0, 10)),
                            forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(algorithm = "newton", convergenceType = "gap",
                                                    tolerance = 1e-8, mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(fit2$return_flag, "SUCCESS")
    expect_gt(fit2$convergence$rankDeficientFaceRounds, 0)
    expect_equal(fit2$convergence$eigenFaceRounds, 0)

    # 3) Rounding-floor face (uniformly large-scale covariates where H * z rounding exceeds kktTol):
    # refinement stall retry must break after at most 1 pivoted retry per outer Hessian, not oscillate for 200 rounds
    Xu <- 1e5 * (X[, 1:10] - 0.5)
    nzBig <- which(Xu != 0, arr.ind = TRUE)
    covBig <- data.frame(rowId = nzBig[, 1], covariateId = nzBig[, 2], covariateValue = Xu[nzBig])
    covBig <- covBig[order(covBig$covariateId, covBig$rowId), ]
    cdBig <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = y2), covBig,
                                  modelType = "lr", addIntercept = FALSE, quiet = TRUE)
    fitBig <- fitCyclopsModel(cdBig, prior = createPrior("laplace", variance = 1e8),
                              forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(algorithm = "newton", convergenceType = "gap",
                                                      tolerance = 1e-6, maxIterations = 15, noiseLevel = "silent"))
    expect_gt(fitBig$convergence$hessians, 0)
    expect_lte(fitBig$convergence$faceRounds, 6 * fitBig$convergence$hessians)

    # 4) High rank deficiency exceeding maxDependent (40 collinear copies in p = 60) falls back cleanly
    X3 <- cbind(X[, 1:20], matrix(rep(X[, 1], 40), n, 40))
    nz3 <- which(X3 != 0, arr.ind = TRUE)
    cov3 <- data.frame(rowId = nz3[, 1], covariateId = nz3[, 2], covariateValue = X3[nz3])
    cov3 <- cov3[order(cov3$rowId, cov3$covariateId), ]
    cd3 <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = y), cov3,
                                modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    newt3 <- fitCyclopsModel(cd3, prior = createPrior("laplace", variance = 2, exclude = 0),
                             forceNewObject = TRUE, warnings = FALSE,
                             control = createControl(algorithm = "newton", convergenceType = "gap",
                                                     tolerance = 1e-8, mergeDuplicates = FALSE, noiseLevel = "silent"))
    ccd3 <- fitCyclopsModel(cd3, prior = createPrior("laplace", variance = 2, exclude = 0),
                            forceNewObject = TRUE, warnings = FALSE,
                            control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                    tolerance = 1e-8, maxIterations = 50000,
                                                    mergeDuplicates = FALSE, noiseLevel = "silent"))
    expect_equal(newt3$return_flag, "SUCCESS")
    expect_gt(newt3$convergence$rankDeficientFaceRounds, 0)
    expect_equal(newt3$convergence$eigenFaceRounds, 0)
    expect_lte(objective(newt3) - objective(ccd3), 1e-6 + 1e-9 * abs(objective(ccd3)))
})

test_that("SCCS with unpenalized collinear splines converges cleanly", {
    set.seed(81)
    nPatients <- 400
    periods <- 6L
    K <- nPatients * periods
    pid <- rep(seq_len(nPatients), each = periods)
    age <- rep(seq_len(periods), times = nPatients)
    season <- ((pid + age) %% 3L) + 1L
    exp1 <- rbinom(K, 1, 0.25)
    exp2 <- rbinom(K, 1, 0.20)
    dur <- runif(K, 0.5, 2.0)
    lp <- 0.6 * exp1 - 0.3 * exp2 + 0.15 * age + 0.1 * season
    y <- integer(K)
    for (i in seq_len(nPatients)) {
        idx <- ((i - 1L) * periods + 1L):(i * periods)
        w <- dur[idx] * exp(lp[idx])
        y[idx] <- as.integer(rmultinom(1, 1L + rbinom(1, 1, 0.2), w / sum(w)))
    }
    covs <- list()
    cid <- 0L
    addInd <- function(rows) { cid <<- cid + 1L; covs[[cid]] <<- data.frame(rowId = rows, covariateId = cid, covariateValue = 1) }
    addInd(which(exp1 == 1L)); addInd(which(exp2 == 1L))
    # Full partition-of-unity age (6 cols) + season (3 cols): collinear unpenalized splines (rank deficiency = 2)
    for (a in 1:6) addInd(which(age == a))
    for (s in 1:3) addInd(which(season == s))
    splineIds <- 3:cid
    covDf <- do.call(rbind, covs)
    covDf$stratumId <- pid[covDf$rowId]
    covDf <- covDf[order(covDf$covariateId, covDf$stratumId, covDf$rowId), ]
    outDf <- data.frame(stratumId = pid, rowId = seq_len(K), y = y, time = dur)
    cd <- convertToCyclopsData(outDf, covDf, modelType = "sccs", addIntercept = FALSE, quiet = TRUE)

    for (priorType in c("laplace", "normal")) {
        prior <- createPrior(priorType, 0.2, exclude = splineIds)
        newt <- fitCyclopsModel(cd, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                control = createControl(algorithm = "newton", convergenceType = "gap",
                                                        tolerance = 1e-8, mergeDuplicates = FALSE,
                                                        noiseLevel = "silent"))
        ccd <- fitCyclopsModel(cd, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                               control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                       tolerance = 1e-8, maxIterations = 50000,
                                                       mergeDuplicates = FALSE, noiseLevel = "silent"))
        expect_equal(newt$return_flag, "SUCCESS")
        expect_equal(newt$convergence$fallbacks, 0)
        if (priorType == "laplace") {
            expect_gt(newt$convergence$rankDeficientFaceRounds, 0)
        } else {
            expect_gt(newt$convergence$droppedDirections, 0)
        }
        expect_equal(newt$convergence$eigenFaceRounds, 0)
        expect_lte(objective(newt) - objective(ccd), 1e-6 + 1e-9 * abs(objective(ccd)))
    }
})

test_that("unpenalized hDelta floor avoids warm-up damping and stays safe under separation and small initialBound", {
    # 1) Far starting intercept + Laplace covariates converges quickly and accurately
    set.seed(91)
    n <- 2000
    X <- matrix(rbinom(n * 30, 1, 0.2), n, 30)
    y <- rbinom(n, 1, plogis(-3.5 + 0.6 * X[, 1] - 0.5 * X[, 2]))
    cd <- sparseLogisticData(X, y)
    prior <- createPrior("laplace", variance = 0.2, exclude = 0)
    fit <- fitCyclopsModel(cd, prior = prior, startingCoefficients = c(0.0, rep(0, 30)),
                           forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(algorithm = "newton", convergenceType = "gap",
                                                   tolerance = 1e-8, noiseLevel = "silent"))
    ccd <- fitCyclopsModel(cd, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                           control = createControl(algorithm = "ccd", convergenceType = "kkt",
                                                   tolerance = 1e-8, maxIterations = 50000, noiseLevel = "silent"))
    expect_equal(fit$return_flag, "SUCCESS")
    expect_lte(fit$convergence$newtonIterations, 10)
    expect_equal(coef(fit), coef(ccd), tolerance = 1e-5)

    # 2) Small initialBound is still respected cleanly
    fitSmall <- fitCyclopsModel(cd, prior = prior, forceNewObject = TRUE, warnings = FALSE,
                                control = createControl(algorithm = "newton", convergenceType = "gap",
                                                        tolerance = 1e-8, initialBound = 1e-2, noiseLevel = "silent"))
    expect_equal(fitSmall$return_flag, "SUCCESS")
    expect_equal(coef(fitSmall), coef(ccd), tolerance = 1e-5)

    # 3) 2D separation on excluded unpenalized covariates + Laplace covariates stays finite
    Xsep <- X
    Xsep[, 1] <- as.integer(y == 1)
    cdSep <- sparseLogisticData(Xsep, y)
    fitSep <- fitCyclopsModel(cdSep, prior = createPrior("laplace", variance = 0.2, exclude = c(0, 1)),
                              forceNewObject = TRUE, warnings = FALSE,
                              control = createControl(algorithm = "auto", convergenceType = "gap",
                                                      tolerance = 1e-6, noiseLevel = "silent"))
    expect_true(all(is.finite(coef(fitSep, ignoreConvergence = TRUE))))
})
