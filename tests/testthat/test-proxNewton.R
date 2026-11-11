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
                        control = createControl(threads = threads, maxIterations = 300, noiseLevel = "silent"))
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

    fits <- lapply(list(c(0.5, 0.3), c(0.7, -0.3)), function(start) { # both columns nonzero: same and opposite signs
        fitCyclopsModel(sparseLogisticData(X, y), prior = prior, control = control, weights = weights,
                        startingCoefficients = c(0, start, rep(0, p - 2)), forceNewObject = TRUE, warnings = FALSE)
    })
    for (fit in fits) {
        b <- coef(fit)
        others <- setdiff(names(b), c("1", "2"))
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(b[["1"]] + b[["2"]], reference[["1"]], tolerance = 1e-8)
        expect_equal(b[others], reference[others], tolerance = 1e-8)
        expect_gt(fit$convergence$rankDeficientFaceRounds, 0) # both columns reached the face
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
