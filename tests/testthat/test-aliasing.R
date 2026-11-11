library("testthat")
library("survival")

# Newton plan Phase 5 (2.8, D6): exact-duplicate columns merged into one coefficient for the fit

aliasProblem <- function(seed = 1, K = 2000, p = 10) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.3), K, p)
    X[, 1] <- rnorm(K)
    eta <- drop(X %*% rnorm(p, 0, 0.3))
    frame <- data.frame(start = round(runif(K, 0, 3)), y = rbinom(K, 1, 0.7), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(eta)))
    frame$D1 <- frame$X2
    frame$D2 <- frame$X2
    frame$D3 <- frame$X5
    frame
}
aliasFormula <- as.formula(paste("Surv(start, time, y) ~", paste0("X", 1:10, collapse = " + "), "+ D1 + D2 + D3"))

aliasFit <- function(frame, prior, ...) {
    fitCyclopsModel(createCyclopsData(aliasFormula, data = frame, modelType = "cox_counting"), prior = prior,
                    control = createControl(noiseLevel = "silent", ...), warnings = FALSE)
}
objective <- function(fit) -(fit$log_likelihood + fit$log_prior)

test_that("T12: merged fits reach the unmerged optimum for normal, Laplace, elastic-net and no priors", {
    frame <- aliasProblem()
    for (prior in list(createPrior("normal", variance = 0.5), createPrior("laplace", variance = 0.1),
                       createPrior("elasticnet", variance = 0.1, laplaceFraction = 0.5), createPrior("none"))) {
        unmerged <- aliasFit(frame, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5)
        merged <- aliasFit(frame, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5,
                           mergeDuplicates = TRUE)
        expect_equal(merged$aliasedColumns, list(c("X2", "D1", "D2"), c("X5", "D3")))
        expect_lt(abs(objective(merged) - objective(unmerged)), 1e-8 * abs(objective(unmerged)))
        b <- coef(merged)
        if (prior$priorType == "normal" || isTRUE(prior$laplaceFraction < 1)) { # the equal split, the unique maximum
            expect_equal(unname(b[c("D1", "D2")]), rep(b[["X2"]], 2))
            expect_equal(b, coef(unmerged), tolerance = 1e-6)
        } else { # all on the representative
            expect_equal(unname(b[c("D1", "D2", "D3")]), c(0, 0, 0))
            expect_equal(b[["X2"]], sum(coef(unmerged)[c("X2", "D1", "D2")]), tolerance = 1e-5)
        }
    }
})

test_that("merging is on by default only on auto and newton paths", {
    frame <- aliasProblem(2)
    prior <- createPrior("normal", variance = 0.5)
    expect_length(aliasFit(frame, prior)$aliasedColumns, 2) # the default algorithm is "auto" (Newton plan D1)
    expect_null(aliasFit(frame, prior, algorithm = "ccd")$aliasedColumns)
    expect_length(aliasFit(frame, prior, algorithm = "auto")$aliasedColumns, 2)
    expect_null(aliasFit(frame, prior, algorithm = "auto", mergeDuplicates = FALSE)$aliasedColumns)
})

test_that("duplicates with different priors or a fixed member are not merged", {
    frame <- aliasProblem(3)
    variances <- rep(0.5, 13)
    variances[11] <- 2 # D1 has another prior than X2 and D2
    prior <- createPrior(rep("normal", 13), variance = variances)
    fit <- aliasFit(frame, prior, mergeDuplicates = TRUE)
    expect_equal(fit$aliasedColumns, list(c("X2", "D2"), c("X5", "D3")))

    data <- createCyclopsData(aliasFormula, data = frame, modelType = "cox_counting")
    fixed <- fitCyclopsModel(data, prior = createPrior("normal", variance = 0.5),
                             control = createControl(mergeDuplicates = TRUE, noiseLevel = "silent"),
                             fixedCoefficients = c(rep(FALSE, 12), TRUE), startingCoefficients = rep(0, 13))
    expect_equal(fixed$aliasedColumns, list(c("X2", "D1", "D2")))
    expect_equal(coef(fixed)[["D3"]], 0)
})

test_that("names of merged groups keep each group's members and the order of the groups", {
    set.seed(11)
    n <- 1500
    X <- matrix(rbinom(n * 24, 1, 0.2), n, 24)
    X <- cbind(X, X[, 1:12], X[, 3]) # 12 groups (more than 9: names must not sort as text), the third with 3 members
    ids <- 4e9 + 1000 * seq_len(ncol(X)) + 210 # above the 32-bit range
    entries <- which(X != 0, arr.ind = TRUE)
    data <- convertToCyclopsData(data.frame(rowId = seq_len(n), y = rbinom(n, 1, 0.3)),
                                 data.frame(rowId = entries[, 1], covariateId = ids[entries[, 2]], covariateValue = 1),
                                 modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    fit <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 0.1, exclude = 0),
                           control = createControl(noiseLevel = "silent"), warnings = FALSE)
    expected <- lapply(1:12, function(k) as.character(bit64::as.integer64(ids[c(k, 24 + k, if (k == 3) 37)])))
    expect_equal(fit$aliasedColumns, expected)
    expect_equal(names(coef(fit))[1], "(Intercept)")
})

test_that("getCovariateIds() leaves out the offset and returns integer64", {
    counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
    outcome <- gl(3, 1, 9)
    treatment <- gl(3, 3)
    exposure <- c(2, 3, 4, 2, 3, 4, 2, 3, 4)
    data <- createCyclopsData(counts ~ outcome + treatment + offset(log(exposure)), modelType = "pr")
    ids <- getCovariateIds(data)
    expect_s3_class(ids, "integer64")
    expect_equal(as.character(ids), as.character(1:5))
})

test_that("a fold merges columns equal on its training rows, whatever their order", {
    # Two columns equal except on one row: in the fold that holds that row out they are duplicates for the fit. Merged
    # there (and split evenly), the held-out prediction of that row, and so the cross-validation score, does not depend
    # on which of the two columns comes first; unmerged, a Laplace fold fit put the mass on either
    set.seed(123)
    n <- 400
    x <- rbinom(n, 1, 0.4)
    z <- matrix(rbinom(n * 5, 1, 0.3), n, 5)
    y <- rbinom(n, 1, plogis(-1 + 1.5 * x + drop(z %*% rep(0.3, 5))))
    twin <- x
    twin[which(x == 1 & y == 1)[1]] <- 0
    make <- function(xId, twinId) {
        cov <- rbind(data.frame(rowId = which(x == 1), covariateId = xId, covariateValue = 1),
                     data.frame(rowId = which(twin == 1), covariateId = twinId, covariateValue = 1),
                     data.frame(rowId = row(z)[z == 1], covariateId = 10 + col(z)[z == 1], covariateValue = 1))
        cov <- cov[order(cov$rowId, cov$covariateId), ]
        convertToCyclopsData(data.frame(rowId = 1:n, y = y), cov, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
    }
    cv <- function(data) getCrossValidationInfo(fitCyclopsModel(data, prior = createPrior("laplace", useCrossValidation = TRUE),
        control = createControl(cvType = "grid", lowerLimit = 0.1, upperLimit = 10, gridSteps = 3, fold = 5, seed = 1,
                                noiseLevel = "silent"), warnings = FALSE))
    a <- cv(make(1, 2))
    b <- cv(make(2, 1))
    expect_equal(b$point, a$point)
    expect_equal(b$ordinate, a$ordinate, tolerance = 1e-8)

    # The same for weights given to a fit: the row of weight 0 does not enter it, so the two columns are one group,
    # reported as an equal split
    w <- as.numeric(x == twin)
    fit <- fitCyclopsModel(make(1, 2), prior = createPrior("laplace", variance = 1), weights = w,
                           control = createControl(noiseLevel = "silent"), warnings = FALSE)
    expect_equal(fit$aliasedColumns, list(c("1", "2")))
    expect_equal(unname(coef(fit)["1"]), unname(coef(fit)["2"]))
    expect_true(coef(fit)["1"] != 0)
})

test_that("columns equal on the rows of positive weight stay apart where rows of weight 0 still count", {
    # Conditional logistic regression: a row of weight 0 stays in its stratum's denominator, so a = (1, 0) and
    # b = (1, 1) are not interchangeable there although they agree on the row of weight 1
    outcomes <- data.frame(rowId = 1:4, stratumId = c(1, 1, 2, 2), y = c(1, 0, 1, 0))
    covariates <- data.frame(rowId = c(1, 3, 1, 2, 3), covariateId = c(1, 1, 2, 2, 2), covariateValue = 1)
    covariates$stratumId <- outcomes$stratumId[covariates$rowId]
    covariates <- covariates[order(covariates$stratumId, covariates$rowId, covariates$covariateId), ]
    data <- convertToCyclopsData(outcomes, covariates, modelType = "clr", quiet = TRUE)
    fit <- function(merge) fitCyclopsModel(data, prior = createPrior("normal", variance = 1), weights = c(1, 0, 1, 1),
                                           control = createControl(mergeDuplicates = merge, noiseLevel = "silent"),
                                           forceNewObject = TRUE, warnings = FALSE)
    merged <- fit(TRUE)
    expect_null(merged$aliasedColumns)
    expect_equal(coef(merged), coef(fit(FALSE)))

    # Columns without a prior: a group's total could diverge, with a limit set over all rows, where the columns differ
    outcomes <- data.frame(rowId = 1:4, y = c(1, 1, 0, 0))
    make <- function(aId, bId) {
        cov <- data.frame(rowId = c(1, 2, 4, 1, 2), covariateId = c(aId, aId, aId, bId, bId), covariateValue = c(1, 1, 1e6, 1, 1))
        convertToCyclopsData(outcomes, cov[order(cov$rowId, cov$covariateId), ], modelType = "lr", quiet = TRUE)
    }
    flat <- function(data) fitCyclopsModel(data, prior = createPrior("none"), weights = c(1, 1, 1, 0),
                                           control = createControl(noiseLevel = "silent"), warnings = FALSE)
    a <- flat(make(1, 2))
    b <- flat(make(2, 1))
    expect_null(a$aliasedColumns) # separated with a prior "none" on every coefficient: ILLCONDITIONED, by design
    expect_equal(unname(coef(b, ignoreConvergence = TRUE)[c("2", "1")]), unname(coef(a, ignoreConvergence = TRUE)[c("1", "2")]))
})
