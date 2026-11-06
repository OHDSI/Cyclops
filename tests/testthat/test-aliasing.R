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

test_that("T12: merged fits reach the unmerged optimum for normal, Laplace and no priors", {
    frame <- aliasProblem()
    for (prior in list(createPrior("normal", variance = 0.5), createPrior("laplace", variance = 0.1),
                       createPrior("none"))) {
        unmerged <- aliasFit(frame, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5)
        merged <- aliasFit(frame, prior, convergenceType = "kkt", tolerance = 1e-10, maxIterations = 1e5,
                           mergeDuplicates = TRUE)
        expect_equal(merged$aliasedColumns, list(c("X2", "D1", "D2"), c("X5", "D3")))
        expect_lt(abs(objective(merged) - objective(unmerged)), 1e-8 * abs(objective(unmerged)))
        b <- coef(merged)
        if (prior$priorType == "normal") { # the equal split, the unique maximum
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
