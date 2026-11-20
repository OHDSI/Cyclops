library("testthat")

context("test-threadsOneFit.R")

# Threads inside one lr or plr fit: a large column's coordinate-descent gradient and update, and the second-order
# kernels' passes over rows, run on the threads with results the same bit for bit on any number of threads. The
# designs are just large enough for those paths: plr columns of at least 4096 subjects, lr fits of more than two
# blocks of 8192 rows

# 5000 subjects with up to 3 intervals (implicit rows, or explicit ones with row offsets), 15 indicators (two in about
# 95% of subjects), a linear and a 0/1 time basis, and an interaction of the first dense indicator with the linear basis
threadPlr <- function(link = "logit", explicit = FALSE) {
    set.seed(31)
    n <- 5000
    subjects <- data.frame(rowId = 1:n, entry = 1, exit = sample(1:3, n, TRUE), y = rbinom(n, 1, 0.2))
    X <- matrix(rbinom(n * 15, 1, 0.1), n, 15)
    X[, 1:2] <- rbinom(2 * n, 1, 0.95)
    covariates <- data.frame(rowId = row(X)[X == 1], covariateId = col(X)[X == 1], covariateValue = 1)
    timeBasis <- rbind(data.frame(interval = 1:3, covariateId = 1000, covariateValue = (1:3) / 3),
                       data.frame(interval = 1, covariateId = 1001, covariateValue = 1))
    map <- data.frame(covariateId = 1, timeCovariateId = 1000, interactionId = 2000)
    outcomes <- subjects
    if (explicit) {
        outcomes <- convertToLongOutcome(transform(subjects, y = 0))
        outcomes$y <- rbinom(nrow(outcomes), 1, 0.1)
        outcomes$offset <- rnorm(nrow(outcomes), 0, 0.3)
    }
    convertToCyclopsData(outcomes, covariates, modelType = "plr", timeBasis = timeBasis, timeEffectMap = map,
                         link = link, quiet = TRUE)
}

# 20000 rows, 30 indicators (two in about 90% of rows) and an offset
threadLr <- function() {
    set.seed(32)
    n <- 20000
    X <- matrix(rbinom(n * 30, 1, 0.05), n, 30)
    X[, 1:2] <- rbinom(2 * n, 1, 0.9)
    outcomes <- data.frame(rowId = 1:n, y = rbinom(n, 1, 0.2), offset = rnorm(n, 0, 0.3))
    covariates <- data.frame(rowId = row(X)[X == 1], covariateId = col(X)[X == 1], covariateValue = 1)
    convertToCyclopsData(outcomes, covariates, modelType = "lr", addIntercept = TRUE, quiet = TRUE)
}

threadFit <- function(data, threads, algorithm = "auto", weights = NULL, maxIterations = 50, exclude = c(0, 1000, 1001),
                      variance = 0.05) {
    fitCyclopsModel(data, prior = createPrior("laplace", variance = variance, exclude = exclude), weights = weights,
                    control = createControl(algorithm = algorithm, maxIterations = maxIterations, threads = threads,
                                            noiseLevel = "silent"),
                    forceNewObject = TRUE)
}

expectSameFit <- function(a, b) {
    expect_identical(coef(a, ignoreConvergence = TRUE), coef(b, ignoreConvergence = TRUE))
    expect_identical(a$log_likelihood, b$log_likelihood)
}

plr <- threadPlr()

test_that("plr: one fit on 1 and 3 threads is the same", {
    expectSameFit(threadFit(plr, 3), threadFit(plr, 1))
})

test_that("plr: coordinate descent on 1 and 3 threads is the same, with the same certified zero steps", {
    # a small variance: the dense indicators stay at 0, and their visits are certified from Hessians computed on
    # the threads
    Sys.setenv(CYCLOPS_ZVERIFY = "1") # every certified visit checked, and counted
    on.exit(Sys.unsetenv("CYCLOPS_ZVERIFY"))
    one <- threadFit(plr, 1, algorithm = "ccd", variance = 0.001)
    three <- threadFit(plr, 3, algorithm = "ccd", variance = 0.001)
    expectSameFit(three, one)
    expect_gt(one$zeroCertificates, 0)
    expect_identical(three$zeroCertificates, one$zeroCertificates)
})

lr <- threadLr()

test_that("lr: one fit on 1 and 3 threads is the same", {
    expectSameFit(threadFit(lr, 3, exclude = 0), threadFit(lr, 1, exclude = 0))
})

test_that("lr: one fit on 2 and 7 threads (uneven parts) is the same", {
    expectSameFit(threadFit(lr, 7, exclude = 0), threadFit(lr, 2, exclude = 0))
})

test_that("plr under the cloglog link: coordinate descent on 1 and 3 threads is the same", {
    data <- threadPlr(link = "cloglog")
    expectSameFit(threadFit(data, 3, algorithm = "ccd", maxIterations = 30),
                  threadFit(data, 1, algorithm = "ccd", maxIterations = 30))
})

test_that("plr with zero weights and row offsets: 1 and 3 threads give the same fit", {
    data <- threadPlr(explicit = TRUE)
    set.seed(33)
    weights <- rbinom(5000, 1, 0.9)
    expectSameFit(threadFit(data, 3, weights = weights), threadFit(data, 1, weights = weights))
})

test_that("lr with zero weights: 1 and 3 threads give the same fit", {
    set.seed(34)
    weights <- rbinom(20000, 1, 0.9)
    expectSameFit(threadFit(lr, 3, weights = weights, exclude = 0), threadFit(lr, 1, weights = weights, exclude = 0))
})
