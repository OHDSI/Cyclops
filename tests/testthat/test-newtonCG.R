library("testthat")
library("survival")

# Newton plan Phase 4: Newton-CG, the dense / CG choice, threaded H

cgProblem <- function(seed, K = 3000, p = 60) {
    set.seed(seed)
    X <- matrix(rbinom(K * p, 1, 0.2), K, p)
    X[, 1:3] <- rnorm(3 * K)
    eta <- drop(X %*% rnorm(p, 0, 0.2))
    frame <- data.frame(start = round(runif(K, 0, 3)), y = rbinom(K, 1, 0.7), X)
    frame$time <- frame$start + ceiling(rexp(K, 0.05 * exp(eta)))
    list(frame = frame, formula = as.formula(paste("Surv(start, time, y) ~", paste0("X", 1:p, collapse = " + "))))
}

withSolver <- function(solver, expr) {
    old <- Sys.getenv("CYCLOPS_NEWTON_SOLVER", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("CYCLOPS_NEWTON_SOLVER") else Sys.setenv(CYCLOPS_NEWTON_SOLVER = old))
    Sys.setenv(CYCLOPS_NEWTON_SOLVER = solver)
    force(expr)
}

cgFit <- function(problem, prior = createPrior("normal", variance = 0.5), ...) {
    fitCyclopsModel(createCyclopsData(problem$formula, data = problem$frame, modelType = "cox_counting"),
                    prior = prior, control = createControl(noiseLevel = "silent", ...), warnings = FALSE)
}

test_that("Newton-CG reaches the dense Newton optimum with a certified gap", {
    problem <- cgProblem(1)
    dense <- withSolver("dense", cgFit(problem, algorithm = "newton", tolerance = 1e-10, convergenceType = "gap"))
    cg <- withSolver("cg", cgFit(problem, algorithm = "newton", tolerance = 1e-10, convergenceType = "gap"))
    expect_equal(cg$convergence$solver, "cg")
    expect_equal(dense$convergence$solver, "dense")
    expect_true(cg$convergence$certified)
    expect_gt(cg$convergence$hessianVectorProducts, 0)
    expect_equal(coef(cg), coef(dense), tolerance = 1e-7)
})

test_that("Newton-CG and threaded H give the same fit for any number of threads", {
    problem <- cgProblem(2)
    for (solver in c("cg", "dense")) {
        one <- withSolver(solver, cgFit(problem, algorithm = "newton", threads = 1))
        four <- withSolver(solver, cgFit(problem, algorithm = "newton", threads = 4))
        expect_identical(coef(four), coef(one))
        expect_identical(four$iterations, one$iterations)
    }
})

test_that("Newton-CG without a normal prior on every coefficient stops without a certificate", {
    problem <- cgProblem(3, p = 30)
    fit <- withSolver("cg", cgFit(problem, prior = createPrior("none"), algorithm = "newton"))
    expect_equal(fit$return_flag, "SUCCESS")
    expect_false(fit$convergence$certified)
    gold <- coxph(problem$formula, data = problem$frame, ties = "breslow")
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1e-5)
})

test_that("Hessian kernels with G |F|^2 above 2^31", {
    skip_on_cran()
    set.seed(4)
    K <- 60000
    p <- 200 # 60,000 event-time groups x 200^2 > 2^31
    X <- matrix(rnorm(K * p), K, p)
    frame <- data.frame(time = seq_len(K) + runif(K, 0, 0.5), y = 1, X)
    formula <- as.formula(paste("Surv(time, y) ~", paste0("X", 1:p, collapse = " + ")))
    data <- createCyclopsData(formula, data = frame, modelType = "cox")
    fitCyclopsModel(data, startingCoefficients = rep(0.01, p), fixedCoefficients = rep(TRUE, p))
    v <- rnorm(p)
    so <- Cyclops:::.cyclopsSecondOrder(data$cyclopsInterfacePtr, NULL, v, 0, 1)
    expect_true(so$supported)
    expect_lt(max(abs(so$hessian %*% v - so$hv)), 1e-10 * max(abs(so$hv)))
})
