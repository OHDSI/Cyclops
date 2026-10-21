library("testthat")
library("survival")
library("gnm")

context("test-conditionalPoisson.R")

test_that("Check simple SCCS as conditional logistic regression", {
#     source("helper-conditionalPoisson.R")
    tolerance <- 1E-6
    gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                          data = Cyclops::oxford)

    dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                  data = Cyclops::oxford,
                                  modelType = "clr")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("none"))
    expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
    expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)
})

test_that("Check simple SCCS as indicator conditional logistic regression", {
	#     source("helper-conditionalPoisson.R")
	tolerance <- 1E-6
	gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
												data = Cyclops::oxford)

	dataPtr <- createCyclopsData(event ~ strata(indiv) + offset(loginterval),
															 indicatorFormula = ~ exgr + agegr,
															 data = Cyclops::oxford,
															 modelType = "clr")
	cyclopsFit <- fitCyclopsModel(dataPtr,
																prior = createPrior("none"))
	expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
	expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)
})

test_that("Check simple SCCS as sparse conditional logistic regression", {
	#     source("helper-conditionalPoisson.R")
	tolerance <- 1E-6
	gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
												data = Cyclops::oxford)

	dataPtr <- createCyclopsData(event ~ strata(indiv) + offset(loginterval),
															 sparseFormula = ~ exgr + agegr,
															 data = Cyclops::oxford,
															 modelType = "clr")
	cyclopsFit <- fitCyclopsModel(dataPtr,
																prior = createPrior("none"))
	expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
	expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)
})

test_that("Check simple SCCS as conditional Poisson regression", {
#     source("helper-conditionalPoisson.R")
    tolerance <- 1E-3
    gold.cp <- gnm(event ~ exgr + agegr + offset(loginterval),
                   family = poisson, eliminate = indiv,
                   data = Cyclops::oxford)

    dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                  data = Cyclops::oxford,
                                  modelType = "cpr")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("none"))

    expect_equal(coef(cyclopsFit)[1:2], coef(gold.cp)[1:2], tolerance = tolerance)
    expect_equal(confint(cyclopsFit, c("exgr1","agegr2"))[,2:3],
                 confint(gold.cp), tolerance = tolerance)
})

test_that("Check simple SCCS as SCCS", {
#     source("helper-conditionalPoisson.R")
    tolerance <- 1E-6
    gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                          data = Cyclops::oxford)

    dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv), time = Cyclops::oxford$interval,
                                  data = Cyclops::oxford,
                                  modelType = "sccs")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("none"))
    expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
    expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)
})

test_that("Check conditional Poisson and SCCS with indicator and few-valued sparse columns", {
    # Grouped models update offsExpXBeta by exp(delta) for an indicator column and through a table of
    # exp(delta * value) for a sparse column with few distinct values (x2); x3 has too many values
    tolerance <- 1E-8
    set.seed(123)
    nStrata <- 300
    stratum <- rep(seq_len(nStrata), sample(3:12, nStrata, replace = TRUE))
    n <- length(stratum)
    x1 <- rbinom(n, 1, 0.3)
    x2 <- sample(c(0.25, 0.5, 1, 2), n, replace = TRUE) * rbinom(n, 1, 0.6)
    x3 <- rnorm(n) * rbinom(n, 1, 0.5)
    time <- sample(1:30, n, replace = TRUE)
    y <- rpois(n, time / 30 * exp(0.5 * x1 - 0.3 * x2 + 0.4 * x3 + rnorm(nStrata)[stratum]))
    data <- data.frame(y, stratum, x1, x2, x3, time)
    control <- createControl(tolerance = 1E-12, noiseLevel = "silent")
    covariates <- c("x1", "x2", "x3")

    dataPtr <- createCyclopsData(y ~ strata(stratum), indicatorFormula = ~ x1, sparseFormula = ~ x2 + x3,
                                 data = data, modelType = "cpr")
    expect_equal(as.character(summary(dataPtr)[covariates, "type"]),
                 c("indicator", "sparse", "sparse"))
    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"), control = control)
    gold.cp <- gnm(y ~ x1 + x2 + x3, family = poisson, eliminate = factor(stratum),
                   data = data, tolerance = 1E-12, verbose = FALSE)
    expect_equal(coef(cyclopsFit)[covariates], coef(gold.cp)[covariates], tolerance = tolerance)

    dataPtr <- createCyclopsData(y ~ strata(stratum), indicatorFormula = ~ x1, sparseFormula = ~ x2 + x3,
                                 time = data$time, data = data, modelType = "sccs")
    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"), control = control)
    gold.sccs <- gnm(y ~ x1 + x2 + x3 + offset(log(time)), family = poisson, eliminate = factor(stratum),
                     data = data, tolerance = 1E-12, verbose = FALSE)
    expect_equal(coef(cyclopsFit)[covariates], coef(gold.sccs)[covariates], tolerance = tolerance)
})

test_that("Conditional Poisson stratum denominators stay exact when they shrink", {
    # x has a large common level, so a start at beta = 5 multiplies every stratum denominator by about
    # exp(50); incremental updates used to cancel down to wrong or negative denominators on the way back
    set.seed(123)
    data <- data.frame(stratum = rep(1:100, each = 4), x = 10 + runif(400), time = runif(400, 10, 100))
    data$y <- rpois(400, data$time * exp(0.5 * data$x - 5) / 100)
    formula <- y ~ x + strata(stratum) + offset(log(time))
    gold <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"))
    for (start in c(3, 4, 5, 8)) {
        fit <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"),
                               startingCoefficients = start, warnings = FALSE)
        expect_equal(fit$return_flag, "SUCCESS")
        expect_equal(fit$estimation$estimate, coef(gold)[["x"]], tolerance = 1E-6)
    }
})

test_that("Conditional Poisson fit does not stop while steps are cut short by their bounds", {
    formula <- event ~ exgr + agegr + strata(indiv) + offset(loginterval)
    gold <- fitCyclopsModel(createCyclopsData(formula, data = Cyclops::oxford, modelType = "cpr"))
    # The first cycle moves each coefficient by at most initialBound, so the objective barely changes
    small <- fitCyclopsModel(createCyclopsData(formula, data = Cyclops::oxford, modelType = "cpr"),
                             control = createControl(initialBound = 1E-6))
    expect_equal(small$return_flag, "SUCCESS")
    expect_equal(coef(small), coef(gold), tolerance = 1E-6)

    # Each ILLCONDITIONED retry from a distant start shrinks the bound by 10, until a first cycle that
    # hardly moves looked converged
    set.seed(123)
    data <- data.frame(stratum = rep(1:100, each = 4), x = rbinom(400, 1, 0.5), time = runif(400, 10, 100))
    data$y <- rpois(400, data$time * exp(0.5 * data$x) / 100)
    formula <- y ~ x + strata(stratum) + offset(log(time))
    gold <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"))
    far <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"), startingCoefficients = 40,
                           control = createControl(maxBoundCount = 8), warnings = FALSE)
    expect_true(far$return_flag != "SUCCESS" || abs(far$estimation$estimate - coef(gold)[["x"]]) < 1E-6)
})

test_that("Conditional Poisson profile refits keep the profiled coefficient through retries", {
    formula <- event ~ exgr + agegr + strata(indiv) + offset(loginterval)
    fit <- fitCyclopsModel(createCyclopsData(formula, data = Cyclops::oxford, modelType = "cpr"))
    mode <- coef(fit)[["exgr1"]]
    # The refit at the mode starts from the one at 50, fails, and retries from the starting values;
    # the retry used to reset the profiled coefficient to 0 too
    prof <- getCyclopsProfileLogLikelihood(fit, parm = "exgr1", x = c(mode, 50, 60, 70))
    expect_equal(prof$value[prof$point == mode], as.numeric(logLik(fit)), tolerance = 1E-6)
    # exp(800) overflows, so every attempt fails; the value must not be the one at 0
    far <- getCyclopsProfileLogLikelihood(fit, parm = "exgr1", x = 800)
    expect_false(isTRUE(far$value > as.numeric(logLik(fit)) - 100))
})
