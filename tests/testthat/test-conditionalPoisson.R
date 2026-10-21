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
