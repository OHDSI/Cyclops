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

