library(testthat)

test_that("Small CLR using MM", {
  library(ConditionedMM)
  library(survival)

  tolerance <- 1E-6

  gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                        data = Cyclops::oxford)

  dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                               data = Cyclops::oxford, method = "debug",
                               modelType = "clr")

  cyclopsFit <- fitCyclopsModel(dataPtr, forceNewObject = TRUE,
                                prior = createPrior("none"))
#   expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
#   expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)

  cyclopsMM <- fitCyclopsModel(dataPtr, forceNewObject = TRUE,
                               prior = createPrior("none"),
                               control = createControl(algorithm = "mm", maxIterations = 100,
                                                       tolerance = 1E-8,
                                                       noiseLevel = "noisy"))

  coef(cyclopsFit)
  coef(cyclopsMM)
})

test_that("Simulated SCCS using MM", {
    library(survival)
    set.seed(666)
    s <- simulateCyclopsData(nstrata = 10, nrows = 1000, ncovars = 5, eCovarsPerRow = 2)
    dataPtr <- convertToCyclopsData(s$outcomes, s$covariates, modelType = "clr", addIntercept = FALSE)
    cyclopsFit <- fitCyclopsModel(dataPtr, forceNewObject = TRUE,
                                  prior = createPrior("none")
                                  # prior = createPrior("laplace", variance = 10)
                                  )
    coef(cyclopsFit)
    cyclopsFit$iterations

    mm <- fitCyclopsModel(dataPtr, forceNewObject = TRUE,
                          control = createControl(algorithm = "mm",
                                                  maxIterations = 20,
                                                  noiseLevel = "noisy"),
                          prior = createPrior("none")
                         # prior = createPrior("laplace", variance = 10)
                         )
    coef(mm)
    mm$iterations

})
