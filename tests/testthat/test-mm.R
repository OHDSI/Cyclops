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

  cyclopsFit <- fitCyclopsModel(dataPtr,
                                prior = createPrior("none"))
  expect_equivalent(logLik(cyclopsFit), logLik(gold.clogit))
  expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)


#   d <- getXY(dataPtr)
#   fit1 <- mm(d$y, d$x, d$stratum)
#   fit2 <- mm(d$y, d$x, d$stratum, method = "2")

  cyclopsMM <- fitCyclopsModel(dataPtr,
                               prior = createPrior("none"),
                               control = createControl(algorithm = "mm"))
})
