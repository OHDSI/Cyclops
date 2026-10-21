library("testthat")
library("survival")
library("gnm")

test_that("vcov for cpr, sccs and clr counts multiple events per stratum", {
    data <- Cyclops::oxford
    data$event <- 3 * data$event # n_s = 3 in every stratum; same MLE, three times the information
    gold <- gnm(event ~ exgr + agegr + offset(loginterval), family = poisson, eliminate = indiv, data = data)
    cpr <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                             data = data, modelType = "cpr"))
    sccs <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv), time = data$interval,
                                              data = data, modelType = "sccs"))
    expect_equal(as.numeric(vcov(cpr)), as.numeric(vcov(gold)), tolerance = 1E-6)
    expect_equal(as.numeric(vcov(sccs)), as.numeric(vcov(gold)), tolerance = 1E-6)

    binary <- Cyclops::oxford
    binary$event[binary$indiv %in% 1:5 & binary$exgr == 0 & binary$agegr == 1] <- 1 # some strata with 2-3 cases
    goldClr <- survival::clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval), data = binary,
                                method = "breslow")
    clr <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                             data = binary, modelType = "clr"))
    expect_equal(as.numeric(vcov(clr)), as.numeric(vcov(goldClr)), tolerance = 1E-6)
})

test_that("vcov after refitting the same cyclopsData uses the current coefficients", {
    data <- Cyclops::oxford
    data$event <- 2 * data$event
    cd <- createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval), data = data, modelType = "cpr")
    first <- fitCyclopsModel(cd)
    invisible(vcov(first))
    second <- fitCyclopsModel(cd, prior = createPrior("normal", variance = 0.1))
    fresh <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                               data = data, modelType = "cpr"),
                             prior = createPrior("normal", variance = 0.1))
    expect_equal(coef(second), coef(fresh), tolerance = 1E-6)
    expect_equal(vcov(second), vcov(fresh), tolerance = 1E-6)
})

test_that("vcov for cpr ignores strata without events", {
    data <- Cyclops::oxford
    data$indiv <- as.character(data$indiv)
    empty <- data[data$indiv %in% 1:5, ]
    empty$indiv <- paste0("empty", empty$indiv)
    empty$event <- 0 # these strata contribute nothing to the likelihood
    formula <- event ~ exgr + agegr + strata(indiv) + offset(loginterval)
    fit <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"))
    fitWithEmpty <- fitCyclopsModel(createCyclopsData(formula, data = rbind(data, empty), modelType = "cpr"))
    expect_equal(coef(fitWithEmpty), coef(fit), tolerance = 1E-6)
    expect_equal(vcov(fitWithEmpty), vcov(fit), tolerance = 1E-6)
})
