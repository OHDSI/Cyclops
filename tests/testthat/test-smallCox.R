library("testthat")
library("survival")

test_that("Check very small Cox example with no ties, but with/without strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3.5,1,2,0
0, 3,  0,0,1
0, 2.5,1,0,1
0, 2,  1,1,1
0, 1.5,0,1,0
0, 1,  1,1,0
")

    goldCounting <-  coxph( Surv(start, length, event) ~ x1 + x2, test)
    summary(goldCounting)

    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    summary(goldRight)

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                       modelType = "cox")
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)

    dataPtrCounting <- createCyclopsData(Surv(start, length, event) ~ x1 + x2, data = test,
                                          modelType = "cox")
    cyclopsFitCounting <- fitCyclopsModel(dataPtrCounting)

    expect_equal(coef(cyclopsFitRight), coef(cyclopsFitCounting))

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)


    goldStrat <- coxph(Surv(length, event) ~ x1 + strata(x2), test)

    dataPtrStrat <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2),
                                       data = test,
                                       modelType = "cox")
    cyclopsFitStrat <- fitCyclopsModel(dataPtrStrat)
    expect_equal(coef(cyclopsFitStrat), coef(goldStrat), tolerance = tolerance)
})



test_that("Check very small Cox example with time-ties, but no failure ties", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  0,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    summary(goldRight)

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                       modelType = "cox")
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)
})

test_that("Check very small Cox example with failure ties, no risk-set contribution after tie", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 2,  1,0,0
")
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")
    coef(goldRight)

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                       modelType = "cox")
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)
})

test_that("Check very small Cox example with failure ties, with risk-set contribution after tie", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

        # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                           modelType = "cox")
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)
})

test_that("Check sparse Cox example with failure ties and strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test,
                                           modelType = "cox")
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)

    # Attempt sparse
    dataSparse <- createCyclopsData(Surv(length, event) ~ strata(x2),
                                         sparseFormula = ~ x1,
                                         data = test, modelType = "cox")

    cyclopsSparse <- fitCyclopsModel(dataSparse)
    expect_equal(coef(cyclopsSparse), coef(goldRight), tolerance = tolerance)
})

test_that("Check sparse Cox example with failure ties, strata and data weights", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")

    test2 <- rbind(data.frame(test, index = 1:7), data.frame(test, index = 1:7))
    test2 <- test2[order(test2$index),]
    test2[1,"x1"] <- 5
    test2[5,"x1"] <- 6

    dataPtrRight <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test2,
                                      modelType = "cox")

    cyclopsFitRight <- fitCyclopsModel(dataPtrRight,
                                       weights = rep(c(0,1), 7))

    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)

    test2clean <- rbind(data.frame(test, index = 1:7), data.frame(test, index = 1:7))
    test2clean <- test2clean[order(test2clean$index),]

    dataPtrClean <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test2clean,
                                      modelType = "cox")

    cyclopsFitClean <- fitCyclopsModel(dataPtrClean,
                                       weights = rep(c(0,1), 7))

    weights <- rep(c(1,0), 7)
    # predictiveLogLik <- getCyclopsPredictiveLogLikelihood(cyclopsFitClean, weights)
    #
    # expect_equal(predictiveLogLik, logLik(goldRight)[[1]])

})

test_that("Check SQL interface for a very small Cox example with failure ties and strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")

    data <- test
    data$row_id <- 1:nrow(data)
    data$covariate_id = 1

    data <- data[order(data$x2, -data$length, data$event),] # Must sort by: strata, into risk set (with events before censorsed)

    dataPtr <- createSqlCyclopsData(modelType = "cox")

    count <- appendSqlCyclopsData(dataPtr,
                                  data$x2,
                                  data$row_id,
                                  data$event,
                                  data$length,
                                  data$row_id,
                                  data$covariate_id,
                                  data$x1)

    cyclopsFitStrat <- fitCyclopsModel(dataPtr)

    tolerance <- 1E-4
    # SQL interface provides different names ('1' instead of 'x1')
    t1 <- coef(cyclopsFitStrat)
    t2 <- coef(goldRight)
    names(t1) <- NULL
    names(t2) <- NULL
    expect_equal(t1, t2, tolerance = tolerance)
})

test_that("Check confidence intervals with fixed coefficients",{
    set.seed(666)
    simulant <- simulateCyclopsData(nstrata = 1,
                                    ncovars = 5,
                                    model = "survival")

    # Fix covariate 2:
    data <- convertToCyclopsData(simulant$outcomes, simulant$covariates, modelType = "cox")
    fixed <- c(FALSE, TRUE, FALSE, FALSE, FALSE)
    names(fixed) <- c("1","2","3","4","5")
    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           forceNewObject = TRUE,
                           fixedCoefficients = fixed)
    expect_error(confint(fit, parm = c("1","2","3","4","5")))
    expect_equal(nrow(confint(fit, parm = c("1","3","4","5"))), 4)
    expect_error(confint(fit, parm = c(2)))
})

#
# test_that("More SQL checks for stratified cox models", {
#     data(lung)
#     lung$status = lung$status -1
#     lung <- lung[!is.na(lung$ph.ecog),]
#
#     goldRight <- coxph(Surv(time, status) ~ age + ph.ecog + strata(sex), lung, ties = "breslow")
#
#     dataPtrRight <- createCyclopsData(Surv(time, status) ~ age + ph.ecog + strata(sex),
#                                            method = "debug",
#                                            data = lung, modelType = "cox")
#     #This crashed R:
#     cyclopsFitRight <- fitCyclopsModel(dataPtrRight,
#                                        control = createControl(noiseLevel = "silent"))
#
#     lung$row_id <- 1:nrow(lung)
#     out <- data.frame(row_id = lung$row_id, stratum_id = lung$sex, time = lung$time, y = lung$status)
#     covAge <- data.frame(row_id = lung$row_id, stratum_id = lung$sex, time = lung$time, y = lung$status, covariate_value = lung$age)
#     covAge$covariate_id = 1
#     covPhEcog <- data.frame(row_id = lung$row_id, stratum_id = lung$sex, time = lung$time, y = lung$status, covariate_value = lung$ph.ecog)
#     covPhEcog$covariate_id = 2
#     cov <- rbind(covAge,covPhEcog)
#
#     # Sometimes, rows match on all (stratum, time, outcome): in which case sort may differ between out/cov
#     out <- out[order(out$stratum_id, -out$time, out$y, out$row_id),] # Must sort by: strata, into risk set (with events before censorsed)
#     cov <- cov[order(cov$stratum_id, -cov$time, cov$y, cov$row_id),] # Must sort by: strata, into risk set (with events before censorsed)
#
#     dataPtr <- createSqlCyclopsData(modelType = "cox")
#
#     count <- appendSqlCyclopsData(dataPtr,
#                                   out$stratum_id,
#                                   out$row_id,
#                                   out$y,
#                                   out$time,
#                                   cov$row_id,
#                                   cov$covariate_id,
#                                   cov$covariate_value)
#     #This crashes R
#     cyclopsFitStrat <- fitCyclopsModel(dataPtr)
# })
