library("testthat")
library("survival")

test_that("Check very small Cox example with ties, but without weights",{
    test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1, x2
                       0, 5, 0, 0, 1
                       0, 4, 1, 1, 2
                       0, 3, 0, 0, 1
                       0, 2, 0, 1, 0
                       0, 2, 1, 0, 0
                       0, 2, 1, 0, 1
                       0, 1, 0, 0, 0
                       0, 1, 1, 2, 1 ")
    tolerance <- 1E-4

    gold <- coxph(Surv(length, status) ~ x1 + x2, test, ties = "breslow")

    cyclopsData <- createCyclopsData(Surv(length, status) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData)

    expect_equal(coef(gold), coef(cyclopsFit), tolerance = tolerance)
    expect_equivalent(logLik(gold),logLik(cyclopsFit))

})

test_that("Check very small Cox example without ties, but with weights",{
    test <- read.table(header=T, sep = ",", text = "
                   start, length, status, x1, x2
                       0, 5,  0,0,1
                       0, 4,  1,1,2
                       0, 3,  0,0,1
                       0, 2,  0,1,0
                       0, 2,  1,0,0
                       0, 1,  0,1,0
                       0, 1,  1,2,1 ")
    weights <- c(1,2,1,2,3,2,1)

    gold <- coxph(Surv(length, status) ~ x1 + x2, test, weights = weights)

    cyclopsData <- createCyclopsData(Surv(length, status) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights)

    expect_equal(coef(gold), coef(cyclopsFit))
    expect_equivalent(logLik(gold),logLik(cyclopsFit))
})

test_that("Check very small Cox example with ties, with weights",{
    test <- read.table(header=T, sep = ",", text = "
                       start, length, status, x1, x2
                       0, 5, 0, 0, 1
                       0, 4, 1, 1, 2
                       0, 3, 0, 0, 1
                       0, 2, 0, 1, 0
                       0, 2, 1, 0, 0
                       0, 2, 1, 0, 1
                       0, 1, 0, 0, 0
                       0, 1, 1, 2, 1 ")
    weights <- c(1,2,1,2,4,3,2,1)

    gold <- coxph(Surv(length, status) ~ x1 + x2, test, weights, ties = "breslow")

    cyclopsData <- createCyclopsData(Surv(length, status) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData, weights=weights)

    expect_equal(coef(gold), coef(cyclopsFit))
    expect_equivalent(logLik(gold),logLik(cyclopsFit))

})

test_that("Check predictive log likelihood",{
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3,  1,2,0
                       0, 3,  0,0,1
                       0, 2,  1,0,1
                       0, 2,  1,1,1
                       0, 1,  0,1,0
                       0, 1,  1,1,0")

    gold <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")

    # Double the data
    test2 <- rbind(data.frame(test, index = 1:7), data.frame(test, index = 1:7))
    test2 <- test2[order(test2$index),]

    data <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test2,
                              modelType = "cox")

    # Fit with the second set
    weights = rep(c(0,1), 7)
    fit <- fitCyclopsModel(data, weights = weights)

    tolerance <- 1E-4
    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
    expect_equivalent(logLik(fit), logLik(gold))

    # Get predictive log likelihood of first set
    pred <- Cyclops:::.cyclopsGetNewPredictiveLogLikelihood(fit$interface, weights = 1 - weights)
    expect_equal(pred, as.numeric(logLik(gold)), tolerance)
})

test_that("Check very small Cox example with weighting", {
    test <- read.table(header=T, sep = ",", text = "
                       start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,1,2,0
                       0, 3,  0,0,1
                       0, 2.5,1,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    gold <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")

    # Duplicate some rows, so we can reweigh later:
    testDup <- rbind(test, test[c(1,3,4),])
    weights <- c(0.5, 1, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5)

    goldDup <- coxph(Surv(length, event) ~ x1 + x2, testDup, weights = weights, ties = "breslow")

    expect_equal(coef(gold), coef(goldDup))

    cyclopsData <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = testDup, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights)

    expect_equal(coef(gold), coef(cyclopsFit), tolerance = 1E-6)
    expect_equal(coef(goldDup), coef(cyclopsFit), tolerance = 1E-6)

    # Weights with sparse and indicator values
    cyclopsData4 <- createCyclopsData(Surv(length, event) ~ 1, sparseFormula = ~ x1,
                                      indicatorFormula = ~ x2,
                                      data = testDup, modelType = "cox",
                                      weights = weights)
    cyclopsFit4 <- fitCyclopsModel(cyclopsData4)
    expect_equal(coef(goldDup), coef(cyclopsFit4), tolerance = 0.000001)

})

test_that("Small Poisson dense regression with weighting", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    weights <- c(1,2,1,2,4,3,2,1,1)

    glmFit <- glm(counts ~ outcome + treatment, data = dobson, weights, family = poisson()) # gold standard

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                  modelType = "pr")
    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"), weights = weights)

    expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitD), predict(glmFit, type = "response"), tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c("(Intercept)","outcome3")), confint(cyclopsFitD, c(1,3)))

    fit <- fitCyclopsModel(dataPtrD,
                           prior = createPrior("laplace", useCrossValidation = TRUE),
                           weights = weights,
                           control = createControl(minCVData = 1, noiseLevel = "quiet"))
})

test_that("Small Bernoulli dense regression with weighting", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    weights <- as.numeric(gl(5,1,length(y)))

    tolerance <- 1E-4

    glmFit <- glm(y ~ log_bid, family = binomial(), weights = weights) # gold standard

    dataPtrD <- createCyclopsData(y ~ log_bid, modelType = "lr")
    cyclopsFitD <- fitCyclopsModel(dataPtrD, prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"), weights = weights)

    expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c(1:2))[,2:3], confint(glmFit, c(1:2)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitD), predict(glmFit, type = "response"), tolerance = tolerance)
})

test_that("Small conditional logistic regression with weighting", {

    weights <- as.numeric(gl(5,1,length(infert$case)))

    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert, weights = weights, method=c("approximate"))

    dataPtr <- createCyclopsData(case ~ spontaneous + induced + strata(stratum),
                                 data = infert,
                                 modelType = "clr")

    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"), weights = weights)

    tolerance <- 1E-4

    expect_equal(coef(cyclopsFit), coef(gold), tolerance = tolerance)
    expect_equal(cyclopsFit$log_likelihood, logLik(gold)[[1]], tolerance = tolerance)

    # expect_equal(vcov(cyclopsFit), vcov(gold), tolerance = tolerance)
    #
    # expect_equal(aconfint(cyclopsFit), confint(gold), tolerance = tolerance)

    expect_equal(confint(cyclopsFit, c(1:2), includePenalty = TRUE),
                 confint(cyclopsFit, c(1:2), includePenalty = FALSE))

})

test_that("Small Bernoulli dense regression with zero-weights", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

    weights <- as.numeric(gl(2,1,length(y))) - 1

    y_sub <- y[which(weights == 1)]
    log_bid_sub <- log_bid[which(weights == 1)]

    glmFit <- glm(y ~ log_bid, family = binomial(), weights = weights) # gold standard

    tolerance <- 1E-4

    data <- createCyclopsData(y ~ log_bid, modelType = "lr")
    data_sub <- createCyclopsData(y_sub ~ log_bid_sub, modelType = "lr")

    fit <- fitCyclopsModel(data, prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"),
                           weights = weights)

    fit_sub <- fitCyclopsModel(data_sub, prior = createPrior("none"),
                               control = createControl(noiseLevel = "silent"))

    expect_equal(coef(fit), coef(glmFit), tolerance = tolerance)
    expect_equal(fit$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(predict(glmFit, type = "response"), predict(fit), tolerance = tolerance)

    expect_equal(coef(fit), coef(fit_sub), check.attributes = FALSE)
    expect_equal(fit$log_likelihood, fit_sub$log_likelihood)

    expect_equal(predict(fit)[which(weights == 1)], predict(fit_sub), check.attributes = FALSE)
})

test_that("Multi-core weights", {
    test <- read.table(header=T, sep = ",", text = "
                   start, length, status, x1, x2
                       0, 5,  0,0,1
                       0, 4,  1,1,2
                       0, 3,  0,0,1
                       0, 2,  0,1,0
                       0, 2,  1,0,0
                       0, 1,  0,1,0
                       0, 1,  1,2,1 ")
    weights <- c(1,2,1,2,3,2,1)

    cyclopsData <- createCyclopsData(Surv(length, status) ~ x1 + x2, data = test, modelType = "cox")
    cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights,
                                  control = createControl(noiseLevel = "silent"))
    ci1 <- confint(cyclopsFit, "x1")
    cyclopsFit <- fitCyclopsModel(cyclopsData, weights = weights,
                                  control = createControl(threads = 2,
                                                          noiseLevel = "silent"))
    ci2 <- confint(cyclopsFit, "x1")

    expect_equal(ci1, ci2)
})
