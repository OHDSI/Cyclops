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
                                   control = createControl(noiseLevel = "silent"),weights=weights)

    expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitD), predict(glmFit, type = "response"), tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c("(Intercept)","outcome3")), confint(cyclopsFitD, c(1,3)))
})

test_that("Small conditional logistic regression with weighting", {

    weights = as.numeric(gl(5,1,length(infert$case)))

    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert, weights = weights, method=c("approximate"))

    dataPtr <- createCyclopsData(case ~ spontaneous + induced + strata(stratum),
                                 data = infert,
                                 modelType = "clr")

    cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"), weights=weights)

    tolerance <- 1E-4

    expect_equal(coef(cyclopsFit), coef(gold), tolerance = tolerance)
    expect_equal(cyclopsFit$log_likelihood, logLik(gold)[[1]], tolerance = tolerance)

    # expect_equal(vcov(cyclopsFit), vcov(gold), tolerance = tolerance)
    #
    # expect_equal(aconfint(cyclopsFit), confint(gold), tolerance = tolerance)
    #
    # expect_equal(confint(cyclopsFit, c(1:2), includePenalty = TRUE),
    #              confint(cyclopsFit, c(1:2), includePenalty = FALSE))

})
