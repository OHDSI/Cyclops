library("testthat")

#
# Small Poisson MLE regression
#

test_that("Small Poisson dense regression", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    glmFit <- glm(counts ~ outcome + treatment, data = dobson, family = poisson()) # gold standard

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                   modelType = "pr")
    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitD$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitD), predict(glmFit, type = "response"), tolerance = tolerance)
    expect_equal(confint(cyclopsFitD, c("(Intercept)","outcome3")), confint(cyclopsFitD, c(1,3)))
})

test_that("Small Poisson dense regression in 32bit", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    gold <- glm(counts ~ outcome + treatment, data = dobson, family = poisson()) # gold standard

    data <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                  modelType = "pr", floatingPoint = 32)
    expect_equal(getFloatingPointSize(data), 32)

    fit <- fitCyclopsModel(data,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))

    expect_equal(coef(fit), coef(gold), tolerance = tolerance)

    expect_equal(Cyclops:::.cyclopsGetMeanOffset(data), 0)
})

test_that("Errors for different criteria", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-3

    gold <- glm(counts ~ outcome + treatment, data = dobson, family = poisson()) # gold standard

    data <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                              modelType = "pr")

    fit <- fitCyclopsModel(data,
                           prior = createPrior("none"),
                           control = createControl(convergenceType = "mittal",
                                                   noiseLevel = "silent"))

    expect_equal(coef(fit), coef(gold), tolerance = tolerance)
})

test_that("Small Poisson dense regression with offset", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = as.numeric(gl(3,1,9)),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    gold <- glm(counts ~  treatment, offset = outcome, data = dobson, family = poisson()) # gold standard

    data <- createCyclopsData(counts ~  treatment, offset = as.numeric(outcome),
                                    data = dobson, modelType = "pr")

    fit <- fitCyclopsModel(data,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))

    expect_equal(coef(fit), coef(gold), tolerance = tolerance)

    expect_error(Cyclops:::.cyclopsGetMeanOffset(fit),
                 "Input must be a cyclopsData object")

    expect_equal(Cyclops:::.cyclopsGetMeanOffset(data), 2)
})

test_that("Small Poisson fixed beta", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                  modelType = "pr")

    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                                   startingCoefficients = rep(0, 5),
                                   fixedCoefficients = c(FALSE, TRUE, FALSE, TRUE, FALSE),
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))
    expect_equivalent(coef(cyclopsFitD)[2], 0)
    expect_equivalent(coef(cyclopsFitD)[4], 0)
})

# test_that("Parallel confint", {
#     dobson <- data.frame(
#         counts = c(18,17,15,20,10,20,25,13,12),
#         outcome = gl(3,1,9),
#         treatment = gl(3,3)
#     )
#     tolerance <- 1E-4
#
#     glmFit <- glm(counts ~ outcome + treatment, data = dobson, family = poisson()) # gold standard
#
#     dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
#                                   modelType = "pr")
#     cyclopsFit1 <- fitCyclopsModel(dataPtrD,
#                                    prior = createPrior("none"),
#                                    control = createControl(noiseLevel = "silent",
#                                                            threads = 1))
#     cyclopsFit2 <- fitCyclopsModel(dataPtrD,
#                                    prior = createPrior("none"),
#                                    control = createControl(noiseLevel = "silent",
#                                                            threads = 2))
#
#     expect_equal(confint(cyclopsFit1, c(1:3)), confint(cyclopsFit2, c(1:3)))
#     ## TODO Check output of confint for "Using 2 thread(s)"
# })

test_that("Specify CI level", {
###function(object, parm, level, ...)
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    dataPtr <- createCyclopsData(counts ~ outcome + treatment,
                                   modelType = "pr")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))

    expect_equal(
        confint(cyclopsFit, c(1:3), level = 0.99)[,2:3],
        confint(glmFit, c(1:3), level = 0.99),
        tolerance = tolerance)


})

test_that("Small Poisson indicator regression", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    dataPtrI <- createCyclopsData(counts ~ outcome, indicatorFormula =  ~ treatment,
                                   modelType = "pr")

    cyclopsFitI <- fitCyclopsModel(dataPtrI,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitI), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitI$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)

    expect_equal(logLik(cyclopsFitI), logLik(glmFit))

    expect_equal(confint(cyclopsFitI, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitI), predict(glmFit, type = "response"), tolerance = tolerance)
})

test_that("Small Poisson sparse regression", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    dataPtrS <- createCyclopsData(counts ~ outcome, sparseFormula =  ~ treatment,
                                   modelType = "pr")
    cyclopsFitS <- fitCyclopsModel(dataPtrS,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitS), coef(glmFit), tolerance = tolerance)
    expect_equal(cyclopsFitS$log_likelihood, logLik(glmFit)[[1]], tolerance = tolerance)
    expect_equal(confint(cyclopsFitS, c(1:3))[,2:3], confint(glmFit, c(1:3)), tolerance = tolerance)
    expect_equal(predict(cyclopsFitS), predict(glmFit, type = "response"), tolerance = tolerance)
})


test_that("Get SEs in small Poisson model", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    gold <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard
    goldSE <- summary(gold)$coefficients[,2]

    dataPtr <- createCyclopsData(counts ~ outcome + treatment,
                                  modelType = "pr")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("none"))

    cyclopsSE <- getSEs(cyclopsFit, c(1:5))

    expect_equal(goldSE, cyclopsSE, tolerance = tolerance)
})

# test_that("Playing with standardization", {
#     counts <- c(18,17,15,20,10,20,25,13,12)
#     outcome <- gl(3,1,9)
#     treatment <- gl(3,3)
#     tolerance <- 1E-4
#
#     dataPtr <- createCyclopsData(counts ~ outcome + treatment,
#                                       modelType = "pr")
#     cyclopsFit <- fitCyclopsModel(dataPtr,
#                                   prior = createPrior("none"))
#
#     dataPtrS <- createCyclopsData(counts ~ outcome + treatment,
#                                        modelType = "pr")
#     cyclopsFitS <- fitCyclopsModel(dataPtrS,
#                                    prior = createPrior("none"))
#
#     coef(cyclopsFit)
#     coef(cyclopsFitS)
# })
