library("testthat")

context("test-multitypePoisson.R")

#
# Small Poisson MLE regression
#

test_that("Small multi-type Poisson dense regression", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson <- rbind(dobson1, dobson1)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4

    goldFit <- glm(counts ~ outcome + treatment, data = dobson1, family = poisson())

    # glmFit <- glm(counts ~ outcome + treatment, data = dobson, contrasts = dobson$type,
    #               family = poisson()) # gold standard

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                       type = dobson$type,
                                       modelType = "pr", method = "debug")

    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))

    dataPtrE <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,
                                       modelType = "pr", method = "debug")

    cyclopsFitE <- fitCyclopsModel(dataPtrE,
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))

    expect_equal(coef(cyclopsFitD), coef(cyclopsFitE))

    dataPtrI <- createCyclopsData(Multitype(counts, type) ~ 1, indicatorFormula = ~ outcome + treatment, data = dobson,
                                       modelType = "pr", method = "debug")

    cyclopsFitI <- fitCyclopsModel(dataPtrI,
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))

    expect_equal(coef(cyclopsFitI), coef(cyclopsFitD))

    dataPtrS <- createCyclopsData(Multitype(counts, type) ~ 1, sparseFormula = ~ outcome + treatment, data = dobson,
                                       modelType = "pr", method = "debug")

    cyclopsFitS <- fitCyclopsModel(dataPtrS,
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))

    expect_equal(coef(cyclopsFitS), coef(cyclopsFitD))
})

test_that("coef throws error when not converged", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4

    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,
                                       modelType = "pr")

    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                                   prior = createPrior(c("normal","normal"), c(0.0001,10), graph = "type"),
                                   control = createControl(noiseLevel = "silent"))
    expect_error(coef(cyclopsFitD), "did not converge")
})


test_that("confirm dimension check", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4

    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,
                                       modelType = "pr")


    expect_error(fitCyclopsModel(dataPtrD,
                                 prior = createPrior(c("normal"), c(0.0001,10), graph = "type"),
                                 control = createControl(noiseLevel = "silent")), "dimensionality mismatch")
    expect_error(fitCyclopsModel(dataPtrD,
                                 prior = createPrior(c("normal", "normal"), c(0.0001), graph = "type"),
                                 control = createControl(noiseLevel = "silent")), "dimensionality mismatch")
})


test_that("Small multi-type Poisson with hierarchical prior", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4

    # glmFit <- glm(counts ~ outcome + treatment, data = dobson, contrasts = dobson$type,
    #               family = poisson()) # gold standard

    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,
                                       modelType = "pr")

    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                                   prior = createPrior(c("normal","normal"), c(0.0001,10), graph = "type"),
                                   control = createControl(noiseLevel = "silent", maxIterations = 2000))

    cyclopsFitE <- fitCyclopsModel(dataPtrD,
                                   prior = createPrior(c("normal","normal"), c(0.0001,0.0001), graph = "type"),
                                   control = createControl(noiseLevel = "silent"))


})


test_that("Check multitype SCCS", {
#
#   stratumID must be unique by type and all patient entries must be replicated by type
#     (how to specify this in input) ???
#
#     source("helper-conditionalPoisson.R")
#     tolerance <- 1E-6
#     gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
#                           data = oxford)
#
#     dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv), time = oxford$interval,
#                                       data = oxford,
#                                       modelType = "sccs")
#     cyclopsFit <- fitCyclopsModel(dataPtr,
#                                   prior = createPrior("none"))
#     expect_equal(logLik(cyclopsFit), logLik(gold.clogit)[1])
#     expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)
})



test_that("Several outcome types give one coefficient per type, whatever type is", {
    # type indexed a contrast matrix, which gave the codes the data need only for a factor of two levels: three levels
    # merged two types and added empty columns (the fit failed), and a character vector or 0/1 codes crashed R
    set.seed(3)
    n <- 900
    k <- factor(rep(c("A", "B", "C"), n / 3))
    z <- rnorm(n)
    frame <- data.frame(y = rpois(n, exp(c(0.2, 0.8, 1.4)[k] + c(0.1, -0.3, 0.5)[k] * z)), z = z, k = k)
    gold <- glm(y ~ 0 + k + k:z, data = frame, family = poisson()) # an intercept and a slope per type
    fit <- function(data) coef(fitCyclopsModel(data, prior = createPrior("none"),
                                               control = createControl(noiseLevel = "silent")))

    byFactor <- createCyclopsData(y ~ z, type = frame$k, data = frame, modelType = "pr")
    expect_equal(getNumberOfTypes(byFactor), 3)
    a <- fit(byFactor)
    expect_equal(names(a), c("(Intercept):A", "(Intercept):B", "(Intercept):C", "z:A", "z:B", "z:C"))
    expect_equal(unname(a), unname(coef(gold)), tolerance = 1e-6)
    expect_equal(fit(createCyclopsData(y ~ z, type = as.character(frame$k), data = frame, modelType = "pr")), a)
    expect_equal(fit(createCyclopsData(Multitype(y, k) ~ z, data = frame, modelType = "pr")), a)

    two <- frame[frame$k != "C", ]
    b <- fit(createCyclopsData(y ~ z, type = droplevels(two$k), data = two, modelType = "pr"))
    expect_equal(unname(fit(createCyclopsData(y ~ z, type = as.integer(two$k == "B"), data = two, modelType = "pr"))),
                 unname(b))

    expect_error(createCyclopsData(y ~ z, type = frame$k[-1], data = frame, modelType = "pr"), "one value per row")
    expect_error(createCyclopsData(y ~ z, type = replace(frame$k, 5, NA), data = frame, modelType = "pr"),
                 "missing values")
})
