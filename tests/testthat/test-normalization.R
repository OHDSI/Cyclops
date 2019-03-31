library("testthat")

test_that("Simple normalization", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    gold <- glm(counts ~ outcome + treatment, family = poisson())

    dataPtr <- createCyclopsData(counts ~ 1,
                                 sparseFormula = ~outcome,
                                 indicatorFormula = ~treatment,
                                 modelType = "pr")
    expect_null(dataPtr$scales)
    fit1 <- fitCyclopsModel(dataPtr, prior = createPrior("none"))

    # Normalize
    Cyclops:::.normalizeCovariates(dataPtr, type = "stdev")

    expect_error(confint(fit1, parm = "outcome2")) # Data are now invalid for fit1

    sum2 <- summary(dataPtr)
    expect_equal(sum(sum2$nzMean != 1.0), 2) # Only sparse covariates are normalized
    fit2 <- fitCyclopsModel(dataPtr, prior = createPrior("none"))

    # Normalizing again should not change values
    Cyclops:::.normalizeCovariates(dataPtr, type = "stdev")
    sum3 <- summary(dataPtr)
    expect_equal(sum2$scale, sum3$scale, tolerance = tolerance * tolerance)
    fit3 <- fitCyclopsModel(dataPtr, prior = createPrior("none"))

    # Check equality of first results at end
    expect_equal(coef(gold), coef(fit1), tolerance = tolerance)
    # And second after rescaling
    expect_equal(coef(gold), coef(fit2, rescale = TRUE), tolerance = tolerance)
    # And third after rescaling
    expect_equal(confint(gold, parm = "outcome2"),
                 confint(fit3, parm = "outcome2", rescale = TRUE)[,2:3], tolerance = sqrt(tolerance))

    # Repeat process with normalization during construction
    dataPtr2 <- createCyclopsData(counts ~ 1,
                                 sparseFormula = ~outcome,
                                 indicatorFormula = ~treatment,
                                 modelType = "pr", normalize = "stdev")

    fit4 <- fitCyclopsModel(dataPtr2, prior = createPrior("none"))
    expect_equal(coef(fit2), coef(fit4))

    # Normalize via max
    covariate <- c(1:length(counts))
    dataPtr3 <- createCyclopsData(counts ~ 1,
                                  sparseFormula = ~covariate,
                                  indicatorFormula = ~treatment,
                                  modelType = "pr", normalize = "max")
    sum4 <- summary(dataPtr3)
    expect_equal(sum4$scale[2], 1 / max(covariate))

    # Normalize via median
    dataPtr4 <- createCyclopsData(counts ~ 1,
                                  sparseFormula = ~covariate,
                                  indicatorFormula = ~treatment,
                                  modelType = "pr", normalize = "median")
    sum5 <- summary(dataPtr4)
    expect_equal(sum5$scale[2], 1 / median(covariate))
})
