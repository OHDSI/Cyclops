library("testthat")
library(MASS)

context("test-offsetPoisson.R")

test_that("Check offset in model formula", {

    tolerance <- 1E-4
    Insurance$logHolders <- log(Insurance$Holders)

    glmFit <- glm(Claims ~ District + Group + Age + offset(logHolders),
                  data = Insurance,
                  family = poisson()) # gold standard

    dataPtr <- createCyclopsData(Claims ~ District + Group + Age + logHolders,
                                  data = Insurance,
                                  modelType = "pr")
    finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = "logHolders", offsetAlreadyOnLogScale = TRUE)

    ## Test new number of covariates

    cyclopsFit <- fitCyclopsModel(dataPtr,
                          prior = createPrior("none"),
                          control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFit), coef(glmFit), tolerance = tolerance)

    dataPtr2 <- createCyclopsData(Claims ~ District + Group + Age + offset(logHolders),
                                   data = Insurance,
                                   modelType = "pr")

    cyclopsFit2 <- fitCyclopsModel(dataPtr2,
                                   startingCoefficients = rep(0.5,10),
                                   prior = createPrior("none"))

    expect_equal(coef(cyclopsFit2), coef(glmFit), tolerance = tolerance)

    # Need to test now using finalize to (1) add intercept and (2) log-transform
})

test_that("Check active set", {
    skip("Current not working")
    tolerance <- 1E-4
    Insurance$logHolders <- log(Insurance$Holders)

    dataPtr2 <- createCyclopsData(Claims ~ District + Group + Age + offset(logHolders),
                                  data = Insurance,
                                  modelType = "pr")

    out <- capture.output(cyclopsFit2 <- fitCyclopsModel(dataPtr2,
                                   control = createControl(useKKTSwindle = TRUE,
                                                           noiseLevel = "quiet",
                                                            tuneSwindle = 4),
                                   prior = createPrior("laplace", exclude=c(1,3))))

    expect_equal(length(out), 8) # Should have 3 (+ prior line) swindle sets
})
