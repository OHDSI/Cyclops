library(Cyclops)
library("testthat")
library("survival")


GpuDevice <- listGPUDevices()[1]
tolerance <- 1E-4


# small cox
test_that("Check small Cox on GPU", {
    skip_if(length(listGPUDevices()) == 0, "GPU not available")
    test <- read.table(header=T, sep = ",", text = "
                   start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3.5,1,2,0
                       0, 3,  0,0,1
                       0, 2.5,1,0,1
                       0, 2,  1,1,1
                       0, 1.5,0,1,0
                       0, 1,  1,1,0")

    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    dataPtrRight_CPU <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                          modelType = "cox", floatingPoint = 32)
    cyclopsFitRight_CPU <- fitCyclopsModel(dataPtrRight_CPU)

    dataPtrRight_GPU <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                          modelType = "cox", floatingPoint = 32)
    cyclopsFitRight_GPU <- fitCyclopsModel(dataPtrRight_GPU, computeDevice = GpuDevice)

    expect_equal(coef(cyclopsFitRight_CPU), coef(goldRight), tolerance = tolerance)
    expect_equal(coef(cyclopsFitRight_GPU), coef(cyclopsFitRight_CPU), tolerance = tolerance)
})

test_that("Check very small Cox example with time-ties", {
    skip_if(length(listGPUDevices()) == 0, "GPU not available")
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
                       0, 4,  1,0,0
                       0, 3,  1,2,0
                       0, 3,  0,0,1
                       0, 2,  1,0,1
                       0, 2,  0,1,1
                       0, 1,  0,1,0
                       0, 1,  1,1,0")
    # cpu
    dataPtrRight_CPU <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                          modelType = "cox", floatingPoint = 32)
    cyclopsFitRight_CPU <- fitCyclopsModel(dataPtrRight_CPU)

    # gpu
    dataPtrRight_GPU <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                                          modelType = "cox", floatingPoint = 32)
    cyclopsFitRight_GPU <- fitCyclopsModel(dataPtrRight_GPU, computeDevice = GpuDevice)

    expect_equal(coef(cyclopsFitRight_GPU), coef(cyclopsFitRight_CPU), tolerance = tolerance)
})


# large cox
test_that("Check Cox on GPU", {
    skip_if(length(listGPUDevices()) == 0, "GPU not available")
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1,
                               nrows = 100000,
                               ncovars = 100,
                               effectSizeSd = 1,
                               zeroEffectSizeProp = 0.8,
                               eCovarsPerRow = 0.2,
                               model = "survival")
    set.seed(12)
    sim$outcomes$time <- sim$outcomes$time + rnorm(100000, mean = 0, sd = 0.00001)

    cyclopsData_CPU <- convertToCyclopsData(sim$outcomes, sim$covariates,
                                            modelType = "cox",floatingPoint = 64,
                                            addIntercept = TRUE)
    fit_CPU <- fitCyclopsModel(cyclopsData_CPU) # iterations: 6 lastObjFunc: 733.17
    cyclopsData_GPU <- convertToCyclopsData(sim$outcomes, sim$covariates,
                                            modelType = "cox",floatingPoint = 64,
                                            addIntercept = TRUE)
    fit_GPU <- fitCyclopsModel(cyclopsData_GPU, computeDevice = GpuDevice) # iterations: 6 lastObjFunc: 733.17
    expect_equal(coef(fit_GPU), coef(fit_CPU), tolerance = tolerance)
})

# lasso cv
test_that("Check cross-validation for lasso Cox on GPU", {
    skip_if(length(listGPUDevices()) == 0, "GPU not available")
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1,
                               nrows = 900,
                               ncovars = 35,
                               effectSizeSd = 1,
                               zeroEffectSizeProp = 0.8,
                               eCovarsPerRow = 1,
                               model = "survival")
    set.seed(123)
    sim$outcomes$time <- sim$outcomes$time + rnorm(900, mean = 0, sd = 0.00001)

    prior <- createPrior("laplace", useCrossValidation = TRUE)
    control <- createControl(noiseLevel = "quiet", lowerLimit = 0.000001, upperLimit = 100,
                                cvType = "auto", fold = 10, cvRepetitions = 1, startingVariance = 0.01, threads = 1,
                                seed = 123)

    cyclopsData_CPU <- convertToCyclopsData(sim$outcomes, sim$covariates,modelType = "cox",floatingPoint = 64,addIntercept = TRUE)
    fit_CPU <- fitCyclopsModel(cyclopsData_CPU, prior = prior, control = control)

    cyclopsData_GPU <- convertToCyclopsData(sim$outcomes, sim$covariates,modelType = "cox",floatingPoint = 64,addIntercept = TRUE)
    fit_GPU <- fitCyclopsModel(cyclopsData_GPU, prior = prior, control = control, computeDevice = GpuDevice)

    expect_equal(getHyperParameter(fit_GPU), getHyperParameter(fit_CPU), tolerance = tolerance)
    expect_equal(coef(fit_GPU), coef(fit_CPU), tolerance = tolerance)

})

# multi-core
test_that("Check multi-core cross-validation for lasso Cox on GPU", {
    skip_if(length(listGPUDevices()) == 0, "GPU not available")
    set.seed(123)
    sim <- simulateCyclopsData(nstrata = 1,
                               nrows = 900,
                               ncovars = 35,
                               effectSizeSd = 1,
                               zeroEffectSizeProp = 0.8,
                               eCovarsPerRow = 1,
                               model = "survival")
    set.seed(123)
    sim$outcomes$time <- sim$outcomes$time + rnorm(900, mean = 0, sd = 0.00001)

    prior <- createPrior("laplace", useCrossValidation = TRUE)
    control <- createControl(noiseLevel = "quiet", lowerLimit = 0.000001, upperLimit = 100,
                                cvType = "auto", fold = 10, cvRepetitions = 1, startingVariance = 0.01, threads = 2,
                                seed = 123)

    cyclopsData_CPU <- convertToCyclopsData(sim$outcomes, sim$covariates,modelType = "cox",floatingPoint = 64,addIntercept = TRUE)
    fit_CPU <- fitCyclopsModel(cyclopsData_CPU, prior = prior, control = control)

    cyclopsData_GPU <- convertToCyclopsData(sim$outcomes, sim$covariates,modelType = "cox",floatingPoint = 64,addIntercept = TRUE)
    fit_GPU <- fitCyclopsModel(cyclopsData_GPU, prior = prior, control = control, computeDevice = GpuDevice)

    expect_equal(getHyperParameter(fit_GPU), getHyperParameter(fit_CPU), tolerance = tolerance)
    expect_equal(coef(fit_GPU), coef(fit_CPU), tolerance = tolerance)

})



# test_that("Check small Cox example with failure ties and strata on GPU", {
#     test <- read.table(header=T, sep = ",", text = "
#                        start, length, event, x1, x2
#                        0, 4,  1,0,0
#                        0, 3,  1,2,0
#                        0, 3,  0,0,1
#                        0, 2,  1,0,1
#                        0, 2,  1,1,1
#                        0, 1,  0,1,0
#                        0, 1,  1,1,0")
#
#     # We get the correct answer when last entry is censored
#     goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")
#
#     dataPtrRight_CPU <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test,
#                                            modelType = "cox", floatingPoint = 32)
#     cyclopsFitRight_CPU <- fitCyclopsModel(dataPtrRight_CPU)
#
#     dataPtrRight_GPU <- createCyclopsData(Surv(length, event) ~ x1 + strata(x2), data = test,
#                                           modelType = "cox", floatingPoint = 32)
#     cyclopsFitRight_GPU <- fitCyclopsModel(dataPtrRight_GPU, computeDevice = GpuDevice)
#
#     tolerance <- 1E-4
#     expect_equal(coef(cyclopsFitRight_CPU), coef(goldRight), tolerance = tolerance)
#     expect_equal(coef(cyclopsFitRight_CPU), coef(cyclopsFitRight_GPU), tolerance = tolerance)
# })

# make sure logistic regression still works
# test_that("Small Bernoulli dense regression using GPU", {
#     binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
#     binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
#     binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)
# 
#     log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
#     y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
#     tolerance <- 1E-4

#     # gold standard
#     glmFit <- glm(y ~ log_bid, family = binomial())
# 
#     # # cpu
#     # dataPtrD_c <- createCyclopsData(y ~ log_bid, modelType = "lr", floatingPoint = 32)
#     # cyclopsFitD_c <- fitCyclopsModel(dataPtrD_c, prior = createPrior("none"),
#     #                                  control = createControl(noiseLevel = "silent"))
# 
#     # gpu
#     dataPtrD <- createCyclopsData(y ~ log_bid, modelType = "lr", floatingPoint = 32)
#     cyclopsFitD <- fitCyclopsModel(dataPtrD, prior = createPrior("none"),
#                                    control = createControl(noiseLevel = "silent"),
#                                    computeDevice = GpuDevice)
# 
#     expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
# })



