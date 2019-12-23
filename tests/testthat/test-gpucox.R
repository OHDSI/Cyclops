library(Cyclops)
library("testthat")
library("survival")

GpuDevice <- listOpenCLDevices()[2]

# make sure logistic regression still works
test_that("Small Bernoulli dense regression using GPU", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
    tolerance <- 1E-4

    # gold standard
    glmFit <- glm(y ~ log_bid, family = binomial())

    # gpu
    dataPtrD <- createCyclopsData(y ~ log_bid, modelType = "lr", floatingPoint = 32)
    cyclopsFitD <- fitCyclopsModel(dataPtrD, prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"),
                                   computeDevice = GpuDevice)

    expect_equal(coef(cyclopsFitD), coef(glmFit), tolerance = tolerance)
})
