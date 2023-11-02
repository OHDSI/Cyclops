library(survival)
library(testthat)

GpuDevice <- listGPUDevices()[1]
tolerance <- 1E-4

test_that("Check very small Cox example with time-varying coefficient as stratified model", {
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
    test_split <- survSplit(Surv(length, event)~., data = test,
                            cut = c(2),
                            episode = "stratumId")

    # assume x1 has time-varying effect on the outcome
    test_cyclops <- test_split
    test_cyclops$time <- test_cyclops$length - test_cyclops$tstart
    test_cyclops$x1.1 <- ifelse(test_cyclops$stratumId == 1, test_cyclops$x1, 0)
    test_cyclops$x1.2 <- ifelse(test_cyclops$stratumId == 2, test_cyclops$x1, 0)
    test_cyclops <- test_cyclops[, c("stratumId", "time", "event",
                                     "x1.1", "x1.2", "x2")]

    goldStrat <- coxph(Surv(tstart, length, event) ~ x2 + x1:strata(stratumId), test_split)

    dataPtrStrat_CPU <- createCyclopsData(Surv(time, event) ~ x2 + x1.1 + x1.2 + strata(stratumId),
                                      data = test_cyclops,
                                      modelType = "cox_time")
    cyclopsFitStrat_CPU <- fitCyclopsModel(dataPtrStrat_CPU)

    dataPtrStrat_GPU <- createCyclopsData(Surv(time, event) ~ x2 + x1.1 + x1.2 + strata(stratumId),
                                      data = test_cyclops,
                                      modelType = "cox_time")
    cyclopsFitStrat_GPU <- fitCyclopsModel(dataPtrStrat_GPU, computeDevice = GpuDevice)

    expect_equal(unname(coef(cyclopsFitStrat_CPU)), unname(coef(goldStrat)), tolerance = tolerance)
    expect_equal(coef(cyclopsFitStrat_CPU), coef(cyclopsFitStrat_GPU), tolerance = tolerance)


    # short sparse cov
    sparseShortCov <- data.frame(stratumId = 1,
                                 rowId = rep(1:nrow(test), 2),
                                 covariateId = rep(1:2, each = nrow(test)),
                                 covariateValue = c(test$x2, test$x1))
    sparseShortCov <- sparseShortCov[sparseShortCov$covariateValue != 0,]

    # long out
    longOut <- splitTime(data.frame(time = test$length,
                                    y = test$event),
                         cut = c(2))

    # long sparse cov
    sparseLongCov <- convertToTimeVaryingCoef(sparseShortCov, longOut, timeVaryCoefId = c(2))

    # CPU: long out and long cov
    dataPtrSparse_CPU <- convertToCyclopsData(outcomes = longOut,
                                              covariates = sparseLongCov,
                                              modelType = "cox_time")
    cyclopsFitSparse_CPU <- fitCyclopsModel(dataPtrSparse_CPU)

    # GPU: long out and long cov
    dataPtrSparse_GPU <- convertToCyclopsData(outcomes = longOut,
                                              covariates = sparseLongCov,
                                              modelType = "cox_time")
    cyclopsFitSparse_GPU <- fitCyclopsModel(dataPtrSparse_GPU, computeDevice = GpuDevice)

    expect_equal(unname(coef(cyclopsFitSparse_CPU)), unname(coef(goldStrat)), tolerance = tolerance)
    expect_equal(coef(cyclopsFitSparse_CPU), coef(cyclopsFitSparse_GPU), tolerance = tolerance)

    # GPU: long out and short cov
    shortCovPtrSparse_GPU <- convertToCyclopsData(outcomes = longOut,
                                                  covariates = sparseShortCov,
                                                  timeEffectMap = data.frame(covariateId = c(2)),
                                                  modelType = "cox_time")
    shortCovFitSparse_GPU <- fitCyclopsModel(shortCovPtrSparse_GPU, computeDevice = GpuDevice)
    expect_equal(unname(coef(cyclopsFitSparse_GPU)), unname(coef(shortCovFitSparse_GPU)), tolerance = tolerance)

})

