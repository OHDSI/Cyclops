library(survival)
library(testthat)

test_that("Check very small Cox example with time-varying coefficient as stratified model", {
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

    tolerance <- 1E-4
    expect_equal(unname(coef(cyclopsFitStrat_CPU)), unname(coef(goldStrat)), tolerance = tolerance)
})

