library("testthat")
library("survival")

context("test-infiniteMLE.R")

test_that("Check for infinite MLE in Cox example with no outcomes in one treatment arm", {

    set.seed(123)
    n <- 1000
    timeToCensor <- rexp(n, 0.01)
    exposure <- runif(n) < 0.5
    timeToEvent <- rep(Inf, n)
    timeToEvent[!exposure] <- rexp(sum(!exposure), 0.01)
    time <- pmin(timeToCensor, timeToEvent)
    outcome <- timeToEvent < timeToCensor

    expect_warning(gold <- coxph(Surv(time, outcome) ~ exposure), regexp = ".*coefficient may be infinite.*")

    cyclopsData <- createCyclopsData(Surv(time, outcome) ~ exposure , modelType = "cox")
    expect_warning(fit <- fitCyclopsModel(cyclopsData), regexp =".*coefficient may be infinite.*")

    ci <- confint(fit, parm = "exposureTRUE", level = 0.9)
    expect_true(is.na(ci[2]))
})
