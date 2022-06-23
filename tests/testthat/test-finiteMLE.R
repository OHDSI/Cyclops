library("testthat")
library("survival")

context("test-infiniteMLE.R")

simulate <- function() {
    set.seed(123)
    n <- 1000
    timeToCensor <- rexp(n, 0.01)
    exposure <- runif(n) < 0.5
    timeToEvent <- rep(Inf, n)
    timeToEvent[!exposure] <- rexp(sum(!exposure), 0.01)
    time <- pmin(timeToCensor, timeToEvent)
    outcome <- timeToEvent < timeToCensor
    list(time = time, outcome = outcome, exposure = exposure)
}

test_that("Check for infinite MLE in Cox example with no outcomes in one treatment arm", {

    data <- simulate()

    expect_warning(gold <- coxph(Surv(time, outcome) ~ exposure, data = data),
                   regexp = ".*coefficient may be infinite.*")

    cyclopsData <- createCyclopsData(Surv(time, outcome) ~ exposure, data = data, modelType = "cox")
    expect_warning(fit <- fitCyclopsModel(cyclopsData), regexp =".*coefficient may be infinite.*")

    ci <- confint(fit, parm = "exposureTRUE", level = 0.9)
    expect_true(is.na(ci[2]))
})

test_that("Check multivariable Jeffreys prior", {

    data <- simulate()
    data$x2 <- runif(length(data$time), 0.5) < 0.5

    cyclopsData <- createCyclopsData(Surv(time, outcome) ~ exposure + x2, data = data, modelType = "cox")
    expect_error(fit <- fitCyclopsModel(cyclopsData,
                                        prior = createPrior(priorType = "jeffreys")),
                 regexp = ".*1 covariate.*")
})

test_that("Check Jeffreys prior with correct control", {

    data <- simulate()

    cyclopsData <- createCyclopsData(Surv(time, outcome) ~ exposure, data = data, modelType = "cox")
    expect_warning(fit <- fitCyclopsModel(cyclopsData,
                                        prior = createPrior(priorType = "jeffreys")),
                 regexp = "BLR convergence")

    fit <- fitCyclopsModel(cyclopsData = cyclopsData, forceNewObject = TRUE,
                           prior = createPrior(priorType = "jeffreys"),
                           control = createControl(convergenceType = "lange"))
    expect_true(fit$return_flag == "SUCCESS")
})

test_that("Check Jeffreys prior with indicator only", {

    data <- data.frame(time = c(1,2),
                       outcome = c(1,0),
                       exposure = c(1,2))

    cyclopsData <- createCyclopsData(Surv(time, outcome) ~ exposure, data = data, modelType = "cox")
    expect_error(fitCyclopsModel(cyclopsData, prior = createPrior(priorType = "jeffreys")),
                 regexp = "*.for indicator covariates")

})

test_that("Check univariable Jeffreys prior", {

    data <- simulate()

    cyclops1 <- createCyclopsData(Surv(time, outcome) ~ exposure, data = data, modelType = "cox")
    cyclops2 <- createCyclopsData(Surv(time, outcome) ~ exposure, data = data, modelType = "cox")

    expect_warning(uniform <- fitCyclopsModel(cyclops1))

    jeffreys <- fitCyclopsModel(cyclops2,
                                prior = createPrior(priorType = "jeffreys"),
                                control = createControl(convergenceType = "lange",
                                                        tolerance = 1E-7),
                                forceNewObject = TRUE)

    jci <- confint(jeffreys, parm = "exposureTRUE", level = 0.95, overrideNoRegularization = TRUE)
    expect_false(is.na(jci[2]))

    # grid <- seq(from=-8, to = -2, length.out = 1000)
    # uniformProfile <- getCyclopsProfileLogLikelihood(uniform, parm = "exposureTRUE", x = grid)
    # jeffreysProfile <- getCyclopsProfileLogLikelihood(jeffreys, parm = "exposureTRUE", x = grid)
    #
    # plot(uniformProfile$point, uniformProfile$value - max(uniformProfile$value), type = "l")
    # lines(jeffreysProfile$point, jeffreysProfile$value - max(uniformProfile$value, lwd = 2))
    #
    # plot(jeffreysProfile$point, jeffreysProfile$value - max(jeffreysProfile$value), type = "l")
    #
    # maxProfile <- function(profile) {
    #     idx <- which(profile$value == max(profile$value))
    #     list(point = profile$point[idx], value = profile$value[idx])
    # }
    #
    # maxProfile(jeffreysProfile)
})
