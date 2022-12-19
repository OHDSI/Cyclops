library("testthat")

context("test-cv.R")
suppressWarnings(RNGversion("3.5.0"))

# ### COX ERROR
#
# library(Cyclops)
# set.seed(100)
# data <- simulateCyclopsData(nstrata=1,nrows=1000,ncovars=200,model="survival")
# cyclopsData <- convertToCyclopsData(data$outcomes,data$covariates,modelType = "cox")
#
# prior <- createPrior("laplace", useCrossValidation = TRUE)
# control <- createControl(noiseLevel = "quiet",lowerLimit = 0.000001,upperLimit = 100,
#                          cvType = "auto", startingVariance = 0.1,
#                          seed = 100,
#                          selectorType = "default")
# fit <- fitCyclopsModel(cyclopsData,prior=prior,control=control)
#
#
# fit <- fitCyclopsModel(cyclopsData,
#                        prior = createPrior("laplace", variance = 0.0464159),
#                        control=control)
#
# # This generates nan on first eval
# library(Cyclops)
# set.seed(100)
# data <- simulateCyclopsData(nstrata=1,nrows=1000,ncovars=200,model="survival")
# cyclopsData <- convertToCyclopsData(data$outcomes,data$covariates,modelType = "cox")
# prior <- createPrior("laplace", useCrossValidation = TRUE)
# control <- createControl(noiseLevel = "quiet",lowerLimit = 0.0464159,upperLimit = 0.0464159,gridSteps=1,
#                          seed = 100,
#                          selectorType = "default")
# fit <- fitCyclopsModel(cyclopsData,prior=prior,control=control)

test_that("Grid in R and auto-search in C++", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")

    seed <- 666
    set.seed(seed)

    nrows <- 1000
    ncovars <- 200

    data <- simulateCyclopsData(nstrata = 1,
                                nrows = nrows,
                                ncovars = ncovars,
                                model = "logistic")

    cyclopsData <- convertToCyclopsData(data$outcomes,
                                        data$covariates,
                                        modelType = "lr",
                                        addIntercept = TRUE)

    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    control <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed)

    out <- capture.output(fit <- fitCyclopsModel(cyclopsData, prior, control))

    cv <- Cyclops:::getCrossValidationInfo(fit)

    printValue <- as.numeric(gsub(".* log likelihood \\((.*)\\) estimated.*", "\\1",
                                  capture.output(cat(out))))

    expect_equal(cv$ordinate, printValue)

    outerGrid <- c(0.5, 1, 2)
    expect_error(createAutoGridCrossValidationControl(outerGrid, autoPosition = 0), "Auto-position")
    expect_error(createAutoGridCrossValidationControl(outerGrid, autoPosition = 3), "Auto-position")

    expect_error(
        fitCyclopsModel(cyclopsData,
                        prior = prior,
                        control = createAutoGridCrossValidationControl(outerGrid)),
        "Auto-grid"
    )

    prior <-
        createParameterizedPrior(
            c("none", rep("laplace", ncovars)),
            parameterize = function(x) {
                lapply(1:(ncovars + 1),
                       function(i) {
                           if (i < (ncovars / 2)) {
                               return(c(0, x[1]))
                           } else {
                               return(c(0, x[1] * x[2]))
                           }
                       })
            },
            values = c(1, 0.5),
            useCrossValidation = TRUE
        )

    fit <- fitCyclopsModel(cyclopsData,
                    prior = prior,
                    control = createAutoGridCrossValidationControl(outerGrid,
                                                                   autoPosition = 1))
    expect_equal(length(fit$searchResults), 3)
    expect_equal(length(Cyclops:::getCrossValidationInfo(fit)$point), 2)

})

test_that("Specify starting variance with auto-search", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")
    seed <- 666
    set.seed(seed)
    ntrain <- 100
    data <- simulateCyclopsData(nstrata = 1,
                         nrows = ntrain,
                         ncovars = 2000,
                         model = "logistic")
    cyclopsData <- convertToCyclopsData(data$outcomes,
                                        data$covariates,
                                        modelType = "lr",
                                        addIntercept = TRUE)
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)
    control1 <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed)
    out1 <- capture.output(fit1 <- fitCyclopsModel(cyclopsData,
                            prior = prior,
                            control = control1))

    control2 <- createControl(noiseLevel = "quiet",
                              cvType = "auto",
                              cvRepetitions = 1,
                              seed = seed,
                              startingVariance = 0.1)

    out2 <- capture.output(fit2 <- fitCyclopsModel(cyclopsData,
                                                   prior = prior,
                                                   control = control2))

    expect_true(grep("default", out1[3]) == 1)
    expect_true(length(grep("default", out2[3])) == 0)
    expect_false(out1[3] == out2[3])
})

test_that("Using multi-core CV", {
    skip_on_cran() # Do not run on CRAN
    skip("Do not run")
    ntest <- 1000
    ntrain <- 1000

    set.seed(666)

    data <- simulateCyclopsData(nstrata=1,nrows=ntest+ntrain,ncovars=2000,model="logistic")
    test <- list(outcomes = data$outcomes[1:ntest,], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[1:ntest],])
    train <- list(outcomes = data$outcomes[(ntest+1):(ntest+ntrain),], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[(ntest+1):(ntest+ntrain)],])
    cyclopsData <- convertToCyclopsData(train$outcomes,train$covariates,modelType = "lr",addIntercept = TRUE)
    prior <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    # 1 thread, cold start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 1, resetCoefficients = TRUE)
    time1 <- system.time(fit1 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 4 thread, cold start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 4, resetCoefficients = TRUE)
    time2 <- system.time(fit2 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 1 thread, warm start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 1)
    time3 <- system.time(fit3 <- fitCyclopsModel(cyclopsData,prior=prior,control=control, forceNewObject = TRUE))

    # 4 thread, warm start
    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 4)
    time4 <- system.time(fit4 <- fitCyclopsModel(cyclopsData,prior=prior,control=control,  forceNewObject = TRUE))

    cat("Variance estimates\n")
    cat(fit1$variance,"\n")
    cat(fit2$variance,"\n")
    cat(fit3$variance,"\n")
    cat(fit4$variance,"\n")

    # Cold starting each model fit will return equal single- to multi-core estimates
    expect_equal(fit1$variance, fit2$variance)

    cat("Times\n")
    cat(time1,"\n")
    cat(time2,"\n")
    cat(time3,"\n")
    cat(time4,"\n")

    # Multi-core should be faster
    expect_less_than(time2[3], time1[3])
    # Warm starting should be faster
    expect_less_than(time3[3], time1[3])
})

test_that("Seed gets returned", {
    y <- 0
    x <- 1
    data <- createCyclopsData(y ~ x, modelType = "lr")
    fit <- fitCyclopsModel(data, control = createControl(seed = 123), warnings = FALSE)
    expect_equal(fit$seed, 123)

    fit <- fitCyclopsModel(data, forceNewObject = TRUE,
                           control = createControl(seed = NULL), warnings = FALSE)
    expect_true(!is.null(fit$seed))
})
