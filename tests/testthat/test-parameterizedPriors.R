library("testthat")

context("test-parameterizedPriors.R")

test_that("lazy parameterization evaluation", {

    # initially invalid
    test <- Cyclops:::.cyclopsTestParameterizedPrior(priorFunction = function(x) { list(c(x[1],x[2])) },
                                             startingParameters = c(0, 0),
                                             indices = c(0), # no set
                                             values = c(1))
    expect_equal(test$valid, 0)

    # invalid after one-update and then returns correct evaluation
    test <- Cyclops:::.cyclopsTestParameterizedPrior(priorFunction = function(x) { list(c(x[1],x[2])) },
                                                     startingParameters = c(0, 0),
                                                     indices = c(1),
                                                     values = c(1))
    expect_equal(test$valid, 0)
    expect_equal(test$evaluation, list(c(1,0)))

    # multiple hits
    test <- Cyclops:::.cyclopsTestParameterizedPrior(priorFunction = function(x) { list(c(x[1],x[2])) },
                                                     startingParameters = c(0, 0),
                                                     indices = c(1,2,0),
                                                     values = c(1,2,0))
    expect_equal(test$valid, c(0,0,1)) # already valid at last index
    expect_equal(test$evaluation[[3]], c(1,2))
})

test_that("Specify parameterized L1 regularization", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)

    data <- createCyclopsData(counts ~ outcome + treatment,
                              modelType = "pr")

    expect_error(
        prior <- createParameterizedPrior(priorType = "laplace",
                                          parameterize =  function(x) { list(c(0,x)) })
        , "provide parameter values")

    expect_error(
        prior <- createParameterizedPrior(priorType = "laplace",
                                          values = c(1),
                                          parameterize =  function(x) { c(0,x) })
        , "return a list")

    expect_error(
        prior <- createParameterizedPrior(priorType = c("laplace", "laplace"),
                                          values = c(1),
                                          parameterize =  function(x) { list(c(0,x)) })
        , "dimensionality mismatch")

    prior <- createParameterizedPrior(priorType = c(rep("laplace", 5)),
                                      values = c(2),
                                      parameterize = function(x) {
                                          lapply(1:5, function(i) {
                                              c(0,x)
                                          })
                                      })

    expect_warning(
        fit <- fitCyclopsModel(data, prior, forceNewObject = TRUE),
        "Excluding intercept")

    comp <- fitCyclopsModel(data, prior = createPrior("laplace", variance = 2), forceNewObject = TRUE)

    expect_equal(coef(fit), coef(comp))

    prior <- createParameterizedPrior(priorType = c(rep("laplace", 5)),
                                      values = c(0.2, 0.1),
                                      parameterize = function(x) {
                                          lapply(1:5, function(i) {
                                              c(x[1],x[2])
                                          })
                                      })

    fit <- fitCyclopsModel(data, prior, forceNewObject = TRUE)

    expect_equivalent(coef(fit)[4:5], c(0.2, 0.2))
})

test_that("Using parameterized cross-validation", {
    skip_on_cran() # Do not run on CRAN

    ntest <- 1000
    ntrain <- 1000
    ncovars <- 2000

    set.seed(666)

    data <- simulateCyclopsData(nstrata=1,nrows=ntest+ntrain,ncovars=ncovars,model="logistic")
    test <- list(outcomes = data$outcomes[1:ntest,], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[1:ntest],])
    train <- list(outcomes = data$outcomes[(ntest+1):(ntest+ntrain),], covariates = data$covariates[data$covariates$rowId %in% data$outcomes$rowId[(ntest+1):(ntest+ntrain)],])
    cyclopsData <- convertToCyclopsData(train$outcomes,train$covariates,modelType = "lr",addIntercept = TRUE)

    prior1 <- createParameterizedPrior(c("none", rep("laplace", ncovars)),
                                      parameterize = function(x) {
                                          lapply(1:(ncovars + 1), function(i) { c(0,x) })
                                      },
                                      values = c(1),
                                      useCrossValidation = TRUE)

    prior2 <- createPrior("laplace", exclude = c(0), useCrossValidation = TRUE)

    control <- createControl(noiseLevel = "silent", cvType = "auto", cvRepetitions = 1, seed = 666, threads = 1, resetCoefficients = TRUE)

    time1 <- system.time(fit1 <- fitCyclopsModel(cyclopsData,
                                                 prior=prior1,
                                                 control=control,
                                                 forceNewObject = TRUE))

    time2 <- system.time(fit2 <- fitCyclopsModel(cyclopsData,
                                                 prior=prior2,
                                                 control=control,
                                                 forceNewObject = TRUE))
    expect_equal(fit1$variance, fit2$variance)

    prior3 <- createParameterizedPrior(c("none", rep("laplace", ncovars)),
                                       parameterize = function(x) {
                                           lapply(1:(ncovars + 1),
                                                  function(i) {
                                                      if (i < (ncovars / 2)) {
                                                          return(c(0,x[1]))
                                                      } else {
                                                          return(c(0,2 * x[2]))
                                                      }
                                                  }
                                           )
                                       },
                                       values = c(1, 0.5),
                                       useCrossValidation = TRUE)

    # time3 <- system.time(fit3 <- fitCyclopsModel(cyclopsData,
    #                                              prior=prior3,
    #                                              control=control,
    #                                              forceNewObject = TRUE))


})

