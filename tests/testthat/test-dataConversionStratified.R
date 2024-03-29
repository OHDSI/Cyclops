library("testthat")
library("survival")
library("gnm")
library("Andromeda")

context("test-dataConversionStratified.R")
suppressWarnings(RNGversion("3.5.0"))

test_that("Test clr", {

  gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)

  #Convert infert dataset to Cyclops format:
  covariates <- data.frame(stratumId = rep(infert$stratum,2),
                           rowId = rep(1:nrow(infert),2),
                           covariateId = rep(1:2,each=nrow(infert)),
                           covariateValue = c(infert$spontaneous,infert$induced))
  outcomes <- data.frame(stratumId = infert$stratum,
                         rowId = 1:nrow(infert),
                         y = infert$case)
  #Make sparse:
  covariates <- covariates[covariates$covariateValue != 0,]

  andr <- andromeda(outcomes = outcomes, covariates = covariates)
  cyclopsDataAndr <- convertToCyclopsData(andr$outcomes, andr$covariates, modelType = "clr", addIntercept = FALSE)
  fitAndr <- fitCyclopsModel(cyclopsDataAndr,prior = createPrior("none"))

  cyclopsData <- convertToCyclopsData(outcomes,covariates,modelType = "clr",addIntercept = FALSE)
  fit <- fitCyclopsModel(cyclopsData,prior = createPrior("none"))

  tolerance <- 1E-4
  expect_equal(as.vector(coef(fit)), as.vector(coef(gold)), tolerance = tolerance)
  expect_equal(as.vector(coef(fitAndr)), as.vector(coef(gold)), tolerance = tolerance)
})

test_that("Test stratified cox", {

  test1 <- data.frame(time=c(4,3,1,1,2,2,3),
                      status=c(1,1,1,0,1,1,0),
                      x=c(0,2,1,1,1,0,0),
                      sex=c(0,0,0,0,1,1,1))

  gold <- coxph(Surv(time, status) ~ x + strata(sex), test1, method="breslow")

  cyclopsDataFormula <- createCyclopsData(Surv(time, status) ~ x + strata(sex), data=test1,modelType="cox")
  fitFormula <- fitCyclopsModel(cyclopsDataFormula)

  #Convert to data.frames for Cyclops:
  covariates <- data.frame(stratumId = test1$sex,
                           rowId = 1:nrow(test1),
                           covariateId = rep(1,nrow(test1)),
                           covariateValue = test1$x)
  outcomes <- data.frame(stratumId = test1$sex,
                         rowId = 1:nrow(test1),
                         y = test1$status,
                         time = test1$time)

  #Make sparse:
  covariates <- covariates[covariates$covariateValue != 0,]

  andr <- andromeda(outcomes = outcomes, covariates = covariates)
  cyclopsDataAndr <- convertToCyclopsData(andr$outcomes, andr$covariates ,modelType = "cox")
  fitAndr <- fitCyclopsModel(cyclopsDataAndr,prior = createPrior("none"))

  cyclopsData <- convertToCyclopsData(outcomes,covariates,modelType = "cox")
  fit <- fitCyclopsModel(cyclopsData,prior = createPrior("none"))

  tolerance <- 1E-4
  expect_equal(as.vector(coef(fit)), as.vector(coef(fitFormula)), tolerance = tolerance)
  expect_equal(as.vector(coef(fit)), as.vector(coef(gold)), tolerance = tolerance)
  expect_equal(as.vector(coef(fitAndr)), as.vector(coef(gold)), tolerance = tolerance)
})

test_that("Test stratified cox using lung dataset ", {

  test <- lung
  test[is.na(test)] <- 0 # Don't want to bother with missing values

  gold <- coxph(Surv(time, status) ~ age + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss + strata(sex), test, method="breslow")

  cyclopsDataFormula <- createCyclopsData(Surv(time, status) ~ age + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss + strata(sex), data=test,modelType="cox")
  fitFormula <- fitCyclopsModel(cyclopsDataFormula)

  #Convert to data.frames for Cyclops:
  nCovars = 6
  covariates <- data.frame(stratumId = rep(test$sex,nCovars),
                           rowId = rep(1:nrow(test),nCovars),
                           covariateId = rep(1:nCovars,each = nrow(test)),
                           covariateValue = c(test$age,test$ph.ecog,test$ph.karno,test$pat.karno,test$meal.cal,test$wt.loss))
  outcomes <- data.frame(stratumId = test$sex,
                         rowId = 1:nrow(test),
                         y = test$status-1,
                         time = test$time)

  #Make sparse:
  covariates <- covariates[covariates$covariateValue != 0,]

  andr <- andromeda(outcomes = outcomes, covariates = covariates)
  cyclopsDataAndr <- convertToCyclopsData(andr$outcomes,andr$covariates,modelType = "cox")
  fitAndr <- fitCyclopsModel(cyclopsDataAndr,prior = createPrior("none"))

  cyclopsData <- convertToCyclopsData(outcomes,covariates,modelType = "cox")
  fit <- fitCyclopsModel(cyclopsData,prior = createPrior("none"))

  tolerance <- 1E-4
  expect_equal(as.vector(coef(fit)), as.vector(coef(fitFormula)), tolerance = tolerance)
  expect_equal(as.vector(coef(fit)), as.vector(coef(gold)), tolerance = tolerance)
  expect_equal(as.vector(coef(fitAndr)), as.vector(coef(gold)), tolerance = tolerance)
})


test_that("Test conditional poisson regression", {
  skip_on_cran() # Do not run on CRAN
  #skip("Do not run")
  set.seed(123)
  sim <- simulateCyclopsData(nstrata = 2, nrows = 10000, ncovars = 2, eCovarsPerRow = 0.5, effectSizeSd = 1,model = "poisson")
  covariates <- sim$covariates
  outcomes <- sim$outcomes

  #Convert to data format for gnm:
  ncovars <- max(covariates$covariateId)
  nrows <- nrow(outcomes)
  m <- matrix(0,nrows,ncovars)
  for (i in 1:nrow(covariates)){
    m[covariates$rowId[i],covariates$covariateId[i]] <- 1
  }
  data <- as.data.frame(m)

  data$rowId <- 1:nrow(data)
  data <- merge(data,outcomes)
  data <- data[order(data$stratumId,data$rowId),]
  formula <- as.formula(paste(c("y ~ V1",paste("V",2:ncovars,sep="")),collapse=" + "))

  gold <- gnm(formula, family=poisson, offset=log(time), eliminate=as.factor(stratumId), data = data)

  formula <- as.formula(paste(c("y ~ V1",paste("V",2:ncovars,sep=""),"strata(stratumId)"),collapse=" + "))
  cyclopsDataFormula <- createCyclopsData(formula, offset=log(time), data=data,modelType="cpr")
  fitFormula <- fitCyclopsModel(cyclopsDataFormula)

  andr <- andromeda(outcomes = outcomes, covariates = covariates)
  cyclopsDataAndr <- convertToCyclopsData(andr$outcomes,andr$covariates,modelType = "cpr",addIntercept = FALSE)
  fitAndr <- fitCyclopsModel(cyclopsDataAndr,prior = createPrior("none"))

  cyclopsData <- convertToCyclopsData(outcomes,covariates,modelType = "cpr",addIntercept = FALSE)
  fit <- fitCyclopsModel(cyclopsData,prior = createPrior("none"))

  tolerance <- 1E-4
  expect_equal(as.vector(sort(coef(fit))), as.vector(sort(coef(fitFormula))), tolerance = tolerance)
  expect_equal(as.vector(sort(coef(fit))), as.vector(sort(coef(gold))), tolerance = tolerance)
  expect_equal(as.vector(sort(coef(fitAndr))), as.vector(sort(coef(gold))), tolerance = tolerance)
})
