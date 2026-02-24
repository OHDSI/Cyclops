library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check Schoenfeld residuals and PH test, no strata", {
    gfit <- coxph(Surv(futime, fustat) ~ age,
                  data=ovarian, method = "breslow")
    gres <- residuals(gfit, "schoenfeld")

    ovarian$mage <- ovarian$age - mean(ovarian$age)

    data <- createCyclopsData(Surv(futime, fustat) ~ age,
                              data=ovarian, modelType = "cox")
    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld")

    expect_equal(cres, gres)

    gtest <- cox.zph(gfit, transform = "identity", global = FALSE)
    ctest <- testProportionality(cfit, parm = NULL, transform = "identity")

    expect_equal(ctest$table, gtest$table)
})

test_that("Check Schoenfeld residuals and PH test, with strata", {
    gfit <- coxph(Surv(futime, fustat) ~ age + strata(ecog.ps),
                  data=ovarian, method = "breslow")
    gres <- residuals(gfit, "schoenfeld")

    data <- createCyclopsData(Surv(futime, fustat) ~ age + strata(ecog.ps),
                              data=ovarian, modelType = "cox")
    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld")

    expect_equivalent(cres, gres)

    gtest <- cox.zph(gfit, transform = "identity", global = FALSE)
    ctest <- testProportionality(cfit, parm = NULL, transform = "identity")

    expect_equal(ctest$table, gtest$table)
})

test_that("Check Schoenfeld residuals and PH test, with sparse covariates", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0.2,0
0, 3,  0,0,1
0, 3,  1,2,0 # was x1 = 2
0, 2,  1,0.5,1
0, 2,  1,0.1,1
0, 2,  1,1.2,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    gfit <- coxph(Surv(length, event) ~ x1, test, ties = "breslow")
    gres <- residuals(gfit, "schoenfeld")
    gtest <- cox.zph(gfit, transform = "km", global = FALSE)

    data <- createCyclopsData(Surv(length, event) ~ 1,
                              sparseFormula = ~ x1,
                              data = test, modelType = "cox")

    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld")

    expect_equal(gres, cres)

    ctest <- testProportionality(cfit, parm = NULL, transform = "km")

    expect_equal(gtest$table, ctest$table)

    ctest2 <- testProportionality(cfit, parm = NULL)

    expect_equal(ctest, ctest2)

#     #  check in Stata
#
#     clear
#     input start length event x1 x2
#     0 4 1   0   0
#     0 3 0   0   1
#     0 3 1   2   0   // was x1 = 2
#     0 2 1   0.1 1
#     0 2 1   1.2 1
#     0 1 0   1   0   // was 0
#     0 1 1   1   0   // was 1
#     end
#
#     stset length, failure(event)
#     stcox x1
#     predict sch_x1, schoenfeld
#     show sch_x1
})

test_that("Check residuals on large Cox regression with weighting",{
    tolerance <- 1E-4
    set.seed(123)
    sim <- simulateCyclopsData(nstrata=1000, #1
                               ncovars=1,
                               nrows=10000, #30
                               effectSizeSd=0.5,
                               eCovarsPerRow=2,
                               model="survival")
    sim$outcomes$weights <- 1/sim$outcomes$rr
    # sim$outcomes$weights <- rep(1, length(sim$outcomes$weights))
    sim$outcomes$weights <- runif(length(sim$outcomes$weights))

    # Gold standard
    covariates <- sim$covariates
    ncovars <- max(covariates$covariateId)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$rowId[i],covariates$covariateId[i]] <- 1
    }
    data <- as.data.frame(m)

    data$rowId <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratumId,data$rowId),]
    formula <- as.formula(paste(c("Surv(time,y) ~ strata(stratumId)",paste("V",1:ncovars,sep="")),collapse=" + "))
    fitCoxph <- survival::coxph(formula, data = data, weights = data$weights, ties = "breslow")

    # Cyclops
    cyclopsData <- convertToCyclopsData(outcomes = sim$outcomes,
                                        covariates = sim$covariates,
                                        modelType = "cox")
    fitCyclops <- fitCyclopsModel(cyclopsData = cyclopsData)

    expect_equivalent(coef(fitCyclops), coef(fitCoxph), tolerance = tolerance)

    resCoxph <- residuals(fitCoxph, type = "schoenfeld")
    resCyclops <- residuals(fitCyclops, type = "schoenfeld")

    expect_equivalent(resCoxph, resCyclops)

    gtest1 <- cox.zph(fitCoxph, transform = "km", global = FALSE)
    ctest1 <- testProportionality(fitCyclops, transform = "km")
    expect_equivalent(gtest1$table, ctest1$table)

    gtest2 <- cox.zph(fitCoxph, transform = "identity", global = FALSE)
    ctest2 <- testProportionality(fitCyclops, transform = "identity")
    expect_equivalent(gtest2$table, ctest2$table)
})

test_that("Check Schoenfeld residuals and PH test, with sparse covariates", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0
")

    gfit <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")
    gres <- residuals(gfit, "schoenfeld")

    data <- createCyclopsData(Surv(length, event) ~ strata(x2),
                                    sparseFormula = ~ x1,
                                    data = test, modelType = "cox")

    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld")

    expect_equivalent(cres, gres)
})
