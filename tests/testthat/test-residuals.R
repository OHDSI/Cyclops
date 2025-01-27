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

    ttimes <- ovarian$futime - mean(ovarian$futime[ovarian$fustat == 1])

    ctest <- testProportionality(cfit, parm = NULL, transformedTimes = ttimes)

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

    ttimes <- ovarian$futime - mean(ovarian$futime[ovarian$fustat == 1])

    ctest <- testProportionality(cfit, parm = NULL, transformedTimes = ttimes)

    expect_equal(ctest$table, gtest$table)
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

    gfit <- coxph(Surv(length, event) ~ x1, test, ties = "breslow")
    gres <- residuals(gfit, "schoenfeld")
    gtest <- cox.zph(gfit, transform = "identity", global = FALSE)


    data <- createCyclopsData(Surv(length, event) ~ x1,
                              # sparseFormula = ~ x1,
                              data = test, modelType = "cox")

    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld") # TODO broken (and not even sparse yet)

    ttimes <- test$length - mean(test$length[test$event == 1])
    ctest <- testProportionality(cfit, parm = NULL, transformedTimes = ttimes)


    # expect_equivalent(cres, gres) # TODO
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

    data <- createCyclopsData(Surv(length, event) ~ x1+ strata(x2),
                                    # sparseFormula = ~ x1,
                                    data = test, modelType = "cox")

    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld") # TODO broken (and not even sparse yet)
    # expect_equivalent(cres, gres) # TODO
})
