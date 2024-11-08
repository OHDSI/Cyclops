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
