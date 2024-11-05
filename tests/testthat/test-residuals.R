library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check Schoenfeld residual tests", {
    gfit <- coxph(Surv(futime, fustat) ~ age + ecog.ps,
                  data=ovarian)
    gres <- residuals(gfit, "schoenfeld")[,1]

    data <- createCyclopsData(Surv(futime, fustat) ~ age + ecog.ps,
                              data=ovarian, modelType = "cox")
    cfit <- fitCyclopsModel(data)
    cres <- residuals(cfit, "schoenfeld")

    expect_equal(cres, gres)

    gold <- cox.zph(gfit, transform = "identity")

    summary(lm(cres ~ as.numeric(names(cres)) - 1))
    logLik(lm(cres ~ as.numeric(names(cres)) - 1))

    testProportionality(cfit, parm = NULL, transformedTimes = ovarian$futime)

})

