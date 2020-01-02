library("testthat")
library("survival")

suppressWarnings(RNGversion("3.5.0"))

test_that("Check very small Cox example with no ties, but with/without strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3.5,1,2,0
0, 3,  0,0,1
0, 2.5,1,0,1
0, 2,  1,1,1
0, 1.5,0,1,0
0, 1,  1,1,0
")

    data <- createCyclopsData(Surv(length, event) ~ x1 + x2, data = test,
                              modelType = "cox")
    fit <- fitCyclopsModel(data) # TODO Fix so not needed

    x <- seq(from = -2, to = 2, length = 10)
    out <- getCyclopsProfileLogLikelihood(fit, "x1", x)

    expect_equal(out$value, x)
})

