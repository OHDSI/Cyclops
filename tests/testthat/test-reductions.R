library("testthat")

context("test-reductions.R")

test_that("Simple reductions", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    dataPtr <- createCyclopsData(counts ~ 1,
                                  indicatorFormula = ~ outcome + treatment,
                                  modelType = "pr")

    expect_error(reduce(dataPtr, 0))
    expect_error(reduce(dataPtr, "BAD"))
    expect_equal(reduce(dataPtr, c(1,2)),
                 c(9,3))

    expect_equivalent(reduce(dataPtr, 4, groupBy = 3),
                      as.data.frame(c(2,1)))
    expect_equivalent(reduce(dataPtr, 3, groupBy = "treatment2"),
                      as.data.frame(c(2,1)))
    expect_equal(dim(reduce(dataPtr, 3, groupBy = "stratum")),
                 c(9,1))
    expect_error(reduce(dataPtr, 4, groupBy = c(3,1)))

    #throw error? when # strata = # row
})
