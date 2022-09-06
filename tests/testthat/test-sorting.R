library("testthat")

context("test-sorting.R")

#
# Sort functions
#

test_that("Check sorting functions", {
    set.seed(123)
    vector <- c(1:100)

    expect_true(Cyclops:::.isSortedVectorList(vectorList = list(vector[1]), ascending = FALSE))
    expect_true(Cyclops:::.isSortedVectorList(vectorList = list(vector), ascending = c(TRUE)))
    expect_false(Cyclops:::.isSortedVectorList(vectorList = list(vector), ascending = c(FALSE)))
})
