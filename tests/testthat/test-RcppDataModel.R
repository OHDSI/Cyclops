library(testthat)

test_that("quantile", {
    set.seed(123)
    vector <- runif(100)

    expect_equal(
        Cyclops:::.cyclopsQuantile(vector, q = 0.5),
        median(vector))
    expect_equal(
        Cyclops:::.cyclopsMedian(vector),
        median(vector))

    expect_error(
        Cyclops:::.cyclopsQuantile(vector, q = -0.1),
        "Invalid quantile")
})

test_that("null ptr", {
    expect_error(
        Cyclops:::.isRcppPtrNull(1),
        "Input must be an Rcpp externalptr")
})

test_that("print MatrixMarket format", {
    binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
    binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
    binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

    log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
    y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))
    data <- createCyclopsData(y ~ log_bid, modelType = "lr")

    tempFile <- tempfile()
    Cyclops:::printMatrixMarket(data, file = tempFile)
    text <- readr::read_delim(file = tempFile, delim = " ", skip = 3, col_names = FALSE, col_types = readr::cols())
    expect_equal(nrow(text), getNumberOfCovariates(data) * getNumberOfRows(data))
    expect_equal(ncol(text), 1 + getNumberOfCovariates(data))

    unlink(tempFile)
})
