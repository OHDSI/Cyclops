library("testthat")

#
# Tests for the XY constructor of ModelData
#

test_that("Test constructor", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)

    oStratumId <- c(1:9)
    oRowId <- c(1:9)
    oY <- c(18,17,15,20,10,20,25,13,12)
    oTime <- rep(0,9)
    cRowId <- c(
        1,
        2,2,
        3,3,
        4,4,
        5,5,5,
        6,6,6,
        7,7,
        8,8,8,
        9,9,9)
    cCovariateId <- c(
        1,
        1,2,
        1,3,
        1,4,
        1,2,4,
        1,3,4,
        1,5,
        1,2,5,
        1,3,5)
    cCovariateValue <- rep(1, 21)

    dataPtr <- createSqlCyclopsData(modelType = "pr")

    count <- loadNewSqlCyclopsDataY(dataPtr, oStratumId, oRowId, oY, oTime)

    count <- loadNewSqlCyclopsDataX(dataPtr, 0, c(1:9), cCovariateValue[1:9],
                                    reload = FALSE,
                                    append = FALSE)

    count <- loadNewSqlCyclopsDataX(dataPtr, 1, as.integer(c()), cCovariateValue[1:9],
                                    reload = FALSE,
                                    append = FALSE)

    count <- loadNewSqlCyclopsDataX(dataPtr, 2, c(1:9), as.numeric(c()),
                                    reload = FALSE,
                                    append = FALSE)

    count <- loadNewSqlCyclopsDataX(dataPtr, 3, as.integer(c()), as.numeric(c()),
                                    reload = FALSE,
                                    append = FALSE)

    summary(dataPtr)
    dataPtr




})
