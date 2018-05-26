library("testthat")

context("test-sqlConstructor.R")

#
# Tests for the SQL constructor of ModelData
#

test_that("Test constructor and append", {
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

    # Should have no data
    expect_equal(getNumberOfRows(dataPtr), 0)
    expect_equal(getNumberOfStrata(dataPtr), 0)
    expect_equal(getNumberOfCovariates(dataPtr), 0)
    expect_error(fitCyclopsModel(dataPtr))

    count <- appendSqlCyclopsData(dataPtr,
                              oStratumId,
                              oRowId,
                              oY,
                              oTime,
                              cRowId,
                              cCovariateId,
                              cCovariateValue)
    finalizeSqlCyclopsData(dataPtr) # Not yet implemented

    expect_equal(count, 9)
    expect_equal(getNumberOfRows(dataPtr), 9)
    expect_equal(getNumberOfStrata(dataPtr), 9)
    expect_equal(getNumberOfCovariates(dataPtr), 5)

    cyclopsFit <- fitCyclopsModel(dataPtr, control = createControl(noiseLevel = "silent"))

    dataPtrF <- createCyclopsData(counts ~ outcome + treatment, modelType = "pr")
    cyclopsFitF <- fitCyclopsModel(dataPtrF, control = createControl(noiseLevel = "silent"))
    expect_equivalent(coef(cyclopsFit), coef(cyclopsFitF)) # Have different coefficient names

    # Test chucked append
    dataPtrC <- createSqlCyclopsData(modelType = "pr")
    count <- appendSqlCyclopsData(dataPtrC,
                              oStratumId[1:5],
                              oRowId[1:5],
                              oY[1:5],
                              oTime[1:5],
                              cRowId[1:10],
                              cCovariateId[1:10],
                              cCovariateValue[1:10])

    count <- appendSqlCyclopsData(dataPtrC,
                              oStratumId[6:9],
                              oRowId[6:9],
                              oY[6:9],
                              oTime[6:9],
                              cRowId[11:21],
                              cCovariateId[11:21],
                              cCovariateValue[11:21])
    finalizeSqlCyclopsData(dataPtrC) # Not yet implemented

    expect_equal(count, 4)
    expect_equal(getNumberOfRows(dataPtrC), 9)
    expect_equal(getNumberOfStrata(dataPtrC), 9)
    expect_equal(getNumberOfCovariates(dataPtrC), 5)

    cyclopsFitC <- fitCyclopsModel(dataPtrC, control = createControl(noiseLevel = "silent"))
    expect_equal(coef(cyclopsFitC), coef(cyclopsFit))
})

test_that("Test bad stratum IDs", {
   binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
   binomial_n <- c(31,29,27,25,23,21,19,17,15,15,15)
   binomial_y <- c(0,3,6,7,9,13,17,12,11,14,13)

   log_bid <- log(c(rep(rep(binomial_bid, binomial_n - binomial_y)), rep(binomial_bid, binomial_y)))
   y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

   dataPtr <- createSqlCyclopsData(modelType = "lr")
   count <- appendSqlCyclopsData(dataPtr,
                             rep(0, length(y)),
                             1:length(y),
                             y,
                             rep(0,length(y)),
                             1:length(y),
                             rep(0,length(y)),
                             log_bid)
   finalizeSqlCyclopsData(dataPtr) # Not yet implemented

#     fitCyclopsModel(dataPtr, prior = createPrior("none")) #crashes R
})
