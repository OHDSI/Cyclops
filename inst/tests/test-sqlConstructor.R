library("testthat")

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
        
    dataPtr <- createSqlCcdData(modelType = "pr")
    
    # Should have no data
    expect_equal(getNumberOfRows(dataPtr), 0)
    expect_equal(getNumberOfStrata(dataPtr), 0)
    expect_equal(getNumberOfCovariates(dataPtr), 0)    
    expect_error(fitCcdModel(dataPtr))

		count <- appendSqlCcdData(dataPtr,
															oStratumId,
															oRowId,
															oY,
															oTime,
															cRowId,
															cCovariateId,
															cCovariateValue)
		finalizeSqlCcdData(dataPtr) # Not yet implemented

		expect_equal(count, 9)
		expect_equal(getNumberOfRows(dataPtr), 9)
		expect_equal(getNumberOfStrata(dataPtr), 9)
		expect_equal(getNumberOfCovariates(dataPtr), 5)

		ccdFit <- fitCcdModel(dataPtr, control = control(noiseLevel = "silent"))

		dataPtrF <- createCcdDataFrame(counts ~ outcome + treatment, modelType = "pr")
		ccdFitF <- fitCcdModel(dataPtrF, control = control(noiseLevel = "silent"))
		expect_equivalent(coef(ccdFit), coef(ccdFitF)) # Have different coefficient names

		# Test chucked append
		dataPtrC <- createSqlCcdData(modelType = "pr")
		count <- appendSqlCcdData(dataPtrC,													
															oStratumId[1:5],													
															oRowId[1:5],													
															oY[1:5],													
															oTime[1:5],													
															cRowId[1:10],													
															cCovariateId[1:10],													
															cCovariateValue[1:10])

		count <- appendSqlCcdData(dataPtrC,															
															oStratumId[6:9],																										
															oRowId[6:9],																										
															oY[6:9],																										
															oTime[6:9],																										
															cRowId[11:21],																										
															cCovariateId[11:21],																										
															cCovariateValue[11:21])
		finalizeSqlCcdData(dataPtrC) # Not yet implemented

		expect_equal(count, 4)
		expect_equal(getNumberOfRows(dataPtrC), 9)
		expect_equal(getNumberOfStrata(dataPtrC), 9)
		expect_equal(getNumberOfCovariates(dataPtrC), 5)
		
		ccdFitC <- fitCcdModel(dataPtrC, control = control(noiseLevel = "silent"))
		expect_equal(coef(ccdFitC), coef(ccdFit))
})

