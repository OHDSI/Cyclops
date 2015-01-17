test_that("Make covariates dense", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4
    
    dataPtr <- createCyclopsData(counts ~ outcome, indicatorFormula =  ~ treatment, 
                                  modelType = "pr")
    
    expect_equal(as.character(summary(dataPtr)["treatment2","type"]),
                 "indicator")
    
    finalizeSqlCyclopsData(dataPtr, makeCovariatesDense = "treatment2")
    
    expect_equal(as.character(summary(dataPtr)["treatment2","type"]),
                 "dense")    
})
