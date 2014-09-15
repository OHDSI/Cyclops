library("testthat")
library("survival")

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
    
    goldCounting <-  coxph( Surv(start, length, event) ~ x1 + x2, test)    
    summary(goldCounting)
    
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    summary(goldRight)
    
    dataPtrRight <- createCyclopsDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                      
                                       modelType = "cox")    
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight)
    
    dataPtrCounting <- createCyclopsDataFrame(Surv(start, length, event) ~ x1 + x2, data = test,                                      
                                          modelType = "cox")    
    cyclopsFitCounting <- fitCyclopsModel(dataPtrCounting)
    
    expect_equal(coef(cyclopsFitRight), coef(cyclopsFitCounting))
    
    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)            
    
    
    goldStrat <- coxph(Surv(length, event) ~ x1 + strata(x2), test)
    
    dataPtrStrat <- createCyclopsDataFrame(Surv(length, event) ~ x1 + strata(x2),
                                       data = test,                                      
                                       modelType = "cox")    
    cyclopsFitStrat <- fitCyclopsModel(dataPtrStrat)         
    expect_equal(coef(cyclopsFitStrat), coef(goldStrat), tolerance = tolerance)
})



test_that("Check very small Cox example with time-ties, but no failure ties", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  0,1,1
0, 1,  0,1,0
0, 1,  1,1,0                       
")
        
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    summary(goldRight)

    dataPtrRight <- createCyclopsDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                      
                                       modelType = "cox")    
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight) 
    
    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)      
})

test_that("Check very small Cox example with failure ties, no risk-set contribution after tie", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1    
0, 2,  1,0,0
")
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")
    coef(goldRight)
      
    dataPtrRight <- createCyclopsDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                                                
                                       modelType = "cox")    
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight) 
    
    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)     
})

test_that("Check very small Cox example with failure ties, with risk-set contribution after tie", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0 
")
    
        # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test, ties = "breslow")
       
    dataPtrRight <- createCyclopsDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                                                                 
                                           modelType = "cox")    
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight) 
    
    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)
})

test_that("Check sparse Cox example with failure ties and strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0 
")
    
    # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")
    
    dataPtrRight <- createCyclopsDataFrame(Surv(length, event) ~ x1 + strata(x2), data = test,                                                                                 
                                           modelType = "cox")    
    cyclopsFitRight <- fitCyclopsModel(dataPtrRight) 
    
    tolerance <- 1E-4
    expect_equal(coef(cyclopsFitRight), coef(goldRight), tolerance = tolerance)
    
    # Attempt sparse
    dataSparse <- createCyclopsDataFrame(Surv(length, event) ~ strata(x2), 
                                         sparseFormula = ~ x1,
                                         data = test, modelType = "cox")
    
    cyclopsSparse <- fitCyclopsModel(dataSparse)
    expect_equal(coef(cyclopsSparse), coef(goldRight), tolerance = tolerance)
})

test_that("Check SQL interface for a very small Cox example with failure ties and strata", {
    test <- read.table(header=T, sep = ",", text = "
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  1,1,1
0, 1,  0,1,0
0, 1,  1,1,0 
")
    
    # We get the correct answer when last entry is censored
    goldRight <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")
      
    data <- test    
    data$row_id <- 1:nrow(data)
    data$covariate_id = 1
    
    data <- data[order(data$x2, -data$length, data$event),] # Must sort by: strata, into risk set (with events before censorsed)
    
    dataPtr <- createSqlCyclopsData(modelType = "cox")
    
    count <- appendSqlCyclopsData(dataPtr,
                                  data$x2,
                                  data$row_id,
                                  data$event,
                                  data$length,
                                  data$row_id,
                                  data$covariate_id,
                                  data$x1)

    cyclopsFitStrat <- fitCyclopsModel(dataPtr)
    
    tolerance <- 1E-4
    # SQL interface provides different names ('1' instead of 'x1')
    t1 <- coef(cyclopsFitStrat)
    t2 <- coef(goldRight)
    names(t1) <- NULL
    names(t2) <- NULL
    expect_equal(t1, t2, tolerance = tolerance)
})
