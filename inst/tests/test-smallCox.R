library("testthat")
library("survival")

test_that("Check small Cox example", {
    test2 <- list(start=c(1, 2, 5, 2, 1, 7, 3, 4, 8, 8),
                  stop =c(2, 3, 6, 7, 8, 9, 9, 9,14,17),
                  event=c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                  x    =c(1, 0, 0, 1, 0, 1, 1, 1, 0, 0) )
    
    gold <-  coxph( Surv(start, stop, event) ~ x, test2)
    
    summary(gold)
    
    
})

test_that("Check very small Cox example", {
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
    expect_equal(coef(ccdFitRight), coef(goldRight), tolerance = tolerance)            
    
    
    goldStrat <- coxph(Surv(length, event) ~ x1 + strata(x2), test)
    
    dataPtrStrat <- createCcdDataFrame(Surv(length, event) ~ x1 + strata(x2),
                                       data = test,                                      
                                       modelType = "cox")    
    ccdFitStrat <- fitCcdModel(dataPtrStrat)    
#     expect_equal(coef(ccdFitStrat), coef(goldStrat))  # Names are still wrong    
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
     
    dataPtrRight <- createCcdDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                      
                                       modelType = "cox")    
    ccdFitRight <- fitCcdModel(dataPtrRight) 
    
    tolerance <- 1E-4
    expect_equal(coef(ccdFitRight), coef(goldRight), tolerance = tolerance)      
})

test_that("Check very small Cox example with failure ties", {
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
    goldRight <- coxph(Surv(length, event) ~ x1 + x2, test)
    coef(goldRight)
    
    dataPtrRight <- createCcdDataFrame(Surv(length, event) ~ x1 + x2, data = test,                                      
                                       modelType = "cox")    
    ccdFitRight <- fitCcdModel(dataPtrRight)   ## BROKEN 
    coef(ccdFitRight)
})
