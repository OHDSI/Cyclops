library("testthat")
library("survival")
library("gnm")

test_that("Check simple SCCS as conditional logistic regression", {
    source("helper-conditionalPoisson.R")
    tolerance <- 1E-6    
    gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval), 
                          data = chopdat)
    
    dataPtr <- createCcdDataFrame(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                  data = chopdat,
                                  modelType = "clr")        
    ccdFit <- fitCcdModel(dataPtr,
                          prior = prior("none"))
    expect_equal(logLik(ccdFit), logLik(gold.clogit)[1])
    expect_equal(coef(ccdFit), coef(gold.clogit), tolerance = tolerance)            
})

test_that("Check simple SCCS as conditional Poisson regression", {
    source("helper-conditionalPoisson.R")
    tolerance <- 1E-3    
    gold.cp <- gnm(event ~ exgr + agegr + offset(loginterval), 
                   family = poisson, eliminate = indiv, 
                   data = chopdat)
    
    dataPtr <- createCcdDataFrame(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                  data = chopdat,
                                  modelType = "cpr")        
    ccdFit <- fitCcdModel(dataPtr,
                          prior = prior("none"))
    
    expect_equal(coef(ccdFit)[1:2], coef(gold.cp)[1:2], tolerance = tolerance)     
    expect_equal(confint(ccdFit, c("exgr1","agegr2"))[,2:3],
                 confint(gold.cp), tolerance = tolerance)    
})

test_that("Check simple SCCS as SCCS", {
    source("helper-conditionalPoisson.R")
    tolerance <- 1E-6    
    gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval), 
                          data = chopdat)
    
    dataPtr <- createCcdDataFrame(event ~ exgr + agegr + strata(indiv), time = chopdat$interval,
                                  data = chopdat,
                                  modelType = "sccs")        
    ccdFit <- fitCcdModel(dataPtr,
                          prior = prior("none"))
    expect_equal(logLik(ccdFit), logLik(gold.clogit)[1])
    expect_equal(coef(ccdFit), coef(gold.clogit), tolerance = tolerance)            
})

