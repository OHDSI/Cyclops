library("testthat")
library("survival")

#
# Small exact, conditional logistic regression
#

test_that("Small exact, conditional logistic regression with no ties", {
    
    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)  
    
    dataPtrNoTies <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
                                            data = infert,
                                            modelType = "clr")
    
    cyclopsFitNoTies <- fitCyclopsModel(dataPtrNoTies, prior = createPrior("none"))    
    
    dataPtrExact <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
                                           data = infert,
                                           modelType = "clr_exact")
    
    cyclopsFitExact <- fitCyclopsModel(dataPtrExact, prior = createPrior("none"))
    
    tolerance <- 1E-4
    
    expect_equal(coef(cyclopsFitNoTies), coef(gold), tolerance = tolerance)
    expect_equal(coef(cyclopsFitExact), coef(gold), tolerance = tolerance)
})
    
 test_that("Small, exact conditinal logistic regression with ties" , {   
     
    withTies <- read.table(system.file("extdata/test1-clr.txt", package="Cyclops"), sep=",")
    names(withTies) <- c("stratum", "y",paste("x", 1:10, sep=""))
    
    goldWithTies <- clogit(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + strata(stratum),
                           data = withTies, method="exact")
    
    goldWithTiesBreslow <- clogit(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + strata(stratum),
                                  data = withTies, method="breslow")    
    
    dataPtrWithTies <-createCyclopsDataFrame(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + strata(stratum),
                                             data = withTies,
                                             modelType = "clr_exact")
    
    cyclopsFitWithTies <- fitCyclopsModel(dataPtrWithTies, prior = createPrior("none"))     
    
    dataPtrWithTiesBreslow <-createCyclopsDataFrame(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + strata(stratum),
                                                    data = withTies,
                                                    modelType = "clr")
    
    cyclopsFitWithTiesBreslow <- fitCyclopsModel(dataPtrWithTiesBreslow, prior = createPrior("none"))  
    
    tolerance <- 1E-4   
        
    expect_equal(coef(cyclopsFitWithTies), coef(goldWithTies), tolerance = tolerance)
    expect_equal(coef(cyclopsFitWithTiesBreslow), coef(goldWithTiesBreslow), tolerance = tolerance)    
})

# test_that("Evaluate speed of exact method without ties (should be same as Breslow)", { 
#     gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)  
#     
#     dataPtrBreslow <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
#                                              data = infert,
#                                              modelType = "clr") 
#     
#     cyclopsFitBreslow <- fitCyclopsModel(dataPtrBreslow, prior = createPrior("none"), forceColdStart = TRUE)
#     
#     dataPtrExact <- createCyclopsDataFrame(case ~ strata(stratum), sparseFormula = ~spontaneous + induced,
#                                            data = infert,
#                                            modelType = "clr_exact")
#     
#     cyclopsFitExact <- fitCyclopsModel(dataPtrExact, prior = createPrior("none"), forceColdStart = TRUE)
#     
#     library(microbenchmark)    
#     microbenchmark(
#         clogit(case ~ spontaneous + induced + strata(stratum), data=infert),
#         {
#             dataPtrBreslow <- createCyclopsDataFrame(case ~ spontaneous + induced + strata(stratum),
#                                                      data = infert,
#                                                      modelType = "clr") 
#             fitCyclopsModel(dataPtrBreslow, prior = createPrior("none"), forceColdStart = TRUE) 
#         },
#         fitCyclopsModel(dataPtrExact, prior = createPrior("none"), forceColdStart = TRUE),
#         times = 100L
#     )    
# })

