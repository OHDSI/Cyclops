library("testthat")

#
# Small Poisson MLE regression
#

test_that("Small multi-type Poisson dense regression", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson <- rbind(dobson1, dobson1)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4
    
    goldFit <- glm(counts ~ outcome + treatment, data = dobson1, family = poisson())
    
    glmFit <- glm(counts ~ outcome + treatment, data = dobson, contrasts = dobson$type, 
                  family = poisson()) # gold standard
    
    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                       type = dobson$type,
                                       modelType = "pr", method = "debug")
    
    cyclopsFitD <- fitCyclopsModel(dataPtrD, 
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))
    
    dataPtrE <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,                                      
                                       modelType = "pr", method = "debug")
    
    cyclopsFitE <- fitCyclopsModel(dataPtrE, 
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))    
    
    expect_equal(coef(cyclopsFitD), coef(cyclopsFitE))
    
    dataPtrI <- createCyclopsData(Multitype(counts, type) ~ 1, indicatorFormula = ~ outcome + treatment, data = dobson,                                      
                                       modelType = "pr", method = "debug")
    
    cyclopsFitI <- fitCyclopsModel(dataPtrI, 
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))    
    
    expect_equal(coef(cyclopsFitI), coef(cyclopsFitD))
    
    dataPtrS <- createCyclopsData(Multitype(counts, type) ~ 1, sparseFormula = ~ outcome + treatment, data = dobson,                                      
                                       modelType = "pr", method = "debug")
    
    cyclopsFitS <- fitCyclopsModel(dataPtrS, 
                                   prior = createPrior("none"),
                                   control = createControl(noiseLevel = "silent"))    
    
    expect_equal(coef(cyclopsFitS), coef(cyclopsFitD))
})

test_that("coef throws error when not converged", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )    
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4
        
    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,                                                                              
                                       modelType = "pr")
    
    cyclopsFitD <- fitCyclopsModel(dataPtrD, 
                                   prior = createPrior(c("normal","normal"), c(0.0001,10), graph = "type"),
                                   control = createControl(noiseLevel = "silent"))
    expect_error(coef(cyclopsFitD), "did not converge")
})


test_that("confirm dimension check", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )    
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4
    
    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,                                                                              
                                       modelType = "pr")
    
    
    expect_error(fitCyclopsModel(dataPtrD, 
                                 prior = createPrior(c("normal"), c(0.0001,10), graph = "type"),
                                 control = createControl(noiseLevel = "silent")), "dimensionality mismatch")
    expect_error(fitCyclopsModel(dataPtrD, 
                                 prior = createPrior(c("normal", "normal"), c(0.0001), graph = "type"),
                                 control = createControl(noiseLevel = "silent")), "dimensionality mismatch")    
})


test_that("Small multi-type Poisson with hierarchical prior", {
    dobson1 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    dobson2 <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12)-10,
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )    
    dobson <- rbind(dobson1, dobson2)
    dobson$type = as.factor(c(rep("A",9),rep("B",9)))
    tolerance <- 1E-4
    
    glmFit <- glm(counts ~ outcome + treatment, data = dobson, contrasts = dobson$type, 
                  family = poisson()) # gold standard
    
    dataPtrD <- createCyclopsData(Multitype(counts, type) ~ outcome + treatment, data = dobson,                                                                              
                                       modelType = "pr")
    
    cyclopsFitD <- fitCyclopsModel(dataPtrD, 
                                   prior = createPrior(c("normal","normal"), c(0.0001,10), graph = "type"),
                                   control = createControl(noiseLevel = "silent", maxIterations = 2000))
    
    cyclopsFitE <- fitCyclopsModel(dataPtrD, 
                                   prior = createPrior(c("normal","normal"), c(0.0001,0.0001), graph = "type"),
                                   control = createControl(noiseLevel = "silent"))    
    
    
})


test_that("Check multitype SCCS", {
#    
#   stratumID must be unique by type and all patient entries must be replicated by type
#     (how to specify this in input) ???
#    
#     source("helper-conditionalPoisson.R")
#     tolerance <- 1E-6    
#     gold.clogit <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval), 
#                           data = oxford)
#     
#     dataPtr <- createCyclopsData(event ~ exgr + agegr + strata(indiv), time = oxford$interval,
#                                       data = oxford,
#                                       modelType = "sccs")        
#     cyclopsFit <- fitCyclopsModel(dataPtr,
#                                   prior = createPrior("none"))
#     expect_equal(logLik(cyclopsFit), logLik(gold.clogit)[1])
#     expect_equal(coef(cyclopsFit), coef(gold.clogit), tolerance = tolerance)            
})


