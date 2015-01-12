library("testthat")

test_that("Try loading cohort via SQL", {
   load(system.file("extdata/cpr_test.RData", 
                    package="Cyclops")) # loads "test" and "cov"
   
   test$TIME[which(test$TIME == 0)] <- 1

   cd <- createSqlCyclopsData(modelType="cpr")

   appendSqlCyclopsData(cd, test$STRATUM_ID, test$ROW_ID, 
                    test$Y, 
                    test$TIME, 
                    #rep(1, length(test$Y)),
                    cov$ROW_ID, cov$COVARIATE_ID, 
                    cov$COVARIATE_VALUE)

   finalizeSqlCyclopsData(cd, 
                          useOffsetCovariate=-1, 
                          addIntercept=FALSE)

   fit <- fitCyclopsModel(cd, prior = createPrior("laplace",0.1))   
   coef(fit)[which(coef(fit) != 0)]
})

