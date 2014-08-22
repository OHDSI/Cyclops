library("testthat")

test_that("Try loading cohort via SQL", {
   load(system.file("extdata/cpr_test.RData", 
                    package="Cyclops")) # loads "test" and "cov"

   cd <- createSqlCyclopsData(modelType="cpr")

   appendSqlCyclopsData(cd, test$STRATUM_ID, test$ROW_ID, 
                    test$Y, test$TIME, cov$ROW_ID, cov$COVARIATE_ID, 
                    cov$COVARIATE_VALUE)

   finalizeSqlCyclopsData(cd, addIntercept=FALSE)

   fit <- fitCyclopsModel(cd, prior=prior("laplace",0.1))                 
})

