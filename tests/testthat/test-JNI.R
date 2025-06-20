library("testthat")

#
# Small Poisson MLE regression
#

test_that("JNI with small Poisson regression", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                   modelType = "pr")
    cyclopsFitD <- fitCyclopsModel(dataPtrD,
                           prior = createPrior("none"),
                           control = createControl(noiseLevel = "silent"))

    instance <- cacheCyclopsModelForJava(cyclopsFitD)

    instance2 <- cacheCyclopsModelForJava(cyclopsFitD)
    expect_equal(instance, instance2)

   })
