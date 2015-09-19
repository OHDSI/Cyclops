library("testthat")

#
# Small Poisson MLE regression
#

test_that("univariable correlation", {
    dobson <- data.frame(
        counts = c(18,17,15,20,10,20,25,13,12),
        outcome = gl(3,1,9),
        treatment = gl(3,3)
    )
    tolerance <- 1E-4

    dataPtrD <- createCyclopsData(counts ~ outcome + treatment, data = dobson,
                                  modelType = "pr")

    allCorrelations <- univariableCorrelation(dataPtrD)
    expect_equal(length(allCorrelations), 5)

    gold <- sapply(1:5, function(i) {
        with(dobson, cor(counts, model.matrix(~outcome + treatment)[,i]))
    })
    expect_equal(gold, allCorrelations, tolerance)

    someCorrelations <- univariableCorrelation(dataPtrD, c("outcome2","outcome3"))
    expect_equal(length(someCorrelations), 2)
})

test_that("Playing with standardization", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    tolerance <- 1E-4

    dataPtr <- createCyclopsData(counts ~ outcome + treatment,
                                      modelType = "pr")
    cyclopsFit <- fitCyclopsModel(dataPtr,
                                  prior = createPrior("none"))

    dataPtrS <- createCyclopsData(counts ~ outcome + treatment,
                                       modelType = "pr")
    cyclopsFitS <- fitCyclopsModel(dataPtrS,
                                   prior = createPrior("none"))

    coef(cyclopsFit)
    coef(cyclopsFitS)
})
