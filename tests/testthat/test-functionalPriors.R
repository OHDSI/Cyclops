library("testthat")

test_that("Specify functional L1 regularization", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)

    dataPtr <- createCyclopsData(counts ~ outcome + treatment,
                                 modelType = "pr")

    # prior <- createPrior(priorType = c("none", "laplace", "none", "laplace", "none"),
    #                      variance = c(0, 1, 0, 1, 1))
    #
    # cyclopsFit <- fitCyclopsModel(dataPtr, prior = prior)
    # expect_equal(length(strsplit(cyclopsFit$prior_info, ' ')[[1]]), 5) # 5 different covariates
    #
    # expect_true(coef(cyclopsFit)[4] == 0)
    # expect_true(coef(cyclopsFit)[5] != 0)
})

