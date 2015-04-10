library("testthat")

#
# Tests for the XY constructor of ModelData
#

test_that("Test basic constructor", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)
    rowId <- c(1:9)

    # gold standard
    tolerance <- 1E-4
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    # Dense interface
    dataPtrD <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrD, NULL, NULL, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrD, 0, NULL, rep(1,9), name = "(Intercept)")
    loadNewSqlCyclopsDataX(dataPtrD, 1, NULL, rep(c(0,1,0),3), name = "outcome2")
    loadNewSqlCyclopsDataX(dataPtrD, 2, NULL, rep(c(0,0,1),3), name = "outcome3")
    loadNewSqlCyclopsDataX(dataPtrD, 3, NULL, c(0,0,0,1,1,1,0,0,0), name = "treatment2")
    loadNewSqlCyclopsDataX(dataPtrD, 4, NULL, c(0,0,0,0,0,0,1,1,1), name = "treatment3")
    expect_equal(summary(dataPtrD)[,"type"], as.factor(rep("dense",5)))
    expect_equal(coef(fitCyclopsModel(dataPtrD)), coef(glmFit), tolerance = tolerance)

    # Indicator interface
    dataPtrI <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrI, NULL, rowId, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrI, 0, c(1:9), NULL, name = "(Intercept)")
    loadNewSqlCyclopsDataX(dataPtrI, 1, c(2,5,8), NULL, name = "outcome2")
    loadNewSqlCyclopsDataX(dataPtrI, 2, c(3,6,9), NULL, name = "outcome3")
    loadNewSqlCyclopsDataX(dataPtrI, 3, c(4:6), NULL, name = "treatment2")
    loadNewSqlCyclopsDataX(dataPtrI, 4, c(7:9), NULL, name = "treatment3")
    expect_equal(summary(dataPtrI)[,"type"], as.factor(rep("indicator",5)))
    expect_equal(coef(fitCyclopsModel(dataPtrI)), coef(glmFit), tolerance = tolerance)

    # Sparse interface
    dataPtrS <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrS, NULL, rowId, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrS, 0, c(1:9), rep(1,9), name = "(Intercept)")
    loadNewSqlCyclopsDataX(dataPtrS, 1, c(2,5,8), rep(1,3), name = "outcome2")
    loadNewSqlCyclopsDataX(dataPtrS, 2, c(3,6,9), rep(1,3), name = "outcome3")
    loadNewSqlCyclopsDataX(dataPtrS, 3, c(4:6), rep(1,3), name = "treatment2")
    loadNewSqlCyclopsDataX(dataPtrS, 4, c(7:9), rep(1,3), name = "treatment3")
    expect_equal(summary(dataPtrS)[,"type"], as.factor(rep("sparse",5)))
    expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(glmFit), tolerance = tolerance)
})

# test_that("Replace covariate", {
#     counts <- c(18,17,15,20,10,20,25,13,12)
#     outcome <- gl(3,1,9)
#     treatment <- gl(3,3)
#     rowId <- c(1:9)
#
#     # gold standard
#     tolerance <- 1E-4
#     glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard
#
#     # Sparse interface
#     dataPtrS <- createSqlCyclopsData(modelType = "pr")
#     loadNewSqlCyclopsDataY(dataPtrS, NULL, rowId, counts, NULL)
#     loadNewSqlCyclopsDataX(dataPtrS, 0, c(1:9), rep(1,9), name = "(Intercept)")
#     loadNewSqlCyclopsDataX(dataPtrS, 1, c(2,5,8), rep(1,3), name = "outcome2")
#     loadNewSqlCyclopsDataX(dataPtrS, 2, c(3,6,9), rep(2,3), name = "outcome3")
#     loadNewSqlCyclopsDataX(dataPtrS, 3, c(4:6), rep(1,3), name = "treatment2")
#     loadNewSqlCyclopsDataX(dataPtrS, 4, c(7:9), rep(1,3), name = "treatment3")
#     expect_equal(summary(dataPtrS)[,"type"], as.factor(rep("sparse",5)))
#     expect_equal(coef(fitCyclopsModel(dataPtrS))[3] * 2, coef(glmFit)[3], tolerance = tolerance)
#
#     # Replace variable
#     loadNewSqlCyclopsDataX(dataPtrS, 2, c(3,6,9), NULL, name = "outcome3", replace = TRUE)
#     summary(dataPtrS)
#
#     expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(glmFit), tolerance = tolerance)
# })

# test_that("Intercept covariate", {
#     # TODO Fix crash; need to write Range.h for InterceptTag
#     counts <- c(18,17,15,20,10,20,25,13,12)
#     dataPtr <- createSqlCyclopsData(modelType = "pr")
#     loadNewSqlCyclopsDataY(dataPtr, NULL, NULL, counts, NULL)
#     loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL)
#     fitCyclopsModel(dataPtr)
# })


