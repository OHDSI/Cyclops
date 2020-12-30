library("testthat")
library("survival")

context("test-xyConstructor.R")
suppressWarnings(RNGversion("3.5.0"))

#
# Tests for the XY constructor of ModelData
#

test_that("Test basic XY constructor", {
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
    expect_equal(as.character(summary(dataPtrD)[,"type"]), rep("dense",5))
    expect_equal(coef(fitCyclopsModel(dataPtrD)), coef(glmFit), tolerance = tolerance)

    # Indicator interface
    dataPtrI <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrI, NULL, rowId, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrI, 0, c(1:9), NULL, name = "(Intercept)")
    loadNewSqlCyclopsDataX(dataPtrI, 1, c(2,5,8), NULL, name = "outcome2")
    loadNewSqlCyclopsDataX(dataPtrI, 2, c(3,6,9), NULL, name = "outcome3")
    loadNewSqlCyclopsDataX(dataPtrI, 3, c(4:6), NULL, name = "treatment2")
    loadNewSqlCyclopsDataX(dataPtrI, 4, c(7:9), NULL, name = "treatment3")
    expect_equal(as.character(summary(dataPtrI)[,"type"]), rep("indicator",5))
    expect_equal(coef(fitCyclopsModel(dataPtrI)), coef(glmFit), tolerance = tolerance)

    # Sparse interface
    dataPtrS <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrS, NULL, rowId, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrS, 0, c(1:9), rep(1,9), name = "(Intercept)", forceSparse = TRUE)
    loadNewSqlCyclopsDataX(dataPtrS, 1, c(2,5,8), rep(1,3), name = "outcome2", forceSparse = TRUE)
    loadNewSqlCyclopsDataX(dataPtrS, 2, c(3,6,9), rep(1,3), name = "outcome3", forceSparse = TRUE)
    loadNewSqlCyclopsDataX(dataPtrS, 3, c(4:6), rep(1,3), name = "treatment2", forceSparse = TRUE)
    loadNewSqlCyclopsDataX(dataPtrS, 4, c(7:9), rep(1,3), name = "treatment3", forceSparse = TRUE)
    expect_equal(as.character(summary(dataPtrS)[,"type"]), rep("sparse",5))
    expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(glmFit), tolerance = tolerance)
})

test_that("Test strata XY constructor", {
    # Gold standard
    gold <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)
    tolerance <- 1E-4

    permute <- order(infert$stratum)
    pInfert <- infert[permute,]

    # Sparse interface
    dataPtrS <- createSqlCyclopsData(modelType = "clr")
    loadNewSqlCyclopsDataY(dataPtrS, pInfert$stratum, c(1:248), pInfert$case, NULL)
    loadNewSqlCyclopsDataX(dataPtrS, 0,
                           which(pInfert$spontaneous != 0),
                           pInfert$spontaneous[which(pInfert$spontaneous != 0)],
                           name = "spontaneous")
    loadNewSqlCyclopsDataX(dataPtrS, 1,
                           which(pInfert$induced != 0),
                           pInfert$induced[which(pInfert$induced != 0)],
                           name = "induced")
    expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(gold), tolerance = tolerance)
})

test_that("Check stratified Cox XY constructor", {
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
    gold <- coxph(Surv(length, event) ~ x1 + strata(x2), test, ties = "breslow")
    tolerance <- 1E-4

    permute <- order(test$x2)
    pTest <- test[permute,]

    # Sparse interface
    dataPtrS <- createSqlCyclopsData(modelType = "cox")
    loadNewSqlCyclopsDataY(dataPtrS, pTest$x2, c(1:7), pTest$event, pTest$length)
    loadNewSqlCyclopsDataX(dataPtrS, 0,
                           which(pTest$x1 != 0),
                           pTest$x1[which(pTest$x1 != 0)],
                           name = "x1")
    cyclops <- fitCyclopsModel(dataPtrS)
    expect_equal(coef(cyclops), coef(gold), tolerance = tolerance)
    expect_equal(logLik(cyclops)[1], logLik(gold)[1], tolerance = tolerance)
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
#     expect_equal(as.character(summary(dataPtrS)[,"type"]), rep("sparse",5))
#     expect_equal(coef(fitCyclopsModel(dataPtrS))[3] * 2, coef(glmFit)[3], tolerance = tolerance)
#
#     # Replace variable
#     loadNewSqlCyclopsDataX(dataPtrS, 2, c(3,6,9), NULL, name = "outcome3", replace = TRUE)
#     summary(dataPtrS)
#
#     expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(glmFit), tolerance = tolerance)
# })

test_that("Intercept covariate", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)

    # gold standard
    tolerance <- 1E-4
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    # Dense interface
    dataPtrD <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrD, NULL, NULL, counts, NULL)
    loadNewSqlCyclopsDataX(dataPtrD, 0, NULL, NULL, name = "(Intercept)")
    loadNewSqlCyclopsDataX(dataPtrD, 1, NULL, rep(c(0,1,0),3), name = "outcome2")
    loadNewSqlCyclopsDataX(dataPtrD, 2, NULL, rep(c(0,0,1),3), name = "outcome3")
    loadNewSqlCyclopsDataX(dataPtrD, 3, NULL, c(0,0,0,1,1,1,0,0,0), name = "treatment2")
    loadNewSqlCyclopsDataX(dataPtrD, 4, NULL, c(0,0,0,0,0,0,1,1,1), name = "treatment3")
    expect_equal(as.character(summary(dataPtrD)[1,"type"]), "intercept")
    expect_equal(coef(fitCyclopsModel(dataPtrD)), coef(glmFit), tolerance = tolerance)
})

test_that("Test COO-constructor", {
    counts <- c(18,17,15,20,10,20,25,13,12)
    outcome <- gl(3,1,9)
    treatment <- gl(3,3)

    # gold standard
    tolerance <- 1E-4
    glmFit <- glm(counts ~ outcome + treatment, family = poisson()) # gold standard

    rowId <- c(
        1,2,3,4,5,6,7,8,9,
        2,5,8,
        3,6,9,
        4,5,6,
        7,8,9)
    covariateId <- c(
        1,1,1,1,1,1,1,1,1,
        2,2,2,
        3,3,3,
        4,4,4,
        5,5,5)

    # Indicator interface
    dataPtrI <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrI, NULL, c(1:9), counts, NULL)  # TODO Crashes without row IDs
    loadNewSqlCyclopsDataMultipleX(dataPtrI, covariateId, rowId, NULL,
                                    name = c("(Intercept)","outcome2","outcome3","treatment2","treatment3"))
    expect_equal(as.character(summary(dataPtrI)[,"type"]), rep("indicator",5))
    expect_equal(coef(fitCyclopsModel(dataPtrI)), coef(glmFit), tolerance = tolerance)

    # Sparse interface
    dataPtrS <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrS, NULL, c(1:9), counts, NULL)  # TODO Crashes without row IDs
    loadNewSqlCyclopsDataMultipleX(dataPtrS, covariateId, rowId, rep(1,length(covariateId)),
                                    name = c("(Intercept)","outcome2","outcome3","treatment2","treatment3"),
                                    forceSparse = TRUE)
    expect_equal(as.character(summary(dataPtrS)[,"type"]), rep("sparse",5))
    expect_equal(coef(fitCyclopsModel(dataPtrS)), coef(glmFit), tolerance = tolerance)

    # Mixed interface (should use in practice)
    dataPtrM <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrM, NULL, c(1:9), counts, NULL)  # TODO Crashes without row IDs
    loadNewSqlCyclopsDataX(dataPtrM, 0, NULL, NULL, name = "(Intercept)")
    loadNewSqlCyclopsDataMultipleX(dataPtrM, covariateId[10:21], rowId[10:21], NULL,
                                    name = c("outcome2","outcome3","treatment2","treatment3"))
    expect_equal(coef(fitCyclopsModel(dataPtrM)), coef(glmFit), tolerance = tolerance)

    # Multiple adds
    dataPtrA <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(dataPtrA, NULL, c(1:9), counts, NULL)  # TODO Crashes without row IDs
    loadNewSqlCyclopsDataMultipleX(dataPtrA, covariateId[1:11], rowId[1:11], NULL)
    loadNewSqlCyclopsDataMultipleX(dataPtrA, covariateId[12:21], rowId[12:21], NULL, append = TRUE)
    expect_equal(as.character(summary(dataPtrA)[,"type"]), rep("indicator",5))
    cf <- coef(fitCyclopsModel(dataPtrA))
    names(cf) <- c("(Intercept)","outcome2","outcome3","treatment2","treatment3")
    expect_equal(cf, coef(glmFit), tolerance = tolerance)
})

test_that("Data errors and casting in COO-constructor", {
    dataPtr <- createSqlCyclopsData(modelType = "lr")
    expect_error(loadNewSqlCyclopsDataMultipleX(dataPtr, c(1,1), c(1,1), NULL))

    loadNewSqlCyclopsDataMultipleX(dataPtr, c(2,2), c(1,2), c(1,1))
    expect_equal(as.character(summary(dataPtr)[2,"type"]), "indicator")

    loadNewSqlCyclopsDataMultipleX(dataPtr, c(3,3), c(1,2), c(1,2))
    expect_equal(as.character(summary(dataPtr)[3,"type"]), "sparse")

    loadNewSqlCyclopsDataMultipleX(dataPtr, c(4,4), c(1,2), c(0,1))
    expect_equal(as.character(summary(dataPtr)[4,"type"]), "indicator")

    expect_error(loadNewSqlCyclopsDataX(dataPtr, 5, c(1,1)))
})

test_that("Poisson xy-construction with offset", {
    set.seed(666)
    sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 10,
                               model = "poisson")

    cyclopsData1 <- convertToCyclopsData(outcomes = sim$outcomes,
                                         covariates = sim$covariates,
                                         modelType = "pr",
                                         checkRowIds = FALSE)

    sim$covariates <- sim$covariates[order(sim$covariates$covariateId,
                                           sim$covariates$rowId),]
    cyclopsData2 <- createSqlCyclopsData(modelType = "pr")
    loadNewSqlCyclopsDataY(cyclopsData2, NULL, sim$outcomes$rowId,
                           sim$outcomes$y, sim$outcomes$time)
    loadNewSqlCyclopsDataX(cyclopsData2, 0, NULL, NULL,
                           name = "(Intercept)") # names are not necessary
    covarNames <- unique(sim$covariates$covariateId)
    loadNewSqlCyclopsDataMultipleX(cyclopsData2, sim$covariates$covariateId,
                                    sim$covariates$rowId,
                                    sim$covariates$covariateValue,
                                    #NULL, # pass NULL if you want indicators instead of sparse
                                    name = covarNames) # names are not necessary

    finalizeSqlCyclopsData(cyclopsData2, addIntercept = FALSE,
                           useOffsetCovariate = -1)

    fit1 <- fitCyclopsModel(cyclopsData1)
    fit2 <- fitCyclopsModel(cyclopsData2)

    tolerance <- 1E-6
    expect_equal(coef(fit2)[1], coef(fit1)[1], tolerance = tolerance)
})
