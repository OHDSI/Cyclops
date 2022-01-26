library("testthat")

context("test-largeProfileLikelihood.R")
suppressWarnings(RNGversion("3.5.0"))

test_that("Check very large and ill-conditioned profile", {
    skip("Takes much too longer")

    library("SelfControlledCaseSeries")
    set.seed(123)

    settings <- createSccsSimulationSettings(includeAgeEffect = TRUE, includeSeasonality = TRUE)
    sccsData <- simulateSccsData(10000, settings)
    ageSettings <- createAgeCovariateSettings(ageKnots = 5)
    seasonalitySettings <- createSeasonalityCovariateSettings(seasonKnots = 5)
    covarSettings <- createEraCovariateSettings(label = "Exposure of interest",
                                                includeEraIds = c(1, 2),
                                                start = 0,
                                                end = 0,
                                                endAnchor = "era end")
    studyPop <- createStudyPopulation(sccsData = sccsData,
                                      outcomeId = 10,
                                      firstOutcomeOnly = FALSE,
                                      naivePeriod = 0)
    sccsIntervalData <- createSccsIntervalData(studyPopulation = studyPop,
                                               sccsData = sccsData,
                                               eraCovariateSettings = covarSettings,
                                               ageCovariateSettings = ageSettings,
                                               seasonalityCovariateSettings = seasonalitySettings,
                                               minCasesForAgeSeason = 10000)
    cyclopsData <- Cyclops::convertToCyclopsData(sccsIntervalData$outcomes,
                                                 sccsIntervalData$covariates,
                                                 modelType = "cpr",
                                                 addIntercept = FALSE,
                                                 checkRowIds = FALSE,
                                                 quiet = TRUE)

    system.time(
        fit <- Cyclops::fitCyclopsModel(cyclopsData, prior = createPrior("none"))
    )
    # user  system elapsed
    # 27.53    0.29   27.84

    system.time(
        logLikelihoodProfile <- Cyclops::getCyclopsProfileLogLikelihood(object = fit,
                                                                        parm = 1000,
                                                                        x = seq(log(0.1), log(10), length.out = 1000),
                                                                        includePenalty = TRUE)$value
    )
})


