library("testthat")
library("survival")
library("gnm")

test_that("vcov for cpr, sccs and clr counts multiple events per stratum", {
    data <- Cyclops::oxford
    data$event <- 3 * data$event # n_s = 3 in every stratum; same MLE, three times the information
    gold <- gnm(event ~ exgr + agegr + offset(loginterval), family = poisson, eliminate = indiv, data = data)
    cpr <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                             data = data, modelType = "cpr"))
    sccs <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv), time = data$interval,
                                              data = data, modelType = "sccs"))
    expect_equal(as.numeric(vcov(cpr)), as.numeric(vcov(gold)), tolerance = 1E-6)
    expect_equal(as.numeric(vcov(sccs)), as.numeric(vcov(gold)), tolerance = 1E-6)

    binary <- Cyclops::oxford
    binary$event[binary$indiv %in% 1:5 & binary$exgr == 0 & binary$agegr == 1] <- 1 # some strata with 2-3 cases
    goldClr <- survival::clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval), data = binary,
                                method = "breslow")
    clr <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                             data = binary, modelType = "clr"))
    expect_equal(as.numeric(vcov(clr)), as.numeric(vcov(goldClr)), tolerance = 1E-6)
})

test_that("vcov after refitting the same cyclopsData uses the current coefficients", {
    data <- Cyclops::oxford
    data$event <- 2 * data$event
    cd <- createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval), data = data, modelType = "cpr")
    first <- fitCyclopsModel(cd)
    invisible(vcov(first))
    second <- fitCyclopsModel(cd, prior = createPrior("normal", variance = 0.1))
    fresh <- fitCyclopsModel(createCyclopsData(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                                               data = data, modelType = "cpr"),
                             prior = createPrior("normal", variance = 0.1))
    expect_equal(coef(second), coef(fresh), tolerance = 1E-6)
    expect_equal(vcov(second), vcov(fresh), tolerance = 1E-6)
})

test_that("vcov for cpr ignores strata without events", {
    data <- Cyclops::oxford
    data$indiv <- as.character(data$indiv)
    empty <- data[data$indiv %in% 1:5, ]
    empty$indiv <- paste0("empty", empty$indiv)
    empty$event <- 0 # these strata contribute nothing to the likelihood
    formula <- event ~ exgr + agegr + strata(indiv) + offset(loginterval)
    fit <- fitCyclopsModel(createCyclopsData(formula, data = data, modelType = "cpr"))
    fitWithEmpty <- fitCyclopsModel(createCyclopsData(formula, data = rbind(data, empty), modelType = "cpr"))
    expect_equal(coef(fitWithEmpty), coef(fit), tolerance = 1E-6)
    expect_equal(vcov(fitWithEmpty), vcov(fit), tolerance = 1E-6)
})

test_that("vcov with weights reports that weights are not supported", {
    fit <- fitCyclopsModel(createCyclopsData(case ~ spontaneous + induced, data = infert, modelType = "lr"),
                           weights = rep(c(1, 2), length.out = nrow(infert)))
    expect_error(vcov(fit), "Weights are not yet implemented in Fisher Information calculations")
})

test_that("vcov for legacy cox and cox_time matches coxph", {
    control <- createControl(tolerance = 1E-10, noiseLevel = "silent")
    legacy <- createControl(tolerance = 1E-10, noiseLevel = "silent", fitter = "legacy")
    lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
    lung2$status <- lung2$status - 1
    lung2$female <- lung2$sex - 1

    gold <- coxph(Surv(time, status) ~ age + sex + ph.ecog, lung2, ties = "breslow")
    fit <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ age + sex + ph.ecog, data = lung2,
                                             modelType = "cox"), control = legacy)
    expect_equal(unname(vcov(fit)), unname(vcov(gold)), tolerance = 1E-6)

    # strata, with indicator and sparse columns
    gold <- coxph(Surv(time, status) ~ age + female + ph.ecog + strata(ph.ecog > 1), lung2, ties = "breslow")
    fit <- fitCyclopsModel(createCyclopsData(Surv(time, status) ~ age + strata(ph.ecog > 1),
                                             indicatorFormula = ~ female, sparseFormula = ~ ph.ecog,
                                             data = lung2, modelType = "cox"), control = legacy)
    names <- names(coef(gold))
    expect_equal(coef(fit)[names], coef(gold), tolerance = 1E-6)
    expect_equal(vcov(fit)[names, names], vcov(gold), tolerance = 1E-6)

    # time-varying coefficient as one stratum per episode
    split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = 300, episode = "episode")
    gold <- coxph(Surv(tstart, time, status) ~ sex + age:strata(episode), split, ties = "breslow")
    split$length <- split$time - split$tstart
    split$age.1 <- ifelse(split$episode == 1, split$age, 0)
    split$age.2 <- ifelse(split$episode == 2, split$age, 0)
    fit <- fitCyclopsModel(createCyclopsData(Surv(length, status) ~ sex + age.1 + age.2 + strata(episode),
                                             data = split, modelType = "cox_time"), control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(unname(vcov(fit)), unname(vcov(gold)), tolerance = 1E-6)
})

simulateCompetingRisks <- function(n, seed) {
    set.seed(seed)
    data <- data.frame(x1 = rnorm(n), x2 = rbinom(n, 1, 0.4))
    event <- rexp(n, 0.10 * exp(0.5 * data$x1 - 0.4 * data$x2))
    competing <- rexp(n, 0.08 * exp(-0.3 * data$x1 + 0.2 * data$x2))
    censor <- runif(n, 2, 15)
    data$time <- pmin(event, competing, censor)
    data$status <- ifelse(data$time == censor, 0, ifelse(event < competing, 1, 2))
    data
}

test_that("vcov for fgr matches coxph on finegray() data", {
    data <- simulateCompetingRisks(400, 123)
    control <- createControl(tolerance = 1E-10, noiseLevel = "silent")

    expanded <- finegray(Surv(time, factor(status)) ~ ., data = data, etype = "1")
    gold <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x1 + x2, data = expanded, weights = fgwt,
                  ties = "breslow")
    goldVcov <- unname(gold$naive.var) # non-integer weights make coxph's var the robust one

    weights <- getFineGrayWeights(data$time, data$status)
    fit <- fitCyclopsModel(createCyclopsData(weights$surv ~ data$x1 + data$x2, modelType = "fgr",
                                             censorWeights = weights$weights), control = control)
    expect_equal(unname(coef(fit)), unname(coef(gold)), tolerance = 1E-6)
    expect_equal(unname(vcov(fit)), goldVcov, tolerance = 1E-6)

    # an indicator column
    outcomes <- data.frame(rowId = seq_len(nrow(data)), time = data$time, y = data$status)
    covariates <- rbind(data.frame(rowId = outcomes$rowId, covariateId = 1, covariateValue = data$x1),
                        data.frame(rowId = which(data$x2 == 1), covariateId = 2, covariateValue = 1))
    fit <- fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "fgr", quiet = TRUE),
                           control = control)
    expect_equal(unname(vcov(fit)), goldVcov, tolerance = 1E-6)
})

test_that("vcov for fgr with tied times inverts the curvature of the log-likelihood", {
    # finegray() handles ties differently from crr(), which fgr matches; compare against
    # central differences of the fgr log-likelihood instead
    data <- simulateCompetingRisks(400, 123)
    data$time <- ceiling(data$time) # failure, competing-risk and censoring ties
    control <- createControl(tolerance = 1E-10, noiseLevel = "silent")

    weights <- getFineGrayWeights(data$time, data$status)
    createData <- function() {
        createCyclopsData(weights$surv ~ data$x1 + data$x2, modelType = "fgr",
                          censorWeights = weights$weights)
    }
    fit <- fitCyclopsModel(createData(), control = control)
    probe <- createData() # refits move the coefficients that vcov(fit) would read from shared data
    logLikAt <- function(beta) {
        fitCyclopsModel(probe, startingCoefficients = beta, fixedCoefficients = c(TRUE, TRUE),
                        control = control)$log_likelihood
    }
    beta <- coef(fit)
    step <- diag(2) * 1E-3
    hessian <- outer(1:2, 1:2, Vectorize(function(i, j) {
        (logLikAt(beta + step[i, ] + step[j, ]) - logLikAt(beta + step[i, ] - step[j, ]) -
             logLikAt(beta - step[i, ] + step[j, ]) + logLikAt(beta - step[i, ] - step[j, ])) / (4 * 1E-6)
    }))
    expect_equal(unname(vcov(fit)), solve(-hessian), tolerance = 1E-5)
})
