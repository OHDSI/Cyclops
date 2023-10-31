# @file Simulation.R
#
# Copyright 2014 Observational Health Data Sciences and Informatics
#
# This file is part of cyclops
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' @title Simulation Cyclops dataset
#'
#' @description
#' \code{simulateCyclopsData} generates a simulated large, sparse data set for use by \code{fitCyclopsSimulation}.
#'
#' @param nstrata   Numeric: Number of strata
#' @param nrows Numeric: Number of observation rows
#' @param ncovars Numeric: Number of covariates
#' @param effectSizeSd Numeric: Standard derivation of the non-zero simulated regression coefficients
#' @param zeroEffectSizeProp Numeric: Expected proportion of zero effect size
#' @param eCovarsPerRow Number: Effective number of non-zero covariates per data row
#' @param model String: Simulation model. Choices are: \code{logistic}, \code{poisson} or \code{survival}
#'
#' @return A simulated data set
#'
#' @template elaborateExample
#'
#' @export
simulateCyclopsData <- function(nstrata = 200,
                                nrows = 10000,
                                ncovars = 20,
                                effectSizeSd = 1,
                                zeroEffectSizeProp = 0.9,
                                eCovarsPerRow = ncovars/100,
                                model="survival"){

    sd <- rep(effectSizeSd, ncovars) * rbinom(ncovars, 1, 1 - zeroEffectSizeProp)
    effectSizes <- data.frame(covariateId=1:ncovars,rr=exp(rnorm(ncovars,mean=0,sd=sd)))

    covarsPerRow <- rpois(nrows,eCovarsPerRow)
    covarsPerRow[covarsPerRow > ncovars] <- ncovars
    covarsPerRow <- data.frame(covarsPerRow = covarsPerRow)
    covarRows <- sum(covarsPerRow$covarsPerRow)
    covariates <- data.frame(rowId = rep(0,covarRows), covariateId = rep(0,covarRows), covariateValue = rep(1,covarRows))
    cursor <- 1
    for (i in 1:nrow(covarsPerRow)){
        n <- covarsPerRow$covarsPerRow[i]
        if (n != 0){
            covariates$rowId[cursor:(cursor+n-1)] <- i
            covariates$covariateId[cursor:(cursor+n-1)] <- sample.int(size=n,ncovars)
            cursor = cursor+n
        }
    }

    outcomes <- data.frame(rowId = 1:nrows, stratumId = round(runif(nrows,min=1,max=nstrata)), y=0)
    covariates <- merge(covariates,outcomes[,c("rowId","stratumId")])

    rowId_to_rr <- aggregate(rr ~ rowId, data=merge(covariates,effectSizes),prod)

    outcomes <- merge(outcomes,rowId_to_rr,all.x=TRUE)
    outcomes$rr[is.na(outcomes$rr)] <- 1

    if (model == "survival"){
        strataBackgroundProb <- runif(nstrata,min=0.01,max=0.03)
        outcomes$rate <-  strataBackgroundProb[outcomes$stratumId] * outcomes$rr
        outcomes$timeToOutcome <- 1+round(rexp(n=nrow(outcomes),outcomes$rate))
        outcomes$timeToCensor <- 1+round(runif(n=nrow(outcomes),min=0,max=499))
        outcomes$time <- outcomes$timeToOutcome
        outcomes$time[outcomes$timeToCensor < outcomes$timeToOutcome] <- outcomes$timeToCensor[outcomes$timeToCensor < outcomes$timeToOutcome]
        outcomes$y <- as.integer(outcomes$timeToCensor > outcomes$timeToOutcome)
    } else if (model == "logistic") {
        strataBackgroundProb <- runif(nstrata,min=0.1,max=0.3)
        outcomes$prob <-  strataBackgroundProb[outcomes$stratumId] * outcomes$rr
        outcomes$y <- as.integer(runif(nrows,min=0,max=1) < outcomes$prob)
    } else if (model == "poisson"){
        strataBackgroundProb <- runif(nstrata,min=0.01,max=0.03)
        outcomes$rate <-  strataBackgroundProb[outcomes$stratumId] * outcomes$rr
        outcomes$time <- 1+round(runif(n=nrow(outcomes),min=0,max=499))
        outcomes$y <- rpois(nrows,outcomes$rate * outcomes$time)
    } else
        stop(paste("Unknown model:",model))

    outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
    covariates <- covariates[order(covariates$stratumId,covariates$rowId,covariates$covariateId),]
    sparseness <- 1-(nrow(covariates)/(nrows*ncovars))
    intercepts <- data.frame(stratumId = 1:nstrata, prob = strataBackgroundProb, logProb = log(strataBackgroundProb))
    writeLines(paste("Sparseness =",sparseness*100,"%"))
    list(outcomes = outcomes,
         covariates = covariates,
         effectSizes = effectSizes,
         sparseness = sparseness,
         intercepts = intercepts)
}

# .figureOutGlmnetComparison <- function() {
#
#     sim <-simulateCyclopsData(1, 100000, 1000, 0.5,
#                               zeroEffectSizeProp = 0.9, model = "logistic")
#
#     convertCyclopsSimulationToGlmnet <- function(sim) {
#         mat <- Matrix(data = 0,
#                       nrow = nrow(sim$outcomes),
#                       ncol = max(sim$covariates$covariateId),
#                       sparse = TRUE)
#         nnz <- length(sim$covariates$rowId)
#         for (i in 1:nnz) {
#             mat[sim$covariates$rowId[i],
#                 sim$covariates$covariateId[i]] <- sim$covariates$covariateValue[i]
#         }
#         mat
#     }
#
#     mat <- convertCyclopsSimulationToGlmnet(sim)
#
#     mat <- sparseX
#     y <- y
#
#     start <- Sys.time()
#     f <- glmnet(y = y, x = mat,
#                 family = "binomial",
#                 lambda = sqrt(2 / 0.1) / nrow(mat),
#                 intercept = FALSE, standardize = FALSE)
#     delta <- Sys.time() - start
#     writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
#
#     start <- Sys.time()
#     cd <- createCyclopsData(y = y, ix = mat, modelType = "lr")
#     ff <- fitCyclopsModel(cd, prior = createPrior("laplace", 0.1))
#     delta <- Sys.time() - start
#     writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
#
# }

.fitUsingClogit <- function(sim,coverage=TRUE){
    start <- Sys.time()
    covariates <- sim$covariates
    ncovars <- max(covariates$covariateId)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$rowId[i],covariates$covariateId[i]] <- 1
    }
    data <- as.data.frame(m)

    data$rowId <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratumId,data$rowId),]
    formula <- as.formula(paste(c("y ~ strata(stratumId)",paste("V",1:ncovars,sep="")),collapse=" + "))
    fit <- survival::clogit(formula,data=data)
    if (coverage) {
        ci <- confint(fit)
    } else {
        ci <- matrix(0,nrow=1,ncol=2)
    }

    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))

    data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
}

.fitUsingCoxph <- function(sim, coverage = TRUE){
    if (!requireNamespace("survival")) {
        stop("survival library required")
    }
    start <- Sys.time()
    covariates <- sim$covariates
    ncovars <- max(covariates$covariateId)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$rowId[i],covariates$covariateId[i]] <- 1
    }
    data <- as.data.frame(m)

    data$rowId <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratumId,data$rowId),]
    formula <- as.formula(paste(c("Surv(time,y) ~ strata(stratumId)",paste("V",1:ncovars,sep="")),collapse=" + "))
    fit <- survival::coxph(formula,data=data)
    if (coverage) {
        ci <- confint(fit)
    } else {
        ci <- matrix(0,nrow=1,ncol=2)
    }

    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))

    data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
}

# .fitUsingGnm <- function(sim,coverage=TRUE){
#     if (!requireNamespace("gnm")) {
#         stop("gnm library required")
#     }
#     start <- Sys.time()
#     covariates <- sim$covariates
#     ncovars <- max(covariates$covariateId)
#     nrows <- nrow(sim$outcomes)
#     m <- matrix(0,nrows,ncovars)
#     for (i in 1:nrow(covariates)){
#         m[covariates$rowId[i],covariates$covariateId[i]] <- 1
#     }
#     data <- as.data.frame(m)
#
#     data$rowId <- 1:nrow(data)
#     data <- merge(data,sim$outcomes)
#     data <- data[order(data$stratumId,data$rowId),]
#     formula <- as.formula(paste(c("y ~ v1",paste("v",2:ncovars,sep="")),collapse=" + "))
#
#     fit = gnm::gnm(formula, family=poisson, offset=log(time), eliminate=as.factor(data$stratumId), data = data)
#     #Todo: figure out how to do confidence intervals correctly
#     confint(fit)
#     fit0 = gnm::gnm(y ~ 1, family=poisson, offset=log(time), eliminate=as.factor(data$stratumId), data = data)
#     se <- abs(coef(fit)[[1]]/qnorm(1-pchisq(deviance(fit0)-deviance(fit),1)))
#
#
#     fit <- survival::coxph(formula,data=data)
#     if (coverage) {
#         ci <- confint(fit)
#     } else {
#         ci <- matrix(0,nrow=1,ncol=2)
#     }
#
#     delta <- Sys.time() - start
#     writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
#
#     data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
# }


#' @title Fit simulated data
#'
#' @description
#' \code{fitCyclopsSimulation} fits simulated Cyclops data using Cyclops or a standard routine.
#' This function is useful for simulation studies comparing the performance of Cyclops when considering
#' large, sparse datasets.
#'
#' @param sim    A simulated Cyclops dataset generated via \code{simulateCyclopsData}
#' @param useCyclops    Logical: use Cyclops or a standard routine
#' @param model  String: Fitted regression model type
#' @param coverage Logical: report coverage statistics
#' @param includePenalty   Logical: include regularized regression penalty in computing profile likelihood based confidence intervals
#' @param computeDevice String: Name of compute device to employ; defaults to \code{"native"} C++ on CPU
#'
#' @export
fitCyclopsSimulation <- function(sim,
                                 useCyclops = TRUE,
                                 model = "logistic",
                                 coverage = TRUE,
                                 includePenalty = FALSE,
				 computeDevice = "native") {
    if (useCyclops) {
        .fitCyclopsSimulationUsingCyclops(sim, model, coverage = coverage, includePenalty = includePenalty, computeDevice = computeDevice)
    } else {
        .fitCyclopsSimulationUsingOtherThanCyclops(sim, model, coverage)
    }
}

.fitCyclopsSimulationUsingOtherThanCyclops <- function(sim, model="logistic",coverage=TRUE){
    if (model == "logistic"){
        writeLines("Fitting model using clogit")
        .fitUsingClogit(sim,coverage)
    } else if (model == "poisson"){
        stop("Poisson not yet implemented")
    } else if (model == "survival"){
        writeLines("Fitting model using coxpht")
        .fitUsingCoxph(sim,coverage)
    } else {
        stop(paste("Unknown model:",model))
    }
}

.fitCyclopsSimulationUsingCyclops <- function(sim,
                                              model = "logistic",
                                              regularized = FALSE,
                                              coverage = TRUE,
                                              includePenalty = FALSE,
					      computeDevice = computeDevice){
    if (!regularized)
        includePenalty = FALSE
    start <- Sys.time()
    stratified <- max(sim$outcomes$stratumId) > 1
    if (stratified){
        if (model == "logistic") modelType = "clr"
        if (model == "poisson") modelType = "cpr"
        if (model == "survival") modelType = "cox"
    } else {
        if (model == "logistic") modelType = "lr"
        if (model == "poisson") modelType = "pr"
        if (model == "survival") modelType = "cox"
    }

    dataPtr <- convertToCyclopsData(sim$outcomes,sim$covariates,modelType = modelType,addIntercept = !stratified)

    # if (regularized){
        coefCyclops <- rep(0,length(sim$effectSizes$rr))
        lbCi95 <- rep(0,length(sim$effectSizes$rr))
        ubCi95 <- rep(0,length(sim$effectSizes$rr))
        for (i in 1:length(sim$effectSizes$rr)){
            if (model == "survival"){ # For some reason can't get confint twice on same dataPtr object, so recreate:
                dataPtr <- convertToCyclopsData(sim$outcomes,sim$covariates,modelType = modelType,addIntercept = !stratified)
            }
            if (regularized) {
                prior <- createPrior("laplace", 0.1, exclude = i)
            } else {
                prior <- createPrior("none")
            }
            cyclopsFit <- fitCyclopsModel(dataPtr,prior = prior)
            coefCyclops[i] <- coef(cyclopsFit)[names(coef(cyclopsFit)) == as.character(i)]
            if (coverage) {
                if (model == "survival"){
                    ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
                    lbCi95[i] <- ci[,2]
                    ubCi95[i] <- ci[,3]
                } else {
                    ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
                    lbCi95[i] <- ci[,2]
                    ubCi95[i] <- ci[,3]
                }
            }
        }
    # } else {
    #     cyclopsFit <- fitCyclopsModel(dataPtr, prior = createPrior("none"))
    #     coefCyclops <- data.frame(covariateId = as.integer(names(coef(cyclopsFit))),beta=coef(cyclopsFit))
    #     coefCyclops <- coefCyclops[order(coefCyclops$covariateId),]
    #     coefCyclops <- coefCyclops$beta
    #     if (coverage) {
    #         if (model == "survival"){
    #             ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
    #             lbCi95 <- ci[,2]
    #             ubCi95 <- ci[,3]
    #         } else {
    #             ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
    #             lbCi95 <- ci[,2]
    #             ubCi95 <- ci[,3]
    #         }
    #     }
    # }
    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units"),
                     "(",
                     signif(cyclopsFit$timeFit,3), attr(cyclopsFit$timeFit,"units"),
                     ")"))

    if (coverage) {
        df <- data.frame(coef = coefCyclops, lbCi95 = lbCi95, ubCi95 = ubCi95)
    } else {
        df <- data.frame(coef = coefCyclops)
    }
    attr(df, "dataPtr") <- dataPtr
    df
}

#' @title Mean squared error
#'
#' @description
#' \code{mse} computes the mean squared error between two numeric vectors
#'
#' @param goldStandard Numeric vector
#' @param estimates  Numeric vector
#'
#' @return MSE(\code{goldStandard}, \code{estimates})
#'
#' @export
mse <- function(goldStandard, estimates){
    mean((goldStandard - estimates)^2)
}

#' @title Coverage
#'
#' @description
#' \code{coverage} computes the coverage on confidence intervals
#'
#' @param goldStandard Numeric vector
#' @param lowerBounds  Numeric vector. Lower bound of the confidence intervals
#' @param upperBounds  Numeric vector. Upper bound of the confidence intervals
#'
#' @return The proportion of times \code{goldStandard} falls between \code{lowerBound} and \code{upperBound}
#'
#' @export
coverage <- function(goldStandard, lowerBounds, upperBounds){
    sum(goldStandard >= lowerBounds & goldStandard <= upperBounds) / length(goldStandard)
}

#' @title Plot Cyclops simulation model fit
#'
#' @description
#' \code{plotCyclopsSimulationFit} generates a plot that compares \code{goldStandard} coefficients to their
#' Cyclops model \code{fit}.
#'
#' @param fit   A Cyclops simulation fit generated by \code{fitCyclopsSimulation}
#' @param goldStandard Numeric vector.  True relative risks.
#' @param label String. Name of estimate type.
#'
#' @keywords internal
plotCyclopsSimulationFit <- function(fit,goldStandard,label){
    if (requireNamespace("ggplot2")) {
        ggplot2::ggplot(fit, ggplot2::aes(x= goldStandard , y=coef, ymin=fit$lbCi95, ymax=fit$ubCi95), environment=environment()) +
            ggplot2::geom_abline(intercept = 0, slope = 1) +
            ggplot2::geom_pointrange(alpha=0.2) +
            ggplot2::scale_y_continuous(label)
    } else {
        stop("gglot2 package required")
    }
}


# TODO Move to vignette
#
# runSimulation1 <- function(){
#     model = "survival"      # "logistic", "survival", or "poisson"
#     sim <- simulateCyclopsData(nstrata=2000,
#                         ncovars=100,
#                         nrows=10000,
#                         effectSizeSd=0.5,
#                         eCovarsPerRow=2,
#                         model=model)
#     coefGoldStandard <- log(sim$effectSizes$rr)
#
#     fitCyclops <- fitCyclopsSimulation(sim, useCyclops = TRUE, regularized=TRUE,model)
#     fit <- fitCyclopsSimulation(sim, useCyclops = FALSE, model)
#
#     writeLines(paste("MSE Cyclops:", mse(fitCyclops$coef,coefGoldStandard)))
#     writeLines(paste("MSE other:", mse(fit$coef,coefGoldStandard)))
#
#     plotCyclopsSimulationFit(fitCyclops,coefGoldStandard,"Cyclops")
#     plotCyclopsSimulationFit(fit,coefGoldStandard,"Other")
#
#     writeLines(paste("Coverage Cyclops:", coverage(coefGoldStandard,fitCyclops$lbCi95, fitCyclops$ubCi95)))
#     writeLines(paste("Coverage other:", coverage(coefGoldStandard,fit$lbCi95, fit$ubCi95)))
# }
#
# runSimulation2 <- function(){
#     model = "logistic"      # "logistic", "survival", or "poisson"
#     sim <- simulateCyclopsData(nstrata=2000,
#                         ncovars=100,
#                         nrows=10000,
#                         effectSizeSd=0.5,
#                         eCovarsPerRow=2,
#                         model=model)
#     coefGoldStandard <- log(sim$effectSizes$rr)
#
#     fit <- fitCyclopsSimulation(sim, useCyclops = FALSE, model,coverage=FALSE)
#     fitCyclops <- fitCyclopsSimulation(sim, useCyclops = TRUE, regularized=TRUE,model,coverage=FALSE)
#
#     writeLines(paste("MSE Cyclops:", mse(fitCyclops$coef,coefGoldStandard)))
#     writeLines(paste("MSE other:", mse(fit$coef,coefGoldStandard)))
# }
