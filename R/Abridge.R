# @file Abridge.R
#
# Copyright 2016 Observational Health Data Sciences and Informatics
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
#
# @author Marc A. Suchard
# @author Ning Li

#' @title Create an ABRIDGE Cyclops prior object
#'
#' @description
#' \code{createAbridgePrior} creates an ABRIDGE Cyclops prior object for use with \code{\link{fitCyclopsModel}}.
#'
#' @param penalty        Specifies the ABRIDGE penalty; possible values are `BIC` or `AIC` or a numeric value
#' @param exclude        A vector of numbers or covariateId names to exclude from prior
#' @param forceIntercept Logical: Force intercept coefficient into prior
#'
#' @examples
#' prior <- createAbridgePrior(penalty = "bic")
#'
#' @return
#' An ABRIDGE Cyclops prior object of class inheriting from \code{"cyclopsAbridgePrior"}
#' and \code{"cyclopsPrior"} for use with \code{fitCyclopsModel}.
#'
#' @export
createAbridgePrior <- function(penalty = "bic",
                               exclude = c(),
                               forceIntercept = FALSE) {

    structure(list(priorType = "normal",
                   variance = 0,
                   exclude = exclude,
                   graph = NULL,
                   neighborhood = NULL,
                   useCrossValidation = FALSE,
                   forceIntercept = forceIntercept),
              class = c("cyclopsPrior","cyclopsAbridgePrior"))
}

# Below are package-private functions

fitAbridge <- function(cyclopsData,
                       abridgePrior,
                       control,
                       weights,
                       forceNewObject,
                       returnEstimates,
                       startingCoefficients,
                       fixedCoefficients) {

    # TODO Pass as parameters
    tol <- 1E-10
    cutoff <- 1E-20
    maxIterations <- 100

    # Getting starting values
    startFit <- fitCyclopsModel(cyclopsData, prior = createAbridgeStartingPrior(),
                                control, weights, forceNewObject, startingCoefficients, fixedCoefficients)

    ridge <- rep("normal", getNumberOfCovariates(cyclopsData)) # TODO Handle intercept
    pre_coef <- coef(startFit)
    penalty <- getPenalty(cyclopsData, abridgePrior)

    continue <- TRUE
    count <- 0
    converged <- FALSE

    while (continue) {
        count <- count + 1

#         for (i in 1:p) {
#             if (abs(pre_coef[i]) < cutoff) {
#                 if (pre_coef[i] > 0)
#                     pre_coef[i] <- cutoff
#                 else
#                     pre_coef[i] <- -cutoff
#             }
#         }

        prior <- createPrior(ridge, variance = (pre_coef) ^ 2 / penalty)
        fit <- fitCyclopsModel(cyclopsData,
                                      prior = prior,
                                      control, weights, forceNewObject,
                                      startingCoefficients = pre_coef)

        coef <- coef(fit)
        if (max(abs(fit - pre_coef)) < tol) {
            converged <- TRUE
        } else {
            pre_coef <- coef
        }

        if (converged || count >= maxIterations) {
            continue <- FALSE
        }
    }

    fit # TODO Threshold final coefficients
}

createAbridgeStartingPrior <- function() {
    createPrior("normal", useCrossValidation = TRUE)
}

getPenalty <- function(cyclopsData, abridgePrior) {
    if (abridgePrior$penalty == "bic") {
        return(log(getNumberOfRows(cyclopsData))) # TODO Handle stratified models
    } else {
        stop("Unhandled ABRIDGE penalty type")
    }
}


.donotrun <- function() {

p<-20
n<-1000

ridge<-rep("normal",p)


beta1<-c(0.5,0,0,-1,1.2)
beta2<-seq(0,0,length=p-length(beta1))
beta<-c(beta1,beta2)

x<-matrix(rnorm(p*n,mean=0,sd=1),ncol = p)
y<-NULL

for(i in 1:n){
    yi<-rbinom(1,1,exp(x[i,]%*%beta)/(1+exp(x[i,]%*%beta)))
    y<-c(y, yi)
}

cyclopsData <- createCyclopsData(y ~ x - 1,modelType = "lr")
cyclopsFit <- fitCyclopsModel(cyclopsData,prior=createPrior("normal",useCrossValidation = TRUE))
coef(cyclopsFit)

pre_coef <- coef(cyclopsFit)

prior <- createPrior(ridge, variance = (pre_coef)^2/log(n))


cyclopsFit <- fitCyclopsModel(cyclopsData, prior = prior)
coef(cyclopsFit)

pre_coef<-coef(cyclopsFit)

}

abridge <- function() {
    cat("hello")
}
