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

    # TODO Check that penalty is valid

    structure(list(penalty = penalty,
                   exclude = exclude,
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
    tol <- 1E-8
    cutoff <- 1E-16
    maxIterations <- 100

    # Getting starting values
    startFit <- fitCyclopsModel(cyclopsData, prior = createAbridgeStartingPrior(cyclopsData, control),
                                control, weights, forceNewObject, returnEstimates, startingCoefficients, fixedCoefficients)

    ridge <- rep("normal", getNumberOfCovariates(cyclopsData)) # TODO Handle intercept
    pre_coef <- coef(startFit)
    penalty <- getPenalty(cyclopsData, abridgePrior)

    continue <- TRUE
    count <- 0
    converged <- FALSE

    while (continue) {
        count <- count + 1

        working_coef <- ifelse(abs(pre_coef) < cutoff, 0.0, pre_coef)
        fixed <- working_coef == 0.0
        variance <- (working_coef) ^ 2 / penalty

        prior <- createPrior(ridge, variance = variance)
        fit <- fitCyclopsModel(cyclopsData,
                               prior = prior,
                               control, weights, forceNewObject,
                               startingCoefficients = working_coef,
                               fixedCoefficients = fixed)

        coef <- coef(fit)
        if (max(abs(coef - pre_coef)) < tol) {
            converged <- TRUE
        } else {
            pre_coef <- coef
        }

        if (converged || count >= maxIterations) {
            continue <- FALSE
        }
    }

    class(fit) <- c(class(fit), "cyclopsAbridgeFit")
    fit$abridgeConverged <- converged
    fit$abridgeIterations <- count
    fit$abridgeFinalPriorVariance <- variance
    fit
}

createAbridgeStartingPrior <- function(cyclopsData, control) {  # TODO Better starting choices
    if (getNumberOfRows(cyclopsData) < control$minCVData) {
        createPrior("normal", variance = 10)
    } else {
        createPrior("normal", useCrossValidation = TRUE)
    }
}

getPenalty <- function(cyclopsData, abridgePrior) {
    if (abridgePrior$penalty == "bic") {
        return(log(getNumberOfRows(cyclopsData))) # TODO Handle stratified models
    } else {
        stop("Unhandled ABRIDGE penalty type")
    }
}
