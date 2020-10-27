# @file SpecialPriors.R
#
# Copyright 2018 Observational Health Data Sciences and Informatics
#
# This file is part of Cyclops
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

#' @title Create a Cyclops prior object that returns the MLE of non-separable coefficients
#'
#' @description
#' \code{createNonSeparablePrior} creates a Cyclops prior object for use with \code{\link{fitCyclopsModel}}.
#'
#' @param maxIterations Numeric: maximum iterations to achieve convergence
#' @param ... Additional argument(s) for \code{\link{fitCyclopsModel}}
#'
#' @examples
#' prior <- createNonSeparablePrior()
#'
#' @return
#' A Cyclops prior object of class inheriting from
#' \code{"cyclopsPrior"} for use with \code{fitCyclopsModel}.
#'
#' @export
createNonSeparablePrior <- function(maxIterations = 10,
                                    ...) {

    fitHook <- function(...) {
      # closure to capture fitCyclopsModel control
      nonSeparableHook(maxIterations, ...)
    }

    prior <- createPrior(priorType = "none")
    prior$fitHook <- fitHook

    return(prior)
}

# Below are package-private functions

nonSeparableHook <- function(
                    maxIterations,
                    cyclopsData,
                    prior,
                    control,
                    weights,
                    forceNewObject,
                    returnEstimates,
                    startingCoefficients,
                    fixedCoefficients) {

    prior <- createPrior(priorType = "none")

    dim <- length(getCovariateIds(cyclopsData))
    separable <- rep(FALSE, dim)

    continue <- TRUE
    count <- 0

    while (continue) {
        count <- count + 1

        fit <- fitCyclopsModel(cyclopsData,
                                        prior = prior,
                                        control, weights, forceNewObject, returnEstimates,
                                        startingCoefficients,
                                        fixedCoefficients = separable)

        if (fit$return_flag == "ILLCONDITIONED") {
            new_separable <- is.nan(coef(fit, ignoreConvergence = TRUE)) | separable
        } else {
            new_separable <- separable
        }

        if (all(new_separable == separable) || count >= maxIterations) {
            continue <- FALSE
        } else {
            separable <- new_separable
        }
    }

    fit$estimation[separable, "estimate"] <- NaN
    fit$non_separable_iterations <- count

    return(fit)
}

