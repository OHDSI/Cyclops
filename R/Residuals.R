# @file Residuals.R
#
# Copyright 2024 Observational Health Data Sciences and Informatics
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

#' @method residuals cyclopsFit
#' @title Model residuals
#'
#' @description
#' \code{residuals.cyclopsFit} computes model residuals for Cox model-based Cyclops objects
#'
#' @param object    A Cyclops model fit object
#' @param parm      A specification of which parameters require residuals,
#'                  either a vector of numbers or covariateId names
#' @param type      Character string indicating the type of residual desires. Possible
#'                  values are "schoenfeld".
#' @param ...		Additional parameters for compatibility with S3 parent function
#'
#' @importFrom stats residuals
#'
#' @export
residuals.cyclopsFit <- function(object, parm, type = "schoenfeld", ...) {
    modelType <- object$cyclopsData$modelType
    if (modelType != "cox") {
        stop("Residuals for only Cox models are implemented")
    }
    if (type != "schoenfeld") {
        stop("Only Schoenfeld residuals are implemented")
    }

    .checkInterface(object$cyclopsData, testOnly = TRUE)

    res <- .cyclopsGetSchoenfeldResiduals(object$interface, NULL)

    result <- res$residuals
    names(result) <- res$times

    return(rev(result))
}

#' @title Test hazard ratio proportionality assumption
#'
#' @description
#' \code{testProportionality} tests the hazard ratio proportionality assumption
#' of a Cyclops model fit object
#'
#' @param object    A Cyclops model fit object
#' @param parm      A specification of which parameters require residuals,
#'                  either a vector of numbers or covariateId names
#' @param transformedTimes Vector of transformed time
#'
#' @export
testProportionality <- function(object, parm, transformedTimes) {

    .checkInterface(object$cyclopsData, testOnly = TRUE)

    if (object$cyclopsData$modelType != "cox") {
        stop("Proportionality test for only Cox models are implemented")
    }

    if (getNumberOfRows(object$cyclopsData) != length(transformedTimes)) {
        stop("Incorrect 'transformedTime' length")
    }

    transformedTimes <- transformedTimes[object$cyclopsData$sortOrder]
    message("TODO: permute transformedTimes")

    res <- .cyclopsTestProportionality(object$interface, NULL, transformedTimes)

    return(res)
}
