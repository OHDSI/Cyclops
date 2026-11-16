# @file UpdateData.R
#
# Copyright 2026 Observational Health Data Sciences and Informatics
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

#' @title Update the weights of Cyclops data
#'
#' @description
#' \code{updateWeights} replaces the row weights that \code{\link{fitCyclopsModel}} uses for a Cyclops data
#' object, so that a model can be refit with new weights (bootstrap frequency weights, for example) without
#' rebuilding the data. The data object is changed in place.
#'
#' @details
#' A row of weight 0 is left out: the fit equals that of the data without the row. In conditional models
#' (\code{"clr"}, \code{"clr_efron"}, \code{"clr_exact"}, \code{"cpr"} and \code{"sccs"}), a row of weight 0
#' would still enter the denominator of its stratum, so the rows of a stratum must all have weight 0 or all have
#' positive weights.
#'
#' The weights stay with the data for all later fits; weights given to \code{\link{fitCyclopsModel}} take
#' precedence, with a warning.
#'
#' @param cyclopsData A Cyclops data object
#' @param weights     Non-negative weights: for data with row ids (\code{\link{convertToCyclopsData}}), a data
#'                    frame with columns \code{rowId} and \code{weights} and a row for each data row; or a vector
#'                    with the weight of each data row, in the order that the \code{weights} of
#'                    \code{\link{fitCyclopsModel}} take (for data from \code{\link{createCyclopsData}}, the order
#'                    of the data); or \code{NULL}, for weight 1 on every row
#'
#' @return The Cyclops data object, invisibly
#'
#' @examples
#' counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
#' outcome <- gl(3, 1, 9)
#' treatment <- gl(3, 3)
#' cyclopsData <- createCyclopsData(counts ~ outcome + treatment, modelType = "pr")
#' fit <- fitCyclopsModel(cyclopsData)
#' updateWeights(cyclopsData, c(1, 2, 0, 1, 1, 3, 1, 0, 1))
#' refit <- fitCyclopsModel(cyclopsData)
#' coef(refit)
#'
#' @export
updateWeights <- function(cyclopsData, weights) {
    .checkData(cyclopsData)
    if (!is.null(weights)) {
        weights <- as.numeric(.valuesInDataOrder(cyclopsData, weights, "weights"))
        if (any(!is.finite(weights)) || any(weights < 0)) {
            stop("weights must be finite and non-negative")
        }
        if (cyclopsData$modelType %in% c("clr", "clr_efron", "clr_exact", "cpr", "sccs")) {
            stratum <- .cyclopsGetStratumVector(cyclopsData)
            mixed <- tapply(weights > 0, stratum, function(positive) any(positive) && !all(positive))
            if (any(mixed)) {
                stop(sum(mixed), " strata have rows of weight 0 and rows of positive weight; in modelType = \"",
                     cyclopsData$modelType, "\", a row of weight 0 still enters its stratum")
            }
        }
    }
    assign("weights", weights, cyclopsData)
    invisible(cyclopsData)
}

#' @title Update the outcomes of Cyclops data
#'
#' @description
#' \code{updateOutcomes} replaces the outcome \code{y} of each row of a Cyclops data object, so that a model can
#' be fit to new outcomes without rebuilding the data. The rows, strata, times, weights and covariates stay. The data
#' object is changed in place, and the next fit on it starts from new model state.
#'
#' @details
#' Available for \code{modelType} \code{"lr"}, \code{"pr"}, \code{"ls"}, \code{"clr"}, \code{"cpr"}, \code{"cox"}
#' and \code{"cox_counting"}. Cox data keep their rows sorted by stratum, decreasing time and outcome, so among the
#' rows of a stratum with the same time, those with \code{y = 0} must still come before those with \code{y = 1};
#' otherwise rebuild the data.
#'
#' Fits made earlier on the data share them, so their later computations (such as \code{confint} or
#' \code{predict}) use the new outcomes.
#'
#' @param cyclopsData A Cyclops data object
#' @param y           The new outcomes: for data with row ids (\code{\link{convertToCyclopsData}}), a data frame
#'                    with columns \code{rowId} and \code{y} and a row for each data row; or a vector with the
#'                    outcome of each data row, in the order that the \code{weights} of
#'                    \code{\link{fitCyclopsModel}} take (for data from \code{\link{createCyclopsData}}, the order
#'                    of the data)
#'
#' @return The Cyclops data object, invisibly
#'
#' @examples
#' counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
#' outcome <- gl(3, 1, 9)
#' treatment <- gl(3, 3)
#' cyclopsData <- createCyclopsData(counts ~ outcome + treatment, modelType = "pr")
#' fit <- fitCyclopsModel(cyclopsData)
#' updateOutcomes(cyclopsData, c(20, 15, 15, 22, 11, 18, 24, 15, 10))
#' refit <- fitCyclopsModel(cyclopsData)
#' coef(refit)
#'
#' @export
updateOutcomes <- function(cyclopsData, y) {
    .checkData(cyclopsData)
    modelType <- cyclopsData$modelType
    if (!(modelType %in% c("lr", "pr", "ls", "clr", "cpr", "cox", "cox_counting"))) {
        stop("updateOutcomes() is not available for modelType = \"", modelType, "\"; rebuild the data")
    }
    y <- as.numeric(.valuesInDataOrder(cyclopsData, y, "y"))
    if (any(!is.finite(y))) {
        stop("y must be finite")
    }
    time <- NULL
    if (modelType %in% c("cox", "cox_counting")) {
        if (!all(y %in% c(0, 1))) {
            stop("Cox models need outcomes y of 0 or 1")
        }
        time <- getTimeVector(cyclopsData)
        stratum <- .cyclopsGetStratumVector(cyclopsData)
        k <- seq_len(length(y) - 1)
        if (any(stratum[k] == stratum[k + 1] & time[k] == time[k + 1] & y[k] > y[k + 1])) {
            stop("The new outcomes change the order of the rows (by stratum, decreasing time and outcome); ",
                 "rebuild the data")
        }
    }
    loadNewSqlCyclopsDataY(cyclopsData, y = y, time = time)
    # The fitted model state (sufficient statistics, separation, risk sets) came from the old outcomes
    assign("cyclopsInterfacePtr", NULL, cyclopsData)
    invisible(cyclopsData)
}

# Values of `column` in data order, from a data frame with rowId and `column`, or a vector in the order that the
# weights of fitCyclopsModel() take
.valuesInDataOrder <- function(cyclopsData, values, column) {
    n <- getNumberOfRows(cyclopsData)
    if (is.data.frame(values)) {
        if (!all(c("rowId", column) %in% colnames(values))) {
            stop("Need columns rowId and ", column)
        }
        rowIds <- .cyclopsGetRowLabels(cyclopsData)
        if (is.null(rowIds)) {
            stop("These data have no row ids; give a vector with a value for each row in data order")
        }
        rowId <- if (is.factor(values$rowId)) as.character(values$rowId) else values$rowId # not the factor codes
        keys <- as.character(bit64::as.integer64(rowId))
        if (anyDuplicated(keys)) {
            stop("Duplicate rowIds")
        }
        index <- match(rowIds, keys)
        if (anyNA(index) || length(keys) != n) {
            stop("Need a ", column, " for each of the ", n, " data rows, and no other rowIds")
        }
        return(values[[column]][index])
    }
    if (length(values) != n) {
        stop("Need a value for each of the ", n, " data rows")
    }
    if (!is.null(cyclopsData$sortOrder)) {
        values <- values[cyclopsData$sortOrder]
    }
    values
}
