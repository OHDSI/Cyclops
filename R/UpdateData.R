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
#' positive weights. Some weight must be positive. Two cases still differ from data without the rows: without a
#' prior, a covariate that is nonzero only in rows of weight 0 makes the fit ill-conditioned (data without the rows
#' would not have the covariate); and cross-validation draws its random folds over all rows and, with automatic
#' search, starts from a variance computed over all rows (give \code{cvFolds} and a grid for the selection of the
#' data without the rows).
#'
#' The weights stay with the data for all later fits; weights given to \code{\link{fitCyclopsModel}} take
#' precedence, with a warning. A fit object keeps pointing at its data, so after a refit, earlier fits on the data
#' report the latest fit in \code{vcov}, \code{predict} and \code{confint}: take what you need from a fit before
#' you update its data.
#'
#' @param cyclopsData A Cyclops data object
#' @param weights     Non-negative weights: a data frame with columns \code{rowId} and \code{weights} and a row for
#'                    each data row, for data with row ids (\code{\link{convertToCyclopsData}}); a vector with the
#'                    weight of each data row, in the order of the data for data from
#'                    \code{\link{createCyclopsData}}, or of increasing \code{rowId} for data with row ids that keep
#'                    that order (\code{"lr"}, \code{"pr"} and \code{"ls"} data, for example; other data with row
#'                    ids need the data frame); or \code{NULL}, for weight 1 on every row
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
        if (!any(weights > 0)) {
            stop("Need a positive weight")
        }
        if (cyclopsData$modelType %in% .conditionalModelTypes) {
            mixed <- .mixedZeroWeightStrata(cyclopsData, weights)
            if (mixed > 0) {
                stop(mixed, " strata have rows of weight 0 and rows of positive weight; in modelType = \"",
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
#' and \code{"cox_counting"}. Outcomes lie between 0 and 1 in \code{"lr"} and \code{"clr"}, are 0 or 1 in \code{"cox"}
#' and \code{"cox_counting"}, and are non-negative in \code{"pr"} and \code{"cpr"}.
#'
#' Fits made earlier on the data can no longer be used: their \code{vcov}, \code{predict}, \code{confint},
#' \code{survfit} and \code{\link{runBootstrap}} stop, and after the next fit on the data they would report that fit. Take what you need
#' from a fit before you update its data.
#'
#' @param cyclopsData A Cyclops data object
#' @param y           The new outcomes: a data frame with columns \code{rowId} and \code{y} and a row for each
#'                    data row, for data with row ids (\code{\link{convertToCyclopsData}}); or a vector with the
#'                    outcome of each data row, in the order of the data for data from
#'                    \code{\link{createCyclopsData}}, or of increasing \code{rowId} for data with row ids that keep
#'                    that order (\code{"lr"}, \code{"pr"} and \code{"ls"} data, for example; other data with row ids
#'                    need the data frame)
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
    if (modelType %in% c("lr", "clr") && any(y < 0 | y > 1)) {
        stop("Logistic models need outcomes y between 0 and 1")
    }
    if (modelType %in% c("pr", "cpr") && any(y < 0)) {
        stop("Poisson models need non-negative outcomes y")
    }
    time <- NULL
    if (modelType %in% c("cox", "cox_counting")) {
        if (!all(y %in% c(0, 1))) {
            stop("Cox models need outcomes y of 0 or 1")
        }
        # The rows were sorted with y last (stratum, decreasing time, y); the fit does not need that order within ties
        time <- getTimeVector(cyclopsData)
    }
    loadNewSqlCyclopsDataY(cyclopsData, y = y, time = time)
    # The fitted model state (sufficient statistics, separation, risk sets) came from the old outcomes
    assign("cyclopsInterfacePtr", NULL, cyclopsData)
    # Earlier fits keep their own interface, which now sees the new outcomes: the Breslow baseline checks this
    assign("outcomesVersion", if (is.null(cyclopsData$outcomesVersion)) 1 else cyclopsData$outcomesVersion + 1,
           cyclopsData)
    invisible(cyclopsData)
}

# Models whose rows of weight 0 still enter their stratum
.conditionalModelTypes <- c("clr", "clr_efron", "clr_exact", "cpr", "sccs")

# Number of strata with rows of weight 0 and rows of positive weight; weights in the internal order
.mixedZeroWeightStrata <- function(cyclopsData, weights) {
    stratum <- .cyclopsGetStratumVector(cyclopsData)
    positive <- rowsum(as.numeric(weights > 0), stratum, reorder = FALSE)
    rows <- rowsum(rep(1, length(stratum)), stratum, reorder = FALSE)
    sum(positive > 0 & positive < rows)
}

# Values of `column` in the internal order, from a data frame with rowId and `column`, or a vector in data order
# (createCyclopsData) or in increasing rowId when that is the internal order
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
            stop("Need a value of ", column, " for each of the ", n, " data rows, and no other rowIds")
        }
        return(values[[column]][index])
    }
    if (length(values) != n) {
        stop("Need a value for each of the ", n, " data rows")
    }
    rowIds <- .cyclopsGetRowLabels(cyclopsData)
    # A vector in rowId order would be read in the internal order
    if (!is.null(rowIds) && is.unsorted(bit64::as.integer64(rowIds))) {
        stop("These data do not keep their rows in rowId order; give a data frame with columns rowId and ", column)
    }
    if (!is.null(cyclopsData$sortOrder)) {
        values <- values[cyclopsData$sortOrder]
    }
    values
}
