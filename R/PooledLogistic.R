# @file PooledLogistic.R
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

#' @title Expand subjects to their person-interval rows
#'
#' @description
#' \code{convertToLongOutcome} expands the outcomes of a pooled logistic regression (\code{modelType = "plr"})
#' given as one row per subject (implicit rows) into one row per subject and interval at risk (explicit rows),
#' as \code{\link{convertToCyclopsData}} takes either. Covariates stay one row per subject.
#'
#' @param outcomes A data frame with one row per subject and columns \code{rowId}, \code{entry} and \code{exit}
#'                 (the first and last intervals at risk, positive integers) and \code{y} (1 if the event is in
#'                 interval \code{exit}), and optionally \code{weights} and \code{offset}.
#'
#' @return A data frame with one row per subject and interval, sorted by \code{rowId} and \code{interval}, with
#' columns \code{rowId}, \code{interval}, \code{y} (1 in the event's interval only) and the subject's \code{weights}
#' and \code{offset}, if given.
#'
#' @examples
#' outcomes <- data.frame(rowId = 1:3, entry = c(1, 1, 2), exit = c(3, 1, 4), y = c(1, 0, 0))
#' convertToLongOutcome(outcomes)
#' @export
convertToLongOutcome <- function(outcomes) {
    .checkPooledSubjects(outcomes)
    outcomes <- outcomes[order(outcomes$rowId), , drop = FALSE]
    n <- outcomes$exit - outcomes$entry + 1
    subject <- rep(seq_len(nrow(outcomes)), n)
    interval <- outcomes$entry[subject] + sequence(n) - 1
    long <- data.frame(rowId = outcomes$rowId[subject],
                       interval = interval,
                       y = as.numeric(outcomes$y[subject] == 1 & interval == outcomes$exit[subject]))
    for (column in intersect(c("weights", "offset"), colnames(outcomes))) {
        long[[column]] <- outcomes[[column]][subject]
    }
    long
}

.checkPooledSubjects <- function(outcomes) {
    if (!all(c("rowId", "entry", "exit", "y") %in% colnames(outcomes))) {
        stop("Implicit plr outcomes need columns rowId, entry, exit and y", call. = FALSE)
    }
    if (anyDuplicated(outcomes$rowId)) {
        stop("Implicit plr outcomes need one row per subject (rowId)", call. = FALSE)
    }
    if (any(outcomes$entry < 1 | outcomes$exit < outcomes$entry | outcomes$entry != round(outcomes$entry) |
            outcomes$exit != round(outcomes$exit))) {
        stop("entry and exit must be integers with 1 <= entry <= exit", call. = FALSE)
    }
}

# convertToCyclopsData() for modelType = "plr": one row of the data per subject (rowId), with the subject's
# person-interval rows, the time basis and the time x covariate interactions loaded beside it (PooledRows.h)
.convertToPooledCyclopsData <- function(outcomes, covariates, timeBasis, timeEffectMap, addIntercept,
                                        checkRowIds, normalize, quiet, floatingPoint, link = "logit") {
    link <- match.arg(link, c("logit", "cloglog"))
    outcomes <- as.data.frame(outcomes)
    covariates <- as.data.frame(covariates)
    if (!all(outcomes$y %in% c(0, 1))) {
        stop("The outcomes y of plr models must be 0 or 1", call. = FALSE)
    }
    for (column in intersect(c("weights", "offset"), colnames(outcomes))) {
        if (any(!is.finite(outcomes[[column]])) || (column == "weights" && any(outcomes$weights < 0))) {
            stop("The ", column, " of plr outcomes must be finite", if (column == "weights") " and non-negative",
                 call. = FALSE)
        }
    }
    implicitRows <- all(c("entry", "exit") %in% colnames(outcomes))
    rowWeight <- numeric(0)
    rowOffset <- numeric(0)
    subjectOffset <- numeric(0)
    if (implicitRows) {
        .checkPooledSubjects(outcomes)
        outcomes <- outcomes[order(outcomes$rowId), , drop = FALSE]
        subjects <- outcomes$rowId
        ySubject <- outcomes$y
        rowsPerSubject <- outcomes$exit - outcomes$entry + 1
        interval <- outcomes$entry
        lastInterval <- max(outcomes$exit)
        event <- integer(0)
        if ("offset" %in% colnames(outcomes)) subjectOffset <- outcomes$offset
    } else {
        if (!"interval" %in% colnames(outcomes)) {
            if (!is.null(timeBasis)) stop("Explicit plr rows with a time basis need an interval column", call. = FALSE)
            outcomes$interval <- 1
        }
        if (any(outcomes$interval < 1 | outcomes$interval != round(outcomes$interval))) {
            stop("plr intervals must be positive integers", call. = FALSE)
        }
        outcomes <- outcomes[order(outcomes$rowId, outcomes$interval), , drop = FALSE]
        runs <- rle(as.numeric(outcomes$rowId))
        subjects <- runs$values
        rowsPerSubject <- runs$lengths
        ySubject <- as.numeric(rowsum(outcomes$y, rep(seq_along(subjects), rowsPerSubject))[, 1] > 0)
        interval <- outcomes$interval
        lastInterval <- max(outcomes$interval)
        event <- outcomes$y
        if ("weights" %in% colnames(outcomes)) rowWeight <- outcomes$weights
        if ("offset" %in% colnames(outcomes)) rowOffset <- outcomes$offset
    }
    subjectId <- NULL
    if ("subjectId" %in% colnames(outcomes)) { # one per subject
        subjectId <- outcomes$subjectId[!duplicated(outcomes$rowId)]
        if (!implicitRows && any(outcomes$subjectId != rep(subjectId, rowsPerSubject))) {
            stop("subjectId must be constant within a rowId", call. = FALSE)
        }
    }

    basisIds <- numeric(0)
    basis <- matrix(0, 0, 0)
    if (!is.null(timeBasis)) {
        timeBasis <- as.data.frame(timeBasis)
        if (!all(c("interval", "covariateId", "covariateValue") %in% colnames(timeBasis))) {
            stop("timeBasis needs columns interval, covariateId and covariateValue", call. = FALSE)
        }
        if (any(timeBasis$interval < 1 | timeBasis$interval != round(timeBasis$interval)) ||
            anyDuplicated(timeBasis[, c("interval", "covariateId")])) {
            stop("timeBasis intervals must be positive integers, with one value per interval and covariateId",
                 call. = FALSE)
        }
        basisIds <- sort(unique(timeBasis$covariateId))
        basis <- matrix(0, max(lastInterval, timeBasis$interval), length(basisIds))
        basis[cbind(timeBasis$interval, match(timeBasis$covariateId, basisIds))] <- timeBasis$covariateValue
    }

    if (checkRowIds && any(!covariates$rowId %in% subjects)) {
        if (!quiet) writeLines("Removing covariate values with rowIds that are not in outcomes")
        covariates <- covariates[covariates$rowId %in% subjects, , drop = FALSE]
    }
    covariates <- covariates[order(covariates$covariateId, covariates$rowId), , drop = FALSE]
    covariateIds <- unique(covariates$covariateId)

    interactions <- NULL
    if (!is.null(timeEffectMap)) {
        timeEffectMap <- as.data.frame(timeEffectMap)
        if (!all(c("covariateId", "timeCovariateId", "interactionId") %in% colnames(timeEffectMap))) {
            stop("timeEffectMap of plr models needs columns covariateId, timeCovariateId and interactionId",
                 call. = FALSE)
        }
        if (!all(timeEffectMap$covariateId %in% covariateIds) || !all(timeEffectMap$timeCovariateId %in% basisIds)) {
            stop("timeEffectMap names a covariate or a time-basis column that is not in the data", call. = FALSE)
        }
        timeEffectMap <- timeEffectMap[order(timeEffectMap$interactionId), , drop = FALSE]
        interactions <- lapply(seq_len(nrow(timeEffectMap)), function(m) {
            covariates[covariates$covariateId == timeEffectMap$covariateId[m], c("rowId", "covariateValue")]
        })
    }
    allIds <- c(if (addIntercept) 0, covariateIds, basisIds, timeEffectMap$interactionId)
    if (anyDuplicated(allIds)) {
        stop("Covariate, time-basis and interaction ids must be distinct (and not 0 with an intercept)", call. = FALSE)
    }

    dataPtr <- createSqlCyclopsData(modelType = "plr", floatingPoint = floatingPoint)
    loadNewSqlCyclopsDataY(object = dataPtr, rowId = subjects, y = ySubject)
    if (addIntercept) {
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    }
    loadNewSqlCyclopsDataMultipleX(object = dataPtr,
                                   covariateId = covariates$covariateId,
                                   rowId = covariates$rowId,
                                   covariateValue = covariates$covariateValue,
                                   name = covariateIds)
    for (id in basisIds) { # every subject: the time basis gives the values
        loadNewSqlCyclopsDataX(dataPtr, id, subjects, NULL, name = as.character(id))
    }
    for (m in seq_along(interactions)) { # the covariate's values: the time basis multiplies them
        loadNewSqlCyclopsDataX(dataPtr, timeEffectMap$interactionId[m], interactions[[m]]$rowId,
                               interactions[[m]]$covariateValue, name = as.character(timeEffectMap$interactionId[m]))
    }
    .loadCyclopsDataPooledRows(dataPtr, implicitRows, as.integer(rowsPerSubject), as.integer(interval),
                               as.integer(event), as.numeric(rowWeight), as.numeric(rowOffset),
                               as.numeric(subjectOffset), as.vector(basis), nrow(basis),
                               bit64::as.integer64(c(basisIds, timeEffectMap$interactionId)),
                               c(seq_along(basisIds), match(timeEffectMap$timeCovariateId, basisIds)),
                               link == "cloglog")
    dataPtr$numberOfPooledRows <- sum(rowsPerSubject)
    dataPtr$numberOfPooledEvents <- if (implicitRows) sum(ySubject) else sum(event)
    dataPtr$pooledLink <- link
    if (!is.null(subjectId)) {
        .loadCyclopsDataSubjectId(dataPtr, bit64::as.integer64(subjectId))
    }

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }
    if (implicitRows && "weights" %in% colnames(outcomes)) {
        dataPtr$weights <- outcomes$weights # one per subject; explicit rows carry theirs in the rows
    }
    dataPtr
}
