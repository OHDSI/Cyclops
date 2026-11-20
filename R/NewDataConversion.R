# @file NewDataConversion.R
#
# Copyright 2020 Observational Health Data Sciences and Informatics
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

isSorted <- function(data, columnNames, ascending = rep(TRUE, length(columnNames))){
    return(.isSorted(data, columnNames, ascending))
}

# The end of each row's (startTime, time] interval may also be called endTime
.renameEndTime <- function(outcomes) {
    if ("endTime" %in% colnames(outcomes)) {
        if ("time" %in% colnames(outcomes)) {
            stop("Give the end of each row's interval as time or endTime, not both", call. = FALSE)
        }
        outcomes <- rename(outcomes, time = "endTime")
    }
    outcomes
}

#' Convert data from two data frames or ffdf objects into a CyclopsData object
#'
#' @description
#' \code{convertToCyclopsData} loads data from two data frames or ffdf objects, and inserts it into a Cyclops data object.
#'
#' @param outcomes      A data frame or ffdf object containing the outcomes with predefined columns (see below).
#' @param covariates    A data frame or ffdf object containing the covariates with predefined columns (see below),
#'                      or, for \code{"lr"} and \code{"pr"} models with data-frame outcomes, a sparse matrix (see below).
#' @param modelType     Cyclops model type. Current supported types are "pr", "cpr", lr", "clr", "cox", "cox_counting"
#'                      or "plr" (pooled logistic regression; see Details)
#' @param addIntercept  Add an intercept to the model?
#' @param checkSorting  (DEPRECATED) Check if the data are sorted appropriately, and if not, sort.
#' @param checkRowIds   Check if all rowIds in the covariates appear in the outcomes.
#' @param normalize     String: Name of normalization for all non-indicator covariates (possible values: stdev, max, median)
#' @param quiet         If true, (warning) messages are suppressed.
#' @param floatingPoint Specified floating-point representation size (32 or 64)
#' @param timeEffectMap A data frame or ffdf object containing the convariates that have time-varying effects on the outcome
#'                      (for \code{"plr"} models, the time x covariate interactions; see Details)
#' @param timeBasis     For \code{"plr"} models, an optional data frame with columns \code{interval}, \code{covariateId}
#'                      and \code{covariateValue}: the time basis, one covariate (with a coefficient) per basis column,
#'                      whose value in a person-interval row is its value at the row's interval (0 where not given)
#' @param link          For \code{"plr"} models, the link: \code{"logit"} (pooled logistic regression) or \code{"cloglog"}
#'                      (complementary log-log, a grouped-time proportional hazards model). Cloglog fits run on
#'                      coordinate descent, so with correlated time-basis columns they need a tighter \code{tolerance}
#'                      in \code{\link{createControl}} (e.g. 1e-10); their \code{vcov()} is the observed information
#'
#' @details
#' These columns are expected in the outcome object:
#' \tabular{lll}{
#'   \verb{stratumId}    \tab(integer) \tab (optional) Stratum ID for conditional regression models \cr
#'   \verb{rowId}  	\tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{y}    \tab(real) \tab The outcome variable \cr
#'   \verb{time}    \tab(real) \tab For models that use time (e.g. Poisson or Cox regression) this contains time \cr
#'                  \tab        \tab(e.g. number of days); it may instead be called \verb{endTime} \cr
#'   \verb{startTime}    \tab(real) \tab (optional) For \code{"cox_counting"} models, the start of the \code{(startTime, time]} \cr
#'                  \tab        \tab (or \code{(startTime, endTime]}) risk interval of each row; rows without it are at risk \cr
#'                  \tab        \tab from time 0 \cr
#'   \verb{subjectId}    \tab(integer) \tab (optional) For any model, rows with the same subject stay in one cross-validation fold \cr
#'                  \tab        \tab and are resampled together by \code{\link{runBootstrap}} (\code{selectorType = "bySubject"}) \cr
#'   \verb{offset} \tab(real) \tab (optional) For \code{"lr"} models, an offset added to the linear predictor (log-odds \cr
#'                  \tab        \tab scale), as \code{offset} in \code{glm()}; the data object is then finalized \cr
#'   \verb{weights} \tab(real) \tab (optional) Non-negative weights to apply to outcome \cr
#'   \verb{censorWeights} \tab(real) \tab (optional) Non-negative censoring weights for competing risk model; will be computed if not provided.
#' }
#'
#' These columns are expected in the covariates object:
#' \tabular{lll}{
#'   \verb{stratumId}    \tab(integer) \tab (optional) Stratum ID for conditional regression models \cr
#'   \verb{rowId}  	\tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{covariateId}    \tab(integer) \tab A numeric identifier of a covariate  \cr
#'   \verb{covariateValue}    \tab(real) \tab The value of the specified covariate \cr
#' }
#'
#' For \code{"lr"} and \code{"pr"} models the covariates can instead be a sparse matrix from the \pkg{Matrix}
#' package (a \code{dgCMatrix}, \code{lgCMatrix} or \code{ngCMatrix}) with one row per row of \code{outcomes}:
#' row \code{r} holds the covariates of \code{outcomes$rowId[r]}, and column \code{j} is the covariate with
#' \code{covariateId} \code{j}, named by \code{colnames()} if the matrix has them. The matrix is read in place,
#' without a data frame of its entries; columns without non-zero entries are left out, as covariates without rows
#' in a data frame are. The data equal those of the data frame with \code{rowId = outcomes$rowId[x@i + 1]},
#' \code{covariateId} the column number and \code{covariateValue = x@x}.
#'
#' These columns are expected in the timeEffectMap object:
#' \tabular{lll}{
#'   \verb{covariateId}    \tab(integer) \tab A numeric identifier of the covariates that have time-varying effects on the outcome \cr
#' }
#'
#' For \code{modelType = "plr"} (pooled logistic regression), a row of the data is a subject (\code{rowId}, the
#' \code{rowId} of its covariates), and its rows in the likelihood are its intervals at risk, given either as one
#' outcome row per subject and interval (explicit rows: columns \code{rowId}, \code{interval}, \code{y} and optional
#' \code{weights} and \code{offset} per row) or as one outcome row per subject (implicit rows: columns \code{rowId},
#' \code{entry}, \code{exit}, \code{y}, 1 if the event is in interval \code{exit}, and optional \code{weights} and
#' \code{offset} per subject); see \code{\link{convertToLongOutcome}}. Intervals are positive integers. The
#' covariates are not expanded to the intervals: row \code{r} of subject \code{i} has the linear predictor
#' \code{offset + sum_j x_ij beta_j + sum_b T[t_r, b] (alpha_b + sum_c x_ic gamma_cb)}, with \code{T} the
#' \code{timeBasis}, \code{alpha_b} its coefficients and \code{gamma_cb} those of the interactions named in
#' \code{timeEffectMap}, a data frame with columns \code{covariateId}, \code{timeCovariateId} (a \code{covariateId}
#' of \code{timeBasis}) and \code{interactionId} (the id of the interaction's coefficient). The fit equals that of
#' \code{"lr"} on the expanded rows (with \code{link = "cloglog"}, that of \code{glm(binomial("cloglog"))}).
#' Cross-validation folds and the case weights of \code{fitCyclopsModel()} are by subject, as are the score
#' contributions of \code{vcov(type = "robust")}; an optional outcome column \code{subjectId} (constant within a
#' \code{rowId}) groups subjects further, e.g. the clones of one person: folds then never split a \code{subjectId}
#' and the robust variance is clustered by it. Predictions are not yet implemented.
#'
#' @return
#' An object of type cyclopsData
#'
#' @examples
#' #Convert infert dataset to Cyclops format:
#' covariates <- data.frame(stratumId = rep(infert$stratum, 2),
#'                          rowId = rep(1:nrow(infert), 2),
#'                          covariateId = rep(1:2, each = nrow(infert)),
#'                          covariateValue = c(infert$spontaneous, infert$induced))
#' outcomes <- data.frame(stratumId = infert$stratum,
#'                        rowId = 1:nrow(infert),
#'                        y = infert$case)
#' #Make sparse:
#' covariates <- covariates[covariates$covariateValue != 0, ]
#'
#' #Create Cyclops data object:
#' cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "clr",
#'                                     addIntercept = FALSE)
#'
#' #Fit model:
#' fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"))
#'
#' @export
convertToCyclopsData <- function(outcomes,
                                 covariates,
                                 modelType = "lr",
				 timeEffectMap = NULL,
                                 addIntercept = TRUE,
                                 checkSorting = NULL,
                                 checkRowIds = TRUE,
                                 normalize = NULL,
                                 quiet = FALSE,
                                 floatingPoint = 64,
                                 timeBasis = NULL,
                                 link = "logit") {
    UseMethod("convertToCyclopsData")
}

#' @describeIn convertToCyclopsData Convert data from two \code{data.frame}
#' @export
# <<<<<<< HEAD
# convertToCyclopsData.ffdf <- function(outcomes,
#                                       covariates,
#                                       modelType = "lr",
#                                       addIntercept = TRUE,
#                                       checkSorting = TRUE,
#                                       checkRowIds = TRUE,
#                                       normalize = NULL,
#                                       quiet = FALSE,
#                                       floatingPoint = 64){
#     if ((modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron") & addIntercept){
#         if(!quiet) {
#             warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
#         }
# =======
convertToCyclopsData.data.frame <- function(outcomes,
                                            covariates,
                                            modelType = "lr",
					    timeEffectMap = NULL,
                                            addIntercept = TRUE,
                                            checkSorting = NULL,
                                            checkRowIds = TRUE,
                                            normalize = NULL,
                                            quiet = FALSE,
                                            floatingPoint = 64,
                                            timeBasis = NULL,
                                            link = "logit") {
    if (!is.null(checkSorting))
        warning("The 'checkSorting' argument has been deprecated. Sorting is now always checked")

    # covariates as a sparse matrix (lr and pr): row r belongs to outcomes row r, column j is covariate j; the matrix is
    # read in place, without a data frame of its entries
    sparseCovariates <- .isSparseCovariateMatrix(covariates, outcomes, modelType)

    if (modelType == "plr") {
        return(.convertToPooledCyclopsData(outcomes, covariates, timeBasis, timeEffectMap, addIntercept,
                                           checkRowIds, normalize, quiet, floatingPoint, link))
    }
    .checkPooledArguments(modelType, timeBasis, link)

    outcomes <- .renameEndTime(outcomes)

    if ((modelType == "clr" | modelType == "cpr") & addIntercept) {
        if (!quiet)
            warning("Intercepts are not allowed in conditional models, removing intercept", call. = FALSE)
# >>>>>>> master
        addIntercept = FALSE
    }
    if (modelType == "pr" | modelType == "cpr")
        if (any(outcomes$time <= 0))
            stop("time cannot be non-positive", call. = FALSE)

    if (modelType == "cox" && "startTime" %in% colnames(outcomes) && any(outcomes$startTime != 0))
        stop("Cox models require start times of 0; use modelType = \"cox_counting\" for start times", call. = FALSE)

    if (modelType == "lr" | modelType == "pr") {
        outcomes$stratumId <- NULL
        if (!sparseCovariates) covariates$stratumId <- NULL
    }
    if ((modelType == "cox" | modelType == "cox_time" | modelType == "fgr" | modelType == "cox_counting") & !"stratumId" %in% colnames(outcomes)) {
        outcomes$stratumId <- 0
        covariates$stratumId <- 0
    }

    covariateOrder <- NULL # lr and pr: the permutation that sorts the covariates, if they were not sorted
    if (modelType == "lr" | modelType == "pr") {
        if (!isSorted(outcomes, c("rowId"))) {
            if (!quiet)
                writeLines("Sorting outcomes by rowId")
            rowOrder <- order(outcomes$rowId)
            outcomes <- outcomes[rowOrder,]
            if (sparseCovariates) covariates <- covariates[rowOrder, , drop = FALSE] # rows follow their outcomes
        }
        if (!sparseCovariates && !isSorted(covariates, c("covariateId", "rowId"))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId and rowId")
            # the permutation only, read through when loading (a sorted copy of the covariates and order()'s work
            # took about 40 bytes per covariate value); order() and a sorted copy where .sortOrder() does not decide
            covariateOrder <- .sortOrder(covariates$covariateId, covariates$rowId)
            if (is.null(covariateOrder))
                covariates <- covariates[order(covariates$covariateId, covariates$rowId),]
        }
    }

    if (modelType == "clr" | modelType == "cpr") {
        if (!isSorted(outcomes, c("stratumId","rowId"))) {
            if (!quiet)
                writeLines("Sorting outcomes by stratumId and rowId")
            outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
        }
        if (!isSorted(covariates, c("covariateId", "stratumId","rowId"))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId, stratumId, and rowId")
            covariates <- covariates[order(covariates$covariateId, covariates$stratumId, covariates$rowId),]
        }
    }

    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr" | modelType == "cox_counting") {

        if ((modelType == "cox" | modelType == "cox_time" | modelType == "cox_counting") & length(unique(outcomes$y)) > 2) {
            stop("Cox model only accepts one outcome type")
        }
# <<<<<<< HEAD
#         if (modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron"){
#             if (!isSorted(outcomes,c("stratumId","rowId"))){
#                 if(!quiet) {
#                     writeLines("Sorting outcomes by stratumId and rowId")
#                 }
#                 rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
#                 outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","rowId")]),]
#             }
#             if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
#                 if(!quiet) {
#                     writeLines("Sorting covariates by covariateId, stratumId and rowId")
#                 }
#                 rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
#                 covariates <- covariates[ff::ffdforder(covariates[c("covariateId", "stratumId","rowId")]),]
#             }
# =======
        if ("subjectId" %in% colnames(outcomes)) {
            if (!isSorted(outcomes,
                          c("stratumId", "time", "y", "subjectId", "rowId"),
                          c(TRUE, FALSE, TRUE, TRUE, TRUE))) {
                if (!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y, subjectId and rowId")
                outcomes <- outcomes[order(outcomes$stratumId, -outcomes$time, outcomes$y, outcomes$subjectId, outcomes$rowId),]
            }
        } else {
            if (!isSorted(outcomes,
                          c("stratumId", "time", "y", "rowId"),
                          c(TRUE, FALSE, TRUE, TRUE))) {
                if (!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y and rowId")
                    outcomes <- outcomes[order(outcomes$stratumId, -outcomes$time, outcomes$y, outcomes$rowId),]
            }
        }
        if (!"time" %in% colnames(covariates)) {
            covariates$time <- NULL
            covariates$y <- NULL
            covariates$stratumId <- NULL
            covariates <- merge(covariates, outcomes, by = c("rowId"))
        }
        if (!isSorted(covariates,
                      c("covariateId", "stratumId", "time", "y", "rowId"),
                      c(TRUE, TRUE, FALSE, TRUE, TRUE))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId, stratumId, time (descending), y, and rowId")
            covariates <- covariates[order(covariates$covariateId, covariates$stratumId, -covariates$time, covariates$y, covariates$rowId),]
        }
    }

    # (.allIn() decides most cases without match()'s vectors as long as the covariates; NA: not decided there)
    if (!sparseCovariates && checkRowIds && !isTRUE(.allIn(covariates$rowId, outcomes$rowId))) {
        if (!is.null(covariateOrder)) { # sorted, as before
            covariates <- covariates[covariateOrder, ]
            covariateOrder <- NULL
        }
        mapping <- match(covariates$rowId,outcomes$rowId)
        if (any(is.na(mapping))) {
            if (!quiet)
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            covariateRowsWithMapping <- which(!is.na(mapping))
            covariates <- covariates[covariateRowsWithMapping,]
        }
    }

    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    .checkOffsetColumn(outcomes, modelType)
    hasOffset <- modelType == "lr" && "offset" %in% colnames(outcomes) # loaded as the time, then made the offset
    loadNewSqlCyclopsDataY(object = dataPtr,
                           stratumId = if ("stratumId" %in% colnames(outcomes)) outcomes$stratumId else NULL,
                           rowId = outcomes$rowId,
                           y = outcomes$y,
                           time = if (hasOffset) outcomes$offset else if ("time" %in% colnames(outcomes)) outcomes$time else NULL)

    if (modelType == "cox_counting" && "startTime" %in% colnames(outcomes)) {
        .loadCyclopsDataStartTime(dataPtr, outcomes$startTime)
    }

    if (modelType == "cox" || modelType == "cox_counting") { # the stratum of each internal index, for predict()
        dataPtr$stratumIds <- unique(outcomes$stratumId)
    }

    if ("subjectId" %in% colnames(outcomes)) {
        .loadCyclopsDataSubjectId(dataPtr, bit64::as.integer64(outcomes$subjectId))
    }

    if (addIntercept & (modelType != "cox" & modelType != "cox_time" & modelType != "fgr" & modelType != "cox_counting")) {
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    }

    if (sparseCovariates) {
        covarNames <- .loadSparseCovariateMatrix(dataPtr, covariates, outcomes$rowId)
    } else {
    # the covariates are sorted by covariateId first, so its distinct values are the starts of its runs (no hash table
    # as long as the covariates); unique() if they are not
    starts <- .runStarts(covariates$covariateId, covariateOrder)
    covarNames <- if (is.null(starts)) {
        unique(if (is.null(covariateOrder)) covariates$covariateId else covariates$covariateId[covariateOrder])
    } else if (is.null(covariateOrder)) {
        covariates$covariateId[starts]
    } else {
        covariates$covariateId[covariateOrder[starts]]
    }
    rm(starts)
    loadNewSqlCyclopsDataMultipleX(object = dataPtr,
                                   covariateId = covariates$covariateId,
                                   rowId = covariates$rowId,
                                   covariateValue = covariates$covariateValue,
                                   name = covarNames,
                                   order = covariateOrder)
    }

    if (modelType == "cox_time" && !is.null(timeEffectMap)) {
        if (!all(timeEffectMap$covariateId %in% covarNames)) stop("Invalid covariateId for time effects.")
        loadNewSqlCyclopsDataStratTimeEffects(object = dataPtr,
					      stratumId = outcomes$stratumId,
					      rowId = outcomes$rowId,
					      subjectId = outcomes$subjectId,
					      timeEffectCovariateId = sort(timeEffectMap$covariateId))
    }

    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (hasOffset) # a covariate with coefficient fixed at 1
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1, offsetAlreadyOnLogScale = TRUE)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if ("weights" %in% colnames(outcomes)) {
        dataPtr$weights <- outcomes$weights
    } else {
        dataPtr$censorWeights <- NULL
    }

    if ("censorWeights" %in% colnames(outcomes)) {
        dataPtr$censorWeights <- outcomes$censorWeights
    } else {
        if (modelType == "fgr") {
            dataPtr$censorWeights <- getFineGrayWeights(outcomes$time, outcomes$y)$weights
            writeLines("Generating censoring weights")
        } else {
            dataPtr$censorWeights <- NULL
        }
    }

    return(dataPtr)
}

#' @describeIn convertToCyclopsData Convert data from two \code{Andromeda} tables
#' @export
# <<<<<<< HEAD
# convertToCyclopsData.data.frame <- function(outcomes,
#                                             covariates,
#                                             modelType = "lr",
#                                             addIntercept = TRUE,
#                                             checkSorting = TRUE,
#                                             checkRowIds = TRUE,
#                                             normalize = NULL,
#                                             quiet = FALSE,
#                                             floatingPoint = 64){
#     if ((modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron") & addIntercept){
#         if(!quiet)
#             warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
# =======
convertToCyclopsData.tbl_dbi <- function(outcomes,
                                         covariates,
                                         modelType = "lr",
					 timeEffectMap = NULL,
                                         addIntercept = TRUE,
                                         checkSorting = NULL,
                                         checkRowIds = TRUE,
                                         normalize = NULL,
                                         quiet = FALSE,
                                         floatingPoint = 64,
                                         timeBasis = NULL,
                                         link = "logit") {
    if (!is.null(checkSorting))
        warning("The 'checkSorting' argument has been deprecated. Sorting is now always checked")

    if (modelType == "plr") { # one row per subject: collected
        return(.convertToPooledCyclopsData(collect(outcomes), collect(covariates),
                                           if (is.null(timeBasis)) NULL else collect(timeBasis),
                                           if (is.null(timeEffectMap)) NULL else collect(timeEffectMap),
                                           addIntercept, checkRowIds, normalize, quiet, floatingPoint, link))
    }
    .checkPooledArguments(modelType, timeBasis, link)

    outcomes <- .renameEndTime(outcomes)

    if ((modelType == "clr" | modelType == "cpr") & addIntercept) {
        if (!quiet) {
            warning("Intercepts are not allowed in conditional models, removing intercept", call. = FALSE)
        }
# >>>>>>> master
        addIntercept = FALSE
    }

    if (modelType == "pr" | modelType == "cpr") {
        if (any(pull(outcomes, time) <= 0)) {
            stop("time cannot be non-positive", call. = FALSE)
        }
    }

    providedNoStrata <- !"stratumId" %in% colnames(outcomes)

# <<<<<<< HEAD
#         if (modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron"){
#             if (!isSorted(outcomes,c("stratumId","rowId"))){
#                 if(!quiet)
#                     writeLines("Sorting outcomes by stratumId and rowId")
#                 outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
#             }
#             if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
#                 if(!quiet)
#                     writeLines("Sorting covariates by covariateId, stratumId, and rowId")
#                 covariates <- covariates[order(covariates$covariateId, covariates$stratumId,covariates$rowId),]
#             }
# =======
    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr" | modelType == "cox_counting") {
        if (providedNoStrata) {
            outcomes <- outcomes %>%
                mutate(stratumId = 0)
            covariates <- covariates %>%
                mutate(stratumId = 0)
# >>>>>>> master
        }
    }

    if (checkRowIds) {
        covariateRowIds <- covariates %>%
            distinct(.data$rowId) %>%
            pull()
        outcomeRowIds <- select(outcomes, "rowId") %>%
            pull()
        mapping <- match(covariateRowIds, outcomeRowIds)
        if (any(is.na(mapping))) {
            if (!quiet) {
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            }
            covariates <- covariates %>%
                filter(.data$rowId %in% outcomeRowIds)
        }
    }

    # Sorting should be last, as other operations may change ordering.
    # Also, should always explicitly define sorting, else not guaranteed.
    if (modelType == "lr" | modelType == "pr") {
        outcomes <- outcomes %>%
            arrange(.data$rowId)

        covariates <- covariates %>%
            arrange(.data$covariateId, .data$rowId)
    }

    if (modelType == "clr" | modelType == "cpr") {
        outcomes <- outcomes %>%
            arrange(.data$stratumId, .data$rowId)

        covariates <- covariates %>%
            arrange(.data$covariateId, .data$stratumId, .data$rowId)
    }

    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr" | modelType == "cox_counting") {

        if ((modelType == "cox" | modelType == "cox_time" | modelType == "cox_counting") &
            (select(outcomes, "y") %>% distinct() %>% count() %>% collect() > 2)) {
            stop("Cox model only accepts one outcome type")
        }
        if (!"time" %in% colnames(covariates)) {
            covariates <- covariates %>%
                inner_join(select(outcomes, "rowId", "time", "y"), by = "rowId")
        }
        if ("subjectId" %in% colnames(outcomes)) {
            outcomes <- outcomes %>%
                arrange(.data$stratumId, desc(.data$time), .data$y, .data$subjectId, .data$rowId)
        } else {
            outcomes <- outcomes %>%
                arrange(.data$stratumId, desc(.data$time), .data$y, .data$rowId)
	}
        covariates <- covariates %>%
            arrange(.data$covariateId, .data$stratumId, desc(.data$time), .data$y, .data$rowId)
    }

    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    outcomes <- collect(outcomes)
    if (modelType == "lr" | modelType == "pr") {
        outcomes$stratumId <- NULL
    }

    if (modelType == "cox" && "startTime" %in% colnames(outcomes) && any(outcomes$startTime != 0)) {
        stop("Cox models require start times of 0; use modelType = \"cox_counting\" for start times", call. = FALSE)
    }

    .checkOffsetColumn(outcomes, modelType)
    hasOffset <- modelType == "lr" && "offset" %in% colnames(outcomes) # loaded as the time, then made the offset
    loadNewSqlCyclopsDataY(object = dataPtr,
                           stratumId = if ("stratumId" %in% colnames(outcomes)) outcomes$stratumId else NULL,
                           rowId = outcomes$rowId,
                           y = outcomes$y,
                           time = if (hasOffset) outcomes$offset else if ("time" %in% colnames(outcomes)) (select(outcomes, time) %>% pull()) else NULL)

    if (modelType == "cox_counting" && "startTime" %in% colnames(outcomes)) {
        .loadCyclopsDataStartTime(dataPtr, outcomes$startTime)
    }

    if (modelType == "cox" || modelType == "cox_counting") { # the stratum of each internal index, for predict()
        dataPtr$stratumIds <- unique(outcomes$stratumId)
    }

    if ("subjectId" %in% colnames(outcomes)) {
        .loadCyclopsDataSubjectId(dataPtr, bit64::as.integer64(outcomes$subjectId))
    }

    if (addIntercept & (modelType != "cox" & modelType != "cox_time" & modelType != "fgr" & modelType != "cox_counting")) {
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    }

    loadCovariates <- function(batch) {
        covarNames <- unique(batch$covariateId)
        loadNewSqlCyclopsDataMultipleX(object = dataPtr,
                                        covariateId = batch$covariateId,
                                        rowId = batch$rowId,
                                        covariateValue = batch$covariateValue,
                                        name = covarNames,
                                        append = TRUE)
    }

    Andromeda::batchApply(covariates,
                          loadCovariates,
                          batchSize = 100000) # TODO Pick magic number

    if (modelType == "cox_time" && !is.null(timeEffectMap)) {
        covarNames <- unique(pull(covariates, .data$covariateId))
        if (!all(timeEffectMap$covariateId %in% covarNames)) stop("Invalid covariateId for time effects.")
        loadNewSqlCyclopsDataStratTimeEffects(object = dataPtr,
					      stratumId = outcomes$stratumId,
					      rowId = outcomes$rowId,
					      subjectId = outcomes$subjectId,
					      timeEffectCovariateId = sort(timeEffectMap$covariateId))
    }

    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (hasOffset) # a covariate with coefficient fixed at 1
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1, offsetAlreadyOnLogScale = TRUE)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if ("weights" %in% colnames(outcomes)) {
        dataPtr$weights <- outcomes %>% pull(.data$weights)
    } else {
        dataPtr$weights <- NULL
    }

    if ("censorWeights" %in% colnames(outcomes)) {
        dataPtr$censorWeights <- outcomes %>% pull(.data$censorWeights)
    } else {
        if (modelType == "fgr") {
            dataPtr$censorWeights <- getFineGrayWeights(
                outcomes %>% pull("time"),
                outcomes %>% pull("y")
            )$weights
            writeLines("Generating censoring weights")
        } else {
            dataPtr$censorWeights <- NULL
        }
    }

    return(dataPtr)
}

# Whether covariates is a sparse matrix of covariates (checked: lr or pr, one row per outcome); FALSE for a data frame.
# A dgCMatrix, lgCMatrix or ngCMatrix is read in place; other Matrix classes are refused rather than copied
.isSparseCovariateMatrix <- function(covariates, outcomes, modelType) {
    if (!methods::is(covariates, "Matrix")) {
        return(FALSE)
    }
    if (!class(covariates)[1] %in% c("dgCMatrix", "lgCMatrix", "ngCMatrix")) {
        stop("Sparse matrix covariates must be a dgCMatrix, lgCMatrix or ngCMatrix (column-compressed and general), ",
             "e.g. methods::as(methods::as(x, \"generalMatrix\"), \"CsparseMatrix\")", call. = FALSE)
    }
    if (!modelType %in% c("lr", "pr")) {
        stop("Sparse matrix covariates are supported for modelType \"lr\" and \"pr\"; ",
             "give the covariates as a data frame for other models", call. = FALSE)
    }
    if (nrow(covariates) != nrow(outcomes)) {
        stop("A sparse covariate matrix needs one row per outcome row: ", nrow(covariates), " rows for ",
             nrow(outcomes), " outcomes", call. = FALSE)
    }
    TRUE
}

# Loads the columns of a sparse covariate matrix whose row r belongs to rowIds[r]: column j has covariate id j, and is
# named by colnames() if the matrix has them (else by its id). Columns without non-zero entries are left out, as
# covariates without rows in a data frame. Returns the names of the columns loaded
.loadSparseCovariateMatrix <- function(dataPtr, covariates, rowIds) {
    p <- covariates@p
    kept <- which(diff(p) > 0)
    if (length(kept) == 0) {
        return(character(0))
    }
    index <- .loadCyclopsDataColumns(dataPtr, p, covariates@i, if (methods::.hasSlot(covariates, "x")) covariates@x else NULL,
                                     .idsForLoading(rowIds), seq_len(ncol(covariates)), FALSE)
    names <- if (is.null(colnames(covariates))) as.character(kept) else colnames(covariates)[kept]
    if (is.null(dataPtr$coefficientNames)) {
        dataPtr$coefficientNames <- as.character(c())
    }
    dataPtr$coefficientNames[index + seq_along(names)] <- names
    names
}

# timeBasis and link are for plr models only
.checkPooledArguments <- function(modelType, timeBasis, link) {
    if (!is.null(timeBasis) || !identical(link, "logit")) {
        warning("timeBasis and link are for plr models only; modelType = \"", modelType, "\" ignores them",
                call. = FALSE)
    }
}

# The offset column of the outcomes: lr models only, and finite
.checkOffsetColumn <- function(outcomes, modelType) {
    if (!"offset" %in% colnames(outcomes)) {
        return(invisible())
    }
    if (modelType != "lr") {
        warning("The offset column of the outcomes is for lr models only; modelType = \"", modelType,
                "\" ignores it", call. = FALSE)
    } else if (any(!is.finite(outcomes$offset))) {
        stop("The offset column of the outcomes must be finite (", sum(!is.finite(outcomes$offset)),
             " rows are NA, NaN or infinite)", call. = FALSE)
    }
}
