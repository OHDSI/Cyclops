# @file NewDataConversion.R
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

#' Check if data are sorted by one or more columns
#'
#' @description
#' \code{isSorted} checks wether data are sorted by one or more specified columns.
#'
#' @param data            Either a data.frame of ffdf object.
#' @param columnNames     Vector of one or more column names.
#' @param ascending       Logical vector indicating the data should be sorted ascending or descending
#' according the specified columns.
#'
#' @details
#' This function currently only supports checking for sorting on numeric values.
#'
#' @return
#' True or false
#'
#' @examples
#' x <- data.frame(a = runif(1000), b = runif(1000))
#' x <- round(x, digits=2)
#' isSorted(x, c("a", "b"))
#'
#' x <- x[order(x$a, x$b),]
#' isSorted(x, c("a", "b"))
#'
#' x <- x[order(x$a,-x$b),]
#' isSorted(x, c("a", "b"), c(TRUE, FALSE))
#'
#' @export
isSorted <- function(data,columnNames,ascending=rep(TRUE,length(columnNames))){
    UseMethod("isSorted")
}

#' @describeIn isSorted Check if a \code{data.frame} is sorted by one or more columns
#' @export
isSorted.data.frame <- function(data,columnNames,ascending=rep(TRUE,length(columnNames))){
    return(.isSorted(data,columnNames,ascending))
}

.quickFfdfSubset <- function(data, index, columnNames) {
    # This function does the same thing as default ffdf subsetting, but outputs a list of vectors instead of a
    # data.frame, so a rownames vector does not have to be created. This saves a LOT of time.
    dataSubset <- vector("list", length(columnNames))
    for (i in 1:length(columnNames)){
        dataSubset[[i]] <- data[index,columnNames[i]]
    }
    names(dataSubset) <- columnNames
    return(dataSubset)
}

#' @describeIn isSorted Check if a \code{ffdf} is sorted by one or more columns
#' @export
isSorted.ffdf <- function(data,columnNames,ascending=rep(TRUE,length(columnNames))){
    if (nrow(data)>100000){ #If data is big, first check on a small subset. If that aready fails, we're done
        if (!.isSortedVectorList(.quickFfdfSubset(data, bit::ri(1,1000),columnNames),ascending)) {
            return(FALSE)
        }
    }
    for (i in ff::chunk.ffdf(data)) {
        if (!.isSortedVectorList(.quickFfdfSubset(data, i,columnNames),ascending)) {
            return(FALSE)
        }
    }
    return(TRUE)
}

#' Convert data from two data frames or ffdf objects into a CyclopsData object
#'
#' @description
#' \code{convertToCyclopsData} loads data from two data frames or ffdf objects, and inserts it into a Cyclops data object.
#'
#' @param outcomes      A data frame or ffdf object containing the outcomes with predefined columns (see below).
#' @param covariates    A data frame or ffdf object containing the covariates with predefined columns (see below).
#' @param modelType     Cyclops model type. Current supported types are "pr", "cpr", lr", "clr", or "cox"
#' @param addIntercept  Add an intercept to the model?
#' @param checkSorting  Check if the data are sorted appropriately, and if not, sort.
#' @param checkRowIds   Check if all rowIds in the covariates appear in the outcomes.
#' @param normalize     String: Name of normalization for all non-indicator covariates (possible values: stdev, max, median)
#' @param quiet         If true, (warning) messages are surpressed.
#' @param floatingPoint Specified floating-point representation size (32 or 64)
#'
#' @details
#' These columns are expected in the outcome object:
#' \tabular{lll}{
#'   \verb{stratumId}    \tab(integer) \tab (optional) Stratum ID for conditional regression models \cr
#'   \verb{rowId}  	\tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{y}    \tab(real) \tab The outcome variable \cr
#'   \verb{time}    \tab(real) \tab For models that use time (e.g. Poisson or Cox regression) this contains time \cr
#'                  \tab        \tab(e.g. number of days) \cr
#'   \verb{weight} \tab(real) \tab Non-negative weight to apply to outcome
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
#' Note: If checkSorting is turned off, the outcome table should be sorted by stratumId (if present)
#' and then rowId except for Cox regression when the table should be sorted by
#' stratumId (if present), -time, y, and rowId. The covariate table should be sorted by covariateId, stratumId
#' (if present), rowId except for Cox regression when the table should be sorted by covariateId,
#' stratumId (if present), -time, y, and rowId.
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
                                 addIntercept = TRUE,
                                 checkSorting = TRUE,
                                 checkRowIds = TRUE,
                                 normalize = NULL,
                                 quiet = FALSE,
                                 floatingPoint = 64) {
    UseMethod("convertToCyclopsData")
}

#' @describeIn convertToCyclopsData Convert data from two \code{ffdf}
#' @export
convertToCyclopsData.ffdf <- function(outcomes,
                                      covariates,
                                      modelType = "lr",
                                      addIntercept = TRUE,
                                      checkSorting = TRUE,
                                      checkRowIds = TRUE,
                                      normalize = NULL,
                                      quiet = FALSE,
                                      floatingPoint = 64){
    if ((modelType == "clr" | modelType == "cpr") & addIntercept){
        if(!quiet) {
            warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
        }
        addIntercept = FALSE
    }
    if (modelType == "pr" | modelType == "cpr") {
        if (any(outcomes$time <= 0)) {
            stop("time cannot be non-positive",call.=FALSE)
        }
    }

    if (modelType == "cox"){
        if (is.null(outcomes$stratumId)){
            outcomes$stratumId <- ff::ff(1, vmode="double", length=nrow(outcomes))
            covariates$stratumId <- ff::ff(1, vmode="double", length=nrow(covariates))
        }
    }

    if (checkSorting){
        if (modelType == "lr" | modelType == "pr"){
            if (!isSorted(outcomes,c("rowId"))){
                if(!quiet) {
                    writeLines("Sorting outcomes by rowId")
                }
                rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
                outcomes <- outcomes[ff::ffdforder(outcomes[c("rowId")]),]
            }
            if (!isSorted(covariates,c("covariateId","rowId"))){
                if(!quiet) {
                    writeLines("Sorting covariates by covariateId, rowId")
                }
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates <- covariates[ff::ffdforder(covariates[c("covariateId","rowId")]),]
            }
        }
        if (modelType == "clr" | modelType == "cpr"){
            if (!isSorted(outcomes,c("stratumId","rowId"))){
                if(!quiet) {
                    writeLines("Sorting outcomes by stratumId and rowId")
                }
                rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
                outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","rowId")]),]
            }
            if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
                if(!quiet) {
                    writeLines("Sorting covariates by covariateId, stratumId and rowId")
                }
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates <- covariates[ff::ffdforder(covariates[c("covariateId", "stratumId","rowId")]),]
            }
        }
        if (modelType == "cox"){
            outcomes$minTime <- ff::ff(vmode="double", length=length(outcomes$time))
            for (i in bit::chunk(outcomes$time)){
                outcomes$minTime[i] <- 0-outcomes$time[i]
            }
            if (!isSorted(outcomes,c("stratumId", "time", "y", "rowId"),c(TRUE, FALSE, TRUE, TRUE))){
                if(!quiet) {
                    writeLines("Sorting outcomes by stratumId, time (descending), y, and rowId")
                }
                rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
                outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","minTime", "y", "rowId")]),]
            }
            covariates$minTime <- NULL
            covariates$time <- NULL
            covariates$y <- NULL
            # covariates <- ffbase::merge.ffdf(covariates, outcomes, by = c("stratumId", "rowId"))
            idx <- ffbase::ffmatch(covariates$rowId, outcomes$rowId)
            covariates$minTime <- outcomes$minTime[idx]
            covariates$time <- outcomes$time[idx]
            covariates$y <- outcomes$y[idx]
            if (!isSorted(covariates, c("covariateId", "stratumId", "time", "y", "rowId"), c(TRUE, TRUE, FALSE, TRUE, TRUE))){
                if(!quiet) {
                    writeLines("Sorting covariates by covariateId, stratumId, time (descending), y, and rowId")
                }
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates <- covariates[ff::ffdforder(covariates[c("covariateId", "stratumId", "minTime", "y", "rowId")]),]
            }
        }
    }
    if (checkRowIds){
        mapped <- ffbase::ffmatch(x = covariates$rowId, table = outcomes$rowId)
        if (ffbase::any.ff(ffbase::is.na.ff(mapped))){
            if(!quiet) {
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            }
            rownames(covariates) <- NULL
            covariates <- covariates[ffbase::ffwhich(mapped, is.na(mapped) == FALSE),]
        }
    }

    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    loadNewSqlCyclopsDataY(dataPtr,
                           if (is.null(outcomes$stratumId) | modelType == "lr" | modelType == "pr") {NULL} else {ff::as.ram.ff(outcomes$stratumId)},
                           ff::as.ram.ff(outcomes$rowId),
                           ff::as.ram.ff(outcomes$y),
                           if (is.null(outcomes$time)) {NULL} else {ff::as.ram.ff(outcomes$time)})

    if (addIntercept & modelType != "cox")
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    for (i in bit::chunk(covariates)){
        covarNames <- unique(covariates$covariateId[i,])
        loadNewSeqlCyclopsDataMultipleX(dataPtr,
                                        covariates$covariateId[i,],
                                        covariates$rowId[i,],
                                        covariates$covariateValue[i,],
                                        name = covarNames, # TODO Does this really work?
                                        append = TRUE)
    }
    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if (is.null(outcomes$weight)) {
        dataPtr$weights <- NULL
    } else {
        dataPtr$weights <- ff::as.ram.ff(outcomes$weight)
    }

    return(dataPtr)

}

#' @describeIn convertToCyclopsData Convert data from two \code{data.frame}
#' @export
convertToCyclopsData.data.frame <- function(outcomes,
                                            covariates,
                                            modelType = "lr",
                                            addIntercept = TRUE,
                                            checkSorting = TRUE,
                                            checkRowIds = TRUE,
                                            normalize = NULL,
                                            quiet = FALSE,
                                            floatingPoint = 64){
    if ((modelType == "clr" | modelType == "cpr") & addIntercept){
        if(!quiet)
            warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
        addIntercept = FALSE
    }
    if (modelType == "pr" | modelType == "cpr")
        if (any(outcomes$time <= 0))
            stop("time cannot be non-positive",call.=FALSE)

    if (modelType == "lr" | modelType == "pr"){
        outcomes$stratumId <- NULL
        covariates$stratumId <- NULL
    }
    if (modelType == "cox" & is.null(outcomes$stratumId)){
        outcomes$stratumId <- 0
        covariates$stratumId <- 0 # MAS: Added
    }

    if (checkSorting){
        if (modelType == "lr" | modelType == "pr"){
            if (!isSorted(outcomes,c("rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by rowId")
                outcomes <- outcomes[order(outcomes$rowId),]
            }
            if (!isSorted(covariates,c("covariateId", "rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by covariateId and rowId")
                covariates <- covariates[order(covariates$covariateId, covariates$rowId),]
            }
        }

        if (modelType == "clr" | modelType == "cpr"){
            if (!isSorted(outcomes,c("stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId and rowId")
                outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
            }
            if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by covariateId, stratumId, and rowId")
                covariates <- covariates[order(covariates$covariateId, covariates$stratumId,covariates$rowId),]
            }
        }
        if (modelType == "cox"){
            if (!isSorted(outcomes,c("stratumId", "time", "y", "rowId"),c(TRUE, FALSE, TRUE, TRUE))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y and rowId")
                outcomes <- outcomes[order(outcomes$stratumId, -outcomes$time, outcomes$y, outcomes$rowId),]
            }
            if (is.null(covariates$time)) {
                covariates$time <- NULL
                covariates$y <- NULL
                covariates$stratumId <- NULL
                covariates <- merge(covariates, outcomes, by = c("rowId"))
            }
            if (!isSorted(covariates, c("covariateId", "stratumId", "time", "y", "rowId"), c(TRUE, TRUE, FALSE, TRUE, TRUE))){
                if(!quiet)
                    writeLines("Sorting covariates by covariateId, stratumId, time (descending), y, and rowId")
                covariates <- covariates[order(covariates$covariateId, covariates$stratumId, -covariates$time, covariates$y, covariates$rowId),]
            }
        }
    }
    if (checkRowIds){
        mapping <- match(covariates$rowId,outcomes$rowId)
        if (any(is.na(mapping))){
            if(!quiet)
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            covariateRowsWithMapping <- which(!is.na(mapping))
            covariates <- covariates[covariateRowsWithMapping,]
        }
    }
    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    loadNewSqlCyclopsDataY(dataPtr, outcomes$stratumId, outcomes$rowId, outcomes$y, outcomes$time)

    if (addIntercept & modelType != "cox")
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")

    covarNames <- unique(covariates$covariateId)
    loadNewSeqlCyclopsDataMultipleX(dataPtr, covariates$covariateId, covariates$rowId, covariates$covariateValue, name = covarNames)
    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if (is.null(outcomes$weight)) {
        dataPtr$weights <- NULL
    } else {
        dataPtr$weights <- outcomes$weight
    }

    return(dataPtr)
}
