# @file DataConversion.R
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

.lastRowNotHavingThisValue <- function(column, value){
    if (column[1] == value)
        return(0)
    for (i in length(column):1){
        if (column[i] != value)
            return(i)
    }
    return(0)
}

.constructCyclopsDataFromBatchableSources <- function(resultSetOutcome,
                                                      resultSetCovariate,
                                                      getOutcomeBatch,
                                                      getCovariateBatch,
                                                      isDone,
                                                      modelType = "lr",
                                                      addIntercept = TRUE,
                                                      offsetAlreadyOnLogScale = FALSE,
                                                      makeCovariatesDense = NULL){
    if ((modelType == "clr" | modelType == "cpr") & addIntercept){
        warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
        addIntercept = FALSE
    }

    # Construct empty Cyclops data object:
    dataPtr <- createSqlCyclopsData(modelType = modelType)

    #Fetch data in batches:
    batchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)

    lastUsedOutcome <- 0
    spillOverCovars <- NULL
    while (!isDone(resultSetCovariate)){
        #Get covars:
        batchCovars <- getCovariateBatch(resultSetCovariate,modelType)
        lastRowId <- batchCovars$rowId[nrow(batchCovars)]
        endCompleteRow <- .lastRowNotHavingThisValue(batchCovars$rowId,lastRowId)

        if (endCompleteRow == 0){ #Entire batch is about 1 row
            if (!is.null(spillOverCovars)){
                if (spillOverCovars$rowId[1] == batchCovars$rowId[1]){ #SpilloverCovars contains info on same row
                    spillOverCovars <- rbind(spillOverCovars,batchCovars)
                    covarsToCyclops <- NULL
                } else { #SplilloverCovars contains covars for a different row
                    covarsToCyclops <- spillOverCovars
                    spillOverCovars <- batchCovars
                }
            } else {
                spillOverCovars <- batchCovars
            }
        } else { #Batch is about different rows (so at least one is complete)
            if (!is.null(spillOverCovars)){
                covarsToCyclops <- rbind(spillOverCovars,batchCovars[1:endCompleteRow,])
            } else {
                covarsToCyclops <- batchCovars[1:endCompleteRow,]
            }
            spillOverCovars <- batchCovars[(endCompleteRow+1):nrow(batchCovars),]
        }

        #Get matching outcomes:
        if (!is.null(covarsToCyclops)){ # There is a complete row
            completeRowId = covarsToCyclops$rowId[nrow(covarsToCyclops)]
            endCompleteRowInOutcome <- which(batchOutcome$rowId == completeRowId)
            while (length(endCompleteRowInOutcome) == 0 & !isDone(resultSetOutcome)){
                if (lastUsedOutcome == nrow(batchOutcome)){
                    batchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)
                } else {
                    newBatchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)
                    batchOutcome <- rbind(batchOutcome[(lastUsedOutcome+1):nrow(batchOutcome),],newBatchOutcome)
                }
                lastUsedOutcome = 0
                endCompleteRowInOutcome <- which(batchOutcome$rowId == completeRowId)
            }
            #Append to Cyclops:
            appendSqlCyclopsData(dataPtr,
                                 batchOutcome$stratumId[(lastUsedOutcome+1):endCompleteRowInOutcome],
                                 batchOutcome$rowId[(lastUsedOutcome+1):endCompleteRowInOutcome],
                                 batchOutcome$y[(lastUsedOutcome+1):endCompleteRowInOutcome],
                                 batchOutcome$time[(lastUsedOutcome+1):endCompleteRowInOutcome],
                                 covarsToCyclops$rowId,
                                 covarsToCyclops$covariateId,
                                 covarsToCyclops$covariateValue
            )

            lastUsedOutcome = endCompleteRowInOutcome
        }
    }
    #End of covar batches, add spillover to Cyclops:
    covarsToCyclops <- spillOverCovars

    completeRowId = covarsToCyclops$rowId[nrow(covarsToCyclops)]
    endCompleteRowInOutcome <- which(batchOutcome$rowId == completeRowId)
    while (length(endCompleteRowInOutcome) == 0 & !isDone(resultSetOutcome)){
        if (lastUsedOutcome == nrow(batchOutcome)){
            batchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)
        } else {
            batchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)
            batchOutcome <- rbind(batchOutcome[(lastUsedOutcome+1):nrow(batchOutcome),],newBatchOutcome)
        }
        lastUsedOutcome = 0
        endCompleteRowInOutcome <- which(batchOutcome$rowId == completeRowId)
    }

    #Append to Cyclops:
    appendSqlCyclopsData(dataPtr,
                         batchOutcome$stratumId[(lastUsedOutcome+1):endCompleteRowInOutcome],
                         batchOutcome$rowId[(lastUsedOutcome+1):endCompleteRowInOutcome],
                         batchOutcome$y[(lastUsedOutcome+1):endCompleteRowInOutcome],
                         batchOutcome$time[(lastUsedOutcome+1):endCompleteRowInOutcome],
                         covarsToCyclops$rowId,
                         covarsToCyclops$covariateId,
                         covarsToCyclops$covariateValue
    )

    lastUsedOutcome = endCompleteRowInOutcome

    #Add any outcomes that are left (without matching covar information):
    if (lastUsedOutcome < nrow(batchOutcome)){
        appendSqlCyclopsData(dataPtr,
                             batchOutcome$stratumId[(lastUsedOutcome+1):nrow(batchOutcome)],
                             batchOutcome$rowId[(lastUsedOutcome+1):nrow(batchOutcome)],
                             batchOutcome$y[(lastUsedOutcome+1):nrow(batchOutcome)],
                             batchOutcome$time[(lastUsedOutcome+1):nrow(batchOutcome)],
                             as.numeric(c()),
                             as.numeric(c()),
                             as.numeric(c()))
    }
    while (!isDone(resultSetOutcome)){
        batchOutcome <- getOutcomeBatch(resultSetOutcome,modelType)

        appendSqlCyclopsData(dataPtr,
                             batchOutcome$stratumId,
                             batchOutcome$rowId,
                             batchOutcome$y,
                             batchOutcome$time,
                             as.numeric(c()),
                             as.numeric(c()),
                             as.numeric(c()))
    }
    if (modelType == "pr" | modelType == "cpr")
        useOffsetCovariate = -1
    else
        useOffsetCovariate = NULL

    if (modelType != "cox"){
        finalizeSqlCyclopsData(dataPtr,
                               addIntercept = addIntercept,
                               useOffsetCovariate = useOffsetCovariate,
                               offsetAlreadyOnLogScale = offsetAlreadyOnLogScale,
                               makeCovariatesDense = makeCovariatesDense)
    }
    return(dataPtr)
}

#' @keywords internal
legacyConvertToCyclopsData <- function(outcomes,
                                 covariates,
                                 modelType = "lr",
                                 addIntercept = TRUE,
                                 offsetAlreadyOnLogScale = FALSE,
                                 makeCovariatesDense = NULL,
                                 checkSorting = TRUE,
                                 checkRowIds = TRUE,
                                 quiet = FALSE) {
    UseMethod("legacyConvertToCyclopsData")
}

#' @keywords internal
legacyConvertToCyclopsData.ffdf <- function(outcomes,
                                      covariates,
                                      modelType = "lr",
                                      addIntercept = TRUE,
                                      offsetAlreadyOnLogScale = FALSE,
                                      makeCovariatesDense = NULL,
                                      checkSorting = TRUE,
                                      checkRowIds = TRUE,
                                      quiet = FALSE){
    #    require(ffbase) #Should be superfluous, since the user already has an ffdf object
    if (checkSorting){
        if (modelType == "lr" | modelType == "pr"){
            if (!isSorted(outcomes,c("rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by rowId")
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                outcomes <- outcomes[ff::ffdforder(outcomes[c("rowId")]),]
            }
            if (!isSorted(covariates,c("rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by rowId")
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates <- covariates[ff::ffdforder(covariates[c("rowId")]),]
            }
        }
        if (modelType == "clr" | modelType == "cpr"){
            if (!isSorted(outcomes,c("stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId and rowId")
                rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
                outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","rowId")]),]
            }
            if (!isSorted(covariates,c("stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by stratumId and rowId")
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates <- covariates[ff::ffdforder(covariates[c("stratumId","rowId")]),]
            }
        }
        if (modelType == "cox"){
            if (is.null(outcomes$stratumId)){
                # This does not work without adding ffbase to search path:
                # outcomes$stratumId = 0
                # covariates$stratumId = 0
                # So we do:
                outcomes$stratumId <- ff::ff(vmode="double", length=nrow(outcomes))
                for (i in bit::chunk(outcomes$stratumId)){
                    outcomes$stratumId[i] <- 0
                }
                covariates$stratumId <- ff::ff(vmode="double", length=nrow(covariates))
                for (i in bit::chunk(covariates$stratumId)){
                    covariates$stratumId[i] <- 0
                }

            }
            if (!isSorted(outcomes,c("stratumId","time","y","rowId"),c(TRUE,FALSE,TRUE,TRUE))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y, and rowId")
                rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
                # This does not work without adding ffbase to search path:
                # outcomes$minTime = 0-outcomes$time
                # Therefore, we do:
                outcomes$minTime <- ff::ff(vmode="double", length=length(outcomes$time))
                for (i in bit::chunk(outcomes$time)){
                    outcomes$minTime[i] <- 0-outcomes$time[i]
                }

                outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","minTime","y","rowId")]),]
            }
            if (is.null(covariates$time) | is.null(covariates$y)){ # If time or y not present, add to check if sorted
                covariates$time = NULL
                covariates$y = NULL
                covariates <- merge(covariates,outcomes,by=c("stratumId","rowId"))
            }
            if (!isSorted(covariates,c("stratumId","time","y","rowId"),c(TRUE,FALSE,TRUE,TRUE))){
                if(!quiet)
                    writeLines("Sorting covariates by stratumId, time (descending), y, and rowId")
                rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
                covariates$minTime = 0-covariates$time
                covariates <- covariates[ff::ffdforder(covariates[c("stratumId","minTime","y","rowId")]),]
            }
        }
    }
    if (checkRowIds){
        mapped <- ffbase::ffmatch(x = covariates$rowId, table=outcomes$rowId, nomatch = 0L) > 0L
        minValue <- min(sapply(bit::chunk(mapped), function(i) {
            min(mapped[i])
        }))
        if (minValue == 0){
            if(!quiet)
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            row.names(covariates) <- NULL #Needed or else next line fails
            covariates <- covariates[ffbase::ffwhich(mapped, mapped == TRUE),]
        }
    }

    resultSetOutcome <- new.env()
    assign("data",outcomes,envir=resultSetOutcome)
    assign("chunks",ff::chunk.ffdf(outcomes),envir=resultSetOutcome)
    assign("cursor",1,envir=resultSetOutcome)
    resultSetCovariate <- new.env()
    assign("data",covariates,envir=resultSetCovariate)
    assign("chunks",ff::chunk.ffdf(covariates),envir=resultSetCovariate)
    assign("cursor",1,envir=resultSetCovariate)

    getOutcomeBatch <- function(resultSetOutcome, modelType){
        data <- get("data",envir=resultSetOutcome)
        chunks <- get("chunks",envir=resultSetOutcome)
        cursor <- get("cursor",envir=resultSetOutcome)
        batchOutcome <- data[chunks[[cursor]],]
        assign("cursor",cursor+1,envir=resultSetOutcome)
        if (modelType == "pr" | modelType == "cpr"| modelType == "cox")
            if (any(batchOutcome$time <= 0))
                stop("time cannot be non-positive",call.=FALSE)
        if (modelType == "lr" | modelType == "pr")
            batchOutcome$stratumId = batchOutcome$rowId
        if (modelType == "cox" & is.null(batchOutcome$stratumId))
            batchOutcome$stratumId = 0
        if (modelType == "lr" | modelType == "clr")
            batchOutcome$time = 0
        batchOutcome
    }

    getCovariateBatch <- function(resultSetCovariate, modelType){
        data <- get("data",envir=resultSetCovariate)
        chunks <- get("chunks",envir=resultSetCovariate)
        cursor <- get("cursor",envir=resultSetCovariate)
        batchCovariate <- data[chunks[[cursor]],]
        assign("cursor",cursor+1,envir=resultSetCovariate)
        batchCovariate
    }

    isDone <- function(resultSet){
        chunks <- get("chunks",envir=resultSet)
        cursor <- get("cursor",envir=resultSet)
        cursor > length(chunks)
    }

    result <- .constructCyclopsDataFromBatchableSources(resultSetOutcome,
                                                        resultSetCovariate,
                                                        getOutcomeBatch,
                                                        getCovariateBatch,
                                                        isDone,
                                                        modelType,
                                                        addIntercept,
                                                        offsetAlreadyOnLogScale,
                                                        makeCovariatesDense)
    return(result)
}

#' @keywords internal
legacyConvertToCyclopsData.data.frame <- function(outcomes,
                                            covariates,
                                            modelType = "lr",
                                            addIntercept = TRUE,
                                            offsetAlreadyOnLogScale = FALSE,
                                            makeCovariatesDense = NULL,
                                            checkSorting = TRUE,
                                            checkRowIds = TRUE,
                                            quiet = FALSE){
    if ((modelType == "clr" | modelType == "cpr") & addIntercept){
        if(!quiet)
            warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
        addIntercept = FALSE
    }
    if (modelType == "pr" | modelType == "cpr")
        if (any(outcomes$time <= 0))
            stop("time cannot be non-positive",call.=FALSE)
    if (modelType == "cox" & is.null(outcomes$stratumId)){
        outcomes$stratumId = 0
        covariates$stratumId = 0
    }
    if (modelType == "lr" | modelType == "clr")
        outcomes$time = 0

    if (checkSorting){
        if (modelType == "lr" | modelType == "pr"){
            if (!isSorted(outcomes,c("rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by rowId")
                outcomes <- outcomes[order(outcomes$rowId),]
            }
            if (!isSorted(covariates,c("rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by rowId")
                covariates <- covariates[order(covariates$rowId),]
            }
        }

        if (modelType == "clr" | modelType == "cpr"){
            if (!isSorted(outcomes,c("stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId and rowId")
                outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
            }
            if (!isSorted(covariates,c("stratumId","rowId"))){
                if(!quiet)
                    writeLines("Sorting covariates by stratumId and rowId")
                covariates <- covariates[order(covariates$stratumId,covariates$rowId),]
            }
        }
        if (modelType == "cox"){
            if (is.null(outcomes$stratumId)){
                outcomes$stratumId = 0
                covariates$stratumId = 0
            }
            if (!isSorted(outcomes,c("stratumId","time","y","rowId"),c(TRUE,FALSE,TRUE,TRUE))){
                if(!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y, and rowId")
                outcomes <- outcomes[order(outcomes$stratumId,-outcomes$time,outcomes$y,outcomes$rowId),]
            }
            if (is.null(covariates$time) | is.null(covariates$y)){ # If time or y not present, add to check if sorted
                covariates$time = NULL
                covariates$y = NULL
                covariates <- merge(covariates,outcomes,by=c("stratumId","rowId"))
            }
            if (!isSorted(covariates,c("stratumId","time","y","rowId"),c(TRUE,FALSE,TRUE,TRUE))){
                if(!quiet)
                    writeLines("Sorting covariates by stratumId, time (descending), y, and rowId")
                covariates <- covariates[order(covariates$stratumId,-covariates$time,covariates$y,covariates$rowId),]
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
    dataPtr <- createSqlCyclopsData(modelType = modelType)

    if (modelType == "lr" | modelType == "pr"){
        appendSqlCyclopsData(dataPtr,
                             outcomes$rowId,
                             outcomes$rowId,
                             outcomes$y,
                             outcomes$time,
                             covariates$rowId,
                             covariates$covariateId,
                             covariates$covariateValue
        )
    } else {
        appendSqlCyclopsData(dataPtr,
                             outcomes$stratumId,
                             outcomes$rowId,
                             outcomes$y,
                             outcomes$time,
                             covariates$rowId,
                             covariates$covariateId,
                             covariates$covariateValue
        )
    }


    if (modelType == "pr" | modelType == "cpr")
        useOffsetCovariate = -1
    else
        useOffsetCovariate = NULL

    if (modelType != "cox"){
        finalizeSqlCyclopsData(dataPtr,
                               addIntercept = addIntercept,
                               useOffsetCovariate = useOffsetCovariate,
                               offsetAlreadyOnLogScale = offsetAlreadyOnLogScale,
                               makeCovariatesDense = makeCovariatesDense)
    }
    return(dataPtr)
}
