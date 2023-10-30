# @file DataManagement.R
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

#' @title Create a Cyclops data object
#'
#' @description
#' \code{createCyclopsData} creates a Cyclops data object from an R formula or data matrices.
#'
#' @details
#' This function creates a Cyclops model data object from R \code{"\link{formula}"} or directly from
#' numeric vectors and matrices to define the model response and covariates.
#' If specifying a model using a \code{"\link{formula}"}, then the left-hand side define the model response and the
#' right-hand side defines dense covariate terms.
#' Objects provided with \code{"sparseFormula"} and \code{"indicatorFormula"} must be include left-hand side responses and terms are
#' coersed into sparse and indicator representations for computational efficiency.
#'
#' Items to discuss:
#' * Only use formula or (y,dx,...)
#' * stratum() in formula
#' * offset() in formula
#' * when \code{"stratum"} (renamed from pid) are necessary
#' * when \code{"time"} are necessary
#'
#' @template types
#'
#' @param formula
#' An object of class \code{"\link{formula}"} that provides a symbolic description of the numerically dense model response and terms.
#' @param sparseFormula
#' An object of class \code{"\link{formula}"} that provides a symbolic description of numerically sparse model terms.
#' @param indicatorFormula
#' An object of class \code{"\link{formula}"} that provides a symbolic description of \{0,1\} model terms.
#' @param data
#' An optional data frame, list or environment containing the variables in the model.
#' @param subset
#' Currently unused
#' @param weights
#' Currently unused
#' @param censorWeights
#' Vector of subject-specific censoring weights (between 0 and 1). Currently only supported in \code{modelType = "fgr"}.
#' @param offset
#' Currently unused
#' @param pid
#' Optional vector of integer stratum identifiers. If supplied, all rows must be sorted by increasing identifiers
#' @param y
#' Currently undocumented
#' @param type
#' Currently undocumented
#' @param time
#' Currently undocumented
#' @param dx
#' Optional dense \code{"\link{Matrix}"} of covariates
#' @param sx
#' Optional sparse \code{"\link{Matrix}"} of covariates
#' @param ix
#' Optional \{0,1\} \code{"\link{Matrix}"} of covariates
#' @param model
#' Currently undocumented
#' @param method
#' Currently undocumented
#' @param normalize
#' String: Name of normalization for all non-indicator covariates (possible values: stdev, max, median)
#' @param floatingPoint
#' Integer: Floating-point representation size (32 or 64)
#'
#' @return
#' A list that contains a Cyclops model data object pointer and an operation duration
#'
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
#' outcome <- gl(3, 1, 9)
#' treatment <- gl(3, 3)
#' cyclopsData <- createCyclopsData(
#'      counts ~ outcome + treatment,
#'      modelType = "pr")
#' cyclopsFit <- fitCyclopsModel(cyclopsData)
#'
#' cyclopsData2 <- createCyclopsData(
#'      counts ~ outcome,
#'      indicatorFormula = ~ treatment,
#'      modelType = "pr")
#' summary(cyclopsData2)
#' cyclopsFit2 <- fitCyclopsModel(cyclopsData2)
#'
#' @export
createCyclopsData <- function(formula, sparseFormula, indicatorFormula, modelType,
                              data, subset = NULL, weights = NULL, censorWeights = NULL, offset = NULL, time = NULL, pid = NULL, y = NULL, type = NULL, dx = NULL,
                              sx = NULL, ix = NULL, model = FALSE, normalize = NULL,
                              floatingPoint = 64,
                              method = "cyclops.fit") {
    cl <- match.call() # save to return
    mf.all <- match.call(expand.dots = FALSE)

    if (!.isValidModelType(modelType)) stop("Invalid model type.")

    .checkFloatingPoint(floatingPoint)

    hasIntercept <- FALSE
    colnames <- NULL

    contrasts <- NULL
    sortOrder <- NULL

    if (!missing(formula)) { # Use formula to construct Cyclops matrices
        if (missing(data)) {
            data <- environment(formula)
        }
        mf.all <- match.call(expand.dots = FALSE)
        m.d <- match(c("formula", "data", "subset", "weights", "censorWeights",
                       "offset"), names(mf.all), 0L)
        mf.d <- mf.all[c(1L, m.d)]
        mf.d$drop.unused.levels <- TRUE
        mf.d[[1L]] <- quote(stats::model.frame)
        mf.d <- eval(mf.d, parent.frame())

        outcome <- model.response(mf.d)
        if (inherits(outcome, "Surv")) {
            if (!.isSurvivalModelType(modelType)) {
                stop("Censored outcomes are currently only supported for Cox regression.")
            }
            if (dim(outcome)[2] == 3) {
                time <- as.numeric(outcome[,2] - outcome[,1])
                y <- as.numeric(outcome[,3])
            } else {
                time <- as.numeric(outcome[,1])
                y <- as.numeric(outcome[,2])
            }
        } else if (inherits(outcome, "Multitype")) {
            contrasts <- attr(outcome, "contrasts")
            type <- outcome[,2]
            y <- outcome[,1]
        } else {
            y <- outcome
        }

        mt.d <- attr(mf.d, "terms")

        # Handle strata
        specialTerms <- terms(formula, "strata")
        special <- attr(specialTerms, "special")
        hasStrata <- !is.null(special$strata)
        strata <- NULL

        if (hasStrata) {
            pid <- as.numeric(strata(mf.d[ , special$strata], shortlabel = TRUE))
            nterm <- survival::untangle.specials(specialTerms, "strata")$terms
            mt.d <- mt.d[-nterm]

            ## Must sort outcomes
            if (.isSurvivalModelType(modelType)) {
                sortOrder <- order(pid, -time, y)
            } else {
                sortOrder <- order(pid)
            }
        } else {
            if (.isSurvivalModelType(modelType)) {
                sortOrder <- order(-time, y)
                if (missing(pid)) {
                    pid <- rep(1, length(y))
                }
            } else {
                if (missing(pid)) {
                    pid <- c(1:length(y))
                }
            }
        }

        #         if (.removeIntercept(modelType)) { # This does not work with constrasts
        #             attr(mt.d, "intercept") <- FALSE
        #         }

        dtmp <- model.matrix(mt.d, mf.d)
        dlabels <- labels(dtmp)[[2]]
        if (attr(mt.d, "intercept") && .removeIntercept(modelType)) {
            dx <- Matrix(as.matrix(dtmp[,-1]), sparse = FALSE)
            dlabels <- dlabels[-1]
        } else {
            dx <- Matrix(dtmp, sparse = FALSE)
        }

        off.d <- model.offset(mf.d)
        if (!is.null(time) && !is.null(off.d)) {
            stop("Supplied both 'time' and 'offset' quantities")
        }
        if (!is.null(off.d)) {
            time <- as.vector(off.d)
        }

        #colnames <- c(colnames, dx@Dimnames[[2]])
        colnames <- c(colnames, dlabels)

        if (attr(mt.d, "intercept") && !.removeIntercept(modelType)) { # Has intercept
            hasIntercept <- TRUE
        }

        if (!missing(sparseFormula)) {
            if (missing(data)) {
                data <- environment(sparseFormula)
            }
            m.s <- match(c("sparseFormula", "data", "subset", "weights", "censorWeights",
                           "offset"), names(mf.all), 0L)
            mf.s <- mf.all[c(1L, m.s)]
            mf.s$drop.unused.levels <- TRUE
            mf.s[[1L]] <- quote(stats::model.frame)
            names(mf.s)[2] = "formula"
            mf.s <- eval(mf.s, parent.frame())

            if (!is.null(model.response(mf.s))) {
                stop("Must only provide outcome variable in dense formula.")
            }
            mt.s <- attr(mf.s, "terms")
            stmp <- model.matrix(mt.s, mf.s)
            #stmp <- model.matrix(mf.s, data)
            slabels <- labels(stmp)[[2]]
            if (attr(attr(mf.s, "terms"), "intercept") == 1) { # Remove intercept
                sx <- Matrix(as.matrix(stmp[,-1]), sparse=TRUE)
                slabels <- slabels[-1]
            } else {
                sx <- Matrix(as.matrix(stmp), sparse=TRUE)
            }
            colnames <- c(colnames, slabels)
        }

        if (!missing(indicatorFormula)) {
            if (missing(data)) {
                data <- environment(indicatorFormula)
            }
            m.i <- match(c("indicatorFormula", "data", "subset", "weights", "censorWeights",
                           "offset"), names(mf.all), 0L)
            mf.i <- mf.all[c(1L, m.i)]
            mf.i$drop.unused.levels <- TRUE
            mf.i[[1L]] <- quote(stats::model.frame)
            names(mf.i)[2] = "formula"
            mf.i <- eval(mf.i, parent.frame())

            if (!is.null(model.response(mf.i))) {
                stop("Must only provide outcome variable in dense formula.")
            }

            # TODO Check that all values in mf.i are 0/1
            mt.i <- attr(mf.i, "terms")
            itmp <- model.matrix(mt.i, mf.i)
            #itmp <- model.matrix(mf.i, data)
            ilabels <- labels(itmp)[[2]]
            if (attr(attr(mf.i, "terms"), "intercept") == 1) { # Remove intercept
                ix <- Matrix(as.matrix(itmp[,-1]), sparse=TRUE)
                ilabels <- ilabels[-1]
            } else {
                ix <- Matrix(as.matrix(itmp), sparse=TRUE)
            }
            colnames <- c(colnames, ilabels)
        }

        if (!is.null(sortOrder)) {
            pid <- pid[sortOrder]
            y <- y[sortOrder]
            if (!missing(type)) {
                type <- type[sortOrder]
            }
            if (!missing(weights)) {
                weights <- weights[sortOrder]
            }
            if (!missing(censorWeights)) {
                censorWeights <- censorWeights[sortOrder]
            }
            time <- time[sortOrder]
            dx <- dx[sortOrder, ]
            if (inherits(dx,"numeric")) {
                dx = as(dx,"dgeMatrix")
            }
            sx <- sx[sortOrder, ]
            if (inherits(sx, "numeric")) {
                sx = Matrix(sx, ncol = 1, sparse = TRUE)
            }
            ix <- ix[sortOrder, ]
            if (inherits(ix, "numeric")) {
                ix = Matrix(ix, ncol = 1, sparse = TRUE)
            }
        }

        if (identical(method, "model.frame")) {
            result <- list()
            if (exists("mf.d")) {
                result$dense <- mf.d
                result$dx <- dx
                result$y <- y
                result$time <- time
            }
            if (exists("mf.s")) {
                result$sparse <- mf.s
                result$sx <- sx
            }
            if (exists("mf.i")) {
                result$indicator <- mf.i
                result$ix <- ix
            }
            return(result)
        }
    } else  {
        if (!missing(sparseFormula) || !missing(indicatorFormula)) {
            stop("Must provide a dense formula when specifying sparse or indicator covariates.")
        }
    }

    # Handle multiple outcome types
    numTypes <- 1
    if (!missing(type)) {
        if (is.null(contrasts)) {
            contrasts <- contrasts(as.factor(type))
        }
        numTypes <- length(contrasts)
        type <- contrasts[type]
        if (!is.null(colnames)) {
            colnames <- as.vector(sapply(colnames, function(x, y) { paste(x,y,sep=":")},
                                         y = dimnames(contrasts)[[1]], USE.NAMES=F))
        }
    }

    # TODO Check types and dimensions

    useTimeAsOffset <- FALSE
    if (!is.null(time) && !.useOffsetModelType(modelType)) {
        useTimeAsOffset <- TRUE
    }

    if (is.null(pid)) {
        pid <- c(1:length(y)) # TODO Should not be necessary
    }

    md <- .cyclopsModelData(pid, y, type, time, dx, sx, ix, modelType, useTimeAsOffset, numTypes,
                            floatingPoint)
    result <- new.env(parent = emptyenv())
    result$cyclopsDataPtr <- md$data
    result$modelType <- modelType
    result$timeLoad <- md$timeLoad
    result$call <- cl
    #     if (exists("mf") && model == TRUE) {
    #         result$mf <- mf
    #     }
    result$cyclopsInterfacePtr <- NULL
    result$call <- cl
    result$coefficientNames <- colnames

    if (!is.null(dx)) {
        result$rowNames <- dx@Dimnames[[1]]
    } else if (!is.null(pid)) {
        result$rowNames <- pid
    } else {
        result$rowNames <- c(1:length(y))
    }

    result$sortOrder <- sortOrder
    result$weights <- weights
    result$censorWeights <- censorWeights

    if (identical(method, "debug")) {
        result$debug <- list()
        result$debug$dx <- dx
        result$debug$sx <- sx
        result$debug$ix <- ix
        result$debug$y <- y
        result$debug$pid <- pid
        result$debug$time <- time
    }
    class(result) <- "cyclopsData"

    if (hasIntercept == TRUE) {
        .cyclopsSetHasIntercept(result, hasIntercept = TRUE)
    }

    if (!is.null(normalize)) {
        .normalizeCovariates(result, normalize)
    }

    result
}

.checkFloatingPoint <- function(floatingPoint) {
      if (floatingPoint != 64 && floatingPoint != 32) {
          stop("Invalid floating point precision")
      }
}

.normalizeCovariates <- function(cyclopsData, type) {
    scale <- .cyclopsNormalizeCovariates(cyclopsData, type)

    if (is.null(cyclopsData$scale)) {
        cyclopsData$scale <- scale
    } else {
        cyclopsData$scale <- cyclopsData$scale * scale # Can call function multiple times without bad effects
    }

    cyclopsData$cyclopsInterfacePtr <- NULL # Force new engine for next fit

    invisible(scale)
}

# @title isValidModelType
#
# @description
# \code{isValidModelType} checks for a valid Cyclops model type
#
# @template types
#
# @return TRUE/FALSE
#
# @examples
# isValidModelType("pr")
# #TRUE
#
# isValidModelType("abc")
# #FALSE
#
# @export
.isValidModelType <- function(modelType) {
    types <- .cyclopsGetModelTypeNames()
    modelType %in% types
}

.removeIntercept <- function(modelType) {
    types <- .cyclopsGetRemoveInterceptNames()
    modelType %in% types
}

.isSurvivalModelType <- function(modelType) {
    types <- .cyclopsGetIsSurvivalNames()
    modelType %in% types
}

.useOffsetModelType <- function(modelType) {
    types <- .cyclopsGetUseOffsetNames()
    modelType %in% types
}

#' @title Read Cyclops data from file
#'
#' @description
#' \code{readCyclopsData} reads a Cyclops-formatted text file.
#'
#' @details
#' This function reads a Cyclops-formatted text file and returns a Cyclops data object. The first line of the
#' file may start with '\samp{#}', indicating that it contains header options.  Valid header options are:
#'
#'  \tabular{ll}{
#'  \verb{	row_label}		\tab (assume file contains a numeric column of unique row identifiers) \cr
#'  \verb{	stratum_label}\tab (assume file contains a numeric column of stratum identifiers) \cr
#'  \verb{	weight}				\tab (assume file contains a column of row-specific model weights, currently unused) \cr
#' 	\verb{	offset}				\tab (assume file contains a dense column of linear predictor offsets) \cr
#' 	\verb{	bbr_outcome}	\tab (assume logistic outcomes are encoded -1/+1 following BBR) \cr
#' 	\verb{	log_offset}		\tab (assume file contains a dense column of values x_i for which log(x_i) is the offset) \cr
#' 	\verb{	add_intercept}\tab (automatically include an intercept column of all 1s for each entry) \cr
#' 	\verb{	indicator_only}\tab(assume all covariates 0/1-valued and only covariate name is given) \cr
#' 	\verb{	sparse}				\tab (force all BBR formatted covariates to be represented as sparse, instead of \cr
#'                              \tab sparse-indicator, columns .. really only for debugging) \cr
#' 	\verb{	dense}				\tab (force all BBR formatted covariates to be represented as dense columns.. really\cr
#'                              \tab only for debugging) \cr
#' }
#'
#' Successive lines of the file are white-space delimited and follow the format:
#'
#' \preformatted{	[Row ID] {Stratum ID} [Weight] <Outcome> {Censored} {Offset} <BBR covariates>}
#'
#' \itemize{
#'   	\item \verb{[optional]}
#'   	\item \verb{<required>}
#'   	\item \verb{{required or optional depending on model}}
#'  }
#'
#' Bayesian binary regression (BBR) covariates are white-space delimited and generally in a sparse
#' \samp{<name>:<value>} format, where \samp{name} must (currently) be numeric and \samp{value} is non-zero.
#' If option \samp{indicator_only} is specified, then format is simply \samp{<name>}.
#' \samp{Row ID} and \samp{Stratum ID} must be numeric, and rows must be sorted such that equal \samp{Stratum ID}
#' are consecutive.
#' \samp{Stratum ID} is required for \samp{clr} and \samp{sccs} models.
#' \samp{Censored} is required for a \samp{cox} model.
#' \samp{Offset} is (currently) required for a \samp{sccs} model.
#'
#' @template types
#'
#' @param fileName          Name of text file to be read. If fileName does not contain an absolute path,
#'
#' @return
#' A list that contains a Cyclops model data object pointer and an operation duration
#'
#' @examples
#' \dontrun{
#' dataPtr = readCyclopsData(system.file("extdata/infert_ccd.txt", package="Cyclops"), "clr")
#' }
#' @export
readCyclopsData <- function(fileName, modelType) {
    cl <- match.call() # save to return

    if (!.isValidModelType(modelType)) stop("Invalid model type.")

    read <- .cyclopsReadData(fileName, modelType)
    result <- new.env(parent = emptyenv())
    result$cyclopsDataPtr <- read$cyclopsDataPtr
    result$modelType <- modelType
    result$timeLoad <- read$timeLoad
    result$cyclopsInterfacePtr <- NULL
    result$call <- cl

    class(result) <- "cyclopsData"
    result
}

#' @title Get univariable correlation
#'
#' @description \code{getUnivariableCorrelation} reports covariates that have high correlation with the outcome
#'
#' @param cyclopsData    A Cyclops data object
#' @param covariates Integer or string vector: list of covariates to report; default (NULL) implies all covariates
#' @param threshold Correlation threshold for reporting
#'
#' @return A list of covariates whose absolute correlation with the outcome is greater than or equal to the threshold
#'
#' @export
getUnivariableCorrelation <- function(cyclopsData, covariates = NULL, threshold = 0.0) {
    # Check for valid arguments
    .checkData(cyclopsData)
    if (.isSurvivalModelType(cyclopsData$modelType)) {
        stop("Univariable correlation for time-to-events model is not yet implemented.")
    }

    labels <- covariates
    covariates <- .checkCovariates(cyclopsData, covariates)
    if (is.null(covariates)) {
        covariates <- bit64::as.integer64(c()) # zero-length vector
        labels <- cyclopsData$coefficientNames
    }

    correlations <- .cyclopsUnivariableCorrelation(cyclopsData, covariates)
    if (threshold > 0.0) {
        select <- abs(correlations) >= threshold
        correlations <- correlations[select]
        labels <- labels[select]
    }

    names(correlations) <- labels
    return(correlations)
}

#' @title Get univariable linear separability
#'
#' @description \code{getUnivariableSeparability} reports covariates that are univariably separable with the outcome
#'
#' @param cyclopsData    A Cyclops data object
#' @param covariates Integer or string vector: list of covariates to report; default (NULL) implies all covariates
#'
#' @return A list of covariates that are univariably separable with the outcome
#'
#' @export
getUnivariableSeparability <- function(cyclopsData, covariates = NULL) {
    # Check for valid arguments
    .checkData(cyclopsData)

    labels <- covariates
    covariates <- .checkCovariates(cyclopsData, covariates)
    ids <- covariates
    if (is.null(covariates)) {
        covariates <- bit64::as.integer64(c()) # zero-length vector
        labels <- cyclopsData$coefficientNames
        ids <- getCovariateIds(cyclopsData)
    }

    types <- getCovariateTypes(cyclopsData, ids)
    if (any(types == "sparse") || any(types == "dense")) {
        warning("Sparse and dense types are not yet implemented")
    }

    separability <- .cyclopsUnivariableSeparability(cyclopsData, covariates)

    names(separability) <- labels
    return(separability == 1)
}

#' @title Apply simple data reductions
#'
#' @description \code{reduce} reports the count of non-zero elements, sum and sum-of-squares for specified covariates in a Cyclops data object.
#'
#' @param object    A Cyclops data object
#' @param covariates Integer or string vector: list of covariates to report
#' @param groupBy   Integer or string (optional): generates a segmented reduction stratified by this covariate.  Setting \code{groupBy = "stratum"} segments reduction for strataID
#' @param power Integer: 0 = non-zero count, 1 = sum, 2 = sum-of-squares
#'
#' @return Specified reduction as number or \code{data.frame} if segmented.
#'
#' @keywords internal
reduce <- function(object, covariates, groupBy, power = 1) {
    if (!isInitialized(object)) {
        stop("Object is no longer or improperly initialized.")
    }
    covariates <- .checkCovariates(object, covariates)

    if (!(power %in% c(0,1,2))) {
        stop("Only powers 0, 1 and 2 are allowed.")
    }

    if (missing(groupBy)) {
        .cyclopsSum(object, covariates, power)
    } else {
        if (length(groupBy) != 1L) {
            stop("Only single stratification is currently implemented")
        }
        if (groupBy == "stratum") {
            as.data.frame(.cyclopsSumByStratum(object, covariates, power),
                          row.names = c(1L:getNumberOfStrata(object)))
        } else {
            groupBy <- .checkCovariates(object, groupBy)
            as.data.frame(.cyclopsSumByGroup(object, covariates, groupBy, power),
                          row.names = c(0L,1L))
        }
    }
}

#' @title appendSqlCyclopsData
#'
#' @description
#' \code{appendSqlCyclopsData} appends data to an OHDSI data object.
#'
#' @details Append data using two tables.  The outcomes table is dense and contains ...  The covariates table is sparse and contains ...
#' All entries in the outcome table must be sorted in increasing order by (oStratumId, oRowId).  All entries in the covariate table
#' must be sorted in increasing order by (cRowId). Each cRowId value must match exactly one oRowId value.
#'
#' @param object    OHDSI Cyclops data object to append entries
#' @param oStratumId    Integer vector (optional): non-unique stratum identifier for each row in outcomes table
#' @param oRowId        Integer vector: unique row identifier for each row in outcomes table
#' @param oY            Numeric vector: model outcome variable for each row in outcomes table
#' @param oTime         Numeric vector (optional): exposure interval or censoring time for each row in outcomes table
#' @param cRowId        Integer vector: non-unique row identifier for each row in covariates table that matches a single outcomes table entry
#' @param cCovariateId  Integer vector: covariate identifier
#' @param cCovariateValue   Numeric vector: covariate value
#'
#' @keywords internal
appendSqlCyclopsData <- function(object,
                                 oStratumId,
                                 oRowId,
                                 oY,
                                 oTime,
                                 cRowId,
                                 cCovariateId,
                                 cCovariateValue) {
    if (!isInitialized(object)) {
        stop("Object is no longer or improperly initialized.")
    }

    if (is.unsorted(oStratumId)
        #|| is.unsorted(oRowId) || is.unsorted(cRowId)
    ) {
        stop("All columns must be sorted first by stratumId (if supplied) and then by rowId")
    }

    .appendSqlCyclopsData(object,
                          oStratumId,
                          oRowId,
                          oY,
                          oTime,
                          cRowId,
                          cCovariateId,
                          cCovariateValue)
}

#' @importFrom bit64 as.integer64 is.integer64
#' @keywords internal
loadNewSqlCyclopsDataMultipleX <- function(object,
                                           covariateId, # Vector
                                           rowId, # Vector
                                           covariateValue = NULL, # Vector
                                           name,
                                           append = FALSE,
                                           checkSorting = FALSE,
                                           checkCovariateIds = FALSE,
                                           checkCovariateBounds = FALSE,
                                           forceSparse = FALSE) {

    if (!isInitialized(object)) stop("Object is no longer or improperly initialized.")

    if (length(covariateId) != length(rowId)) stop("Vector length mismatch")

    if (is.null(covariateValue)) {
        if (forceSparse) stop("Must provide covariate values when forcing sparse")
        covariateValue <- as.numeric(c())
    } else {
        if (length(covariateId) != length(covariateValue)) stop ("Vector length mismatch")
    }

    if (checkSorting) {
        # TODO: Check sorted by (1) covariateId, (2) rowId
        stop("Not yet implemented")
    }

    if (!bit64::is.integer64(covariateId)) {
      covariateId <- bit64::as.integer64(covariateId)
    }

    if (!bit64::is.integer64(rowId)) {
      rowId <- bit64::as.integer64(rowId)
    }

    index <- .loadCyclopsDataMultipleX(object,
                                       covariateId,
                                       rowId,
                                       covariateValue,
                                       checkCovariateIds,
                                       checkCovariateBounds,
                                       append,
                                       forceSparse)

    if (!missing(name)) {
        if (is.null(object$coefficientNames)) {
            object$coefficientNames <- as.character(c())
        }
        start <- index + 1
        end <- index + length(name)
        object$coefficientNames[start:end] <- as.character(name)
    }
}

#' @keywords internal
loadNewSqlCyclopsDataX <- function(object,
                                   covariateId, # Scalar
                                   rowId = NULL, # Vector
                                   covariateValue = NULL, # Vector
                                   name,
                                   replace = FALSE,
                                   append = FALSE,
                                   forceSparse = FALSE) {
    if (!isInitialized(object)) {
        stop("Object is no longer or improperly initialized.")
    }

    if (replace == TRUE) {
        stop("Replacing an X column is not currently supported")
    }

    if (is.null(rowId)) rowId <- bit64::as.integer64(c())
    if (is.null(covariateValue)) {
        if (forceSparse) stop("Must provide covariate values when forcing sparse")
        covariateValue <- as.numeric(c())
    }

    if (!bit64::is.integer64(covariateId)) {
      covariateId <- bit64::as.integer64(covariateId)
    }

    if (!bit64::is.integer64(rowId)) {
      rowId <- bit64::as.integer64(rowId)
    }

    # throws error if covariateId already exists and append == FALSE
    index <- .loadCyclopsDataX(object,
                               covariateId,
                               rowId,
                               covariateValue,
                               replace,
                               append,
                               forceSparse)

    if (!missing(name)) {
        if(is.null(object$coefficientNames)) {
            object$coefficientNames <- as.character(c())
        }
        object$coefficientNames[index + 1] <- name
    }
}

#' @keywords internal
loadNewSqlCyclopsDataY <- function(object,
                                   stratumId = NULL,
                                   rowId = NULL,
                                   y,
                                   time = NULL) {
    if (!isInitialized(object)) {
        stop("Object is no longer or improperly initialized.")
    }

    if (is.null(stratumId)) stratumId <- bit64::as.integer64(c())
    if (is.null(rowId)) rowId <- bit64::as.integer64(c())

    if (is.unsorted(stratumId)) {
        stop("All columns must be sorted first by stratumId (if supplied) and then by rowId")
    }

    if (is.null(time)) {
        if (.isSurvivalModelType(object$modelType)) stop("Must provide time for survival model")
        time <- as.numeric(c())
    }

    if (!bit64::is.integer64(stratumId)) {
      stratumId <- bit64::as.integer64(stratumId)
    }

    if (!bit64::is.integer64(rowId)) {
      rowId <- bit64::as.integer64(rowId)
    }

    .loadCyclopsDataY(object,
                      stratumId,
                      rowId,
                      y,
                      time)
}

#' @keywords internal
loadNewSqlCyclopsDataStratTimeEffects <- function(object,
                                                  stratumId = NULL,
                                                  rowId = NULL,
                                                  subjectId = NULL,
                                                  timeEffectCovariateId) {

    if (!isInitialized(object)) stop("Object is no longer or improperly initialized.")

    if (is.unsorted(stratumId)) {
        stop("All columns must be sorted first by stratumId (if supplied) and then by rowId")
    }

    if (!bit64::is.integer64(stratumId)) {
        stratumId <- bit64::as.integer64(stratumId)
    }

    if (!bit64::is.integer64(rowId)) {
        rowId <- bit64::as.integer64(rowId)
    }

    if (!bit64::is.integer64(subjectId)) {
        subjectId <- bit64::as.integer64(subjectId)
    }

    if (!bit64::is.integer64(timeEffectCovariateId)) {
        timeEffectCovariateId <- bit64::as.integer64(timeEffectCovariateId)
    }

    timeEffectCovariatesName <- .loadCyclopsDataStratTimeEffects(object,
                                                                 stratumId,
                                                                 rowId,
                                                                 subjectId,
                                                                 timeEffectCovariateId)
    if (!is.null(object$coefficientNames)) {
        object$coefficientNames <- append(object$coefficientNames, timeEffectCovariatesName)
    }
}

#' @title finalizeSqlCyclopsData
#'
#' @description
#' \code{finalizeSqlCyclopsData} finalizes a Cyclops data object
#'
#' @param object							Cyclops data object
#' @param addIntercept				Add an intercept covariate if one was not imported through SQL
#' @param useOffsetCovariate	Specify is a covariate should be used as an offset (fixed coefficient = 1).
#' 														Set option to \code{-1} to specify the time-to-event column,
#' 														otherwise include a single numeric or character covariate name.
#' @param offsetAlreadyOnLogScale						Set to \code{TRUE} to indicate that offsets were log-transformed before importing into Cyclops data object.
#' @param sortCovariates			Sort covariates in numeric-order with intercept first if it exists.
#' @param makeCovariatesDense List of numeric or character covariates names to densely represent in Cyclops data object.
#' 														For efficiency, we suggest making at least the intercept dense.
##' @keywords internal
#' @export
finalizeSqlCyclopsData <- function(object,
                                   addIntercept = FALSE,
                                   useOffsetCovariate = NULL,
                                   offsetAlreadyOnLogScale = FALSE,
                                   sortCovariates = FALSE,
                                   makeCovariatesDense = NULL) {
    if (!isInitialized(object)) {
        stop("Object is no longer or improperly initialized.")
    }

    savedUseOffsetCovariate <- useOffsetCovariate
    useOffsetCovariate <- .checkCovariates(object, useOffsetCovariate)
    if (length(useOffsetCovariate) > 1) {
        stop("Can only supply one offset")
    }

    makeCovariatesDense <- .checkCovariates(object, makeCovariatesDense)

    .cyclopsFinalizeData(object, addIntercept, useOffsetCovariate,
                         offsetAlreadyOnLogScale, sortCovariates,
                         makeCovariatesDense)

    if (addIntercept == TRUE) {
        if (!is.null(object$coefficientNames)) {
            object$coefficientNames = c("(Intercept)",
                                        object$coefficientNames)
        }
    }
    if (!is.null(useOffsetCovariate) && useOffsetCovariate != -1) {
        if (!is.null(object$coefficientNames)) {
            object$coefficientNames = object$coefficientNames[-as.integer(useOffsetCovariate)]
        }
    }
}

#' @keywords internal
createSqlCyclopsData <- function(modelType, control,
                                 floatingPoint = 64) {
    cl <- match.call() # save to return

    if (!.isValidModelType(modelType)) stop("Invalid model type.")

    noiseLevel <- "silent"
    if (!missing(control)) { # Set up control
        stopifnot(inherits(control, "cyclopsControl"))
        noiseLevel <- control$noiseLevel
    }

    sql <- .cyclopsNewSqlData(modelType, noiseLevel, floatingPoint)
    result <- new.env(parent = emptyenv()) # TODO Remove code duplication with two functions above
    result$cyclopsDataPtr <- sql$cyclopsDataPtr
    result$modelType <- modelType
    result$timeLoad <- 0
    result$cyclopsInterfacePtr <- NULL
    result$call <- cl
    class(result) <- "cyclopsData"
    result
}

#' @title Check if a Cyclops data object is initialized
#'
#' @description
#' \code{isInitialized} determines if an Cyclops data object is properly
#' initialized and remains in memory.  Cyclops data objects do not
#' serialized/deserialize their back-end memory across R sessions.
#'
#' @param object    Cyclops data object to test
#'
#' @export
isInitialized <- function(object) {
    return(!is.null(object$cyclopsDataPtr) && !.isRcppPtrNull(object$cyclopsDataPtr))
}


#' @title Get covariate types
#'
#' @description
#' \code{getCovariateTypes} returns a vector covariate types in a Cyclops data object
#'
#' @param object    A Cyclops data object
#' @param covariateLabel Integer vector: covariate identifiers to return
#'
#' @export
getCovariateTypes <- function(object, covariateLabel) {
  if (!isInitialized(object)) {
    stop("Cyclops data object is no longer or improperly initialized")
  }

  if (!bit64::is.integer64(covariateLabel)) {
    covariateLabel <- bit64::as.integer64(covariateLabel)
  }

  return(.getCovariateTypes(object, covariateLabel))
}

#' @title Cyclops data object summary
#'
#' @method summary cyclopsData
#'
#' @description \code{summary.cyclopsData} summarizes the data held in an Cyclops data object.
#'
#' @param object    A Cyclops data object
#' @param ...       Additional arguments
#'
#' @return
#' Returns a \code{data.frame} that reports simply summarize statistics for each covariate in a Cyclops data object.
#'
#' @export
summary.cyclopsData <- function(object, ...) {
    if (!isInitialized(object)) {
        stop("Cyclops data object is no longer or improperly initialized")
    }
    covariates <- getCovariateIds(object)
    counts <- reduce(object, covariates, power = 0)
    sums <- reduce(object, covariates, power = 1)
    sumsSquared <- reduce(object, covariates, power = 2)
    types <- getCovariateTypes(object, covariates)

    tmean <- sums / counts;
    tvar <- (sumsSquared -  counts * tmean * tmean) / counts
    tvar[tvar < 0] <- 0

    if (is.null(object$scale)) {
        tdf <- data.frame(covariateId = covariates,
                          nzCount = counts,
                          nzMean = tmean,
                          nzVar = tvar,
                          type = types)
    } else {
        tdf <- data.frame(covariateId = covariates,
                          nzCount = counts,
                          nzMean = tmean,
                          nzVar = tvar,
                          type = types,
                          scale = object$scale)
    }

    if (!is.null(object$coefficientNames)) {
        row.names(tdf) <- object$coefficientNames
    }
    tdf
}


#' @method print cyclopsData
#' @title Print a Cyclops data object
#'
#' @description
#' \code{print.cyclopsData} displays information about a Cyclops data model object.
#'
#' @param x    A Cyclops data model object
#' @param show.call Logical: display last call to construct the Cyclops data model object
#' @param ...   Additional arguments
#'
#' @export
print.cyclopsData <- function(x, show.call=TRUE ,...) {
    cat("Cyclops Data Object\n\n")

    if (show.call && !is.null(x$call)) {
        cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")
    }
    cat("     Model: ", x$modelType, "\n", sep="")

    if (isInitialized(x)) {
        nRows <- getNumberOfRows(x)
        cat("      Rows: ", nRows, "\n", sep = "")
        cat("Covariates: ", getNumberOfCovariates(x), "\n", sep = "")
        nStrata <- getNumberOfStrata(x)
        if (nRows != nStrata) {
            cat("    Strata: ", nStrata, "\n", sep = "")
        }
        if (.cyclopsGetHasOffset(x)) {
            cat("    Offset: ", .cyclopsGetMeanOffset(x), " (mean)\n", sep = "")
        }
        cat("   FP size:", getFloatingPointSize(x))
    } else {
        cat("\nObject is no longer or improperly initialized.\n")
    }
    cat("\n")
    if (!is.null(x$cyclopsInterfacePtr) && !.isRcppPtrNull(x$cyclopsInterfacePtr)) {
        cat("Initialized interface (details coming soon).\n")
    } else {
        cat("Uninitialized interface.\n")
    }
    invisible(x)
}
