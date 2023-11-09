# @file ModelFit.R
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

#' @title Fit a Cyclops model
#'
#' @description
#' \code{fitCyclopsModel} fits a Cyclops model data object
#'
#' @details
#' This function performs numerical optimization to fit a Cyclops model data object.
#'
#' @param cyclopsData			A Cyclops data object
#' @template prior
#' @param control  A \code{"cyclopsControl"} object constructed by \code{\link{createControl}}
#' @param weights Vector of 0/1 weights for each data row
#' @param forceNewObject Logical, forces the construction of a new Cyclops model fit object
#' @param returnEstimates Logical, return regression coefficient estimates in Cyclops model fit object
#' @param startingCoefficients Vector of starting values for optimization
#' @param fixedCoefficients Vector of booleans indicating if coefficient should be fix
#' @param warnings Logical, report regularization warnings
#' @param computeDevice String: Name of compute device to employ; defaults to \code{"native"} C++ on CPU
#'
#' @return
#' A list that contains a Cyclops model fit object pointer and an operation duration
#'
#' @references
#' Suchard MA, Simpson SE, Zorych I, Ryan P, Madigan D.
#' Massive parallelization of serial inference algorithms for complex generalized linear models.
#' ACM Transactions on Modeling and Computer Simulation, 23, 10, 2013.
#'
#' Simpson SE, Madigan D, Zorych I, Schuemie M, Ryan PB, Suchard MA.
#' Multiple self-controlled case series for large-scale longitudinal observational databases.
#' Biometrics, 69, 893-902, 2013.
#'
#' Mittal S, Madigan D, Burd RS, Suchard MA.
#' High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis.
#' Biostatistics, 15, 207-221, 2014.
#'
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' cyclopsData <- createCyclopsData(counts ~ outcome + treatment, modelType = "pr")
#' cyclopsFit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"))
#' coef(cyclopsFit)
#' confint(cyclopsFit, c("outcome2","treatment3"))
#' predict(cyclopsFit)
#'
#' @export
fitCyclopsModel <- function(cyclopsData,
                            prior = createPrior("none"),
                            control = createControl(),
                            weights = NULL,
                            forceNewObject = FALSE,
                            returnEstimates = TRUE,
                            startingCoefficients = NULL,
                            fixedCoefficients = NULL,
                            warnings = TRUE,
							computeDevice = "native") {

    # Delegate to control$setHook if exists
    if (!is.null(control$setHook)) {
        return(control$setHook(cyclopsData, prior, control,
                               weights, forceNewObject, returnEstimates,
                               startingCoefficients, fixedCoefficients))
    }

    # Delegate to prior$fitHook if exists
    if (!is.null(prior$fitHook)) {
        return(prior$fitHook(cyclopsData, prior, control,
                             weights, forceNewObject, returnEstimates,
                             startingCoefficients, fixedCoefficients))
    }

    cl <- match.call()

    # Check conditions
    .checkData(cyclopsData)

    if (getNumberOfRows(cyclopsData) < 1 ||
            getNumberOfStrata(cyclopsData) < 1 ||
            getNumberOfCovariates(cyclopsData) < 1) {
        stop("Data are incompletely loaded")
    }

    .checkInterface(cyclopsData, computeDevice =  computeDevice, forceNewObject = forceNewObject)

    # Set up prior
    stopifnot(inherits(prior, "cyclopsPrior"))

    if (!is.null(prior$setHook)) {

        prior$setHook(cyclopsData, warnings) # Call-back

    } else {
        prior$exclude <- .checkCovariates(cyclopsData, prior$exclude)

        if (!is.null(prior$neighborhood)) {
            prior$neighborhood <- lapply(prior$neighborhood,
                                         function(element) {
                                             list(.checkCovariates(cyclopsData, element[[1]]),
                                                  .checkCovariates(cyclopsData, element[[2]]))
                                         })
        }

        if (prior$priorType[1] != "none" &&
            is.null(prior$graph) && # TODO Ignore hierarchical models for now
            .cyclopsGetHasIntercept(cyclopsData) &&
            !prior$forceIntercept) {
            interceptId <- bit64::as.integer64(.cyclopsGetInterceptLabel(cyclopsData))
            warn <- FALSE
            if (is.null(prior$exclude)) {
                prior$exclude <- c(interceptId)
                warn <- TRUE
            } else {
                if (!(interceptId %in% prior$exclude)) {
                    prior$exclude <- c(interceptId, prior$exclude)
                    warn <- TRUE
                }
            }
            if (warn && warnings) {
                warning("Excluding intercept from regularization")
            }
        }

        if (is.null(prior$graph)) {
            graph <- NULL
        } else {
            graph <- .makeHierarchyGraph(cyclopsData, prior$graph)
            if (length(prior$priorType) != length(prior$variance)) {
                stop("Prior types and variances have a dimensionality mismatch")
            }
            if (any(prior$priorType != "normal")) {
                stop("Only normal-normal hierarchies are currently supported")
            }
        }

        if (is.null(prior$neighborhood)) {
            neighborhood <- NULL
        } else {
            neighborhood <- prior$neighborhood
            if (length(prior$priorType) != length(prior$variance)) {
                stop("Prior types and variances have a dimensionality mismatch");
            }
            if (any(prior$priorType  != "laplace")) {
                stop("Only Laplace-Laplace fused neighborhoods are currently supported")
            }
        }

        if (is.null(graph) && is.null(neighborhood) && length(prior$priorType) > 1) {
            if (length(prior$priorType) != getNumberOfCovariates(cyclopsData)) {
                stop("Length of priors must equal the number of covariates")
            }
        }

        if (any(prior$priorType == "jeffreys")) {
            if (Cyclops::getNumberOfCovariates(cyclopsData) > 1) {
                stop("Jeffreys prior is currently only implemented for 1 covariate")
            }

            covariate <- Cyclops::getCovariateIds(cyclopsData)
            if (Cyclops::getCovariateTypes(cyclopsData, covariate) != "indicator") {
                count <- reduce(cyclopsData, covariate, power = 0)
                sum <- reduce(cyclopsData, covariate, power = 1)
                mean <- sum / count
                if (!(mean == 0.0 || mean == 1.0)) {
                    stop("Jeffreys prior is currently only implemented for indicator covariates")
                }
            }
        }

        .cyclopsSetPrior(cyclopsData$cyclopsInterfacePtr, prior$priorType, prior$variance,
                         prior$exclude, graph, neighborhood)
    }

    if (control$selectorType == "auto") {
        if (cyclopsData$modelType %in% c("pr", "lr")) {
            control$selectorType <- "byRow"
        } else {
            rowsPerStratum <- (getNumberOfRows(cyclopsData) / getNumberOfStrata(cyclopsData))
            if (rowsPerStratum < getNumberOfStrata(cyclopsData)) {
                control$selectorType <- "byPid"
            } else {
                control$selectorType <- "byRow"
            }
        }
        if (prior$useCrossValidation && control$noiseLevel != "silent") {
            writeLines(paste("Using cross-validation selector type", control$selectorType))
        }
    }

    if (control$cvRepetitions == "auto") {
        control$cvRepetitions <- .getNumberOfRepetitions(getNumberOfRows(cyclopsData))
    }

    control <- .setControl(cyclopsData$cyclopsInterfacePtr, control)
    threads <- control$threads

    if (!is.null(startingCoefficients)) {

        if (length(startingCoefficients) != getNumberOfCovariates(cyclopsData)) {
            stop("Must provide a value for each coefficient")
        }

        if (.cyclopsGetHasOffset(cyclopsData)) {
            startingCoefficients <- c(1.0, startingCoefficients)
        }

        .cyclopsSetBeta(cyclopsData$cyclopsInterfacePtr, startingCoefficients)
        .cyclopsSetStartingBeta(cyclopsData$cyclopsInterfacePtr, startingCoefficients)
    }

    if (!is.null(fixedCoefficients)) {
        if (length(fixedCoefficients) != getNumberOfCovariates(cyclopsData)) {
            stop("Must provide a boolean for each coefficient")
        }

        offset <- ifelse(.cyclopsGetHasOffset(cyclopsData), 1, 0)
        for (i in 1:length(fixedCoefficients)) {
            .cyclopsSetFixedBeta(cyclopsData$cyclopsInterfacePtr, offset + i, fixedCoefficients[i] == TRUE)
        }
    }

    # Handle weights

    weightsUnsorted <- TRUE
    if (!is.null(cyclopsData$weights)) {
        if (!is.null(weights)) {
            warning("Using weights passed to fitCyclopsModel()")
        } else {
            weights <- cyclopsData$weights
            weightsUnsorted <- FALSE
        }
    }
    if (!is.null(weights)) {
        if (length(weights) != getNumberOfRows(cyclopsData)) {
            stop("Must provide a weight for each data row")
        }
        if (any(weights < 0)) {
            stop("Only non-negative weights are allowed")
        }

        if (weightsUnsorted) {
            if (!is.null(cyclopsData$sortOrder)) {
                weights <- weights[cyclopsData$sortOrder]
            }
        }

        .cyclopsSetWeights(cyclopsData$cyclopsInterfacePtr, weights)
    }

    # censorWeight check for the Fine-Gray model
    if (cyclopsData$modelType == "fgr" & is.null(cyclopsData$censorWeights)) {
        stop("Subject-specific censoring weights must be specified for modelType = 'fgr'.")
    }

    if (!is.null(cyclopsData$censorWeights)) {
        if (cyclopsData$modelType != 'fgr' && warnings) {
            warning(paste0("modelType = '", cyclopsData$modelType, "' does not use censorWeights. These weights will not be passed further."))
        }
        if (length(cyclopsData$censorWeights) != getNumberOfRows(cyclopsData)) {
            stop("Must provide a censorWeight for each data row")
        }
        if (any(cyclopsData$censorWeights < 0) || any(cyclopsData$censorWeights > 1)) {
            stop("Only weights between 0 and 1 are allowed for censorWeights")
        }
        .cyclopsSetCensorWeights(cyclopsData$cyclopsInterfacePtr, cyclopsData$censorWeights)
    }

    if (prior$useCrossValidation) {
        minCVData <- control$minCVData
        if (control$selectorType == "byRow" && minCVData > getNumberOfRows(cyclopsData)) {
            stop("Insufficient data count for cross validation")
        }
        if (control$selectorType == "byPid" && minCVData > getNumberOfStrata(cyclopsData)) {
            stop("Insufficient data count for cross validation")
        }

        fit <- .cyclopsRunCrossValidation(cyclopsData$cyclopsInterfacePtr)
    } else {
        fit <- .cyclopsFitModel(cyclopsData$cyclopsInterfacePtr)
    }

    if (fit$return_flag == "POOR_BLR_STEP" && control$convergenceType == "gradient") {

        if (warnings) {
            warning("BLR convergence criterion failed; coefficient may be infinite")
        }

        control$convergenceType <- "lange"
        return(fitCyclopsModel(cyclopsData = cyclopsData,
                               prior = prior,
                               control = control,
                               weights = weights,
                               forceNewObject = forceNewObject,
                               returnEstimates = returnEstimates,
                               startingCoefficients = startingCoefficients,
                               fixedCoefficients = fixedCoefficients,
                               computeDevice = computeDevice))
    }

    if (returnEstimates) {
        estimates <- .cyclopsLogModel(cyclopsData$cyclopsInterfacePtr)
        fit <- c(fit, estimates)
        fit$estimation <- as.data.frame(fit$estimation)
    }
    fit$call <- cl
    fit$cyclopsData <- cyclopsData
    fit$coefficientNames <- cyclopsData$coefficientNames
    if (!is.null(fixedCoefficients)) {
        fit$fixedCoefficients <- fixedCoefficients
    }
    fit$rowNames <- cyclopsData$rowNames
    fit$scale <- cyclopsData$scale
    fit$threads <- threads
    fit$seed <- control$seed

    if (prior$useCrossValidation) {
        fit$cvRepetitions <- control$cvRepetitions
    }

    class(fit) <- "cyclopsFit"
    return(fit)
}

.checkCovariates <- function(cyclopsData, covariates) {
    if (!is.null(covariates)) {
        saved <- covariates

        indices <- NULL

        if (inherits(covariates, "character")) {
            # Try to match names
            indices <- match(covariates, cyclopsData$coefficientNames)
            covariates <- getCovariateIds(cyclopsData)[indices]
        }

        if (!bit64::is.integer64(covariates)) {
            covariates <- bit64::as.integer64(covariates)
        }

        if (any(is.na(covariates))) {
            stop("Unable to match all covariates: ", paste(saved, collapse = ", "))
        }

        attr(covariates, "indices") <- indices
    }
    covariates
}

.checkData <- function(x) {
    # Check conditions
    if (missing(x) || is.null(x$cyclopsDataPtr) || !inherits(x$cyclopsDataPtr, "externalptr")) {
        stop("Improperly constructed cyclopsData object")
    }
    if (.isRcppPtrNull(x$cyclopsDataPtr)) {
        stop("Data object is no longer initialized")
    }
}

.checkInterface <- function(x, computeDevice = "native", forceNewObject = FALSE, testOnly = FALSE) {
    if (forceNewObject
        || is.null(x$cyclopsInterfacePtr)
        || !inherits(x$cyclopsInterfacePtr, "externalptr")
        || .isRcppPtrNull(x$cyclopsInterfacePtr)
        #|| .cyclopsGetComputeDevice(x$cyclopsInterfacePtr) != computeDevice TODO is this necessary?
    ) {

        if (testOnly == TRUE) {
            stop("Interface object is not initialized")
        }

        if (computeDevice != "native") {
            stopifnot(computeDevice %in% listGPUDevices())
        }

        # Build interface
        interface <- .cyclopsInitializeModel(x$cyclopsDataPtr, modelType = x$modelType, computeDevice, computeMLE = TRUE)
        # TODO Check for errors
        assign("cyclopsInterfacePtr", interface$interface, x)
    }
}



#' @title Extract model coefficients
#'
#' @description
#' \code{coef.cyclopsFit} extracts model coefficients from an Cyclops model fit object
#'
#' @param object    Cyclops model fit object
#' @param rescale   Boolean: rescale coefficients for unnormalized covariate values
#' @param ignoreConvergence Boolean: return coefficients even if fit object did not converge
#' @param ...       Other arguments
#'
#' @return Named numeric vector of model coefficients.
#'
#' @export
coef.cyclopsFit <- function(object, rescale = FALSE, ignoreConvergence = FALSE, ...) {

    if (object$return_flag != "SUCCESS" && !ignoreConvergence) {
        stop("Cyclops estimation is null; suspect that estimation did not converge.")
    }

    result <- object$estimation$estimate
    if (is.null(object$coefficientNames)) {
        names(result) <- object$estimation$column_label
        if ("0" %in% names(result)) {
            names(result)[which(names(result) == "0")] <- "(Intercept)"
        }
    } else {
        names(result) <- object$coefficientNames
    }

    if (!is.null(object$scale) && rescale) {
        result <- result * object$scale
    }
    result
}

#' @title Get hyperparameter
#'
#' @description
#' \code{getHyperParameter} returns the current hyper parameter in a Cyclops model fit object
#'
#' @param object    A Cyclops model fit object
#'
#' @template elaborateExample
#'
#' @export
getHyperParameter <- function(object) {
    if (inherits(object, "cyclopsFit")) {
        object$variance
    } else {
        NULL
    }
}

#' @title Extract log-likelihood
#'
#' @description
#' \code{logLik} returns the current log-likelihood of the fit in a Cyclops model fit object
#'
#' @param object    A Cyclops model fit object
#' @param ...       Additional arguments
#'
#' @template elaborateExample
#'
#' @export
logLik.cyclopsFit <- function(object, ...) {
    out <- object$log_likelihood
    attr(out, 'df') <- sum(!is.na(coefficients(object)))
    attr(out, 'nobs') <- getNumberOfRows(object$cyclopsData)
    class(out) <- 'logLik'
    out
}


#' @method print cyclopsFit
#' @title Print a Cyclops model fit object
#'
#' @description
#' \code{print.cyclopsFit} displays information about a Cyclops model fit object
#'
#' @param x    A Cyclops model fit object
#' @param show.call Logical: display last call to update the Cyclops model fit object
#' @param ...   Additional arguments
#'
#' @export
print.cyclopsFit <- function(x, show.call=TRUE ,...) {
    cat("Cyclops model fit object\n\n")

    if (show.call && !is.null(x$call)) {
        cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")
    }
    cat("           Model: ", x$cyclopsData$modelType, "\n", sep="")
    cat("           Prior: ", x$prior_info, "\n", sep="")
    cat("  Hyperparameter: ", paste(x$variance, collapse=" "), "\n", sep="")
    cat("     Return flag: ", x$return_flag, "\n", sep="")
    if (x$return_flag == "SUCCESS") {
        cat("Log likelikehood: ", x$log_likelihood, "\n", sep="")
        cat("       Log prior: ", x$log_prior, "\n", sep="")
    }
    invisible(x)
}

#' @title Create a Cyclops control object
#'
#' @description
#' \code{createControl} creates a Cyclops control object for use with \code{\link{fitCyclopsModel}}.
#'
#' @param maxIterations			Integer: maximum iterations of Cyclops to attempt before returning a failed-to-converge error
#' @param tolerance					Numeric: maximum relative change in convergence criterion from successive iterations to achieve convergence
#' @param convergenceType		String: name of convergence criterion to employ (described in more detail below)
#' @param cvType						String: name of cross validation search.
#' 													Option \code{"auto"} selects an auto-search following BBR.
#' 													Option \code{"grid"} selects a grid-search cross validation
#' @param fold							Numeric: Number of random folds to employ in cross validation
#' @param lowerLimit				Numeric: Lower prior variance limit for grid-search
#' @param upperLimit				Numeric: Upper prior variance limit for grid-search
#' @param gridSteps					Numeric: Number of steps in grid-search
#' @param cvRepetitions			Numeric: Number of repetitions of X-fold cross validation
#' @param minCVData					Numeric: Minimum number of data for cross validation
#' @param noiseLevel				String: level of Cyclops screen output (\code{"silent"}, \code{"quiet"}, \code{"noisy"})
#' @param threads               Numeric: Specify number of CPU threads to employ in cross-validation; default = 1 (auto = -1)
#' @param seed                  Numeric: Specify random number generator seed. A null value sets seed via \code{\link{Sys.time}}.
#' @param resetCoefficients     Logical: Reset all coefficients to 0 between model fits under cross-validation
#' @param startingVariance      Numeric: Starting variance for auto-search cross-validation; default = -1 (use estimate based on data)
#' @param useKKTSwindle Logical: Use the Karush-Kuhn-Tucker conditions to limit search
#' @param tuneSwindle    Numeric: Size multiplier for active set
#' @param selectorType  String: name of exchangeable sampling unit.
#'                              Option \code{"byPid"} selects entire strata.
#'                              Option \code{"byRow"} selects single rows.
#'                              If set to \code{"auto"}, \code{"byRow"} will be used for all models except conditional models where
#'                              the average number of rows per stratum is smaller than the number of strata.
#' @param initialBound          Numeric: Starting trust-region size
#' @param maxBoundCount         Numeric: Maximum number of tries to decrease initial trust-region size
#' @param algorithm             String: name of fitting algorithm to employ; default is `ccd`
#' @param doItAll               Currently unused
#' @param syncCV                Currently unused
#'
#' Todo: Describe convegence types
#'
#' @return
#' A Cyclops control object of class inheriting from \code{"cyclopsControl"} for use with \code{\link{fitCyclopsModel}}.
#'
#' @template elaborateExample
#'
#' @export
createControl <- function(maxIterations = 1000,
                          tolerance = 1E-6,
                          convergenceType = "gradient",
                          cvType = "auto",
                          fold = 10,
                          lowerLimit = 0.01,
                          upperLimit = 20.0,
                          gridSteps = 10,
                          cvRepetitions = 1,
                          minCVData = 100,
                          noiseLevel = "silent",
                          threads = 1,
                          seed = NULL,
                          resetCoefficients = FALSE,
                          startingVariance = -1,
                          useKKTSwindle = FALSE,
                          tuneSwindle = 10,
                          selectorType = "auto",
                          initialBound = 2.0,
                          maxBoundCount = 5,
                          algorithm = "ccd",
                          doItAll = TRUE,
                          syncCV = FALSE) {
    validCVNames = c("grid", "auto")
    stopifnot(cvType %in% validCVNames)

    validNLNames = c("silent", "quiet", "noisy")
    stopifnot(noiseLevel %in% validNLNames)
    stopifnot(threads == -1 || threads >= 1)
    stopifnot(startingVariance == -1 || startingVariance > 0)
    stopifnot(selectorType %in% c("auto","byPid", "byRow"))

    validAlgorithmNames = c("ccd", "mm")
    stopifnot(algorithm %in% validAlgorithmNames)

    structure(list(maxIterations = maxIterations,
                   tolerance = tolerance,
                   convergenceType = convergenceType,
                   autoSearch = (cvType == "auto"),
                   fold = fold,
                   lowerLimit = lowerLimit,
                   upperLimit = upperLimit,
                   gridSteps = gridSteps,
                   minCVData = minCVData,
                   cvRepetitions = cvRepetitions,
                   noiseLevel = noiseLevel,
                   threads = threads,
                   seed = seed,
                   resetCoefficients = resetCoefficients,
                   startingVariance = startingVariance,
                   useKKTSwindle = useKKTSwindle,
                   tuneSwindle = tuneSwindle,
                   selectorType = selectorType,
                   initialBound = initialBound,
                   maxBoundCount = maxBoundCount,
                   algorithm = algorithm,
                   doItAll = doItAll,
                   syncCV = syncCV),
              class = "cyclopsControl")
}

#' @title Create a Cyclops prior object
#'
#' @description
#' \code{createPrior} creates a Cyclops prior object for use with \code{\link{fitCyclopsModel}}.
#'
#' @param priorType     Character: specifies prior distribution.  See below for options
#' @param variance      Numeric: prior distribution variance
#' @param exclude       A vector of numbers or covariateId names to exclude from prior
#' @param graph         Child-to-parent mapping for a hierarchical prior
#' @param neighborhood  A list of first-order neighborhoods for a partially fused prior
#' @param useCrossValidation    Logical: Perform cross-validation to determine prior \code{variance}.
#' @param forceIntercept  Logical: Force intercept coefficient into prior
#'
#' @section Prior types:
#'
#' We specify all priors in terms of their variance parameters.
#' Similar fitting tools for regularized regression often parameterize the Laplace distribution
#' in terms of a rate \code{"lambda"} per observation.
#' See \code{"glmnet"}, for example.
#'
#' variance = 2 * / (nobs * lambda)^2 or lambda = sqrt(2 / variance) / nobs
#'
#' @template elaborateExample
#'
#' @return
#' A Cyclops prior object of class inheriting from \code{"cyclopsPrior"} for use with \code{fitCyclopsModel}.
#'
#' @export
createPrior <- function(priorType,
                        variance = 1,
                        exclude = c(),
                        graph = NULL,
                        neighborhood = NULL,
                        useCrossValidation = FALSE,
                        forceIntercept = FALSE) {
    validNames = c("none", "laplace","normal", "barupdate", "hierarchical", "jeffreys")
    stopifnot(priorType %in% validNames)
    if (!is.null(exclude)) {
        if (!inherits(exclude, "character") &&
                !inherits(exclude, "numeric") &&
                !inherits(exclude, "integer")
        ) {
            stop(cat("Unable to parse excluded covariates:"), exclude)
        }
    }

    if (length(priorType) != length(variance)) {
        stop("Prior types and variances have a dimensionality mismatch")
    }

    if (all(priorType == "none") && useCrossValidation) {
        stop("Cannot perform cross validation with a flat prior")
    }
    if (any(priorType == "barupdate") && useCrossValidation) {
        stop("Cannot perform cross valudation with BAR updates")
    }
    if (any(priorType == "hierarchical") && missing(graph)) {
        stop("Must provide a graph for a hierarchical prior")
    }
    if (!is.null(neighborhood)) {
        allNames <- unlist(neighborhood)
        if (!inherits(allNames, "character") &&
            !inherits(allNames, "numeric") &&
            !inherits(allNames, "integer")) {
            stop(cat("Unable to parse neighborhood covariates:"), allNames)
        }
    }
    structure(list(priorType = priorType, variance = variance, exclude = exclude,
                   graph = graph,
                   neighborhood = neighborhood,
                   useCrossValidation = useCrossValidation, forceIntercept = forceIntercept),
              class = "cyclopsPrior")
}

# .cyclopsSetCoefficients <- function(object, coefficients) {
#     .checkInterface(object$cyclopsData, testOnly = TRUE)
#
#     if (length(coefficients) != getNumberOfCovariates(object$cyclopsData)) {
#         stop("Must provide a value for each coefficient")
#     }
#
#     if (.cyclopsGetHasOffset(object$cyclopsData)) {
#         coefficients <- c(1.0, coefficients)
#     }
#
#     .cyclopsSetBeta(object$cyclopsData$cyclopsInterfacePtr, coefficients)
# }

#' @title Compute predictive log-likelihood from a Cyclops model fit
#'
#' @description
#' \code{getCyclopsPredictiveLogLikelihood} returns the log-likelihood of a subset of the data in a Cyclops model fit object.
#'
#' @param object    A Cyclops model fit object
#' @param weights   Numeric vector: vector of 0/1 identifying subset (=1) of rows from \code{object} to use in computing the log-likelihood
#' @return The predictive log-likelihood
#'
#' @keywords internal
getCyclopsPredictiveLogLikelihood <- function(object, weights) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)

    if (length(weights) != getNumberOfRows(object$cyclopsData)) {
        stop("Must provide a weight for each data row")
    }
    if (any(weights < 0)) {
        stop("Only non-negative weights are allowed")
    }

    if(!is.null(object$cyclopsData$sortOrder)) {
        weights <- weights[object$cyclopsData$sortOrder]
    }
    # TODO Remove code duplication with weights section of fitCyclopsModel

    .cyclopsGetNewPredictiveLogLikelihood(object$cyclopsData$cyclopsInterfacePtr, weights)
}

#' @title Get cross-validation information from a Cyclops model fit
#'
#' @description {getCrossValidationInfo} returns the predicted optimal cross-validation point and ordinate
#'
#' @param object A Cyclops model fit object
#'
#' @keywords internal
getCrossValidationInfo <- function(object) {
    info <- object$cross_validation

    if (is.na(info) || info == "") {
        stop("No cross-validation information is available")
    }

    values <- as.numeric(unlist(strsplit(info, " ")))
    list(ordinate = values[1],
         point = values[-1])
}

.setControl <- function(cyclopsInterfacePtr, control) {
    if (!missing(control)) {
        stopifnot(inherits(control, "cyclopsControl"))

        if (is.null(control$seed)) {
            control$seed <- as.integer(Sys.time())
        }

        if (is.null(control$algorithm) || is.na(control$algorithm)) { # Provide backwards compatibility
            control$algorithm <- "ccd"
        }

        .cyclopsSetControl(cyclopsInterfacePtr, control$maxIterations, control$tolerance,
                           control$convergenceType, control$autoSearch, control$fold,
                           (control$fold * control$cvRepetitions),
                           control$lowerLimit, control$upperLimit, control$gridSteps,
                           control$noiseLevel, control$threads, control$seed, control$resetCoefficients,
                           control$startingVariance, control$useKKTSwindle, control$tuneSwindle,
                           control$selectorType, control$initialBound, control$maxBoundCount,
                           control$algorithm, control$doItAll, control$syncCV
                          )
        return(control)
    }

    return(NULL)
}

#' @title Extract standard errors
#'
#' @description
#' \code{getSEs} extracts asymptotic standard errors for specific covariates from a Cyclops model fit object.
#'
#' @details This function first computes the (partial) Fisher information matrix for
#' just the requested covariates and then returns the square root of the diagonal elements of
#' the inverse of the Fisher information matrix.  These are the asymptotic standard errors
#' when all possible covariates are included.
#' When the requested covariates do not equate to all coefficients in the model,
#' then interpretation is more challenging.
#'
#' @param object    A Cyclops model fit object
#' @param covariates    Integer or string vector: list of covariates for which asymptotic standard errors are wanted
#'
#' @return Vector of standard error estimates
#'
#' @keywords internal
getSEs <- function(object, covariates) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    covariates <- .checkCovariates(object$cyclopsData, covariates)
    if (getNumberOfCovariates(object$cyclopsData) != length(covariates)) {
        warning("Asymptotic standard errors are only valid if computed for all covariates simultaneously")
    }

    fisherInformation <- .cyclopsGetFisherInformation(object$cyclopsData$cyclopsInterfacePtr, covariates)
    ses <- sqrt(diag(solve(fisherInformation)))
    names(ses) <- object$coefficientNames[as.integer(covariates)]
    ses
}

#' @title Run Bootstrap for Cyclops model parameter
#'
#' @param object    A fitted Cyclops model object
#' @param outFileName     Character: Output file name
#' @param treatmentId     Character: variable to output
#' @param replicates      Numeric: number of bootstrap samples
#'
#' @export
runBootstrap <- function(object, outFileName, treatmentId, replicates) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    bs <- .cyclopsRunBootstrap(object$cyclopsData$cyclopsInterfacePtr, outFileName, treatmentId, replicates)
    bs
}

#' @title Confidence intervals for Cyclops model parameters
#'
#' @description
#' \code{confinit.cyclopsFit} profiles the data likelihood to construct confidence intervals of
#' arbitrary level. Usually it only makes sense to do this for variables that have not been regularized.
#'
#' @param object    A fitted Cyclops model object
#' @param parm      A specification of which parameters require confidence intervals,
#'                  either a vector of numbers of covariateId names
#' @param level     Numeric: confidence level required
## @param control   A Cyclops \code{\link{control}} object
#' @param overrideNoRegularization   Logical: Enable confidence interval estimation for regularized parameters
#' @param includePenalty    Logical: Include regularized covariate penalty in profile
#' @param rescale   Boolean: rescale coefficients for unnormalized covariate values
#' @param ... Additional argument(s) for methods
#'
#' @return
#' A matrix with columns reporting lower and upper confidence limits for each parameter.
#' These columns are labelled as (1-level) / 2 and 1 - (1 - level) / 2 in percent
#' (by default 2.5 percent and 97.5 percent)
#'
#' @template elaborateExample
#'
#' @export
confint.cyclopsFit <- function(object, parm, level = 0.95, #control,
                               overrideNoRegularization = FALSE,
                               includePenalty = TRUE,
                               rescale = FALSE, ...) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    #.setControl(object$cyclopsData$cyclopsInterfacePtr, control)

    parm <- .checkCovariates(object$cyclopsData, parm)
    if (level < 0.01 || level > 0.99) {
        stop("level must be between 0 and 1")
    }
    threshold <- qchisq(level, df = 1) / 2
    threads <- object$threads

    if (!is.null(object$fixedCoefficients)) {
        if (any(object$fixedCoefficients[as.integer(parm)])) {
            stop("Cannot estimate confidence interval for a fixed coefficient")
        }
    }

    prof <- .cyclopsProfileModel(object$cyclopsData$cyclopsInterfacePtr, parm,
                                 threads, threshold,
                                 overrideNoRegularization,
                                 includePenalty)

    indices <- match(parm, getCovariateIds(object$cyclopsData))

    if (!is.null(object$scale) && rescale) {
        prof$lower <- prof$lower * object$scale[indices]
        prof$upper <- prof$upper * object$scale[indices]
    }
    prof <- as.matrix(as.data.frame(prof))
    rownames(prof) <- object$coefficientNames[indices]
    qs <- c((1 - level) / 2, 1 - (1 - level) / 2) * 100
    colnames(prof)[2:3] <- paste(sprintf("%.1f", qs), "%")

    # Change NaN to NA
    prof[which(is.nan(prof[, 2])), 2] <- NA
    prof[which(is.nan(prof[, 3])), 3] <- NA

    prof
}

.initAdaptiveProfile <- function(object, parm, bounds, includePenalty) {
    # If an MLE was found, let's not throw that bit of important information away:
    if (object$return_flag == "SUCCESS" &&
        coef(object)[as.character(parm)] > bounds[1] &&
        coef(object)[as.character(parm)] < bounds[2]) {
        profile <- tibble(point = coef(object)[as.character(parm)],
                          value = fixedGridProfileLogLikelihood(object, parm, coef(object)[as.character(parm)], includePenalty)$value)
    } else {
        profile <- tibble()
    }
}

#' @title Profile likelihood for Cyclops model parameters
#'
#' @description
#' \code{getCyclopsProfileLogLikelihood} evaluates the profile likelihood at a grid of parameter values.
#'
#' @param object    Fitted Cyclops model object
#' @param parm      Specification of which parameter requires profiling,
#'                  either a vector of numbers of covariateId names
#' @param x         Vector of values of the parameter
#' @param bounds    Pair of values to bound adaptive profiling
#' @param tolerance Absolute tolerance allowed for adaptive profiling
#' @param initialGridSize Initial grid size for adaptive profiling
#' @param includePenalty    Logical: Include regularized covariate penalty in profile
#'
#' @return
#' A data frame containing the profile log likelihood. Returns NULL when the adaptive profiling fails
#' to converge.
#'
#' @export
getCyclopsProfileLogLikelihood <- function(object,
                                           parm,
                                           x = NULL,
                                           bounds = NULL,
                                           tolerance = 1E-3,
                                           initialGridSize = 10,
                                           includePenalty = TRUE) {
    maxResets <- 10
    if (!xor(is.null(x), is.null(bounds))) {
        stop("Must provide either `x` or `bounds`, but not both.")
    }

    if (!is.null(bounds)) { # Adaptive profiling using recursive calls

        if (length(bounds) != 2 || bounds[1] >= bounds[2]) {
            stop("Must provide bounds[1] < bounds[2]")
        }
        profile <- .initAdaptiveProfile(object, parm, bounds, includePenalty)

        # Start with sparse grid:
        grid <- seq(bounds[1], bounds[2], length.out = initialGridSize)

        # Iterate until stopping criteria met:
        priorMaxMaxError <- Inf
        resetsPerformed <- 0
        while (length(grid) != 0) {
            ll <- fixedGridProfileLogLikelihood(object, parm, grid, includePenalty)
            profile <- bind_rows(profile, ll) %>% arrange(.data$point)
            invalid <- is.nan(profile$value) | is.infinite(profile$value)
            if (any(invalid)) {
                if (all(invalid)) {
                    warning("Failing to compute likelihood at entire initial grid.")
                    return(NULL)
                }

                start <- min(which(!invalid))
                end <- max(which(!invalid))
                if (start == end) {
                    warning("Failing to compute likelihood at entire grid except one. Giving up")
                    return(NULL)
                }
                profile <- profile[start:end, ]
                invalid <- invalid[start:end]
                if (any(invalid)) {
                    warning("Failing to compute likelihood in non-extreme regions. Giving up.")
                    return(NULL)
                }
                warning("Failing to compute likelihood at extremes. Truncating bounds.")
            }

            deltaX <- profile$point[2:nrow(profile)] - profile$point[1:(nrow(profile) - 1)]
            deltaY <- profile$value[2:nrow(profile)] - profile$value[1:(nrow(profile) - 1)]
            slopes <- deltaY / deltaX

            if (resetsPerformed < maxResets && !all(slopes[2:length(slopes)] < slopes[1:(length(slopes)-1)])) {
                warning("Coefficient drift detected. Resetting Cyclops object and recomputing all likelihood values computed so far.")
                grid <- profile$point
                profile <- tibble()
                interface <- .cyclopsInitializeModel(object$cyclopsData$cyclopsDataPtr, modelType = object$cyclopsData$modelType, "native", computeMLE = TRUE)
                assign("cyclopsInterfacePtr", interface$interface, object)
                resetsPerformed <- resetsPerformed + 1
                next
            }

            # Compute where prior and posterior slopes intersect
            slopes <- c(slopes[1] + (slopes[2] - slopes[3]),
                        slopes,
                        slopes[length(slopes)] - (slopes[length(slopes) - 1] - slopes[length(slopes)]))

            interceptX <- (profile$value[2:nrow(profile)] -
                               profile$point[2:nrow(profile)] * slopes[3:length(slopes)] -
                               profile$value[1:(nrow(profile) - 1)] +
                               profile$point[1:(nrow(profile) - 1)] * slopes[1:(length(slopes) - 2)]) /
                (slopes[1:(length(slopes) - 2)] - slopes[3:length(slopes)])

            # Compute absolute difference between linear interpolation and worst case scenario (which is at the intercept):
            maxError <- abs((profile$value[1:(nrow(profile) - 1)] + (interceptX - profile$point[1:(nrow(profile) - 1)]) * slopes[1:(length(slopes) - 2)]) -
                                (profile$value[1:(nrow(profile) - 1)] + (interceptX - profile$point[1:(nrow(profile) - 1)]) * slopes[2:(length(slopes) - 1)]))

            maxMaxError <- max(maxError, na.rm = TRUE)
            if (is.na(maxMaxError) || maxMaxError > priorMaxMaxError) {
                warning("Failing to converge when using adaptive profiling.")
                return(NULL)
            }
            priorMaxMaxError <- maxMaxError

            exceed <- which(maxError > tolerance)
            grid <- (profile$point[exceed] + profile$point[exceed + 1]) / 2
        }
    } else { # Use x
        profile <- fixedGridProfileLogLikelihood(object, parm, x, includePenalty)
    }

    return(profile)
}

fixedGridProfileLogLikelihood <- function(object, parm, x, includePenalty) {

    .checkInterface(object$cyclopsData, testOnly = TRUE)
    parm <- .checkCovariates(object$cyclopsData, parm)
    threads <- object$threads

    if (getNumberOfCovariates(object$cyclopsData) == 1 || length(x) == 1) {
        grid <- .cyclopsGetProfileLikelihood(object$cyclopsData$cyclopsInterfacePtr, parm, x,
                                             threads, includePenalty)
    } else {
        # Partition sequence
        y <- sort(x)
        midPt <- floor(length(x) / 2)
        lower <- y[midPt:1]
        upper <- y[(midPt + 1):length(x)]

        # Execute: TODO chunk and repeat until ill-conditioned
        gridLower <- .cyclopsGetProfileLikelihood(object$cyclopsData$cyclopsInterfacePtr, parm, lower,
                                                  threads, includePenalty)
        gridUpper <- .cyclopsGetProfileLikelihood(object$cyclopsData$cyclopsInterfacePtr, parm, upper,
                                                  threads, includePenalty)
        # Merge
        grid <- rbind(gridLower, gridUpper)
        grid <- grid[order(grid$point),]
        rownames(grid) <- NULL
    }

    return(grid)
}

#' @title Asymptotic confidence intervals for a fitted Cyclops model object
#'
#' @description
#' \code{aconfinit} constructs confidence intervals of
#' arbitrary level using asymptotic standard error estimates.
#'
#' @param object    A fitted Cyclops model object
#' @param parm      A specification of which parameters require confidence intervals,
#'                  either a vector of numbers of covariateId names
#' @param level     Numeric: confidence level required
#' @param control   A \code{"cyclopsControl"} object constructed by \code{\link{createControl}}
#' @param overrideNoRegularization   Logical: Enable confidence interval estimation for regularized parameters
#' @param ... Additional argument(s) for methods
#'
#' @return
#' A matrix with columns reporting lower and upper confidence limits for each parameter.
#' These columns are labelled as (1-level) / 2 and 1 - (1 - level) / 2 in %
#' (by default 2.5% and 97.5%)
#'
#' @keywords internal
#' @export
aconfint <- function(object, parm, level = 0.95, control,
                     overrideNoRegularization = FALSE, ...) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    .setControl(object$cyclopsData$cyclopsInterfacePtr, control)
    cf <- coef(object)
    if (missing(parm)) {
        parm <- names(cf)
    }
    #parm <- .checkCovariates(object$cyclopsData, parm)
    if (level < 0.01 || level > 0.99) {
        stop("level must be between 0 and 1")
    }
    a <- (1 - level) / 2
    a <- c(a, 1 - a)
    pct <- paste(sprintf("%.1f", a * 100), "%")
    fac <- qnorm(a)
    ci <- array(NA, dim = c(length(parm), 2L), dimnames = list(parm, pct))
    ses <- sqrt(diag(vcov(object)))[parm]
    ci[] <- cf[parm] + ses %o% fac
    ci
}

#' @title Calculate variance-covariance matrix for a fitted Cyclops model object
#'
#' @description
#' \code{vcov.cyclopsFit} returns the variance-covariance matrix for all covariates of a Cyclops model object
#'
#' @param object    A fitted Cyclops model object
#' @param control    A \code{"cyclopsControl"} object constructed by \code{\link{createControl}}
#' @param overrideNoRegularization   Logical: Enable variance-covariance estimation for regularized parameters
#' @param ... Additional argument(s) for methods
#'
#' @return
#' A matrix of the estimates covariances between all covariate estimates.
#'
#' @export
vcov.cyclopsFit <- function(object, control, overrideNoRegularization = FALSE, ...) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    .setControl(object$cyclopsData$cyclopsInterfacePtr, control)
    fisherInformation <- .cyclopsGetFisherInformation(object$cyclopsData$cyclopsInterfacePtr, NULL)
    vcov <- solve(fisherInformation)
    if (!is.null(object$coefficientNames)) {
        rownames(vcov) <- object$coefficientNames
        colnames(vcov) <- object$coefficientNames
    }
    vcov
}

#' @title Convert to Cyclops Prior Variance
#'
#' @description
#' \code{convertToCyclopsVariance} converts the regularization parameter \code{lambda}
#' from \code{glmnet} into a prior variance.
#'
#' @param lambda    Regularization parameter from \code{glmnet}
#' @param nobs      Number of observation rows in dataset
#'
#' @return Prior variance under a Laplace() prior
#'
#' @keywords internal
convertToCyclopsVariance <- function(lambda, nobs) {
    2 / (nobs * lambda)^2
}

#' @title Convert to glmnet regularization parameter
#'
#' @description
#' \code{convertToGlmnetLambda} converts a prior variance
#' from \code{Cyclops} into the regularization parameter \code{lambda}.
#'
#' @param variance  Prior variance
#' @param nobs      Number of observation rows in dataset
#'
#' @return \code{lambda}
#'
#' @keywords internal
convertToGlmnetLambda <- function(variance, nobs) {
    sqrt(2 / variance) / nobs
}

.makeHierarchyGraph <- function(cyclopsData, graph) {
    nTypes <- getNumberOfTypes(cyclopsData)
    if (nTypes < 2 || graph != "type") stop("Only multitype hierarchies are currently supported")
    hasOffset <- .cyclopsGetHasOffset(cyclopsData)
    if (hasOffset) stop("Hierarchies with offset covariates are currently not supported")

    # Build Multitype hierarchy
    nChild <- getNumberOfCovariates(cyclopsData)
    nParents <- nChild / nTypes

    graph <-  lapply(0:(nParents - 1), function(n, types) { 0:(nTypes - 1) + n * nTypes }, types = nTypes)
    graph
}

.getNumberOfRepetitions <- function(nrows) {
    top <- 1E2
    factor <- 0.5
    pmax(pmin(
        ceiling(top / nrows^(factor))
        , 10), 1)
}
