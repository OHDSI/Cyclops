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

    fitter <- .fitter(control)
    if (fitter == "legacy" && cyclopsData$modelType == "cox_counting") {
        stop("The legacy fitter does not handle start times; use createControl(fitter = \"default\")")
    }
    .checkInterface(cyclopsData, computeDevice =  computeDevice, forceNewObject = forceNewObject,
                    fitter = fitter)

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
        if (control$selectorType == "byRow" && .cyclopsGetNumberOfSubjects(cyclopsData) > 0) {
            control$selectorType <- "bySubject"
        }
        if (prior$useCrossValidation && control$noiseLevel != "silent") {
            writeLines(paste("Using cross-validation selector type", control$selectorType))
        }
    }

    if (control$selectorType == "bySubject" && .cyclopsGetNumberOfSubjects(cyclopsData) == 0) {
        stop("selectorType = \"bySubject\" requires subjects from cluster() or outcomes$subjectId")
    }

    if (control$cvRepetitions == "auto") {
        control$cvRepetitions <- .getNumberOfRepetitions(getNumberOfRows(cyclopsData))
    }

    control <- .setControl(cyclopsData$cyclopsInterfacePtr, control, cyclopsData)
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

    if (!is.null(fixedCoefficients) || isTRUE(cyclopsData$hasFixedCoefficients)) {
        if (is.null(fixedCoefficients)) { # release the coefficients an earlier fit on this interface fixed
            fixedCoefficients <- rep(FALSE, getNumberOfCovariates(cyclopsData))
        }
        if (length(fixedCoefficients) != getNumberOfCovariates(cyclopsData)) {
            stop("Must provide a boolean for each coefficient")
        }

        offset <- ifelse(.cyclopsGetHasOffset(cyclopsData), 1, 0)
        for (i in 1:length(fixedCoefficients)) {
            .cyclopsSetFixedBeta(cyclopsData$cyclopsInterfacePtr, offset + i, fixedCoefficients[i] == TRUE)
        }
        assign("hasFixedCoefficients", any(fixedCoefficients), cyclopsData)
    }

    if (.cyclopsGetHasIntercept(cyclopsData) &&
        is.null(prior$fitHook) &&
        is.null(startingCoefficients)) {
        yMean <- mean(getYVector(cyclopsData))
        if (cyclopsData$modelType == "lr") {
            interceptValue <- log(yMean / (1 - yMean))
        } else if (cyclopsData$modelType == "pr") {
            interceptValue <- log(yMean)
        } else if (cyclopsData$modelType == "ls") {
            interceptValue <- yMean
        }
        betas <- c(interceptValue, rep(0.0, getNumberOfCovariates(cyclopsData) - 1))
        if (.cyclopsGetHasOffset(cyclopsData)) {
            betas <- c(1.0, betas)
        }
        .cyclopsSetBeta(cyclopsData$cyclopsInterfacePtr, betas)
        .cyclopsSetStartingBeta(cyclopsData$cyclopsInterfacePtr, betas)
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
        if (control$selectorType == "bySubject" && minCVData > .cyclopsGetNumberOfSubjects(cyclopsData)) {
            stop("Insufficient data count for cross validation")
        }

        fit <- .cyclopsRunCrossValidation(cyclopsData$cyclopsInterfacePtr)
        if (warnings && !is.null(fit$cv_failed_folds) && fit$cv_failed_folds > 0) {
            warning(fit$cv_failed_folds, " cross-validation fold fits failed and were left out of the mean ",
                    "predictive log-likelihood at their variance")
        }
    } else {
        fit <- .cyclopsFitModel(cyclopsData$cyclopsInterfacePtr)
    }

    if (fit$return_flag == "POOR_BLR_STEP" && control$convergenceType == "gradient") {

        if (warnings) {
            warning("BLR convergence criterion failed; coefficient may be infinite")
        }

        control$convergenceType <- "lange"
        if (prior$useCrossValidation) {
            # Refit only the final model at the selected variance; fold fits already fell back to "lange"
            cvFit <- fit
            prior$useCrossValidation <- FALSE
            prior$variance <- cvFit$variance
            fit <- fitCyclopsModel(cyclopsData = cyclopsData,
                                   prior = prior,
                                   control = control,
                                   weights = weights,
                                   forceNewObject = forceNewObject,
                                   returnEstimates = returnEstimates,
                                   startingCoefficients = startingCoefficients,
                                   fixedCoefficients = fixedCoefficients,
                                   warnings = warnings,
                                   computeDevice = computeDevice)
            fit$cross_validation <- cvFit$cross_validation
            fit$cv_failed_folds <- cvFit$cv_failed_folds
            fit$cvRepetitions <- control$cvRepetitions
            return(fit)
        }
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

#' @title Get full path to installed Cyclops library
#'
#' @description
#' \code{cyclopsLibraryFileName} returns the full path to the installed Cyclops .so/.dll file
#'
#' @return String
cyclopsLibraryFileName <- function() {
    normalizePath(system.file("libs", .Platform$r_arch,
                              paste0("Cyclops", .Platform$dynlib.ext),
                              package = "Cyclops"))
}

#' @title Place Cyclops model fit object in a persistent cache
#'
#' @description
#' \code{cacheCyclopsModelForJava} places a Cyclops model fit object into a persistent cache,
#' so that the object may be used across language-barriers.
#'
#' @param object    Cyclops model fit object
#'
#' @return Integer index of object in persistent cache
#'
#' @export
cacheCyclopsModelForJava <- function(object) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    instance <- attr(object, "jniCache")

    if (is.null(instance)) {
        instance <- .cyclopsCacheForJava(object$cyclopsData$cyclopsInterfacePtr)
        bit::setattr(object, "jniCache", instance)
    }

    return(instance)
}

#' @title Clear the Cyclops persistent cache
#'
#' @description
#' \code{clearCyclopsModelCache} clears the persistent cache holding
#' Cyclops model fit objects that may be used across language-barriers.
#'
#' @export
clearCyclopsModelCache <- function() {
    stop("Not yet implemented")
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

.fitter <- function(control) { # controls created before fitter existed lack it
    if (is.null(control$fitter)) "default" else control$fitter
}

.dataFitter <- function(x) { # the fitter that built the interface of a cyclopsData object
    if (is.null(x$fitter)) "default" else x$fitter
}

.checkInterface <- function(x, computeDevice = "native", forceNewObject = FALSE, testOnly = FALSE,
                            fitter = NULL) {
    if (forceNewObject
        || is.null(x$cyclopsInterfacePtr)
        || !inherits(x$cyclopsInterfacePtr, "externalptr")
        || .isRcppPtrNull(x$cyclopsInterfacePtr)
        || (!is.null(fitter) && !identical(.dataFitter(x), fitter))
        #|| .cyclopsGetComputeDevice(x$cyclopsInterfacePtr) != computeDevice TODO is this necessary?
    ) {

        if (testOnly == TRUE) {
            stop("Interface object is not initialized")
        }

        if (computeDevice != "native") {
            stopifnot(computeDevice %in% listGPUDevices())
        }

        # Build interface
        if (is.null(fitter)) {
            fitter <- "default"
        }
        interface <- .cyclopsInitializeModel(x$cyclopsDataPtr, modelType = x$modelType, computeDevice,
                                             computeMLE = TRUE, fitter = fitter)
        # TODO Check for errors
        assign("cyclopsInterfacePtr", interface$interface, x)
        assign("fitter", fitter, x)
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

#' @title Extract gradient
#'
#' @description
#' \code{gradient} returns the current gradient wrt the regression parameters of
#' the log-likelihood of the fit in a Cyclops model fit object
#'
#' @param object    A Cyclops model fit object
#'
#' @export
gradient <- function(object) {

    .checkInterface(object$cyclopsData, testOnly = TRUE)
    gradient <- .cyclopsGetLogLikelihoodGradient(object$interface)
    names(gradient) <- names(coef(object))

    return(gradient)
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
#' @param maxIterations			Integer: maximum iterations of Cyclops to attempt before returning a failed-to-converge error.
#'                              \code{NULL} (default) means 1000, or 5000 with \code{convergenceType = "kkt"}
#' @param tolerance					Numeric: maximum relative change in convergence criterion from successive iterations to achieve
#'                              convergence. \code{NULL} (default) means 1e-6, or 1e-4 with \code{convergenceType = "kkt"},
#'                              which measures it in standard-error units
#' @param convergenceType		String: name of convergence criterion to employ (described in more detail below).
#'                              \code{NULL} (default) means \code{"gradient"}, or \code{"kkt"} for logistic regression with
#'                              \code{fitter = "super"}
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
#' @param threads               Numeric: Specify number of CPU threads to employ in cross-validation, and within a fit
#'                              for Cox models (building risk-set indices and gradients of coefficients at zero,
#'                              with results identical to one thread); default = 1 (auto = -1)
#' @param seed                  Numeric: Specify random number generator seed. A null value sets seed via \code{\link{Sys.time}}.
#' @param resetCoefficients     Logical: start every cross-validation fold fit from the starting coefficients.
#'                              With \code{FALSE} (default), \code{cvWarmStart} defaults to \code{TRUE}, so each
#'                              fold fit starts from that fold's solution at the nearest hyperparameter already
#'                              tried and results do not depend on \code{threads}. Under \code{fitter = "legacy"},
#'                              each model copy instead continues from the last fold fit it ran, as in Cyclops 3.7
#' @param startingVariance      Numeric: Starting variance for auto-search cross-validation; default = -1 (use estimate based on data)
#' @param useKKTSwindle Logical: Use the Karush-Kuhn-Tucker conditions to limit search
#' @param tuneSwindle    Numeric: Size multiplier for active set
#' @param selectorType  String: name of exchangeable sampling unit for cross-validation and bootstrap.
#'                              Option \code{"byPid"} selects entire strata.
#'                              Option \code{"byRow"} selects single rows.
#'                              Option \code{"bySubject"} selects all rows of a subject (Cox models with
#'                              subjects from \code{cluster()} or \code{outcomes$subjectId}).
#'                              If set to \code{"auto"}, \code{"byRow"} will be used for all models except conditional models where
#'                              the average number of rows per stratum is smaller than the number of strata;
#'                              \code{"bySubject"} replaces \code{"byRow"} when the data have subjects.
#' @param initialBound          Numeric: Starting trust-region size
#' @param maxBoundCount         Numeric: Maximum number of tries to decrease initial trust-region size
#' @param algorithm             String: name of fitting algorithm to employ; default is `ccd`
#' @param fitter                String: which fitting code to run. \code{"default"} runs the current code.
#'                              \code{"legacy"} runs the code of Cyclops 3.7, to reproduce earlier results: for
#'                              logistic regression (\code{"lr"}), the grouped models (\code{"cpr"}, \code{"sccs"},
#'                              \code{"clr"}, \code{"clr_efron"}, \code{"clr_exact"}), \code{"cox"} models on the
#'                              CPU, profile likelihoods and confidence intervals. Bug fixes apply to every fitter,
#'                              except that \code{"legacy"} keeps the earlier cross-validation score of
#'                              \code{"lr"} models (held-out cases only). \code{"cox_counting"} models need another
#'                              fitter. \code{"super"} runs the code of \code{"default"} and changes how \code{"lr"}
#'                              models are fit through the defaults of \code{convergenceType} and the
#'                              arguments below: the \code{"kkt"} convergence criterion, cross-validation fold
#'                              fits to a looser \code{"kkt"} tolerance that start from each fold's previous
#'                              solution, an auto-search that steps down first and refines its result,
#'                              active-set cycling for 1000 or more covariates, and adaptive intercept steps
#' @param cvConvergenceType     String: convergence criterion of the cross-validation fold fits only.
#'                              \code{NULL} (default) means that of the final fit, or \code{"kkt"} for logistic
#'                              regression with \code{fitter = "super"}
#' @param cvTolerance           Numeric: tolerance of the cross-validation fold fits only. \code{NULL} (default)
#'                              means 3e-3 with \code{cvConvergenceType = "kkt"}, 1e-6 with another
#'                              \code{cvConvergenceType}, and \code{tolerance} without one
#' @param cvMaxIterations       Integer: maximum iterations of the cross-validation fold fits only.
#'                              \code{NULL} (default) means 5000 with \code{cvConvergenceType = "kkt"}, 1000 with
#'                              another \code{cvConvergenceType}, and \code{maxIterations} without one
#' @param cvMinIterations       Integer: minimum iterations of a cross-validation fold fit when
#'                              \code{cvConvergenceType}, \code{cvTolerance} or \code{cvMaxIterations} applies;
#'                              \code{NULL} (default) means 5
#' @param cvSearchDirection     String: first step of the auto-search from \code{startingVariance}: \code{"up"},
#'                              \code{"down"}, or \code{"auto"} (up below a variance of 1 and down above, as in
#'                              BBR). \code{NULL} (default) means \code{"down"} for logistic regression with
#'                              \code{fitter = "super"} and \code{"auto"} otherwise
#' @param cvRefine              Logical: after the auto-search stops, evaluate its estimate and refine it by
#'                              quadratic fits to the best point and its two neighbours, until the step is under
#'                              15% (at most three more evaluations). \code{NULL} (default) means \code{TRUE}
#'                              for logistic regression with \code{fitter = "super"} and \code{FALSE} otherwise
#' @param cvWarmStart           Logical: start each cross-validation fold fit from that fold's solution at the
#'                              nearest hyperparameter already tried; results do not depend on \code{threads}.
#'                              \code{NULL} (default) means \code{TRUE} for logistic regression with
#'                              \code{fitter = "super"}, and otherwise \code{TRUE} with \code{resetCoefficients =
#'                              FALSE} except under \code{fitter = "legacy"}
#' @param activeSetEvery        Integer: active-set cycling, which needs \code{convergenceType = "kkt"} or
#'                              \code{cvConvergenceType = "kkt"}. Cycles between full sweeps skip coordinates
#'                              that did not move, with a full sweep at least every \code{activeSetEvery}
#'                              cycles; 0 turns it off. \code{NULL} (default) means 10 for logistic regression
#'                              with \code{fitter = "super"} and at least 1000 covariates, and 0 otherwise
#' @param activeSetThreshold    Numeric: with active-set cycling, cycles between full sweeps also skip
#'                              coordinates whose last step, in standard-error units, was below this fraction of
#'                              the largest step of the last full sweep; with 0, they skip only coordinates with
#'                              a Laplace prior that are 0 and did not move. \code{NULL} (default) means 0.01
#' @param adaptiveIntercept     Logical: for logistic regression with an unpenalized intercept, which needs
#'                              \code{convergenceType = "kkt"} or \code{cvConvergenceType = "kkt"}: take an extra
#'                              Newton step on the intercept during a cycle whenever its predicted gain per row
#'                              of work reaches a quarter of that of the coordinate steps in the previous cycle
#'                              and its gradient, in standard-error units, exceeds the tolerance. With many
#'                              covariates the intercept otherwise lags behind them. \code{NULL} (default) means
#'                              \code{TRUE} for logistic regression with \code{fitter = "super"} and \code{FALSE}
#'                              otherwise
#' @param doItAll               Currently unused
#' @param syncCV                Currently unused
#'
#' Todo: Describe convegence types
#'
#' Convergence type \code{"kkt"} stops when the largest minimum-norm subgradient of the
#' log posterior over a cycle, in standard-error units, is below \code{tolerance} for two full cycles in a row
#' and then at the point returned.
#'
#' @return
#' A Cyclops control object of class inheriting from \code{"cyclopsControl"} for use with \code{\link{fitCyclopsModel}}.
#'
#' @template elaborateExample
#'
#' @export
createControl <- function(maxIterations = NULL,
                          tolerance = NULL,
                          convergenceType = NULL,
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
                          fitter = "default",
                          cvConvergenceType = NULL,
                          cvTolerance = NULL,
                          cvMaxIterations = NULL,
                          cvMinIterations = NULL,
                          cvSearchDirection = NULL,
                          cvRefine = NULL,
                          cvWarmStart = NULL,
                          activeSetEvery = NULL,
                          activeSetThreshold = NULL,
                          adaptiveIntercept = NULL,
                          doItAll = TRUE,
                          syncCV = FALSE) {
    validCVNames = c("grid", "auto")
    stopifnot(cvType %in% validCVNames)

    validNLNames = c("silent", "quiet", "noisy")
    stopifnot(noiseLevel %in% validNLNames)
    stopifnot(threads == -1 || threads >= 1)
    stopifnot(startingVariance == -1 || startingVariance > 0)
    stopifnot(selectorType %in% c("auto","byPid", "byRow", "bySubject"))

    validAlgorithmNames = c("ccd", "mm")
    stopifnot(algorithm %in% validAlgorithmNames)
    stopifnot(fitter %in% c("legacy", "default", "super"))
    validConvergenceNames <- c("gradient", "lange", "mittal", "zhang", "onestep", "kkt")
    stopifnot(is.null(convergenceType) || convergenceType %in% validConvergenceNames)
    stopifnot(is.null(tolerance) || tolerance >= 0) # 0 runs maxIterations cycles
    stopifnot(is.null(maxIterations) || maxIterations >= 1)
    stopifnot(is.null(cvConvergenceType) || cvConvergenceType %in% validConvergenceNames)
    stopifnot(is.null(cvTolerance) || cvTolerance > 0)
    stopifnot(is.null(cvMaxIterations) || cvMaxIterations >= 1)
    stopifnot(is.null(cvMinIterations) || cvMinIterations >= 0)
    stopifnot(is.null(cvSearchDirection) || cvSearchDirection %in% c("auto", "up", "down"))
    stopifnot(is.null(cvRefine) || is.logical(cvRefine))
    stopifnot(is.null(cvWarmStart) || is.logical(cvWarmStart))
    stopifnot(is.null(activeSetEvery) || activeSetEvery == 0 || activeSetEvery > 1)
    stopifnot(is.null(activeSetThreshold) || (activeSetThreshold >= 0 && activeSetThreshold < 1))
    stopifnot(is.null(adaptiveIntercept) || is.logical(adaptiveIntercept))

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
                   fitter = fitter,
                   cvConvergenceType = cvConvergenceType,
                   cvTolerance = cvTolerance,
                   cvMaxIterations = cvMaxIterations,
                   cvMinIterations = cvMinIterations,
                   cvSearchDirection = cvSearchDirection,
                   cvRefine = cvRefine,
                   cvWarmStart = cvWarmStart,
                   activeSetEvery = activeSetEvery,
                   activeSetThreshold = activeSetThreshold,
                   adaptiveIntercept = adaptiveIntercept,
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

# Settings that createControl() leaves to the fitter. fitter = "super" changes logistic regression only: the
# "kkt" convergence criterion, cross-validation fold fits by "kkt" at a looser tolerance, a search that steps
# down first and refines its result, per-fold warm starts, active-set cycling for models with many covariates
# and adaptive intercept steps. A setting given to createControl() wins.
.fitterSettings <- function(control, modelType = NULL, nCovariates = NULL) {
    super <- identical(.fitter(control), "super") && identical(modelType, "lr")
    if (super) {
        list(convergenceType = "kkt", cvConvergenceType = "kkt", cvMinIterations = 5,
             cvSearchDirection = "down", cvRefine = TRUE, cvWarmStart = TRUE,
             activeSetEvery = if (!is.null(nCovariates) && nCovariates >= .activeSetMinCovariates) 10 else 0,
             activeSetThreshold = 0.01, adaptiveIntercept = TRUE)
    } else { # warm starts make fold fits without resetCoefficients independent of threads
        warmStart <- !isTRUE(control$resetCoefficients) && !identical(.fitter(control), "legacy")
        list(convergenceType = "gradient", cvConvergenceType = NULL, cvMinIterations = 5,
             cvSearchDirection = "auto", cvRefine = FALSE, cvWarmStart = warmStart,
             activeSetEvery = 0, activeSetThreshold = 0.01, adaptiveIntercept = FALSE)
    }
}

# Active-set cycling pays off only with many covariates (it lost with a support of about 100)
.activeSetMinCovariates <- 1000

.resolveControl <- function(control, modelType = NULL, nCovariates = NULL) {
    settings <- .fitterSettings(control, modelType, nCovariates)
    if (identical(.fitter(control), "super") && identical(modelType, "lr") &&
        !is.null(control$tolerance) && is.null(control$convergenceType)) {
        warning("tolerance is ignored without convergenceType under fitter = \"super\": ",
                "its \"kkt\" criterion measures the tolerance in standard-error units", call. = FALSE)
        control["tolerance"] <- list(NULL)
    }
    filled <- character(0)
    for (name in names(settings)) {
        if (is.null(control[[name]])) {
            control[name] <- list(settings[[name]])
            filled <- c(filled, name)
        }
    }
    kkt <- control$convergenceType == "kkt" # in standard-error units; an honest rule needs more cycles
    if (is.null(control$tolerance)) {
        control$tolerance <- if (kkt) 1E-4 else 1E-6
    }
    if (is.null(control$maxIterations)) {
        control$maxIterations <- if (kkt) 5000 else 1000
    }
    if (!is.null(control$cvConvergenceType)) { # fold fits with their own rule get its defaults; kkt folds stop earlier
        cvKkt <- control$cvConvergenceType == "kkt"
        if (is.null(control$cvTolerance)) {
            control$cvTolerance <- if (cvKkt) 3E-3 else 1E-6
        }
        if (is.null(control$cvMaxIterations)) {
            control$cvMaxIterations <- if (cvKkt) 5000 else 1000
        }
    }
    if (control$convergenceType != "kkt" && !identical(control$cvConvergenceType, "kkt")) {
        # the fitter's settings that need "kkt" lapse; given ones are an error
        if ("activeSetEvery" %in% filled) control$activeSetEvery <- 0
        if ("adaptiveIntercept" %in% filled) control$adaptiveIntercept <- FALSE
        if (control$activeSetEvery != 0) {
            stop("activeSetEvery needs convergenceType = \"kkt\" or cvConvergenceType = \"kkt\"", call. = FALSE)
        }
        if (control$adaptiveIntercept) {
            stop("adaptiveIntercept needs convergenceType = \"kkt\" or cvConvergenceType = \"kkt\"", call. = FALSE)
        }
    }
    control
}

# Cross-validation options of a resolved control
.cvControlOptions <- function(control) {
    valueOr <- function(value, default) if (is.null(value)) default else value
    list(searchDirection = control$cvSearchDirection,
         refine = control$cvRefine,
         convergenceType = valueOr(control$cvConvergenceType, ""),
         tolerance = valueOr(control$cvTolerance, -1),
         maxIterations = as.integer(valueOr(control$cvMaxIterations, -1)),
         minIterations = as.integer(control$cvMinIterations),
         warmStart = control$cvWarmStart)
}

# Mode-finding options of a resolved control
.modeControlOptions <- function(control) {
    list(activeSetEvery = as.integer(control$activeSetEvery),
         activeSetThreshold = as.numeric(control$activeSetThreshold),
         adaptiveIntercept = as.logical(control$adaptiveIntercept))
}

.setControl <- function(cyclopsInterfacePtr, control, cyclopsData = NULL) {
    if (!missing(control)) {
        stopifnot(inherits(control, "cyclopsControl"))
        control <- .resolveControl(control,
                                   modelType = if (is.null(cyclopsData)) NULL else cyclopsData$modelType,
                                   nCovariates = if (is.null(cyclopsData)) NULL else getNumberOfCovariates(cyclopsData))

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
        .cyclopsSetCvControl(cyclopsInterfacePtr, .cvControlOptions(control))
        .cyclopsSetModeControl(cyclopsInterfacePtr, .modeControlOptions(control))
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

#' @title Run bootstrap for Cyclops model parameters
#'
#' @details
#' Bootstrap samples resample the units given by the fit's \code{selectorType} (see \code{\link{createControl}}),
#' so Cox models with subjects resample whole subjects by default. Units whose rows all have weight zero
#' are not resampled: each sample draws as many units as the fit uses.
#'
#' @param object          A fitted Cyclops model object
#' @param replicates      Numeric: number of bootstrap samples
#' @param ciCoverage      Numeric (0-1): coverage proportion for bootstrap confidence intervals
#' @param na.rm           Logical: strip out bootstrap replicates with NA values (most likely without a finite estimate)
#'
#' @export
runBootstrap <- function(object, replicates, ciCoverage = 0.95, na.rm = FALSE) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    bs <- .cyclopsRunBootstrap(object$cyclopsData$cyclopsInterfacePtr, "", "1", replicates) # TODO Remove unneeded arguments

    if (.cyclopsGetHasOffset(object$cyclopsData)) {
        bs$samples <- bs$samples[(replicates + 1):length(bs$samples)]
    }

    bs$samples <- matrix(bs$samples, nrow = replicates)

    colnames(bs$samples) <- names(coef(object))
    bs$samples <- as_tibble(bs$samples)

    if (na.rm) {
        bs$samples <- bs$samples %>% tidyr::drop_na()
    }

    epsilon <- 1 - ciCoverage

    bs$summary <- data.frame(
        original = coef(object),
        bias = sapply(bs$samples, mean) - coef(object),
        std_err = sapply(bs$samples, function(x) { sqrt(stats::var(x)) }),
        bpi_lower = sapply(bs$samples, stats::quantile, probs = epsilon / 2),
        bpi_upper = sapply(bs$samples, stats::quantile, probs = 1 - epsilon / 2)
    )

    return(bs)
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
#' @param maximumCurvature Numeric: maximum curvature of log-likelihood profile required to compute CI
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
                               rescale = FALSE,
                               maximumCurvature = -1E-3,
                               ...) {
    .checkInterface(object$cyclopsData, testOnly = TRUE)
    #.setControl(object$cyclopsData$cyclopsInterfacePtr, control)

    savedParm <- parm
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

    hessianDiagonal <- .cyclopsGetLogLikelihoodHessianDiagonal(object$cyclopsData$cyclopsInterfacePtr,
                                                               parm)

    convergenceType <- .cyclopsGetConvergenceType(object$cyclopsData$cyclopsInterfacePtr)
    prof <- NULL
    if (convergenceType != 1 && any(hessianDiagonal >= maximumCurvature)) {

        .cyclopsSetConvergenceType(object$interface, "lange")
        newFit <- .cyclopsFitModel(object$cyclopsData$cyclopsInterfacePtr)
        newHessianDiagonal <- .cyclopsGetLogLikelihoodHessianDiagonal(object$cyclopsData$cyclopsInterfacePtr,
                                                                      parm)

        if (isTRUE(any(newHessianDiagonal > 0))) {
            warning("Cannot estimate confidence interval for positive-curvature log-likelihood function")

            prof <- data.frame(covariate = parm, # was 'savedParm' but unsure why
                               lower = rep(NA, length(parm)),
                               upper = rep(NA, length(parm)),
                               evaluations = rep(NA, length(parm)))
        }
    }

    if (is.null(prof)) {

        prof <- .cyclopsProfileModel(object$cyclopsData$cyclopsInterfacePtr, parm,
                                     threads, threshold,
                                     overrideNoRegularization,
                                     includePenalty)
    }

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

.initAdaptiveProfile <- function(object, parm, bounds, includePenalty, optimalWarmStart, returnDerivatives) {
    # If an MLE was found, let's not throw that bit of important information away:
    if (object$return_flag == "SUCCESS" &&
        coef(object)[as.character(parm)] > bounds[1] &&
        coef(object)[as.character(parm)] < bounds[2]) {
        eval <- fixedGridProfileLogLikelihood(object, parm, coef(object)[as.character(parm)], includePenalty,
        									  optimalWarmStart, returnDerivatives)
        point <- coef(object)[as.character(parm)]
        if (returnDerivatives) {
            profile <- tibble(point = point,
                              value = eval$value,
                              derivative = eval$derivative)
        } else {
            profile <- tibble(point = point,
                              value = eval$value)
        }
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
#'                  either a vector of numbers or covariateId names
#' @param x         Vector of values of the parameter
#' @param bounds    Pair of values to bound adaptive profiling
#' @param tolerance Absolute tolerance allowed for adaptive profiling
#' @param initialGridSize Initial grid size for adaptive profiling
#' @param maxResets Maximum allowed number of recomputing the likelihood when coefficient drift is
#'                  detected.
#' @param includePenalty    Logical: Include regularized covariate penalty in profile
#' @param optimalWarmStart Logical: also warm-start each point from the solutions at the requested points between
#'                         it and the estimate. Each fit starts from the solution at the nearest such point, the
#'                         estimate or (adaptive profiling) a point already evaluated, so results do not depend
#'                         on the number of threads
#' @param returnDerivatives Return derivative of log-likelihood wrt the parameter `param` at each evaluation point
#' @param maxDrop   Numeric: adaptive profiling does not refine intervals whose two end points both lie more than
#'                  \code{maxDrop} below the largest profile log likelihood found so far; their points remain in
#'                  the result. The default (\code{Inf}) refines everywhere. With wide bounds, a value
#'                  such as 50 saves most evaluations, and the profile stays within \code{tolerance}
#'                  wherever it lies within 50 of its maximum, as evidence synthesis needs
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
                                           maxResets = 10,
                                           includePenalty = TRUE,
                                           optimalWarmStart = TRUE,
                                           returnDerivatives = FALSE,
                                           maxDrop = Inf) {

    if (!xor(is.null(x), is.null(bounds))) {
        stop("Must provide either `x` or `bounds`, but not both.")
    }

    logLik <- .cyclopsGetLogLikelihood(object$interface)
    if (is.na(logLik)) {
        object <- fitCyclopsModel(
            object$cyclopsData,
            fixedCoefficients = rep(TRUE, getNumberOfCovariates(object$cyclopsData)))
    }

    if (!is.null(bounds)) { # Adaptive profiling using recursive calls

        if (length(bounds) != 2 || bounds[1] >= bounds[2]) {
            stop("Must provide bounds[1] < bounds[2]")
        }
        if (!(maxDrop > 0)) {
            stop("Must provide maxDrop > 0")
        }
        profile <- .initAdaptiveProfile(object, parm, bounds, includePenalty, optimalWarmStart, returnDerivatives)

        # Start with sparse grid:
        grid <- seq(bounds[1], bounds[2], length.out = initialGridSize)

        # Solutions at evaluated points warm-start the new points of the next round (not under the
        # legacy fitter, which keeps the profile evaluation of Cyclops 3.7)
        keepStarts <- getNumberOfCovariates(object$cyclopsData) <= 10000 &&
            .dataFitter(object$cyclopsData) != "legacy"
        starts <- NULL

        # Iterate until stopping criteria met:
        priorMaxMaxError <- Inf
        resetsPerformed <- 0
        while (length(grid) != 0) {
            ll <- fixedGridProfileLogLikelihood(object, parm, grid, includePenalty, optimalWarmStart, returnDerivatives,
                                                starts = starts, returnBetas = keepStarts)
            if (keepStarts) {
                valid <- is.finite(ll$value)
                starts <- list(point = c(starts$point, ll$point[valid]),
                               betas = cbind(starts$betas, attr(ll, "betas")[, valid, drop = FALSE]))
                attr(ll, "betas") <- NULL
            }
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
                interface <- .cyclopsInitializeModel(object$cyclopsData$cyclopsDataPtr, modelType = object$cyclopsData$modelType, "native", computeMLE = TRUE,
                                                     fitter = .dataFitter(object$cyclopsData))
                assign("cyclopsInterfacePtr", interface$interface, object)
                resetsPerformed <- resetsPerformed + 1
                starts <- NULL
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

            # Refine only intervals with an end point within maxDrop of the largest value
            relevant <- pmax(profile$value[1:(nrow(profile) - 1)], profile$value[2:nrow(profile)]) >=
                max(profile$value) - maxDrop

            maxMaxError <- max(maxError[relevant], na.rm = TRUE)
            if (is.na(maxMaxError) || maxMaxError > priorMaxMaxError) {
                warning("Failing to converge when using adaptive profiling.")
                return(NULL)
            }
            priorMaxMaxError <- maxMaxError

            exceed <- which(maxError > tolerance & relevant)
            grid <- (profile$point[exceed] + profile$point[exceed + 1]) / 2
            if (keepStarts) { # only end points of refined intervals can be nearest to new points
                keep <- starts$point %in% c(profile$point[exceed], profile$point[exceed + 1])
                starts <- list(point = starts$point[keep], betas = starts$betas[, keep, drop = FALSE])
            }
        }
    } else { # Use x
        profile <- fixedGridProfileLogLikelihood(object, parm, x, includePenalty, optimalWarmStart, returnDerivatives)
    }

    return(profile)
}

fixedGridProfileLogLikelihood <- function(object, parm, x, includePenalty,
                                          optimalWarmStart = TRUE, returnDerivatives = FALSE,
                                          starts = NULL, returnBetas = FALSE) {

    .checkInterface(object$cyclopsData, testOnly = TRUE)
    parm <- .checkCovariates(object$cyclopsData, parm)
    threads <- object$threads

    if (.dataFitter(object$cyclopsData) == "legacy") {
        return(.legacyGridProfileLogLikelihood(object, parm, x, includePenalty, optimalWarmStart,
                                               returnDerivatives, threads))
    }

    # Each point starts from the solution at the nearest of: the current coefficients, `starts` and
    # (optimalWarmStart) the points in `x` between it and the current value
    grid <- .cyclopsGetProfileLikelihood(object$cyclopsData$cyclopsInterfacePtr, parm, x,
                                         threads, includePenalty, returnDerivatives,
                                         if (is.null(starts)) numeric(0) else starts$point,
                                         if (is.null(starts)) numeric(0) else as.vector(starts$betas),
                                         returnBetas, optimalWarmStart)
    betas <- attr(grid, "betas")
    if (optimalWarmStart && length(x) > 1) {
        o <- order(grid$point)
        grid <- grid[o, ]
        rownames(grid) <- NULL
        if (!is.null(betas)) betas <- betas[, o, drop = FALSE]
    }
    attr(grid, "betas") <- betas

    return(grid)
}

# The profile evaluation of Cyclops 3.7, for createControl(fitter = "legacy"): each thread fits its
# points in the order given, each starting from its previous fit, so the grid is split at its middle
# and evaluated outwards in two calls
.legacyGridProfileLogLikelihood <- function(object, parm, x, includePenalty, optimalWarmStart,
                                            returnDerivatives, threads) {
    evaluate <- function(points) {
        .cyclopsGetProfileLikelihood(object$cyclopsData$cyclopsInterfacePtr, parm, points,
                                     threads, includePenalty, returnDerivatives,
                                     numeric(0), numeric(0), FALSE, FALSE)
    }
    if (getNumberOfCovariates(object$cyclopsData) == 1 || length(x) == 1) {
        grid <- evaluate(x)
    } else if (!optimalWarmStart && length(x) / 2 >= threads) {
        grid <- evaluate(x)
    } else {
        y <- sort(x)
        midPt <- floor(length(x) / 2)
        grid <- rbind(evaluate(y[midPt:1]), evaluate(y[(midPt + 1):length(x)]))
        grid <- grid[order(grid$point), ]
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
    .setControl(object$cyclopsData$cyclopsInterfacePtr, control, object$cyclopsData)
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
    .setControl(object$cyclopsData$cyclopsInterfacePtr, control, object$cyclopsData)
    fisherInformation <- .cyclopsGetFisherInformation(object$cyclopsData$cyclopsInterfacePtr, NULL)
    if (.cyclopsGetHasOffset(object$cyclopsData)) {
        fisherInformation <- fisherInformation[2:nrow(fisherInformation), 2:ncol(fisherInformation)]
    }
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
