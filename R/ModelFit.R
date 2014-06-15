#' @title fitCcdModel
#'
#' @description
#' \code{fitCcdModel} fits a CCD model data object
#'
#' @details
#' This function performs numerical optimization to fit a CCD model data object.
#'
#' @param ccdData			An OHDSI data object
#' @template prior
#' 
#' @return
#' A list that contains a CCD model fit object pointer and an operation duration
#' 
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' ccdData <- createCcdDataFrame(counts ~ outcome + treatment)
#' ccdFit <- fitCcdModel(ccdData)
#'
fitCcdModel <- function(ccdData		
		, tolerance = 1E-8
		, prior
		, crossValidation
		, control
		, returnEstimates = TRUE
		, forceColdStart = FALSE
	) {
		
	cl <- match.call()
	
		# Check conditions
		.checkData(ccdData)
	
    .checkInterface(ccdData, forceColdStart)
    
    if (!missing(prior)) { # Set up prior
    	stopifnot(inherits(prior, "ccdPrior"))
    	
    	prior$exclude = as.numeric(prior$exclude) # TODO Search names
    	
    	stopifnot(!any(is.na(prior$exclude)))    
    	.ccdSetPrior(ccdData$ccdInterfacePtr, prior$priorType, prior$variance, prior$exclude)    		
    }
	
		.setControl(ccdData$ccdInterfacePtr, control)
			
 	
	if (!missing(prior) && prior$useCrossValidation) {
		if (missing(control)) {
			minCVData <- control()$minCVData		
		} else {
			minCVData <- control$minCVData
		}
		if (minCVData > getNumberOfRows(ccdData)) { # TODO Not correct; some models CV by stratum
			stop("Insufficient data count for cross validation")
		}
		fit <- .ccdRunCrossValidation(ccdData$ccdInterfacePtr)
	} else {
		fit <- .ccdFitModel(ccdData$ccdInterfacePtr)
	}
	
	if (returnEstimates && fit$return_flag == "SUCCESS") {
		estimates <- .ccdLogModel(ccdData$ccdInterfacePtr)		
		fit <- c(fit, estimates)	
		fit$estimation <- as.data.frame(fit$estimation)
	}	
	fit$call <- cl
	fit$ccdData <- ccdData
	fit$ccdInterfacePtr <- ccdData$ccdInterfacePtr
	fit$coefficientNames <- ccdData$coefficientNames
	fit$rowNames <- ccdData$rowNames
	
# 	if (!missing(prior)) {
# 		fit$prior <- prior
# 	} else {
# 		fit$prior <- NULL # prior("none")
# 	}
	class(fit) <- "ccdFit"
	return(fit)
} 

.checkData <- function(x) {
	# Check conditions
	if (missing(x) || is.null(x$ccdDataPtr) || class(x$ccdDataPtr) != "externalptr") {
		stop("Improperly constructed ccdData object")
	}	
	if (.isRcppPtrNull(x$ccdDataPtr)) {
		stop("Data object is no longer initialized")			
	}
}

.checkInterface <- function(x, forceColdStart = FALSE, testOnly = FALSE) {
	if (forceColdStart 
			|| is.null(x$ccdInterfacePtr) 
			|| class(x$ccdInterfacePtr) != "externalptr" 
			|| .isRcppPtrNull(x$ccdInterfacePtr)
		) {
		
		if (testOnly == TRUE) {
			stop("Interface object is not initialized")
		}
		# Build interface
		interface <- .ccdInitializeModel(x$ccdDataPtr, modelType = x$modelType, computeMLE = TRUE)
		# TODO Check for errors
        assign("ccdInterfacePtr", interface$interface, x)
	}
}

coef.ccdFit <- function(x, ...) {
	result <- x$estimation$estimate
	if (is.null(x$coefficientNames)) {
		names(result) <- x$estimation$column_label
		if ("0" %in% names(result)) {
			names(result)[which(names(result) == "0")] <- "(Intercept)"
		}
	} else {
		names(result) <- x$coefficientNames
	}
	result
}

getHyperParameter <- function(x) {
	x$variance
}

print.ccdFit <- function(x,digits=max(3,getOption("digits")-3),show.call=TRUE,...) {
  cat("OHDSI CCD model fit object\n\n")
  
  if (show.call && !is.null(x$call)) {
    cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")  
  }
  cat("           Model: ", x$ccdData$modelType, "\n", sep="")
  cat("           Prior: ", x$prior_info, "\n", sep="")
  cat("  Hyperparameter: ", x$variance, "\n", sep="")
  cat("     Return flag: ", x$return_flag, "\n", sep="")
  if (x$return_flag == "SUCCESS") {  	
  	cat("Log likelikehood: ", x$log_likelihood, "\n", sep="")
  	cat("       Log prior: ", x$log_prior, "\n", sep="")
  }
  invisible(x)
}

#' @title control
#'
#' @description
#' \code{control} builds a CCD convergence criteria object
#'
#' @param maxIterations			Integer: maximum iterations of CCD to attempt before returning a failed-to-converge error
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
#' @param minCVData					Numeric: Minumim number of data for cross validation
#' @param noiseLevel				String: level of CCD screen output (\code{"silent"}, \code{"quiet"}, \code{"noisy"})
#' 
#' @section Criteria:
#' 
#' @return
#' A CCD convergence criteria object of class inheriting from \code{"ccdConvergence"} for use with \code{fitCcdModel}.
#' 
control <- function(
		maxIterations = 1000, tolerance = 1E-6, convergenceType = "gradient",
		cvType = "grid", fold = 10, lowerLimit = 0.01, upperLimit = 20.0, gridSteps = 10,
		cvRepetitions = 1,
		minCVData = 100, noiseLevel = "noisy") {
	
	validCVNames = c("grid", "auto")
	stopifnot(cvType %in% validCVNames)
	
	validNLNames = c("silent", "quiet", "noisy")
	stopifnot(noiseLevel %in% validNLNames)
	structure(list(maxIterations = maxIterations, tolerance = tolerance, convergenceType = convergenceType,
								 autoSearch = (cvType == "auto"), fold = fold, lowerLimit = lowerLimit, 
								 upperLimit = upperLimit, gridSteps = gridSteps, minCVData = minCVData, 
								 cvRepetitions = cvRepetitions,
								 noiseLevel = noiseLevel),
						class = "ccdControl")
}

#' @title prior
#'
#' @description
#' \code{prior} builds a CCD prior object
#'
#' @param priorType
#' @param variance
#' @param exclude
#' @param useCrossValidation
#' 
#' @section Prior types:
#' 
#' @return
#' A CCD prior object of class inheriting from \code{"ccdPrior"} for use with \code{fitCcdModel}.
#' 
prior <- function(priorType, variance = 1, exclude = c(), useCrossValidation = FALSE) {
	validNames = c("none", "laplace","normal")
	stopifnot(priorType %in% validNames)	
	if (!is.null(exclude)) {
		stopifnot(inherits(exclude, "character"))
	}
	if (priorType == "none" && useCrossValidation) {
		stop("Cannot perform cross validation with a flat prior")
	}
	structure(list(priorType = priorType, variance = variance, exclude = exclude, 
								 useCrossValidation = useCrossValidation), class = "ccdPrior")
}

.clear <- function() {
    cat("\014")  
}

predict.ccdFit <- function(object) {
	.checkInterface(object, testOnly = TRUE)
	pred <- .ccdPredictModel(object$ccdInterfacePtr)
#	pred <- pred$prediction
 	values <- pred$prediction
 	if (is.null(names(values))) {
 		names(values) <- object$rowNames
 	}
 	values
}

.setControl <- function(ccdInterfacePtr, control) {
	if (!missing(control)) { # Set up control
		stopifnot(inherits(control, "ccdControl"))
		.ccdSetControl(ccdInterfacePtr, control$maxIterations, control$tolerance, 
									 control$convergenceType, control$autoSearch, control$fold, 
									 (control$fold * control$cvRepetitions),
									 control$lowerLimit, control$upperLimit, control$gridSteps, control$noiseLevel)		
	}	
}

confint.ccdFit <- function(fitted, covariates, control) {
	.checkInterface(fitted, testOnly = TRUE)
	.setControl(fitted$ccdInterfacePtr, control)
	
	prof <- .ccdProfileModel(fitted$ccdInterfacePtr, covariates)
	prof <- as.matrix(as.data.frame(prof))
	rownames(prof) <- fitted$coefficientNames[covariates]
	colnames(prof)[2:3] <- c("2.5 %", "97.5 %")
	prof
}
