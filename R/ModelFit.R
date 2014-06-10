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
		, returnEstimates = TRUE
		, forceColdStart = FALSE
	) {
		
	# Check conditions
	if (missing(ccdData) || is.null(ccdData$ccdDataPtr) || class(ccdData$ccdDataPtr) != "externalptr") {
		stop("Improperly constructed ccdData object")
	}	
	if (.isRcppPtrNull(ccdData$ccdDataPtr)) {
	    stop("Data object is no longer initialized")			
	}
	
	cl <- match.call()
	
    .checkInterface(ccdData, forceColdStart)
    
    if (!missing(prior)) { # Set up prior
    	stopifnot(inherits(prior, "ccdPrior"))
    	
    	prior$exclude = as.numeric(prior$exclude) # TODO Search names
    	
    	stopifnot(!any(is.na(prior$exclude)))    
    	.ccdSetPrior(ccdData$ccdInterfacePtr, prior$priorType, prior$variance, prior$exclude)    		
    }
    
#    .ccdSetTolerance(tolerance)
	
	fit <- .ccdFitModel(ccdData$ccdInterfacePtr) # TODO Pass along other options	
	if (returnEstimates && fit$return_flag == "SUCCESS") {
		estimates <- .ccdLogModel(ccdData$ccdInterfacePtr)		
		fit <- c(fit, estimates)	
		fit$estimation <- as.data.frame(fit$estimation)
	}	
	fit$call <- cl
	fit$ccdData <- ccdData
	class(fit) <- "ccdFit"
	return(fit)
} 

.checkInterface <- function(x, forceColdStart) {
	if (forceColdStart || is.null(x$ccdInterfacePtr) 
			|| class(x$ccdInterfacePtr) != "externalptr") {
		# Build interface
		interface <- .ccdInitializeModel(x$ccdDataPtr, modelType = x$modelType, computeMLE = TRUE)
		# TODO Check for errors
        assign("ccdInterfacePtr", interface$interface, x)
	}
}

coef.ccdFit <- function(x, ...) {
	x$estimation
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
#   if (!is.null(x$ccdDataPtr) && !.isRcppPtrNull(x$ccdDataPtr)) {
#       nRows <- getNumberOfRows(x)
#       cat("      Rows: ", nRows, "\n", sep="")
#       cat("Covariates: ", getNumberOfCovariates(x), "\n", sep="")
#       nStrata <- getNumberOfStrata(x)
#       if (nRows != nStrata) {
#         cat("    Strata: ", nStrata, "\n", sep="")
#       }
#   } else {
#     cat("\nObject is no longer or improperly initialized.\n")
#   }
#   cat("\n")
#   if (!is.null(x$ccdInterfacePtr) && !.isRcppPtrNull(x$ccdInterfacePtr)) {    
#     cat("Initialized interface (details coming soon).\n")
#   } else {
#     cat("Uninitialized interface.\n")
#   }
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
#' 
#' @section Criteria:
#' 
#' @return
#' A CCD convergence criteria object of class inheriting from \code{"ccdConvergence"} for use with \code{fitCcdModel}.
#' 
control <- function(maxIterations = 1000, tolerance = 1E-8, convergenceType = "gradient") {
	structure(list(maxIterations = maxIterations, tolerance = tolerance, convergenceType = convergenceType),
						class = "ccdConvergence")
}

prior <- function(priorType, variance = 1, exclude = c()) {
	validNames = c("none", "laplace","normal")
	stopifnot(priorType %in% validNames)	
	if (!is.null(exclude)) {
		stopifnot(inherits(exclude, "character"))
	}
	structure(list(priorType = priorType, variance = variance, exclude = exclude), class = "ccdPrior")
}

.clear <- function() {
    cat("\014")  
}
