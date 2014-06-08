#' @title fitCcdModel
#'
#' @description
#' \code{fitCcdModel} fits a CCD model data object
#'
#' @details
#' This function performs numerical optimization to fit a CCD model data object.
#'
#' @param formula			An R formula
#' @param data
#' 
#' @return
#' A list that contains a CCD model fit object pointer and an operation duration
#' 
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' ccdData <- createCcdDataFrame(counts ~ outcome + treatment, modelType="pr")
#' ccdFit <- fitCcdModel(ccdData)
#'
fitCcdModel <- function(ccdData		
		, tolerance = 1E-8
		, variance
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
	
	if (forceColdStart || is.null(ccdData$ccdInterfacePtr) 
			|| class(ccdData$ccdInterfacePtr) != "externalptr") {
		# Build interface
		interface <- .ccdInitializeModel(ccdData$ccdDataPtr, modelType = ccdData$modelType, computeMLE = TRUE)
		# TODO Check for errors
        assign("ccdInterfacePtr", interface$interface, ccdData)
#		ccdData$ccdInterfacePtr <- interface$interface
#		ccdData$timeInit <- interface$timeInit
	}
	
	fit <- .ccdFitModel(ccdData$ccdInterfacePtr) # TODO Pass along other options		
#	fit <- fit[-1] # Remove interface, TODO clean up C++; no need to chain
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
