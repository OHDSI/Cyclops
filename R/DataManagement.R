#' @title createCcdDataFrame
#'
#' @description
#' \code{createCcdDataFrame} creates a CCD model data object from an R formula
#'
#' @details
#' This function creates a CCD model data object from an R formula and data.frame.
#'
#' @template types
#'
#' @param formula			An R formula
#' @param data
#' 
#' @return
#' A list that contains a CCD model data object pointer and an operation duration
#' 
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' ccdData <- createCcdDataFrame(counts ~ outcome + treatment, modelType = "pr")
#' ccdFit <- fitCcdModel(ccdData)
#'
#' ccdData2 <- createCcdDataFrame(counts ~ outcome, indicatorFormula = ~ treatment, modeltype = "pr")
#' ccdFit2 <- fittCcdModel(ccdData2)
#'
#' @export
createCcdDataFrame <- function(formula, sparseFormula, indicatorFormula, modelType,
	data, subset, weights, offs = NULL, y = NULL, z = NULL, dx = NULL, 
	sx = NULL, ix = NULL, model = FALSE, method = "ccd.fit", ...) {	
	cl <- match.call() # save to return
	mf.all <- match.call(expand.dots = FALSE)
			
	if (!isValidModelType(modelType)) stop("Invalid model type.")		
			
	colnames <- NULL
	
	if (!missing(formula)) { # Use formula to construct CCD matrices
		if (missing(data)) {
			data <- environment(formula)
		}					
		m.d <- match(c("formula", "data", "subset", "weights",
				"offset"), names(mf.all), 0L)
		mf.d <- mf.all[c(1L, m.d)]
		mf.d$drop.unused.levels <- TRUE
		mf.d[[1L]] <- quote(stats::model.frame)			
		mf.d <- eval(mf.d, parent.frame())	
		dx <- Matrix(model.matrix(mf.d, data), sparse=FALSE)	# TODO sparse / indicator matrices
		y <- model.response(mf.d) # TODO Update for censored outcomes
		pid <- c(1:length(y)) # TODO Update for stratified models		       
    
		colnames <- c(colnames, dx@Dimnames[[2]])
		
		if (!missing(sparseFormula)) {					
			if (missing(data)) {
				data <- environment(sparseFormula)
			}					
			m.s <- match(c("sparseFormula", "data", "subset", "weights",
					"offset"), names(mf.all), 0L)
			mf.s <- mf.all[c(1L, m.s)]
			mf.s$drop.unused.levels <- TRUE
			mf.s[[1L]] <- quote(stats::model.frame)			
			names(mf.s)[2] = "formula"
			mf.s <- eval(mf.s, parent.frame())			
			
			if (!is.null(model.response(mf.s))) {
				stop("Must only provide outcome variable in dense formula.")
			}
			
			stmp <- model.matrix(mf.s, data)
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
			m.i <- match(c("indicatorFormula", "data", "subset", "weights",
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
			
			itmp <- model.matrix(mf.i, data)
			ilabels <- labels(itmp)[[2]]
			if (attr(attr(mf.i, "terms"), "intercept") == 1) { # Remove intercept
				ix <- Matrix(as.matrix(itmp[,-1]), sparse=TRUE)
				ilabels <- ilabels[-1]
			} else {
				ix <- Matrix(as.matrix(itmp), sparse=TRUE)			
			}					
			colnames <- c(colnames, ilabels)
		} 
		
		if (identical(method, "model.frame")) {
			result <- list()
			if (exists("mf.d")) {
				result$dense <- mf.d
				result$dx <- dx
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
    
    # TODO Check types and dimensions        
    
    md <- .ccdModelData(pid, y, z, offs, dx, sx, ix)
	result <- new.env(parent = emptyenv())
	result$ccdDataPtr <- md$data
	result$modelType <- modelType
	result$timeLoad <- md$timeLoad	
	result$call <- cl
	if (exists("mf") && model == TRUE) {
		result$mf <- mf
	}
	result$ccdInterfacePtr <- NULL
	result$call <- cl
	result$coefficientNames <- colnames
	result$rowNames <- dx@Dimnames[[1]]
	result$debug <- list()
	result$debug$dx <- dx
	result$debug$sx <- sx
	result$debug$ix <- ix
	result$debug$y <- y
	result$debug$pid <- pid
	
	
	class(result) <- "ccdData"
	result
}

#' @title isValidModelType
#'
#' @description
#' \code{isValidModelType} checks for a valid CCD model type
#'
#' @template types
#'
#' @return TRUE/FALSE
isValidModelType <- function(modelType) {
    types <- c("ls", "pr", "lr", "clr", "sccs", "cox")
    modelType %in% types
}

#' @title readCcdData
#'
#' @description
#' \code{readCcdData} reads a CCD-formatted text file
#'
#' @details
#' This function reads a CCD-formatted text file and returns a CCD data object. The first line of the
#' file may start with '\samp{#}', indicating that it contains header options.  Valid header options are:
#' 
#'  \tabular{ll}{  
#'   \verb{	row_label}		\tab (assume file contains a numeric column of unique row identifiers) \cr
#'   \verb{	stratum_label}\tab (assume file contains a numeric column of stratum identifiers) \cr
#'   \verb{	weight}				\tab (assume file contains a column of row-specific model weights, currently unused) \cr
#' 	\verb{	offset}				\tab (assume file contains a dense column of linear predictor offsets) \cr
#' 	\verb{	bbr_outcome}	\tab (assume logistic outcomes are encoded -1/+1 following BBR) \cr
#' 	\verb{	log_offset}		\tab (assume file contains a dense column of values x_i for which log(x_i) is the offset) \cr
#' 	\verb{	add_intercept}\tab (automatically include an intercept column of all 1s for each entry) \cr
#' 	\verb{	indicator_only}\tab(assume all covariates 0/1-valued and only covariate name is given) \cr
#' 	\verb{	sparse}				\tab (force all BBR formatted covariates to be represented as sparse, instead of sparse-indicator, columns .. really only for debugging) \cr
#' 	\verb{	dense}				\tab (force all BBR formatted covariates to be represented as dense columns .. really only for debugging) \cr
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
#' 												 the name is relative to the current working directory, \code{\link{getwd}}.
#' @param modelTypeName		 character string: if non-empty, declares the specific model formatting to
#' 												 read.  Valid types are listed below.
#' 
#' @template ccdData
#' 
#' @examples
#' dataPtr = readCcdData(system.file("extdata/infert_ccd.txt", package="CCD"), "clr")
#'
readCcdData <- function(fileName, modelType) {
    cl <- match.call() # save to return
    
    if (!isValidModelType(modelType)) stop("Invalid model type.")    
    
    read <- .ccdReadData(fileName, modelType)
    result <- new.env(parent = emptyenv())
    result$ccdDataPtr <- read$ccdDataPtr
    result$modelType <- modelType
    result$timeLoad <- read$timeLoad
    result$ccdInterfacePtr <- NULL
    result$call <- cl
    
    class(result) <- "ccdData"
    result
}

print.ccdData <- function(x,digits=max(3,getOption("digits")-3),show.call=TRUE,...) {
  cat("OHDSI CCD Data Object\n\n")
  
  if (show.call && !is.null(x$call)) {
    cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")  
  }
  cat("     Model: ", x$modelType, "\n", sep="")
  
  if (!is.null(x$ccdDataPtr) && !.isRcppPtrNull(x$ccdDataPtr)) {
      nRows <- getNumberOfRows(x)
      cat("      Rows: ", nRows, "\n", sep="")
      cat("Covariates: ", getNumberOfCovariates(x), "\n", sep="")
      nStrata <- getNumberOfStrata(x)
      if (nRows != nStrata) {
        cat("    Strata: ", nStrata, "\n", sep="")
      }
  } else {
    cat("\nObject is no longer or improperly initialized.\n")
  }
  cat("\n")
  if (!is.null(x$ccdInterfacePtr) && !.isRcppPtrNull(x$ccdInterfacePtr)) {    
    cat("Initialized interface (details coming soon).\n")
  } else {
    cat("Uninitialized interface.\n")
  }
  invisible(x)
}
