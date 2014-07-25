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
#' ccdData2 <- createCcdDataFrame(counts ~ outcome, indicatorFormula = ~ treatment, modelType = "pr")
#' summary(ccdData2)
#' ccdFit2 <- fitCcdModel(ccdData2)
#'
#' @export
createCcdDataFrame <- function(formula, sparseFormula, indicatorFormula, modelType,
                               data, subset, weights, offset, time = NULL, pid = NULL, y = NULL, z = NULL, dx = NULL, 
                               sx = NULL, ix = NULL, model = FALSE, method = "ccd.fit", ...) {	
    cl <- match.call() # save to return
    mf.all <- match.call(expand.dots = FALSE)
    
    if (!isValidModelType(modelType)) stop("Invalid model type.")
    
    hasIntercept <- FALSE		
    colnames <- NULL
    
    if (!missing(formula)) { # Use formula to construct CCD matrices
        if (missing(data)) {
            data <- environment(formula)
        }					
        mf.all <- match.call(expand.dots = FALSE)
        m.d <- match(c("formula", "data", "subset", "weights",
                       "offset"), names(mf.all), 0L)
        mf.d <- mf.all[c(1L, m.d)]
        mf.d$drop.unused.levels <- TRUE
        mf.d[[1L]] <- quote(stats::model.frame)			
        mf.d <- eval(mf.d, parent.frame())
        
        y <- model.response(mf.d)
        if (class(y) == "Surv") {
            if (modelType != "cox") {
                stop("Censored outcomes are currently only support for Cox regression.")
            }
            if (dim(y)[2] == 3) {
                time <- as.numeric(y[,2] - y[,1])
                y <- as.numeric(y[,3])
            } else {
                time <- as.numeric(y[,1])
                y <- as.numeric(y[,2])
            }            
        }
        
        mt.d <- attr(mf.d, "terms")
        
        # Handle strata
        specialTerms <- terms(formula, "strata")
        special <- attr(specialTerms, "special")
        hasStrata <- !is.null(special$strata)
        strata <- NULL
        sortOrder <- NULL
        if (hasStrata) {
            pid <- as.numeric(strata(mf.d[ , special$strata], shortlabel = TRUE))
            nterm <- untangle.specials(specialTerms, "strata")$terms
            mt.d <- mt.d[-nterm]
            
            ## Must sort outcomes
            if (modelType == "cox") {
                sortOrder <- order(-time, pid)    
            } else {
                sortOrder <- order(pid)
            }
        } else {
            pid <- c(1:length(y))
            if (modelType == "cox") {
                sortOrder <- order(-time)
            }
        }      
        
        #         if (.removeIntercept(modelType)) { # This does not work with constrasts
        #             attr(mt.d, "intercept") <- FALSE
        #         }
        
        dtmp = model.matrix(mt.d, mf.d)
        if (attr(mt.d, "intercept") && .removeIntercept(modelType)) {
            dx <- Matrix(as.matrix(dtmp[,-1]), sparse = FALSE)
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
        
        colnames <- c(colnames, dx@Dimnames[[2]])
        
        if (attr(mt.d, "intercept") && !.removeIntercept(modelType)) { # Has intercept
            hasIntercept <- TRUE
        }
        
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
            z <- z[sortOrder] 
            time <- time[sortOrder]
            dx <- dx[sortOrder, ]
            if (class(dx) == "numeric")
                dx = as(dx,"dgeMatrix")
            sx <- sx[sortOrder, ]
            ix <- ix[sortOrder, ]            
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
    
    # TODO Check types and dimensions        
    
    useTimeAsOffset <- FALSE
    if (!is.null(time) && modelType != "sccs" && modelType != "cox") { # TODO Generic check
        useTimeAsOffset <- TRUE
    }
    
    if (is.null(pid)) {
        pid <- c(1:length(y)) # TODO Should not be necessary
    }
    
    md <- .ccdModelData(pid, y, z, time, dx, sx, ix, modelType, useTimeAsOffset)
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
    
    if (!is.null(dx)) {
        result$rowNames <- dx@Dimnames[[1]]
    } else if (!is.null(pid)) {
        result$rowNames <- pid
    } else {
        result$rowNames <- c(1:length(y))
    }
     
    if (identical(method, "debug")) {
        result$debug <- list()
        result$debug$dx <- dx
        result$debug$sx <- sx
        result$debug$ix <- ix
        result$debug$y <- y
        result$debug$pid <- pid
        result$debug$time <- time
    }
    class(result) <- "ccdData"
    
    if (hasIntercept == TRUE) {
        .ccdSetHasIntercept(result, hasIntercept = TRUE)
    }
    
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
    types <- c("ls", "pr", "lr", "clr", "cpr", "sccs", "cox")
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

reduce <- function(object, covariates, groupBy, power = 1) {
	if (!isInitialized(object)) {
		stop("Object is no longer or improperly initialized.")
	}
	covariates <- .checkCovariates(object, covariates)
	
    if (!(power %in% c(0,1,2))) {
        stop("Only powers 0, 1 and 2 are allowed.")
    }
    
	if (missing(groupBy)) {
		.ccdSum(object, covariates, power)
	} else {
		if (length(groupBy) != 1L) {
			stop("Only single stratification is currently implemented")
		}
		if (groupBy == "stratum") {
			as.data.frame(.ccdSumByStratum(object, covariates, power), 
										row.names = c(1L:getNumberOfStrata(object)))			
		} else {
			groupBy <- .checkCovariates(object, groupBy)
			as.data.frame(.ccdSumByGroup(object, covariates, groupBy, power), 
										row.names = c(0L,1L))			
		}				
	}
}



#' @title appendSqlCcdData
#'
#' @description
#' \code{appendSqlCcdData} appends data to an OHDSI data object.
#' 
appendSqlCcdData <- function(object,
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
    
    if (is.unsorted(oStratumId) || is.unsorted(oRowId) || is.unsorted(cRowId)) {
        stop("All columns must be sorted first by stratumId (if supplied) and then by rowId")
    }
    
    .appendSqlCcdData(object, 
                      oStratumId, 
                      oRowId, 
                      oY, 
                      oTime, 
                      cRowId, 
                      cCovariateId, 
                      cCovariateValue)
}

#' @title finalizeSqlCcdData
#'
#' @description
#' \code{finalizeSqlCcdData} finalizes a CCD data object
#'
#' @param object							CCD data object
#' @param addIntercept				Add an intercept covariate if one was not imported through SQL
#' @param useOffsetCovariate	Specify is a covariate should be used as an offset (fixed coefficient = 1).
#' 														Set option to \code{"useTime"} to specify the time-to-event column, 
#' 														otherwise include a single numeric or character covariate name.
#' @param logOffset						Set to \code{TRUE} to indicate that offsets were log-transformed before importing into CCD data object. 														
#' @param sortCovariates			Sort covariates in numeric-order with intercept first if it exists.
#' @param makeCovariatesDense List of numeric or character covariates names to densely represent in CCD data object.
#' 														For efficiency, we suggest making atleast the intercept dense.
#'
finalizeSqlCcdData <- function(object,
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
        
    .ccdFinalizeData(object, addIntercept, useOffsetCovariate,
                        offsetAlreadyOnLogScale, sortCovariates,
                        makeCovariatesDense)
    
    if (addIntercept == TRUE) {
        if (!is.null(object$coefficientNames)) {
            object$coefficientNames = c("(Intercept)", 
                                        object$coefficientNames)
        }
    }
    if (!is.null(useOffsetCovariate)) {
        if (!is.null(object$coefficientNames)) {
            object$coefficientNames = object$coefficientNames[-useOffsetCovariate]
        }
    }
}

createSqlCcdData <- function(modelType, control) {
	cl <- match.call() # save to return
	
	if (!isValidModelType(modelType)) stop("Invalid model type.")  
    
    noiseLevel <- "silent"
	if (!missing(control)) { # Set up control
	    stopifnot(inherits(control, "ccdControl"))
        noiseLevel <- control$noiseLevel
	}
	
	sql <- .ccdNewSqlData(modelType, noiseLevel)
	result <- new.env(parent = emptyenv()) # TODO Remove code duplication with two functions above
	result$ccdDataPtr <- sql$ccdDataPtr
	result$modelType <- modelType
	result$timeLoad <- 0
	result$ccdInterfacePtr <- NULL
	result$call <- cl
	class(result) <- "ccdData"
	result
}

#' @title isInitialized
#'
#' @description
#' \code{isInitialized} determines if an OHDSI data object is properly 
#' initialized and remains in memory.  OHSDI data objects do not 
#' serialized/deserialize their back-end memory across R sessions.
#' 
isInitialized <- function(object) {
	return(!is.null(object$ccdDataPtr) && !.isRcppPtrNull(object$ccdDataPtr))	
}


.removeIntercept <- function(modelType) {
    if (modelType == "clr" || modelType == "cpr" || modelType == "sccs" || modelType == "cox") {
        return(TRUE)
    } else {
        return(FALSE)
    }
}

summary.ccdData <- function(x,
                            digits = max(3, getOptions("digiits") - 3),
                            show.call = TRUE,
                            ...) {
    if (!isInitialized(x)) {
        stop("OHDSI data object is no longer or improperly initialized")
    }
    covariates <- getCovariateIds(x)
    counts <- reduce(x, covariates, power = 0)
    sums <- reduce(x, covariates, power = 1)
    sumsSquared <- reduce(x, covariates, power = 2)
    types <- getCovariateTypes(x, covariates)    
    
    tmean <- sums / counts;
    
    tdf <- data.frame(covariateId = covariates,
                      nzCount = counts,
                     nzMean = tmean,
                     nzVar = (sumsSquared -  counts * tmean * tmean) / counts,
                     type = types)

    if (!is.null(x$coefficientNames)) {
#         if(.ccdGetHasIntercept(x)) {
#             row.names(tdf) <- x$coefficientNames[-1]            
#         } else {
            row.names(tdf) <- x$coefficientNames
#         }
    }
    tdf
}

print.ccdData <- function(x,digits=max(3,getOption("digits")-3),show.call=TRUE,...) {
  cat("OHDSI CCD Data Object\n\n")
  
  if (show.call && !is.null(x$call)) {
    cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")  
  }
  cat("     Model: ", x$modelType, "\n", sep="")
  
  if (isInitialized(x)) {
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
