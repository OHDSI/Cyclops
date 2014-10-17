#' @title fitCyclopsModel
#'
#' @description
#' \code{fitCyclopsModel} fits a Cyclops model data object
#'
#' @details
#' This function performs numerical optimization to fit a Cyclops model data object.
#'
#' @param cyclopsData			An OHDSI data object
#' @template prior
#' @param control OHDSI control object, see \code{"\link{control}"}                        
#' @param forceColdStart Logical, forces fitting algorithm to restart at regression coefficients = 0
#' @param returnEstimates Logical, return regression coefficient estimates in Cyclops model fit object 
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
#' cyclopsData <- createCyclopsDataFrame(counts ~ outcome + treatment, modelType = "pr")
#' cyclopsFit <- fitCyclopsModel(cyclopsData, prior = prior("none"))
#' coef(cyclopsFit)
#' confint(cyclopsFit, c("outcome2","treatment3"))
#' predict(cyclopsFit)
#'
fitCyclopsModel <- function(cyclopsData, 
                        prior,
                        control,                        
                        forceColdStart = FALSE,
                        returnEstimates = TRUE) {
		
	cl <- match.call()
	
	# Check conditions
	.checkData(cyclopsData)
	
	if (getNumberOfRows(cyclopsData) < 1 ||
				getNumberOfStrata(cyclopsData) < 1 ||
				getNumberOfCovariates(cyclopsData) < 1) {
		stop("Data are incompletely loaded")
	}
	
	.checkInterface(cyclopsData, forceColdStart)
    
	if (!missing(prior)) { # Set up prior
	    stopifnot(inherits(prior, "cyclopsPrior"))    	
	    prior$exclude <- .checkCovariates(cyclopsData, prior$exclude)
        
# 	    if (prior$priorType != "none" && .cyclopsGetHasIntercept(cyclopsData)) {           	        
# 	        interceptId <- .cyclopsGetInterceptLabel(cyclopsData)
# 	        if (!(interceptId %in% prior$exclude) && !prior$forceIntercept) {	           
# 	           warning("Excluding intercept from regularization")
# 	           prior$exclude <- c(interceptId, prior$exclude)	         	           
#                browser()
# 	        }	        
# 	    }
        
        if (is.null(prior$graph)) {
            graph <- NULL
        } else {
            graph <- .makeHierarchyGraph(cyclopsData, prior$graph)
            if (length(prior$priorType) != length(prior$variance)){
                stop("Prior types and variances have a dimensionality mismatch")
            }
            if (any(prior$priorType != "normal")) {
                stop("Only normal-normal hierarchies are currently supported")
            }
        }

	    .cyclopsSetPrior(cyclopsData$cyclopsInterfacePtr, prior$priorType, prior$variance, 
                         prior$exclude, graph)    		
	}
	
    if (!missing(control)) {
	    .setControl(cyclopsData$cyclopsInterfacePtr, control)
    }
 	
	if (!missing(prior) && prior$useCrossValidation) {
		if (missing(control)) {
			minCVData <- control()$minCVData		
		} else {
			minCVData <- control$minCVData
		}
		if (minCVData > getNumberOfRows(cyclopsData)) { # TODO Not correct; some models CV by stratum
			stop("Insufficient data count for cross validation")
		}
		fit <- .cyclopsRunCrossValidation(cyclopsData$cyclopsInterfacePtr)
	} else {
		fit <- .cyclopsFitModel(cyclopsData$cyclopsInterfacePtr)
	}
	
	if (returnEstimates && fit$return_flag == "SUCCESS") {
		estimates <- .cyclopsLogModel(cyclopsData$cyclopsInterfacePtr)		
		fit <- c(fit, estimates)	
		fit$estimation <- as.data.frame(fit$estimation)
	}	
	fit$call <- cl
	fit$cyclopsData <- cyclopsData
	fit$cyclopsInterfacePtr <- cyclopsData$cyclopsInterfacePtr
	fit$coefficientNames <- cyclopsData$coefficientNames
	fit$rowNames <- cyclopsData$rowNames
	class(fit) <- "cyclopsFit"
	return(fit)
} 

.checkCovariates <- function(cyclopsData, covariates) {
	if (!is.null(covariates)) {
		saved <- covariates
		if (inherits(covariates, "character")) {
			# Try to match names
			covariates <- match(covariates, cyclopsData$coefficientNames)
		}
		covariates = as.numeric(covariates) 
	 
		if (any(is.na(covariates))) {
			stop("Unable to match all covariates: ", paste(saved, collapse = ", "))
		}
	}
	covariates
}

.checkData <- function(x) {
	# Check conditions
	if (missing(x) || is.null(x$cyclopsDataPtr) || class(x$cyclopsDataPtr) != "externalptr") {
		stop("Improperly constructed cyclopsData object")
	}	
	if (.isRcppPtrNull(x$cyclopsDataPtr)) {
		stop("Data object is no longer initialized")			
	}	
}

.checkInterface <- function(x, forceColdStart = FALSE, testOnly = FALSE) {
	if (forceColdStart 
			|| is.null(x$cyclopsInterfacePtr) 
			|| class(x$cyclopsInterfacePtr) != "externalptr" 
			|| .isRcppPtrNull(x$cyclopsInterfacePtr)
		) {
		
		if (testOnly == TRUE) {
			stop("Interface object is not initialized")
		}
		# Build interface
		interface <- .cyclopsInitializeModel(x$cyclopsDataPtr, modelType = x$modelType, computeMLE = TRUE)
		# TODO Check for errors
        assign("cyclopsInterfacePtr", interface$interface, x)
	}
}



#' @title Extract model coefficients
#' 
#' @description
#' \code{coef.cyclopsFit} extracts model coefficients from an OHDSI Cyclops model fit object
#' 
#' @param object    OHDSI Cyclops model fit object
#' @param ...       Other arguments
#' 
#' @return Named numeric vector of model coefficients.
coef.cyclopsFit <- function(object, ...) {
    if (is.null(object$estimation)) {
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
	result
}

#' @title Get hyperparameter
#' 
#' @description
#' \code{getHyperParameter} returns the current hyper parameter in an OHDSI Cyclops model fit object
#' 
#' @param object    An OHDSI Cyclops model fit object
#'
getHyperParameter <- function(object) {
    if (class(object) == "cyclopsFit") {
        object$variance
    } else {
        NULL
    }
}

#' @title Extract log-likelihood
#' 
#' @description
#' \code{logLik} returns the current log-likelihood of the fit in an OHDSI Cyclops model fit object
#' 
#' @param object    An OHDSI Cyclops model fit object
#' @param ...       Additional arguments
#'
logLik.cyclopsFit <- function(object, ...) {
    out <- object$log_likelihood
    attr(out, 'df') <- sum(!is.na(coefficients(object)))
    attr(out, 'nobs') <- getNumberOfRows(object$cyclopsData)
    class(out) <- 'logLik'
    out
}


#' @method print cyclopsFit
#' @title Print an OHDSI Cyclops model fit object
#' 
#' @description
#' \code{print.cyclopsFit} displays information about an OHDSI Cyclops model fit object
#' 
#' @param x    An OHDSI Cyclops model fit object
#' @param show.call Logical: display last call to update the OHDSI Cyclops model fit object
#' @param ...   Additional arguments
#' 
print.cyclopsFit <- function(x, show.call=TRUE ,...) {
  cat("OHDSI Cyclops model fit object\n\n")
  
  if (show.call && !is.null(x$call)) {
    cat("Call: ",paste(deparse(x$call),sep="\n",collapse="\n"),"\n\n",sep="")  
  }
  cat("           Model: ", x$cyclopsData$modelType, "\n", sep="")
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
#' \code{control} builds a Cyclops control object
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
#' @param minCVData					Numeric: Minumim number of data for cross validation
#' @param noiseLevel				String: level of Cyclops screen output (\code{"silent"}, \code{"quiet"}, \code{"noisy"})
#' @param seed                  Numeric: Specify random number generator seed. A null value sets seed via \code{\link{Sys.time}}.
#' 
#' @section Criteria:
#' TODO
#' 
#' @return
#' A Cyclops convergence criteria object of class inheriting from \code{"cyclopsConvergence"} for use with \code{fitCyclopsModel}.
#' 
#' @examples \dontrun{
#' # Add cross-validation example
#' }
control <- function(
		maxIterations = 1000, tolerance = 1E-6, convergenceType = "gradient",
		cvType = "grid", fold = 10, lowerLimit = 0.01, upperLimit = 20.0, gridSteps = 10,
		cvRepetitions = 1,
		minCVData = 100, noiseLevel = "silent",
        seed = NULL) {
	
	validCVNames = c("grid", "auto")
	stopifnot(cvType %in% validCVNames)
	
	validNLNames = c("silent", "quiet", "noisy")
	stopifnot(noiseLevel %in% validNLNames)
	structure(list(maxIterations = maxIterations, tolerance = tolerance, convergenceType = convergenceType,
								 autoSearch = (cvType == "auto"), fold = fold, lowerLimit = lowerLimit, 
								 upperLimit = upperLimit, gridSteps = gridSteps, minCVData = minCVData, 
								 cvRepetitions = cvRepetitions,
								 noiseLevel = noiseLevel,
                                 seed = seed),
						class = "cyclopsControl")
}

#' @title prior
#'
#' @description
#' \code{prior} builds a Cyclops prior object
#'
#' @param priorType     Character: specifies prior distribution.  See below for options
#' @param variance      Numeric: prior distribution variance
#' @param exclude       A vector of numbers or covariateId names to exclude from prior
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
#' @return
#' A Cyclops prior object of class inheriting from \code{"cyclopsPrior"} for use with \code{fitCyclopsModel}.
#' 
prior <- function(priorType, 
                  variance = 1, 
                  exclude = c(), 
                  graph = NULL,
                  useCrossValidation = FALSE,
                  forceIntercept = FALSE) {
	validNames = c("none", "laplace","normal", "hierarchical")
	stopifnot(priorType %in% validNames)	
	if (!is.null(exclude)) {
		if (!inherits(exclude, "character") &&
					!inherits(exclude, "numeric") &&
					!inherits(exclude, "integer")
					) {
			stop(cat("Unable to parse excluded covariates:"), exclude)
		}
	}
	if (priorType == "none" && useCrossValidation) {
		stop("Cannot perform cross validation with a flat prior")
	}
    if (priorType == "hierarchical" && missing(graph)) {
        stop("Must provide a graph for a hierarchical prior")
    }
	structure(list(priorType = priorType, variance = variance, exclude = exclude, 
                   graph = graph,
	               useCrossValidation = useCrossValidation, forceIntercept = forceIntercept), 
              class = "cyclopsPrior")
}

#' @method predict cyclopsFit
#' @title Model predictions
#' 
#' @description
#' \code{predict.cyclopsFit} computes model response-scale predictive values for all data rows
#' 
#' @param object    An OHDSI Cyclops model fit object
#' @param ...   Additional arguments
#' 
predict.cyclopsFit <- function(object, ...) {
	.checkInterface(object, testOnly = TRUE)
	pred <- .cyclopsPredictModel(object$cyclopsInterfacePtr)
 	values <- pred$prediction
 	if (is.null(names(values))) {
 		names(values) <- object$rowNames
 	}
 	values
}

.setControl <- function(cyclopsInterfacePtr, control) {
	if (!missing(control)) { # Set up control
		stopifnot(inherits(control, "cyclopsControl"))
        if (is.null(control$seed)) {
            control$seed <- as.integer(Sys.time())
        }
		.cyclopsSetControl(cyclopsInterfacePtr, control$maxIterations, control$tolerance, 
									 control$convergenceType, control$autoSearch, control$fold, 
									 (control$fold * control$cvRepetitions),
									 control$lowerLimit, control$upperLimit, control$gridSteps, 
                                     control$noiseLevel, control$seed)		
	}	
}

#' @title Extract standard errors
#' 
#' @description
#' \code{getSEs} extracts asymptotic standard errors for specific covariates from an OHDSI Cyclops model fit object.
#' 
#' @details This function first computes the (partial) Fisher information matrix for
#' just the requested covariates and then returns the square root of the diagonal elements of
#' the inverse of the Fisher information matrix.  These are the asymptotic standard errors
#' when all possible covariates are included.
#' When the requested covariates do not equate to all coefficients in the model,
#' then interpretation is more challenging.
#' 
#' @param object    An OHDSI Cyclops model fit object
#' @param covariates    Integer or string vector: list of covariates for which asymptotic standard errors are wanted
#' 
#' @return Vector of standard error estimates
#'
getSEs <- function(object, covariates) {
    .checkInterface(object, testOnly = TRUE)    
    covariates <- .checkCovariates(object$cyclopsData, covariates)
    if (getNumberOfCovariates(object$cyclopsData) != length(covariates)) {
        warning("Asymptotic standard errors are only valid if computed for all covariates simultaneously")
    }
    fisherInformation <- .cyclopsGetFisherInformation(object$cyclopsInterfacePtr, covariates)
    ses <- sqrt(diag(solve(fisherInformation)))
    names(ses) <- object$coefficientNames[covariates]
    ses
}

#' @title confint.cyclopsFit
#'
#' @description
#' \code{confinit.cyclopsFit} profiles the data likelihood to construct confidence intervals of
#' arbitrary level.   TODO: Profile data likelihood or joint distribution of remaining parameters.
#' 
#' @param object    A fitted Cyclops model object
#' @param parm      A specification of which parameters require confidence intervals,
#'                  either a vector of numbers of covariateId names
#' @param level     Numeric: confidence level required
#' @param control   A Cyclops \code{\link{control}} object
#' @param overrideNoRegularization   Logical: Enable confidence interval estimation for regularized parameters
#' @param includePenalty    Logical: Include regularized covariate penalty in profile
#' @param ... Additional argument(s) for methods
#' 
#' @return
#' A matrix with columns reporting lower and upper confidence limits for each parameter.
#' These columns are labelled as (1-level) / 2 and 1 - (1 - level) / 2 in % 
#' (by default 2.5% and 97.5%)
#' 
confint.cyclopsFit <- function(object, parm, level = 0.95, control, 
                               overrideNoRegularization = FALSE,
                               includePenalty = FALSE, ...) {
    .checkInterface(object, testOnly = TRUE)
    .setControl(object$cyclopsInterfacePtr, control)
    parm <- .checkCovariates(object$cyclopsData, parm)
    if (level < 0.01 || level > 0.99) {
        stop("level must be between 0 and 1")
    }
    threshold <- qchisq(level, df = 1) / 2
    
    prof <- .cyclopsProfileModel(object$cyclopsInterfacePtr, parm, threshold,
                                 overrideNoRegularization,
                                 includePenalty)
    prof <- as.matrix(as.data.frame(prof))
    rownames(prof) <- object$coefficientNames[parm]
    qs <- c((1 - level) / 2, 1 - (1 - level) / 2) * 100    
    colnames(prof)[2:3] <- paste(sprintf("%.1f", qs), "%")
    prof
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
#' @param control   A Cyclops \code{\link{control}} object
#' @param overrideNoRegularization   Logical: Enable confidence interval estimation for regularized parameters
#' @param ... Additional argument(s) for methods
#' 
#' @return
#' A matrix with columns reporting lower and upper confidence limits for each parameter.
#' These columns are labelled as (1-level) / 2 and 1 - (1 - level) / 2 in % 
#' (by default 2.5% and 97.5%)
#' 
aconfint <- function(object, parm, level = 0.95, control, 
                               overrideNoRegularization = FALSE, ...) {
    .checkInterface(object, testOnly = TRUE)
    .setControl(object$cyclopsInterfacePtr, control)
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
#' @param control   A Cyclops \code{\link{control}} object
#' @param overrideNoRegularization   Logical: Enable variance-covariance estimation for regularized parameters
#' @param ... Additional argument(s) for methods
#' 
#' @return
#' A matrix of the estimates covariances between all covariate estimates.
#' 
vcov.cyclopsFit <- function(object, control, overrideNoRegularization = FALSE, ...) {
    .checkInterface(object, testOnly = TRUE)
    .setControl(object$cyclopsInterfacePtr, control)
    fisherInformation <- .cyclopsGetFisherInformation(object$cyclopsInterfacePtr, NULL)    
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
    
    graph <-  lapply(0:(nParents - 1), function(n, types) { 0:(nTypes-1) + n * nTypes }, types = nTypes)   
    graph
}
