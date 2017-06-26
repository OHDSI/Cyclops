#' @title Create a Cyclops parameterized prior object
#'
#' @description
#' \code{createParameterizedPrior} creates a Cyclops prior object for use with \code{\link{fitCyclopsModel}}
#' in which arbitrary \code{R} functions parameterize the prior location and variance.
#'
#' @param priorType     Character vector: specifies prior distribution.  See below for options
#' @param parameterize  Function list: parameterizes location and variance
#' @param values        Numeric vector: initial parameter values
#' @param useCrossValidation    Logical: Perform cross-validation to determine \code{parameters}.
#' @param forceIntercept  Logical: Force intercept coefficient into prior
#'
#' @return
#' A Cyclops prior object of class inheriting from \code{"cyclopsPrior"} and \code{"cyclopsFunctionalPrior"}
#' for use with \code{fitCyclopsModel}.
#'
#' @export
createParameterizedPrior <- function(priorType,
                                  parameterize,
                                  values,
                                  useCrossValidation = FALSE,
                                  forceIntercept = FALSE) {

    validNames = c("none", "laplace","normal")
    stopifnot(priorType %in% validNames)

    if (missing(values)) {
        stop("Must provide parameter values")
    }

    parameterizationEval <- parameterize(values)
    if (!inherits(parameterizationEval, "list")) {
        stop("Parameterization function should return a list")
    }

    if (length(priorType) != length(parameterizationEval)) {
        stop("Prior types and functions have a dimensionality mismatch")
    }

    if (all(priorType == "none") && useCrossValidation) {
        stop("Cannot perform cross validation with a flat prior")
    }

    setHook <- function(cyclopsData) {
        # closure to capture arguments
        if (length(priorType) > 1) {
            if (length(priorType) != getNumberOfCovariates(cyclopsData)) {
                stop("Length of priors must equal the number of covariates")
            }
        }

        if (priorType[1] != "none" && .cyclopsGetHasIntercept(cyclopsData) && !forceIntercept) {
            priorType[1] <- "none"
            warning("Excluding intercept from regularization")
        }

        .cyclopsSetParameterizedPrior(cyclopsData$cyclopsInterfacePtr,
                                      priorType,
                                      parameterize,
                                      values,
                                      excludeNumeric = NULL)
    }

    structure(list(priorType = priorType,
                   useCrossValidation = useCrossValidation,
                   forceIntercept = forceIntercept,
                   setHook = setHook,
                   parameterize = parameterize,
                   values = values),
              class = c("cyclopsPrior", "cyclopsParameterizedPrior"))
}
