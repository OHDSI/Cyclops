#' @title Create a Cyclops control object that supports in- / out-of-sample hyperparameter search using weights
#'
#' @description
#' \code{createWeightBasedSearchControl} creates a Cyclops control object for use with \code{\link{fitCyclopsModel}}
#' that supports hyperparameter optimization through an auto-search where weight = 1 identifies in-sample observations
#' and weight = 0 identifies out-of-sample observations.
#'
#' @param cvType        Must equal "auto"
#' @param initialValue  Initial value for auto-search parameter
#' @param ...           Additional parameters passed through to \code{\link{createControl}}
#'
#' @return
#' A Cyclops prior object of class inheriting from \code{"cyclopsControl"}
#' for use with \code{fitCyclopsModel}.
#'
#' @export
createWeightBasedSearchControl <- function(cvType = "auto",
                                           initialValue = 1,
                                           ...) {
    if (cvType != "auto") {
        stop("Only auto-search is currently implemented")
    }
    control <- createControl(cvType = cvType, fold = -1, cvRepetitions = 1, minCVData = 2, ...)

    control$setHook <- function(cyclopsData, prior, control, weights, ...) {
        # closure to capture arguments

        if (!prior$useCrossValidation) {
            stop("Prior specification must require cross-validation")
        }

        if (is.null(weights)) {
            stop("Must specify weights")
        }

        control$setHook <- NULL # Do not re-enter call-back

        fit <- fitCyclopsModel(cyclopsData,
                               prior = prior,
                               control = control,
                               weights = weights, ...)

        return(fit)
    }

    return(control)
}
