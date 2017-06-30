#' @title Create a Cyclops control object that supports multiple hyperparameters
#'
#' @description
#' \code{createCrossValidationControl} creates a Cyclops control object for use with \code{\link{fitCyclopsModel}}
#' that supports multiple hyperparameters through an auto-search in one dimension and a grid-search over the remaining
#' dimensions
#'
#' @param outerGrid     List or data.frame of grid parameters to explore
#' @param autoPosition  Vector position for auto-search parameter (concatenated into outerGrid)
#' @param refitAtMaximum Logical: re-fit Cyclops object at maximal cross-validation parameters
#' @param cvType        Must equal "auto"
#' @param initialValue  Initial value for auto-search parameter
#' @param ...           Additional parameters passed through to \code{\link{createControl}}
#'
#' @return
#' A Cyclops prior object of class inheriting from \code{"cyclopsPrior"} and \code{"cyclopsFunctionalPrior"}
#' for use with \code{fitCyclopsModel}.
#'
#' @export
createAutoGridCrossValidationControl <- function(outerGrid,
                                         autoPosition = 1,
                                         refitAtMaximum = TRUE,
                                         cvType = "auto",
                                         initialValue = 1,
                                         ...) {
    if (cvType != "auto") {
        stop("Only auto cross-validation allowed in inner loop")
    }
    control <- createControl(cvType = cvType, ...)

    # Set-up grid
    outerGrid <- as.data.frame(outerGrid)
    numRows <- nrow(outerGrid)
    numCols <- ncol(outerGrid)
    innerAuto <- rep(initialValue, numRows)

    if (autoPosition <= 0 || autoPosition > (numCols + 1)) {
        stop("Auto-position is invalid")
    }

    if (autoPosition == 1) {
        outerGrid <- cbind(innerAuto, outerGrid)
    } else if (autoPosition == (numCols + 1)) {
        outerGrid <- cbind(outerGrid, innerAuto)
    } else {
        outerGrid <- cbind(outerGrid[,1:(autoPosition - 1)],
                           innerAuto,
                           outerGrid[,autoPosition:numCols])
    }

    control$setHook <- function(cyclopsData, prior, control, ...) {
        # closure to capture arguments

        if (!inherits(prior, "cyclopsParameterizedPrior")) {
            stop("Auto-grid cross-validation is only implemented for parameterized priors")
        }

        if (!prior$useCrossValidation) {
            stop("Prior specification must require cross-validation")
        }

        control$setHook <- NULL # Do not re-enter call-back
        # parametrize <- prior$parameterize

        # Delegate parameterized prior functions
        priors <- apply(outerGrid, MARGIN = 1,
                         function(point) {
                             newPrior <- createParameterizedPrior(
                                 priorType = prior$priorType,
                                 parameterize =  function(x) {
                                     point[autoPosition] <- x
                                     prior$parameterize(point)
                                 },
                                 values = prior$values[autoPosition],
                                 useCrossValidation = TRUE,
                                 forceIntercept = prior$forceIntercept)
                             newPrior$point <- point
                             newPrior$point[autoPosition] <- NA
                             return(newPrior)
                         })

        # Run grid search
        searchResults <- lapply(priors, # TODO Could run in parallel
                                function(prior) {
                                    fit <- fitCyclopsModel(cyclopsData,
                                                           prior = prior,
                                                           control = control, ...)
                                    cvInfo <- getCrossValidationInfo(fit)
                                    pt <- cvInfo$point
                                    cvInfo$point <- prior$point
                                    cvInfo$point[is.na(cvInfo$point)] <- pt
                                    list(fit = fit,
                                         cvInfo = cvInfo)
                                })

        gridEvals <- sapply(searchResults,
                            function(fit) {
                                fit$cvInfo$ordinate
                            })

        whichMax <- which(gridEvals == max(gridEvals))
        cvInfo <- searchResults[[whichMax]]$cvInfo

        if (refitAtMaximum) {
            maxFit <- fitCyclopsModel(cyclopsData,
                                      prior = priors[[whichMax]],
                                      control = control, ...)
        } else {
            maxFit <- gridEvals[[whichMax]]
        }

        maxFit$searchResults <- searchResults
        maxFit$cross_validation <- paste0(c(cvInfo$ordinate,
                                            cvInfo$point), collapse = " ")
        return(maxFit)
    }

    return(control)
}
