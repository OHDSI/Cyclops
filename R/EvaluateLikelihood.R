#' @title Evaluate the likelihood of a Cyclops model at given coefficients
#'
#' @description
#' \code{evaluateLikelihood} sets the coefficients of a Cyclops data object and returns the log-likelihood there,
#' with its gradient and Fisher information if requested. It is meant for packages that build their own estimators
#' on Cyclops (for example through \code{prior$fitHook}): one call sets the coefficients and the weights and then
#' evaluates, so no result depends on an earlier fit.
#'
#' @param cyclopsData   A Cyclops data object. It does not need to have been fitted.
#' @param coefficients  Numeric vector with a value for each covariate, in the order of \code{coef()} of a fit (the
#'                      intercept included, an offset covariate not), or named by covariate as \code{coef()} names
#'                      them.
#' @param weights       Optional vector of non-negative weights, one per data row, in the order of the data as
#'                      given. Rows of weight 0 do not count. \code{NULL}: the weights stored with the data, else
#'                      every row once. The weights of an earlier fit are never used.
#' @param gradient      Logical: return the gradient of the log-likelihood for every covariate.
#' @param information   Optional covariates (ids as numeric, integer64 or character, or names as in \code{coef()},
#'                      such as \code{"(Intercept)"}) for which to return the Fisher information matrix.
#' @param informationDiagonal Optional covariates, as for \code{information}, for which to return only the diagonal
#'                      of the Fisher information. Cheaper than the matrix for many covariates.
#' @param threads       Number of threads for the parts that use them (the Fisher information of logistic
#'                      regression); -1 for all cores. Results do not depend on it.
#'
#' @return A list with
#' \item{logLikelihood}{the log-likelihood at \code{coefficients}, without the prior (for least squares, minus the
#'       residual sum of squares)}
#' \item{gradient}{(if \code{gradient}) its gradient, named by covariate}
#' \item{information}{(if \code{information} is given) the Fisher information matrix (minus the Hessian of the
#'       log-likelihood) of those covariates, with their names as dimnames}
#' \item{informationDiagonal}{(if \code{informationDiagonal} is given) the diagonal of the Fisher information of
#'       those covariates, named}
#'
#' @details
#' The Fisher information of Cox and conditional models with weights, and of models with weights that are not
#' whole numbers other than logistic, Poisson and least-squares regression, is not available and stops with an
#' error. A later fit of the same data object starts from these coefficients unless it is given others.
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(200)
#' y <- rbinom(200, 1, plogis(0.5 * x))
#' data <- createCyclopsData(y ~ x, modelType = "lr")
#' evaluateLikelihood(data, c(0, 0.5), information = c("(Intercept)", "x"))
#'
#' @export
evaluateLikelihood <- function(cyclopsData, coefficients, weights = NULL, gradient = TRUE,
                               information = NULL, informationDiagonal = NULL, threads = 1) {
    .checkData(cyclopsData)
    .checkInterface(cyclopsData)
    ptr <- cyclopsData$cyclopsInterfacePtr
    names <- .coefficientNames(cyclopsData)

    if (!is.numeric(coefficients) || length(coefficients) != length(names)) {
        stop("coefficients needs a value for each of the ", length(names), " covariates")
    }
    if (!is.null(names(coefficients))) {
        if (!setequal(names(coefficients), names)) {
            stop("The names of coefficients do not match the covariates")
        }
        coefficients <- coefficients[names]
    }
    beta <- as.numeric(coefficients)
    if (.cyclopsGetHasOffset(cyclopsData)) {
        beta <- c(1.0, beta)
    }

    if (is.null(weights)) {
        weights <- cyclopsData$weights # already in data order
    } else {
        if (length(weights) != getNumberOfRows(cyclopsData)) {
            stop("Must provide a weight for each data row")
        }
        if (any(weights < 0)) {
            stop("Only non-negative weights are allowed")
        }
        if (!is.null(cyclopsData$sortOrder)) {
            weights <- weights[cyclopsData$sortOrder]
        }
    }

    .cyclopsSetThreads(ptr, as.integer(threads))
    .cyclopsSetWeights(ptr, if (is.null(weights)) NULL else as.numeric(weights))
    .cyclopsSetBeta(ptr, beta)

    result <- list(logLikelihood = .cyclopsGetLogLikelihood(ptr))
    if (gradient) {
        result$gradient <- .cyclopsGetLogLikelihoodGradient(ptr)
        names(result$gradient) <- names
    }
    # Least squares: the log-likelihood is -RSS (a fixed variance of 1/2), while the engine's Fisher information is
    # X'X, the unit-variance information that vcov() uses; twice it is minus the Hessian of the log-likelihood here
    scale <- if (cyclopsData$modelType == "ls") 2 else 1
    known <- as.character(getCovariateIds(cyclopsData))
    covariateIds <- function(covariates) {
        ids <- .checkCovariates(cyclopsData, covariates)
        if (!all(as.character(ids) %in% known)) {
            stop("Unknown covariates: ", paste(covariates[!(as.character(ids) %in% known)], collapse = ", "))
        }
        ids
    }
    if (!is.null(information)) {
        ids <- covariateIds(information)
        labels <- .coefficientNames(cyclopsData, ids)
        result$information <- matrix(.cyclopsGetFisherInformation(ptr, ids), length(ids), length(ids),
                                     dimnames = list(labels, labels)) * scale
    }
    if (!is.null(informationDiagonal)) {
        ids <- covariateIds(informationDiagonal)
        result$informationDiagonal <- -.cyclopsGetLogLikelihoodHessianDiagonal(ptr, ids)
        names(result$informationDiagonal) <- .coefficientNames(cyclopsData, ids)
    }
    result
}
