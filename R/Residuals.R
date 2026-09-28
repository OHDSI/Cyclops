# @file Residuals.R
#
# Copyright 2024 Observational Health Data Sciences and Informatics
#
# This file is part of cyclops
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' @method residuals cyclopsFit
#' @title Model residuals
#'
#' @description
#' \code{residuals.cyclopsFit} computes model residuals for Cox model-based Cyclops objects
#'
#' @param object    A Cyclops model fit object
#' @param type      Character string indicating the type of residual desires. Possible
#'                  values are "schoenfeld" and "scaledsch" (scaled Schoenfeld residuals, which
#'                  need all parameters and \code{vcov()})
#' @param parm      A specification of which parameters require residuals,
#'                  either a vector of numbers or covariateId names; default is all
#' @param ...		Additional parameters for compatibility with S3 parent function
#'
#' @return One residual per event, ordered by stratum and time and named by event time: a vector
#' for one parameter, otherwise a matrix with one column per parameter
#'
#' @importFrom stats residuals
#'
#' @export
residuals.cyclopsFit <- function(object, type = "schoenfeld", parm = NULL, ...) {

    .checkInterface(object$cyclopsData, testOnly = TRUE)

    if (object$cyclopsData$modelType != "cox" && object$cyclopsData$modelType != "cox_counting") {
        stop("Residuals for only Cox models are currently implemented")
    }
    if (type != "schoenfeld" && type != "scaledsch") {
        stop("Only Schoenfeld residuals are currently implemented")
    }

    covariateIds <- getCovariateIds(object$cyclopsData)
    covariates <- if (is.null(parm)) covariateIds else .checkCovariates(object$cyclopsData, parm)
    if (type == "scaledsch" && length(covariates) != length(covariateIds)) {
        stop("Scaled Schoenfeld residuals need all parameters")
    }

    # One covariate at a time; every call returns the events in the same order
    columns <- lapply(seq_along(covariates), function(j) {
        .cyclopsGetSchoenfeldResiduals(object$interface, covariates[j])
    })
    res <- columns[[1]]
    ord <- order(res$strata, res$times)
    result <- matrix(unlist(lapply(columns, function(column) column$residuals[ord])),
                     ncol = length(columns))
    colnames(result) <- names(coef(object))[match(as.character(covariates), as.character(covariateIds))]

    if (type == "scaledsch") { # as in survival: residuals %*% variance * events + coefficients
        nEvents <- nrow(result)
        result <- result %*% vcov(object) * nEvents + rep(coef(object), each = nEvents)
    }

    if (ncol(result) == 1) {
        result <- result[, 1]
        names(result) <- res$times[ord]
    } else {
        rownames(result) <- res$times[ord]
    }

    tbl <- table(res$strata)
    if (dim(tbl) > 1) {
        names(tbl) <- paste0("stratum=", names(tbl))
        attr(result, "strata") <- tbl
    }

    return(result)
}

#' @title Test hazard ratio proportionality assumption
#'
#' @description
#' \code{testProportionality} tests the hazard ratio proportionality assumption
#' of a Cyclops model fit object using a score test
#'
#' @param object    A Cyclops model fit object
#' @param parm      A specification of which parameters require a proportionality test,
#'                  either a vector of numbers or covariateId names
#' @param transform Same as `transform` in `cox.zph`
#'
#' @export
testProportionality <- function(object, parm = NULL, transform = "km") {

    .checkInterface(object$cyclopsData, testOnly = TRUE)

    if (object$cyclopsData$modelType != "cox" && object$cyclopsData$modelType != "cox_counting") {
        stop("Proportionality test for only Cox models are currently implemented")
    }

    nCovariates <- getNumberOfCovariates(object$cyclopsData)
    if (nCovariates != 1) {
        stop("Only single-covariate models are currently implemented")
    }

    times <- getTimeVector(object$cyclopsData)
    y <- getYVector(object$cyclopsData)
    startTimes <- getStartTimeVector(object$cyclopsData)
    if (length(startTimes) > 0) {
        survY <- survival::Surv(startTimes, times, y) # Kaplan-Meier with delayed entry
    } else {
        survY <- survival::Surv(time = times, event = y)
    }
    if (is.character(transform)) {
        tname <- transform
        ttimes <- switch(transform, identity = times, rank = rank(times),
                         log = log(times),
                         km = {
                             temp <- survival::survfitKM(
                                 factor(rep(1L, nrow(survY))), survY,
                                 se.fit = FALSE)
                             indx <- findInterval(times, temp$time, left.open = TRUE)
                             1 - c(1, temp$surv)[indx + 1]
                             },
                         stop("Unrecognized transform"))
    } else {
        tname <- deparse(substitute(transform))
        if (length(tname) > 1) {
            tname <- "user"
        }
        ttimes <- transform(times)
    }
    transformedTimes <- ttimes - mean(ttimes[y == 1])

    res <- .cyclopsTestProportionality(object$interface, NULL, transformedTimes)
    nCovariates <- 1 # TODO Remove
    res$hessian <- matrix(res$hessian, nrow = (nCovariates + 1))

    if (any(abs(res$gradient[1:nCovariates]) > 1E-5)) {
        stop("Internal state of Cyclops 'object' is not at its mode: ", res$gradient)
    }

    u <- c(rep(0, nCovariates), res$gradient[nCovariates + 1])
    test <- drop(solve(res$hessian, u) %*% u)
    df <- 1

    tbl <- cbind(test, df, pchisq(test, df, lower.tail = FALSE))

    names <- as.character(getCovariateIds(object$cyclopsData)[1])
    if (!is.null(object$cyclopsData$coefficientNames)) {
        names <- object$coefficientNames[1]
    }

    dimnames(tbl) <- list(names, c("chisq", "df", "p"))
    res$table <- tbl

    class(res) <- "cyclopsZph"

    return(res)
}

#' @method print cyclopsZph
#' @importFrom stats printCoefmat
#'
#' @export
print.cyclopsZph <- function(x, digits = max(options()$digits - 4, 3),
                             signif.stars = FALSE, ...) {
    invisible(printCoefmat(x$table, digits=digits, signif.stars=signif.stars,
                           P.values=TRUE, has.Pvalue=TRUE, ...))
}
