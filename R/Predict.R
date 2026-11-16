# @file Predict.R
#
# Copyright 2016 Observational Health Data Sciences and Informatics
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

#' @method predict cyclopsFit
#' @title Model predictions
#'
#' @description
#' \code{predict.cyclopsFit} computes model response-scale predictive values for all data rows
#'
#' @details
#' For \code{cox} and \code{cox_counting} fits, new data need \code{times}: the prediction is the
#' survival probability S(t | x) = exp(-Lambda0(t) exp(x'beta)) of each new row at each of
#' \code{times}, with the weighted Breslow baseline cumulative hazard Lambda0 of the fit (see
#' \code{\link{survfit.cyclopsFit}}), as \code{survfit(coxph(..., ties = "breslow"), newdata, ctype = 1,
#' stype = 2)}. Lambda0 is a step function that includes the events at t, and is 0 before the first event.
#' For a stratified fit, \code{newOutcomes} needs a \code{stratumId} column naming a stratum of the fitted
#' data, and each row uses the baseline of its stratum.
#'
#' @param object    A Cyclops model fit object
#' @param newOutcomes  An optional data frame or Andromeda table object, similar to the object used in \code{\link{convertToCyclopsData}}.
#' @param newCovariates  An optional data frame or Andromeda table object, similar to the object used in \code{\link{convertToCyclopsData}}.
#' @param times  Numeric vector of times at which to predict survival, for \code{cox} and \code{cox_counting}
#'               fits with new data (required for them, and not used otherwise).
#' @param ...   Additional arguments
#'
#' @return Without new data, the predictions for the rows of the fitted data. With new data, a vector named
#' by \code{rowId}; for \code{cox} and \code{cox_counting}, a matrix of survival probabilities with a row
#' per \code{rowId} (sorted) and a column per element of \code{times}.
#'
#' @importFrom stats predict
#'
#' @export
predict.cyclopsFit <- function(object, newOutcomes, newCovariates, times = NULL, ...) {
    if (!missing(newOutcomes) && (missing(newCovariates) || is.null(newCovariates)))
        stop("Need to specify both newOutcomes and newCovariates")
    if (!missing(newCovariates) && (missing(newOutcomes) || is.null(newOutcomes)))
        stop("Need to specify both newOutcomes and newCovariates")
    if (missing(newOutcomes)) {
        if (!is.null(times))
            stop("`times` needs newOutcomes and newCovariates")
        # Predict for same data that was used to fit the model:
        .checkInterface(object$cyclopsData, testOnly = TRUE)
        pred <- .cyclopsPredictModel(object$cyclopsData$cyclopsInterfacePtr)
        values <- pred$prediction
        if (is.null(names(values))) {
            names(values) <- object$rowNames
        }
        return(values)
    } else {
        # Predict for new data:
        modelType <- object$cyclopsData$modelType
        isCox <- modelType == "cox" || modelType == "cox_counting"
        if (isCox && is.null(times))
            stop("Specify `times` to predict survival for Cox models")
        if (modelType == "cpr" || modelType == "clr")
            stop("Prediction for conditional models not implemented")

        if (any(class(newOutcomes) != class(newCovariates))) {
            stop("`newCovariates` and `newOutcomes` must be of the same type")
        }

        coefficients <- coef(object)
        if (isCox) { # no intercept
            intercept <- 0
        } else {
            intercept <- coefficients[1]
            coefficients <- coefficients[2:length(coefficients)]
        }
        coefficients <- data.frame(beta = as.numeric(coefficients),
                                   covariateId = as.numeric(names(coefficients)))
        coefficients <- coefficients[coefficients$beta != 0, ]

        if (inherits(newCovariates, "tbl_dbi")) {

            # Optimized for Andromeda
            if (nrow(coefficients) == 0) {
                if ("time" %in% colnames(newOutcomes)) {
                    prediction <- newOutcomes %>%
                        select("rowId", "time", any_of("stratumId")) %>%
                        collect()
                } else {
                    prediction <- newOutcomes %>%
                        select("rowId", any_of("stratumId")) %>%
                        collect()
                }
                prediction <- prediction %>%
                    mutate(value = intercept)
            } else {
                prediction <- inner_join(newCovariates,
                                         coefficients, by = "covariateId", copy = TRUE)

                prediction <- prediction %>%
                    mutate(value = .data$covariateValue * .data$beta) %>%
                    group_by(.data$rowId) %>%
                    summarize(value = sum(.data$value, na.rm = TRUE))

                prediction <- left_join(newOutcomes,
                                         prediction, by = "rowId") %>%
                    collect()

                prediction$value[is.na(prediction$value)] <- 0
                prediction$value <- prediction$value + intercept
            }
            prediction <- prediction %>%
                arrange(.data$rowId)
        } else {
            # Not using Andromeda
            if (nrow(coefficients) == 0) {
                prediction <- newOutcomes
                prediction$value <- intercept
            } else {
                prediction <- merge(newCovariates, coefficients, by = "covariateId")
                prediction$value <- prediction$covariateValue * prediction$beta
                if (nrow(prediction) == 0) { # no new row has a covariate with a nonzero coefficient
                    prediction <- data.frame(rowId = newOutcomes$rowId[0], value = numeric(0))
                } else {
                    prediction <- aggregate(value ~ rowId, data = prediction, sum)
                }
                prediction <- merge(newOutcomes, prediction, by = "rowId", all.x = TRUE)
                prediction$value[is.na(prediction$value)] <- 0
                prediction$value <- prediction$value + intercept
            }
        }

        if (modelType == "lr") {
            link <- function(x) {
                return(1/(1 + exp(0 - x)))
            }
            prediction$value <- link(prediction$value)
        } else if (modelType == "pr") {
            prediction$value <- exp(prediction$value) * prediction$time
        } else if (isCox) {
            survival <- .coxSurvival(object, prediction, times)
            dimnames(survival) <- list(prediction$rowId, times)
            return(survival)
        }

        result <- prediction$value
        names(result) <- prediction$rowId
        return(result)
    }

}

# Survival of new rows (rowId, value = linear predictor, and stratumId for a stratified fit) at times, with
# the weighted Breslow baseline of a cox or cox_counting fit
.coxSurvival <- function(object, prediction, times) {
    delta <- mean(predict(object))
    curves <- .breslowCurves(object, delta) # centered at delta, for numerical stability

    if (length(unique(.cyclopsGetStratumVector(object$cyclopsData))) <= 1) {
        stratum <- rep(1L, nrow(prediction))
    } else {
        if (!"stratumId" %in% colnames(prediction))
            stop("newOutcomes needs a stratumId column to predict from a stratified Cox model")
        if (is.null(object$cyclopsData$stratumIds))
            stop("Stratified prediction needs a model fit to data from convertToCyclopsData()")
        # Internal stratum k is the (k + 1)-th stratumId of the fitted data, in sorted order
        stratum <- match(match(prediction$stratumId, object$cyclopsData$stratumIds) - 1, as.integer(names(curves)))
        if (any(is.na(stratum)))
            stop("newOutcomes has stratumIds without rows of positive weight in the fitted data")
    }

    survival <- matrix(NA_real_, nrow = nrow(prediction), ncol = length(times))
    for (s in unique(stratum)) {
        rows <- which(stratum == s)
        curve <- curves[[s]]
        cumhaz <- c(0, curve$cumhaz)[findInterval(times, curve$time) + 1]
        survival[rows, ] <- exp(-outer(exp(prediction$value[rows] - delta), cumhaz))
    }
    return(survival)
}

#' @title Calculates xbar*beta
#' @description
#' \code{meanLinearPredictor} computes xbar*beta for model fit
#'
#' @param cyclopsFit A Cyclops model fit object
#'
#' @export
meanLinearPredictor <- function(cyclopsFit) {
    cyclopsData = cyclopsFit$cyclopsData
    dataSummary = summary(cyclopsData)
    dataSummary$xbar = dataSummary$nzCount*dataSummary$nzMean/getNumberOfRows(cyclopsData)
    dataSummary$beta = coef(cyclopsFit)[match(rownames(dataSummary),names(coef(cyclopsFit)))]
    dataSummary$xbarBeta = dataSummary$xbar * dataSummary$beta
    delta = sum(dataSummary$xbarBeta)
    return (delta)
}
