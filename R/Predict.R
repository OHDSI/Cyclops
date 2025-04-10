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
#' @param object    A Cyclops model fit object
#' @param newOutcomes  An optional data frame or Andromeda table object, similar to the object used in \code{\link{convertToCyclopsData}}.
#' @param newCovariates  An optional data frame or Andromeda table object, similar to the object used in \code{\link{convertToCyclopsData}}.
#' @param ...   Additional arguments
#'
#' @importFrom stats predict
#'
#' @export
predict.cyclopsFit <- function(object, newOutcomes, newCovariates, ...) {
    if (!missing(newOutcomes) && (missing(newCovariates) || is.null(newCovariates)))
        stop("Need to specify both newOutcomes and newCovariates")
    if (!missing(newCovariates) && (missing(newOutcomes) || is.null(newOutcomes)))
        stop("Need to specify both newOutcomes and newCovariates")
    if (missing(newOutcomes)) {
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
        if (modelType == "cox")
            stop("Prediction for Cox models not implemented")
        if (modelType == "cpr" || modelType == "clr")
            stop("Prediction for conditional models not implemented")

        if (any(class(newOutcomes) != class(newCovariates))) {
            stop("`newCovariates` and `newOutcomes` must be of the same type")
        }

        coefficients <- coef(object)
        intercept <- coefficients[1]
        coefficients <- coefficients[2:length(coefficients)]
        coefficients <- data.frame(beta = as.numeric(coefficients),
                                   covariateId = as.numeric(names(coefficients)))
        coefficients <- coefficients[coefficients$beta != 0, ]

        if (inherits(newCovariates, "tbl_dbi")) {

            # Optimized for Andromeda
            if (nrow(coefficients) == 0) {
                if ("time" %in% colnames(newOutcomes)) {
                    prediction <- newOutcomes %>%
                        select("rowId", "time") %>%
                        collect()
                } else {
                    prediction <- newOutcomes %>%
                        select("rowId") %>%
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
                prediction <- aggregate(value ~ rowId, data = prediction, sum)
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
        }

        result <- prediction$value
        names(result) <- prediction$rowId
        return(result)
    }

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
