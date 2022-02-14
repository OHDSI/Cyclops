# @file timeEffects.R
#
# Copyright 2022 Observational Health Data Sciences and Informatics
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

#' @title Creates long outcome table for fitting pooled logistic regression.
#'
#' @description
#' \code{convertToLongOutcome} creates a long outcome table for fitting pooled logistic regression.
#'
#' @param time    Numeric: Observed event time
#' @param status  Numeric: Observed event status
#'
#' @return A long outcome table for fitting pooled logistic regression.
#' @examples
#' time <- c(5, 4, 3, 2, 2, 2, 1, 1)
#' status <- c(0, 1, 0, 0, 1, 1, 0, 1)
#' convertToLongOutcome(time, status)
#' @export
convertToLongOutcome <- function(time, status) {
    if(length(time) != length(status)) stop("Vector length mismatch.")

    n <- length(time)
    maxTime <- max(time[status == 1])

    realTime <- rep(maxTime : 1, n)
    realY <- rep(NA, n * maxTime)
    valid <- 1 * (realTime == 1)

    for(i in 1 : maxTime){
        realY[realTime == i] <- status * (time == i)
        valid[realTime == i] <- (time >= i)
    }

    longOutcome <- data.frame(stratumId = as.numeric(gl(n, maxTime)),
                              time = realTime,
                              y = realY,
                              valid)
    longOutcome <- longOutcome[longOutcome$valid == 1, c("stratumId", "time", "y")]
    longOutcome$rowId <- 1:dim(longOutcome)[1]
    longOutcome <- longOutcome[, c(4, 1:3)]

    return(longOutcome)
}
