# @file getFineGrayWeights.R
#
# Copyright 2020 Observational Health Data Sciences and Informatics
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
#' @title Creates a \code{Surv} object that forces in competing risks and the IPCW needed for Fine-Gray estimation.
#'
#' @description
#' \code{getFineGrayWeights} creates a list \code{Surv} object and vector of weights required for estimation.
#'
#' @param ftime    Numeric: Observed event (failure) times
#' @param fstatus  Numeric: Observed event (failure) types
#' @param cencode Numeric: Code to denote censored observations (Default is 0)
#' @param failcode Numeric: Code to denote event of interest (Default is 1)
#'
#' @return A list that returns both an object of class \code{Surv} that forces in the competing risks indicators and a vector of weights needed for parameter estimation.
#' @examples
#' ftime <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#' fstatus <- c(1, 2, 0, 1, 2, 0, 1, 2, 0, 1)
#' getFineGrayWeights(ftime, fstatus, cencode = 0, failcode = 1)
#' @importFrom survival survfit Surv
#' @importFrom stats approx
#' @export
getFineGrayWeights <- function(ftime, fstatus,
                               cencode = 0, failcode = 1) {

    # Check for errors
    if(!cencode %in% unique(fstatus)) stop("cencode must be a valid value from fstatus")
    if(!failcode %in% unique(fstatus)) stop("failcode must be a valid value from fstatus")
    if(any(ftime < 0)) stop("all values of ftime must be positive valued")

    obj <- suppressWarnings(survival::Surv(ftime, fstatus)) # Suppress warning given by Surv function
    obj[, 2] <- fstatus # Changes NA's to competing risks indicators
    cenind   <- ifelse(fstatus == cencode, 1, 0)
    obj[, 2] <- ifelse(obj[, 2] == failcode, 1, 2 * (1 - cenind)) # Changes competing risks to 2

    # Create IPCW here (see original F&G code)
    u <- do.call('survfit', list(formula = Surv(ftime, cenind) ~ 1,
                                 data = data.frame(ftime, cenind)))
    u <- approx(c(0, u$time, max(u$time) * (1 + 10 * .Machine$double.eps)), c(1, u$surv, 0),
                xout = ftime * (1 - 100 * .Machine$double.eps), method = 'constant',
                f = 0, rule = 2)
    uuu <- u$y
    return(list(surv = obj,
                weights = uuu))
}
