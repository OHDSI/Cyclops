# @file cyclops.R
#
# Copyright 2014 Observational Health Data Sciences and Informatics
#
# This file is part of Cyclops
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
#
# @author Observational Health Data Sciences and Informatics
# @author Marc Suchard
# @author Trevor Shaddox


#' Cyclops: Cyclic coordinate descent for logistic, Poisson and survival analysis
#'
#' The Cyclops package incorporates cyclic coordinate descent and
#' majorization-minimization approaches to fit a variety of regression models
#' found in large-scale observational healthcare data.  Implementations focus
#' on computational optimization and fine-scale parallelization to yield
#' efficient inference in massive datasets.
#'
#' @docType package
#' @name cyclops
#' @import Rcpp Matrix MASS
#'
#' @importFrom methods as
#' @importFrom stats aggregate as.formula coef coefficients confint contrasts deviance model.matrix model.offset model.response pchisq poisson qchisq qnorm rbinom rexp rnorm rpois runif terms time vcov
#'
#' @useDynLib Cyclops, .registration = TRUE
NULL

.onUnload <- function (libpath) {
  library.dynam.unload("Cyclops", libpath)
}
