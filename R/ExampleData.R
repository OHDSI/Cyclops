# @file ExampleData.R
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

#' Oxford self-controlled case series data
#'
#' A dataset containing the MMR vaccination / meningitis in Oxford example from
#' Farrington and Whitaker.  There are 10 patients comprising 38 unique exposure intervals.
#'
#' @name oxford
#' 
#' @docType data
#' 
#' @usage data(oxford)
#' 
#' @format A data frame with 38 rows and 6 variables:
#' \describe{
#'   \item{indiv}{patient identifier}
#'   \item{event}{number of events in interval}
#'   \item{interval}{interval length in days}
#'   \item{agegr}{age group}
#'   \item{exgr}{exposure group}
#'   \item{loginterval}{log interval length}
#'   ...
#' }
#' @source \url{http://statistics.open.ac.uk/sccs/r.htm}
NULL
