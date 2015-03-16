# @file Multitype.R
#
# Copyright 2014 Observational Health Data Sciences and Informatics
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
#' @title Create a multitype outcome object
#' 
#' @description
#' \code{Multitype} creates a multitype outcome object, usually used as a response variable in a 
#' hierarchical Cyclops model fit.
#' 
#' @param y     Numeric: Response count(s)
#' @param type Numeric or factor: Response type
#' 
#' @return An object of class \code{Multitype} with length equal to the length of \code{y} and \code{type}.
#' @examples
#' Multitype(c(0,1,0), as.factor(c("A","A","B")))
#' 
#' @export
Multitype <- function(y, type) {
    if (missing(y) || !is.numeric(y)) stop("Must have outcome counts")
    if (missing(type)) stop("Must have outcome types")
    if (length(y) != length(type)) stop("Must have equal lengths")
    
    #mt <- data.frame(y, type)
    
    #mt$dim <- c(length(y),2)
    #attr(mt, "dim")  <- c(length(y), 2)
    mt <- cbind(y = y, type = type)
    attr(mt, "contrasts") <- contrasts(type)
    class(mt) <- c("Multitype")
    
    mt
}
