/**
 * @file RcppIsSorted.cpp
 *
 * This file is part of Cyclops
 *
 * Copyright 2014 Observational Health Data Sciences and Informatics
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __RcppIsSorted_cpp__
#define __RcppIsSorted_cpp__


#include <Rcpp.h>
#include "IsSorted.h"

using namespace Rcpp;

// [[Rcpp::export(".isSorted")]]
bool isSorted(const DataFrame& dataFrame,const std::vector<std::string>& indexes,const std::vector<bool>& ascending){

  using namespace ohdsi::cyclops;

  return IsSorted::isSorted(dataFrame,indexes,ascending);
}

// [[Rcpp::export(".isSortedVectorList")]]
bool isSortedVectorList(const List& vectorList,const std::vector<bool>& ascending){

  using namespace ohdsi::cyclops;

  return IsSorted::isSorted(vectorList, ascending);
}

#endif // __RcppIsSorted_cpp__
