/**
 * @file IsSorted.cpp
 *
 * This file is part of Cyclops
 *
 * Copyright 2020 Observational Health Data Sciences and Informatics
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

 #ifndef __IsSorted_cpp__
 #define __IsSorted_cpp__

 #include <Rcpp.h>
 #include <iostream>
 #include "IsSorted.h"


 using namespace Rcpp ;

 namespace ohdsi {
     namespace cyclops {

         namespace {
             // Reads a numeric column as doubles without a copy (a NumericVector of an integer column is a coerced
             // copy); an integer NA reads as NaN, as in that copy
             struct DoubleReader {
                 const int* ints = nullptr;
                 const double* reals = nullptr;
                 explicit DoubleReader(SEXP x) {
                     if (TYPEOF(x) == INTSXP) {
                         ints = INTEGER_RO(x);
                     } else if (TYPEOF(x) == LGLSXP) {
                         ints = LOGICAL_RO(x);
                     } else {
                         reals = REAL_RO(x);
                     }
                 }
                 double operator[](size_t i) const {
                     if (ints != nullptr) {
                         return ints[i] == NA_INTEGER ? NA_REAL : static_cast<double>(ints[i]);
                     }
                     return reals[i];
                 }
             };

             // as the NumericVector conversion: other types are coerced (copied); the RObject keeps the copy protected
             RObject numericColumn(SEXP x) {
                 if (TYPEOF(x) == INTSXP || TYPEOF(x) == LGLSXP || TYPEOF(x) == REALSXP) return RObject(x);
                 return RObject(NumericVector(x));
             }
         }

         bool IsSorted::isSorted(const DataFrame& dataFrame,const std::vector<std::string>& indexes,const std::vector<bool>& ascending){
             const R_xlen_t nrows = dataFrame.nrows(); // (an int: data frames have fewer than 2^31 rows)
             unsigned int ncols = indexes.size();
             if (nrows == 1){
                 return true;
             }
             std::vector<RObject> keep(ncols);
             std::vector<DoubleReader> columns;
             columns.reserve(ncols);

             for (unsigned int column=0; column<ncols; column++){
                 keep[column] = numericColumn(dataFrame[indexes.at(column)]);
                 columns.emplace_back(keep[column]);
             }

             for(R_xlen_t row=1; row<nrows; row++) {
                 for (unsigned int column=0; column<ncols; column++){
                     if (ascending.at(column)){
                         if (columns[column][row] > columns[column][row-1]){
                             break;
                         } else if (columns[column][row] < columns[column][row-1]){
                             return false;
                         }
                     } else {
                         if (columns[column][row] < columns[column][row-1]){
                             break;
                         } else if (columns[column][row] > columns[column][row-1]){
                             return false;
                         }
                     }
                 }
             }
             return true;
         }

         bool IsSorted::isSorted(const List& vectorList, const std::vector<bool>& ascending){
             unsigned int ncols = vectorList.size();
             std::vector<RObject> keep(ncols);
             std::vector<DoubleReader> columns;
             columns.reserve(ncols);
             for (unsigned int column=0; column<ncols; column++){
                 keep[column] = numericColumn(vectorList[column]);
                 columns.emplace_back(keep[column]);
             }
             const R_xlen_t nrows = Rf_xlength(keep[0]);
             if (nrows == 1){
                 return true;
             }

             for(R_xlen_t row=1; row<nrows; row++) {
                 for (unsigned int column=0; column<ncols; column++){
                     if (ascending.at(column)){
                         if (columns[column][row] > columns[column][row-1]){
                             break;
                         } else if (columns[column][row] < columns[column][row-1]){
                             return false;
                         }
                     } else {
                         if (columns[column][row] < columns[column][row-1]){
                             break;
                         } else if (columns[column][row] > columns[column][row-1]){
                             return false;
                         }
                     }
                 }
             }
             return true;
         }

     }
 }

 #endif // __IsSorted_cpp__
