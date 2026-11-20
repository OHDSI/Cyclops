/**
 * @file RcppIsSorted.cpp
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

#ifndef __RcppIsSorted_cpp__
#define __RcppIsSorted_cpp__


#include <Rcpp.h>
#include "IsSorted.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

namespace {

// A plain integer or double vector: no class (not integer64, factor, Date, ...)
bool isPlain(SEXP v) {
  return (TYPEOF(v) == INTSXP || TYPEOF(v) == REALSXP) && Rf_getAttrib(v, R_ClassSymbol) == R_NilValue;
}

// Reads a plain vector as doubles, as match() and order() compare them: an integer NA as NaN, -0 as 0
struct PlainReader {
  const int* ints = nullptr;
  const double* reals = nullptr;
  explicit PlainReader(SEXP v) {
    if (TYPEOF(v) == INTSXP) ints = INTEGER_RO(v); else reals = REAL_RO(v);
  }
  double operator()(R_xlen_t i) const {
    if (ints != nullptr) return ints[i] == NA_INTEGER ? R_NaN : static_cast<double>(ints[i]);
    return reals[i] == 0.0 ? 0.0 : reals[i];
  }
};

// Whole numbers in [minimum, minimum + range] with range at most limit: their offsets index a direct table
bool wholeRange(const PlainReader& value, R_xlen_t n, double limit, double& minimum, size_t& range) {
  double lo = R_PosInf, hi = R_NegInf;
  for (R_xlen_t i = 0; i < n; ++i) {
    const double v = value(i);
    if (std::isnan(v) || v != std::trunc(v)) return false;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  if (n == 0 || hi - lo > limit) return false;
  minimum = lo;
  range = static_cast<size_t>(hi - lo);
  return true;
}

// The ranks of the values of a plain vector (equal values share a rank), in increasing order of value. A direct table
// when the values are whole numbers in a range at most twice as long as the vector, else a hash map of at most n / 8
// distinct values; false when neither applies, or with NA or NaN (memory stays below about 8 bytes per element)
bool valueRanks(const PlainReader& value, R_xlen_t n, std::vector<int>& rank, int& distinct) {
  double minimum = 0.0;
  size_t range = 0;
  if (wholeRange(value, n, 2.0 * static_cast<double>(n) + 1024.0, minimum, range)) {
    std::vector<int> table(range + 1, -1);
    for (R_xlen_t i = 0; i < n; ++i) table[static_cast<size_t>(value(i) - minimum)] = 0;
    int next = 0;
    for (int& slot : table) if (slot == 0) slot = next++;
    distinct = next;
    rank.resize(static_cast<size_t>(n));
    for (R_xlen_t i = 0; i < n; ++i) rank[static_cast<size_t>(i)] = table[static_cast<size_t>(value(i) - minimum)];
    return true;
  }
  const size_t cap = static_cast<size_t>(n / 8) + 16;
  std::unordered_map<double, int> rankOf;
  for (R_xlen_t i = 0; i < n; ++i) {
    const double v = value(i);
    if (std::isnan(v)) return false;
    rankOf.emplace(v, 0);
    if (rankOf.size() > cap) return false;
  }
  std::vector<double> values;
  values.reserve(rankOf.size());
  for (const auto& entry : rankOf) values.push_back(entry.first);
  std::sort(values.begin(), values.end());
  for (size_t r = 0; r < values.size(); ++r) rankOf[values[r]] = static_cast<int>(r);
  distinct = static_cast<int>(values.size());
  rank.resize(static_cast<size_t>(n));
  for (R_xlen_t i = 0; i < n; ++i) rank[static_cast<size_t>(i)] = rankOf[value(i)];
  return true;
}

// Stable counting sort of the 1-based positions in (nullptr: 1..n), by key[position - 1]; result in out
void countingSort(const std::vector<int>& key, int distinct, const int* in, int* out, R_xlen_t n) {
  std::vector<R_xlen_t> start(static_cast<size_t>(distinct) + 1, 0);
  auto position = [in](R_xlen_t i) { return in == nullptr ? static_cast<int>(i + 1) : in[i]; };
  for (R_xlen_t i = 0; i < n; ++i) ++start[static_cast<size_t>(key[static_cast<size_t>(position(i) - 1)]) + 1];
  for (size_t r = 1; r < start.size(); ++r) start[r] += start[r - 1];
  for (R_xlen_t i = 0; i < n; ++i) out[start[static_cast<size_t>(key[static_cast<size_t>(position(i) - 1)])]++] = position(i);
}

} // namespace

// 1-based positions (of x[order], or of x without an order) where a run of equal values starts, read in place; NULL
// when x is not non-decreasing (then unique(x), which equals x[runStarts(x)] for sorted x, is needed). Integer,
// integer64 or double vectors; order: NULL or an integer vector of positions 1..length(x)
// [[Rcpp::export(".runStarts")]]
SEXP runStarts(SEXP x, SEXP order = R_NilValue) {
  const R_xlen_t n = Rf_xlength(x);
  if (n > std::numeric_limits<int>::max()) return R_NilValue; // positions are integers
  const int* index = nullptr;
  if (!Rf_isNull(order)) {
    if (TYPEOF(order) != INTSXP || Rf_xlength(order) != n) Rcpp::stop("order must be an integer vector as long as x");
    index = INTEGER_RO(order);
    for (R_xlen_t i = 0; i < n; ++i) {
      if (index[i] < 1 || index[i] > n) Rcpp::stop("order must hold positions of x");
    }
  }
  auto at = [index](R_xlen_t i) { return index == nullptr ? i : static_cast<R_xlen_t>(index[i] - 1); };
  std::vector<int> starts;
  auto scan = [&](auto value) -> bool {
    for (R_xlen_t i = 0; i < n; ++i) {
      if (i == 0 || value(at(i)) != value(at(i - 1))) {
        if (i > 0 && !(value(at(i)) > value(at(i - 1)))) return false; // decreasing, or NaN
        starts.push_back(static_cast<int>(i + 1));
      }
    }
    return true;
  };
  bool sorted = false;
  if (TYPEOF(x) == INTSXP) {
    const int* v = INTEGER_RO(x);
    sorted = scan([v](R_xlen_t i) { return v[i] == NA_INTEGER ? R_NaN : static_cast<double>(v[i]); });
  } else if (TYPEOF(x) == REALSXP && Rf_inherits(x, "integer64")) {
    const char* bytes = reinterpret_cast<const char*>(REAL_RO(x));
    sorted = scan([bytes](R_xlen_t i) {
      int64_t v;
      std::memcpy(&v, bytes + i * sizeof(int64_t), sizeof(int64_t));
      return v;
    });
  } else if (TYPEOF(x) == REALSXP) {
    const double* v = REAL_RO(x);
    sorted = scan([v](R_xlen_t i) { return v[i]; });
  }
  if (!sorted) return R_NilValue;
  return wrap(starts);
}

// TRUE when a plain double vector holds a value that is neither NA/NaN nor inside the int64 range (bit64's
// as.integer64() makes it NA with an overflow warning)
// [[Rcpp::export(".anyIdOverflow")]]
bool anyIdOverflow(SEXP x) {
  if (TYPEOF(x) != REALSXP || Rf_inherits(x, "integer64")) return false;
  const double* v = REAL_RO(x);
  const R_xlen_t n = Rf_xlength(x);
  for (R_xlen_t i = 0; i < n; ++i) {
    if (!std::isnan(v[i]) && !(v[i] > -9223372036854775808.0 && v[i] < 9223372036854775808.0)) return true;
  }
  return false;
}

// TRUE when every value of x is in table (both plain vectors without NA or NaN), compared as match() compares them;
// NA when that is not decided here (the caller then uses match()). A bit table when table holds whole numbers in a
// range at most 64 times its length plus the length of x, else a hash set when table is at most an eighth as long as x
// [[Rcpp::export(".allIn")]]
LogicalVector allIn(SEXP x, SEXP table) {
  if (!isPlain(x) || !isPlain(table)) return LogicalVector::create(NA_LOGICAL);
  const PlainReader t(table), r(x);
  const R_xlen_t nt = Rf_xlength(table), n = Rf_xlength(x);
  double minimum = 0.0;
  size_t range = 0;
  if (wholeRange(t, nt, 64.0 * static_cast<double>(nt + n) + 1024.0, minimum, range)) {
    std::vector<bool> present(range + 1, false);
    for (R_xlen_t i = 0; i < nt; ++i) present[static_cast<size_t>(t(i) - minimum)] = true;
    for (R_xlen_t i = 0; i < n; ++i) {
      const double v = r(i);
      if (std::isnan(v)) return LogicalVector::create(NA_LOGICAL);
      const double offset = v - minimum;
      if (v != std::trunc(v) || offset < 0.0 || offset > static_cast<double>(range) || !present[static_cast<size_t>(offset)]) {
        return LogicalVector::create(false);
      }
    }
    return LogicalVector::create(true);
  }
  if (nt > n / 8 + 16) return LogicalVector::create(NA_LOGICAL);
  std::unordered_set<double> values;
  values.reserve(static_cast<size_t>(nt));
  for (R_xlen_t i = 0; i < nt; ++i) {
    const double v = t(i);
    if (std::isnan(v)) return LogicalVector::create(NA_LOGICAL);
    values.insert(v);
  }
  for (R_xlen_t i = 0; i < n; ++i) {
    const double v = r(i);
    if (std::isnan(v)) return LogicalVector::create(NA_LOGICAL);
    if (values.find(v) == values.end()) return LogicalVector::create(false);
  }
  return LogicalVector::create(true);
}

// order(first, second) for plain vectors (ties by position, -0 as 0), by two stable counting sorts over the ranks of
// their values: at most about 12 bytes per element beside nothing else, where order() takes about 23 and a sorted copy
// of the data frame follows. NULL when not decided here (other types or classes, NA or NaN, values too spread for
// valueRanks()): the caller then uses order()
// [[Rcpp::export(".sortOrder")]]
SEXP sortOrder(SEXP first, SEXP second) {
  const R_xlen_t n = Rf_xlength(first);
  if (!isPlain(first) || !isPlain(second) || Rf_xlength(second) != n || n > std::numeric_limits<int>::max()) {
    return R_NilValue;
  }
  const PlainReader firstValue(first), secondValue(second); // (data pointers before any allocation here)
  std::vector<int> firstRank, secondRank;
  int firstDistinct = 0, secondDistinct = 0;
  if (!valueRanks(firstValue, n, firstRank, firstDistinct)) return R_NilValue;
  if (!valueRanks(secondValue, n, secondRank, secondDistinct)) return R_NilValue;
  std::vector<int> bySecond(static_cast<size_t>(n));
  countingSort(secondRank, secondDistinct, nullptr, bySecond.data(), n); // by second, ties by position
  std::vector<int>().swap(secondRank);
  IntegerVector result(n);
  countingSort(firstRank, firstDistinct, bySecond.data(), result.begin(), n); // by first, ties as sorted by second
  return result;
}

#endif // __RcppIsSorted_cpp__
