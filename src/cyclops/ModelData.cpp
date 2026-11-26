/*
 * ModelData.cpp
 *
 *  Created on: August, 2012
 *      Author: msuchard
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <list>
#include <functional>

// #include <boost/iterator/permutation_iterator.hpp>
// #include <boost/iterator/transform_iterator.hpp>

#include "ModelData.h"

namespace bsccs {

using std::string;
using std::vector;

template <typename RealType>
ModelData<RealType>::ModelData(
    ModelType _modelType,
    loggers::ProgressLoggerPtr _log,
    loggers::ErrorHandlerPtr _error
    ) : modelType(_modelType), nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false), isFinalized(false),
        lastStratumMap(0,0), sparseIndexer(X), log(_log), error(_error), touchedY(true), touchedX(true) {
	// Do nothing
}

template <typename RealType>
size_t ModelData<RealType>::getColumnIndex(const IdType covariate) const {
    int index = getColumnIndexByName(covariate);
    if (index == -1) {
        std::ostringstream stream;
        stream << "Variable " << covariate << " is unknown";
        error->throwError(stream);
    }
    return index;
}

template <typename RealType>
void ModelData<RealType>::moveTimeToCovariate(bool takeLog) {
    X.push_back(NULL, make_shared<RealVector>(offs.begin(), offs.end()), DENSE); // TODO Remove copy?
}

namespace { // anonymous

template <typename RealType>
void copyAssign(std::vector<RealType>& destination, const std::vector<double>& source) {
    if (destination.size() != source.size()) {
        destination.resize(source.size());
    }
    std::copy(std::begin(source), std::end(source), std::begin(destination));
}

}; // namespace anonymous

template <typename RealType>
void ModelData<RealType>::loadY(
		const std::vector<IdType>& oStratumId,
		const std::vector<IdType>& oRowId,
		const std::vector<double>& oY,
		const std::vector<double>& oTime) {

    bool previouslyLoaded = y.size() > 0;

    if (   (oStratumId.size() > 0 && oStratumId.size() != oY.size())
        || (oRowId.size() > 0 && oRowId.size() != oY.size())
        || (oTime.size() > 0 && oTime.size() != oY.size())
        || (previouslyLoaded && y.size() != oY.size())
    ) {
        std::ostringstream stream;
        stream << "Mismatched outcome column dimensions";
        error->throwError(stream);
    }

    copyAssign(y, oY); // y = oY; // copy assignment
	if (oTime.size() == oY.size()) {
		copyAssign(offs, oTime); // offs = oTime; // copy assignment
	}
	touchedY = true;

	if (!previouslyLoaded) { // Load stratum and row IDs

        if (oRowId.size() > 0) {
		    pid.reserve(oRowId.size());
        }

		bool processStrata = oStratumId.size() > 0;

		bool consecutive = oRowId.size() > 0;
		for (size_t i = 1; i < oRowId.size() && consecutive; ++i) {
			consecutive = static_cast<uint64_t>(oRowId[i]) - static_cast<uint64_t>(oRowId[i - 1]) == 1; // no overflow
		}
		consecutiveRowIds = consecutive;
		firstRowId = consecutive ? oRowId[0] : 0;

		for (size_t i = 0; i < oRowId.size(); ++i) { // ignored if oRowId.size() == 0
			IdType currentRowId = oRowId[i];
			if (!consecutive) {
				rowIdMap[currentRowId] = i;
			}

			// Begin code duplication
			if (processStrata) {
				IdType cInStratum = oStratumId[i];
				if (getX().nRows == 0) {
					lastStratumMap.first = cInStratum;
					lastStratumMap.second = 0;
					nPatients++;
				} else {
					if (cInStratum != lastStratumMap.first) {
						lastStratumMap.first = cInStratum;
						lastStratumMap.second++;
						nPatients++;
					}
				}
				pid.push_back(lastStratumMap.second);
			}
			++getX().nRows;

			if (!consecutive) {
				// TODO Check timing on adding label as string
				std::stringstream ss;
				ss << currentRowId;
				labels.push_back(ss.str());
			}
			// End code duplication
		}

		if (oRowId.size() == 0) getX().nRows = y.size();

		if (!processStrata) nPatients = getX().nRows;
	} else {
		if (oStratumId.size() > 0 || oRowId.size() > 0) {
			std::ostringstream stream;
			stream << "Strata or row IDs are already loaded";
			error->throwError(stream);
		}
	}

// 	std::cerr << "Row ID map:" << std::endl;;
// 	for (const auto& pair : rowIdMap) {
// 		std::cerr << "\t" << pair.first << " -> " << pair.second << std::endl;
// 	}
}

template <typename RealType>
void ModelData<RealType>::loadStartTime(const std::vector<double>& oStartTime) {

    if (oStartTime.size() != y.size() || oStartTime.size() != offs.size()) {
        std::ostringstream stream;
        stream << "Mismatched start time dimensions";
        error->throwError(stream);
    }

    for (size_t i = 0; i < oStartTime.size(); ++i) {
        if (!(oStartTime[i] < offs[i])) {
            std::ostringstream stream;
            stream << "Start times must be less than stop times";
            error->throwError(stream);
        }
    }

    copyAssign(startTime, oStartTime);
}

template <typename RealType>
void ModelData<RealType>::loadSubjectId(const std::vector<IdType>& oSubjectId) {

    if (oSubjectId.size() != y.size()) {
        std::ostringstream stream;
        stream << "Mismatched subject ID dimensions";
        error->throwError(stream);
    }

    // Consecutive indices in order of first appearance, so one row per subject gives index = row
    unordered_map<IdType, int> index;
    subjectIndex.resize(oSubjectId.size());
    for (size_t i = 0; i < oSubjectId.size(); ++i) {
        const int next = static_cast<int>(index.size());
        subjectIndex[i] = index.insert(std::make_pair(oSubjectId[i], next)).first->second;
    }
}

namespace {

// Room for extra more entries: exactly for a new vector, else at least doubling (a column appended to batch by batch
// is not copied once per batch)
template <typename Vector>
void reserveFor(Vector& vector, size_t extra) {
	const size_t needed = vector.size() + extra;
	if (vector.capacity() < needed) {
		const size_t doubled = vector.capacity() <= vector.max_size() / 2 ? 2 * vector.capacity() : vector.max_size();
		vector.reserve(vector.empty() ? needed : std::max(needed, doubled));
	}
}

} // namespace

template <typename RealType>
void ModelData<RealType>::loadFineGray(const std::vector<int>& group, const std::vector<int>& censoringGroup,
        const std::vector<double>& censoringTime, const std::vector<double>& censoringSurvival) {

    if (group.size() != y.size()) {
        std::ostringstream stream;
        stream << "Mismatched Fine-Gray group dimensions";
        error->throwError(stream);
    }
    if (censoringTime.size() != censoringGroup.size() || censoringSurvival.size() != censoringGroup.size()) {
        std::ostringstream stream;
        stream << "Mismatched Fine-Gray censoring distribution dimensions";
        error->throwError(stream);
    }
    int groups = 0;
    for (size_t k = 0; k < group.size(); ++k) {
        if (group[k] < 0) {
            std::ostringstream stream;
            stream << "Fine-Gray groups must be non-negative";
            error->throwError(stream);
        }
        if (group[k] > 0 && y[k] != static_cast<RealType>(0)) {
            std::ostringstream stream;
            stream << "A Fine-Gray row (a competing event) cannot have an outcome event";
            error->throwError(stream);
        }
        groups = std::max(groups, group[k]);
    }
    for (size_t i = 0; i < censoringGroup.size(); ++i) {
        if (censoringGroup[i] < 1) {
            std::ostringstream stream;
            stream << "Fine-Gray censoring groups must be positive";
            error->throwError(stream);
        }
        if (!(censoringSurvival[i] >= 0.0 && censoringSurvival[i] <= 1.0)) {
            std::ostringstream stream;
            stream << "Fine-Gray censoring survival must lie in [0, 1]";
            error->throwError(stream);
        }
        groups = std::max(groups, censoringGroup[i]);
    }

    std::vector<std::vector<std::pair<double, double>>> steps(groups);
    for (size_t i = 0; i < censoringGroup.size(); ++i) {
        steps[censoringGroup[i] - 1].push_back(std::make_pair(censoringTime[i], censoringSurvival[i]));
    }
    fineGrayTimes.assign(groups, std::vector<double>());
    fineGraySurvival.assign(groups, std::vector<double>());
    for (int g = 0; g < groups; ++g) {
        std::sort(steps[g].begin(), steps[g].end());
        for (size_t i = 0; i < steps[g].size(); ++i) {
            if (i > 0 && steps[g][i].first == steps[g][i - 1].first) {
                std::ostringstream stream;
                stream << "Fine-Gray censoring times must be distinct within a group";
                error->throwError(stream);
            }
            fineGrayTimes[g].push_back(steps[g][i].first);
            fineGraySurvival[g].push_back(steps[g][i].second);
        }
    }
    fineGrayGroup.assign(group.begin(), group.end());
}

template <typename RealType>
double ModelData<RealType>::fineGraySurvivalBefore(int group, double t) const {
    const std::vector<double>& times = fineGrayTimes[group - 1];
    const size_t before = std::lower_bound(times.begin(), times.end(), t) - times.begin(); // times < t
    return (before == 0) ? 1.0 : fineGraySurvival[group - 1][before - 1];
}

template <typename RealType>
int ModelData<RealType>::loadMultipleX(
		const NumberView& covariateIds,
		const NumberView& rowIds,
		const NumberView& covariateValues,
		const bool checkCovariateIds,
		const bool checkCovariateBounds,
		const bool append,
		const bool forceSparse) {

	const size_t n = covariateIds.size;
	if (n == 0) { // no covariates (an intercept-only design): nothing to load; read below as the first id
		return getNumberOfColumns();
	}

	int firstColumnIndex = getNumberOfColumns();

	auto index = getColumnIndexByName(covariateIds.id(0));
	if (index >= 0) {
		if (!append) {
            std::ostringstream stream;
            stream << "Variable " << covariateIds.id(0) << " already exists";
            error->throwError(stream);
		}
		firstColumnIndex = index;
	}

	int64_t maxId = covariateIds.id(0);

	size_t i = 0;
	while (i < n) {
		const int64_t columnId = covariateIds.id(i);
		maxId = std::max(maxId, columnId);
		// the column's entries are [i, end)
		size_t end = i + 1;
		while (end < n && covariateIds.id(end) == columnId) ++end;
		appendEntries(index, columnId, rowIds, covariateValues, i, end, forceSparse);
		i = end;
		index = -1; // new column
	}

	maxCovariateId = maxId;

	touchedX = true;
	return firstColumnIndex;
}

template <typename RealType>
void ModelData<RealType>::appendEntries(int index, const int64_t columnId, const NumberView& rowIds,
		const NumberView& values, const size_t begin, const size_t end, const bool forceSparse) {
	const bool hasCovariateValues = values.size > 0;
	const bool useRowMap = consecutiveRowIds || rowIdMap.size() > 0;

	if (index < 0) { // Create new column if necessary
	    const auto format =
	        (hasCovariateValues &&
	            (!(values.value(begin) == 1.0 || values.value(begin) == 0.0) // not 0 or 1
                    || forceSparse)) ?
	                SPARSE : INDICATOR;
        X.push_back(format);
        index = getNumberOfColumns() - 1;
        X.getColumn(index).add_label(columnId);
	}

	// Append data into CompressedDataColumn
	CompressedDataColumn<RealType>& column = X.getColumn(index);
	FormatType format = column.getFormatType();
	if (format == INDICATOR || format == SPARSE) { // (intercept and dense columns keep no row indices)
		size_t entries = end - begin; // the entries kept: values of 0 are not
		if (hasCovariateValues) {
			for (size_t k = begin; k < end; ++k) entries -= values.value(k) == 0.0 ? 1 : 0;
		}
		reserveFor(column.getColumnsVector(), entries);
		if (format == SPARSE) {
			reserveFor(column.getDataVector(), entries);
		}
	}

	for (size_t k = begin; k < end; ++k) {
	    const int64_t rowId = rowIds.id(k);
	    if (k != begin && rowId == rowIds.id(k - 1)) {
	        std::ostringstream stream;
	        stream << "Repeated row-column entry at ";
	        stream << rowId << " - " << columnId;
	        throw std::range_error(stream.str());
	    }
	    const auto mappedRow = (useRowMap) ? mapRowId(rowId) : rowId;
	    if (hasCovariateValues) {
	        const double value = values.value(k);
                if (value != 0.0) {
                    if (format == INDICATOR && value != 1.0) {
                        // Upcast
                        column.convertColumnToSparse();
                        format = SPARSE;
                        reserveFor(column.getDataVector(), column.getColumnsVector().capacity() - column.getDataVector().size());
                    }
                    if (format == SPARSE) {
                        column.getDataVector().push_back(value);
                    }
                    column.getColumnsVector().push_back(mappedRow);
                }
	    } else {
	        column.getColumnsVector().push_back(mappedRow);
	    }
	}
}

template <typename RealType>
int ModelData<RealType>::loadColumns(
		const int* columnStarts,
		const size_t columns,
		const NumberView& columnIds,
		const NumberView& rowIds,
		const NumberView& values,
		const bool forceSparse) {

	const int firstColumnIndex = getNumberOfColumns();
	bool first = true;
	int64_t maxId = maxCovariateId;
	for (size_t j = 0; j < columns; ++j) {
		const size_t begin = static_cast<size_t>(columnStarts[j]);
		const size_t end = static_cast<size_t>(columnStarts[j + 1]);
		if (begin == end) continue; // no entries: no column, as a covariate without rows in a data frame
		const int64_t columnId = columnIds.id(j);
		if (first) { // as loadMultipleX(append = false): the first id must be new
			if (getColumnIndexByName(columnId) >= 0) {
				std::ostringstream stream;
				stream << "Variable " << columnId << " already exists";
				error->throwError(stream);
			}
			maxId = columnId;
			first = false;
		}
		maxId = std::max(maxId, columnId);
		appendEntries(-1, columnId, rowIds, values, begin, end, forceSparse);
	}
	if (!first) {
		maxCovariateId = maxId;
		touchedX = true;
	}
	return firstColumnIndex;
}

namespace {

struct NoValueTag {};
struct BoolTag {};

template <typename ValueT>
inline double readScalar(const void* data, size_t k) {
	return static_cast<double>(static_cast<const ValueT*>(data)[k]);
}

template <>
inline double readScalar<NoValueTag>(const void*, size_t) {
	return 1.0;
}

template <>
inline double readScalar<BoolTag>(const void* data, size_t k) {
	return static_cast<const uint8_t*>(data)[k] != 0 ? 1.0 : 0.0;
}

template <typename RealType, typename ValueT>
void copyStridedValues(std::vector<RealType>& dst, const void* data, size_t n, ptrdiff_t byteStride) {
	dst.resize(n);
	if (n == 0) return;
	if (byteStride == static_cast<ptrdiff_t>(sizeof(ValueT))) {
		const auto* src = static_cast<const ValueT*>(data);
		if (std::is_same<RealType, ValueT>::value) {
			std::memcpy(dst.data(), src, n * sizeof(RealType));
		} else {
			for (size_t i = 0; i < n; ++i) {
				dst[i] = static_cast<RealType>(src[i]);
			}
		}
		return;
	}
	const auto* base = static_cast<const char*>(data);
	for (size_t i = 0; i < n; ++i) {
		ValueT v;
		std::memcpy(&v, base + static_cast<ptrdiff_t>(i) * byteStride, sizeof(ValueT));
		dst[i] = static_cast<RealType>(v);
	}
}

template <typename RealType>
void copyStridedBoolValues(std::vector<RealType>& dst, const void* data, size_t n, ptrdiff_t byteStride) {
	dst.resize(n);
	if (n == 0) return;
	const auto* base = static_cast<const char*>(data);
	for (size_t i = 0; i < n; ++i) {
		const uint8_t b = *reinterpret_cast<const uint8_t*>(base + static_cast<ptrdiff_t>(i) * byteStride);
		dst[i] = b != 0 ? static_cast<RealType>(1) : static_cast<RealType>(0);
	}
}

template <typename RealType>
void copyPositionalValues(std::vector<RealType>& dst, const PositionalValueView& view, ptrdiff_t byteStride) {
	switch (view.kind) {
	case PositionalValueView::FLOAT64:
		copyStridedValues<RealType, double>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::FLOAT32:
		copyStridedValues<RealType, float>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::BOOL:
		copyStridedBoolValues<RealType>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::INT8:
		copyStridedValues<RealType, int8_t>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::UINT8:
		copyStridedValues<RealType, uint8_t>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::INT32:
		copyStridedValues<RealType, int32_t>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::INT64:
		copyStridedValues<RealType, int64_t>(dst, view.data, view.size, byteStride);
		break;
	case PositionalValueView::NONE:
		dst.clear();
		break;
	}
}

inline void copyAndValidateRowSlice(
		IntVector& dst,
		const int32_t* src,
		size_t begin,
		size_t len,
		size_t nRows,
		int64_t colId,
		const loggers::ErrorHandlerPtr& error) {
	static_assert(sizeof(int) == sizeof(int32_t), "Cyclops row indices require 32-bit int");
	dst.resize(len);
	if (len == 0) return;
	std::memcpy(dst.data(), src + begin, len * sizeof(int));
	const int nRowsInt = static_cast<int>(nRows);
	const int* out = dst.data();
	for (size_t i = 0; i < len; ++i) {
		const int r = out[i];
		if (r < 0 || r >= nRowsInt) {
			std::ostringstream stream;
			stream << "Row index out of range";
			error->throwError(stream);
		}
		if (i > 0) {
			const int prev = out[i - 1];
			if (r == prev) {
				std::ostringstream stream;
				stream << "Repeated row-column entry at " << r << " - " << colId;
				error->throwError(stream);
			}
			if (r < prev) {
				std::ostringstream stream;
				stream << "row_indices must be sorted in ascending order within each column";
				error->throwError(stream);
			}
		}
	}
}

inline void copyAndValidateRowSlice(
		IntVector& dst,
		const int64_t* src,
		size_t begin,
		size_t len,
		size_t nRows,
		int64_t colId,
		const loggers::ErrorHandlerPtr& error) {
	dst.resize(len);
	int* out = dst.data();
	const int64_t nRows64 = static_cast<int64_t>(nRows);
	int64_t prev = -1;
	for (size_t t = 0; t < len; ++t) {
		const int64_t r = src[begin + t];
		if (r < 0 || r >= nRows64) {
			std::ostringstream stream;
			stream << "Row index out of range";
			error->throwError(stream);
		}
		if (t > 0) {
			if (r == prev) {
				std::ostringstream stream;
				stream << "Repeated row-column entry at " << r << " - " << colId;
				error->throwError(stream);
			}
			if (r < prev) {
				std::ostringstream stream;
				stream << "row_indices must be sorted in ascending order within each column";
				error->throwError(stream);
			}
		}
		prev = r;
		out[t] = static_cast<int>(r);
	}
}

template <typename RealType, typename ValueT>
struct SliceCopier {
	static void copy(std::vector<RealType>& dst, const void* src, size_t begin, size_t len) {
		dst.resize(len);
		if (len == 0) return;
		const auto* typed = static_cast<const ValueT*>(src) + begin;
		if (std::is_same<RealType, ValueT>::value) {
			std::memcpy(dst.data(), typed, len * sizeof(RealType));
		} else {
			RealType* out = dst.data();
			for (size_t t = 0; t < len; ++t) {
				out[t] = static_cast<RealType>(typed[t]);
			}
		}
	}
};

template <typename RealType>
struct SliceCopier<RealType, BoolTag> {
	static void copy(std::vector<RealType>& dst, const void* src, size_t begin, size_t len) {
		dst.resize(len);
		const auto* bytes = static_cast<const uint8_t*>(src) + begin;
		RealType* out = dst.data();
		for (size_t t = 0; t < len; ++t) {
			out[t] = bytes[t] != 0 ? static_cast<RealType>(1) : static_cast<RealType>(0);
		}
	}
};

template <typename RealType>
struct SliceCopier<RealType, NoValueTag> {
	static void copy(std::vector<RealType>& dst, const void*, size_t, size_t len) {
		dst.assign(len, static_cast<RealType>(1));
	}
};

template <typename RealType, typename IndexT, typename ValueT>
void loadPositionalCSCImpl(
		CompressedDataMatrix<RealType>& X,
		const PositionalIndexView& indptr,
		const IndexT* rowIndices,
		const size_t nnz,
		const void* valuesData,
		const PositionalIndexView& covariateIds,
		const size_t nRows,
		const bool forceSparse,
		const uint8_t* isIndicator,
		const loggers::ErrorHandlerPtr& error,
		int64_t& maxCovariateId,
		std::atomic<bool>& touchedX) {

	const size_t nCols = covariateIds.size;
	if (nCols == 0) return;

	const bool hasValues = !std::is_same<ValueT, NoValueTag>::value;
	const bool isBoolValues = std::is_same<ValueT, BoolTag>::value;
	const int64_t nRows64 = static_cast<int64_t>(nRows);

	struct PendingColumn {
		IntVectorPtr cols;
		std::shared_ptr<std::vector<RealType>> vals;
		FormatType format;
		int64_t id;
	};
	std::vector<PendingColumn> pending;
	pending.reserve(nCols);

	auto validatePrivateRows = [&](const IntVector& cols, const int64_t colId) {
		const int nRowsInt = static_cast<int>(nRows);
		for (size_t i = 0; i < cols.size(); ++i) {
			const int r = cols[i];
			if (r < 0 || r >= nRowsInt) {
				std::ostringstream stream;
				stream << "Row index out of range";
				error->throwError(stream);
			}
			if (i > 0) {
				const int prev = cols[i - 1];
				if (r == prev) {
					std::ostringstream stream;
					stream << "Repeated row-column entry at " << r << " - " << colId;
					error->throwError(stream);
				}
				if (r < prev) {
					std::ostringstream stream;
					stream << "row_indices must be sorted in ascending order within each column";
					error->throwError(stream);
				}
			}
		}
	};

	const int64_t nnzSigned = static_cast<int64_t>(nnz);
	int64_t prevEnd = 0;

	for (size_t j = 0; j < nCols; ++j) {
		// Snapshot indptr bounds into locals so concurrent mutation cannot underflow or read OOB.
		const int64_t beginRaw = indptr.at(j);
		const int64_t endRaw = indptr.at(j + 1);
		if (beginRaw < 0 || endRaw < beginRaw || endRaw > nnzSigned || beginRaw != prevEnd) {
			std::ostringstream stream;
			stream << "indptr was modified concurrently during load";
			error->throwError(stream);
		}
		prevEnd = endRaw;
		const size_t begin = static_cast<size_t>(beginRaw);
		const size_t end = static_cast<size_t>(endRaw);
		const size_t len = end - begin;
		const int64_t colId = covariateIds.at(j);

		if (len == 0) {
			if (forceSparse) {
				pending.push_back({make_shared<IntVector>(), make_shared<std::vector<RealType>>(), SPARSE, colId});
			} else {
				pending.push_back({make_shared<IntVector>(), nullptr, INDICATOR, colId});
			}
			continue;
		}

		// Known-indicator fast path: pure pattern matrix or caller-certified indicator column
		if (!forceSparse && (!hasValues || (isIndicator != nullptr && isIndicator[j] != 0))) {
			auto cols = make_shared<IntVector>();
			copyAndValidateRowSlice(*cols, rowIndices, begin, len, nRows, colId, error);
			pending.push_back({std::move(cols), nullptr, INDICATOR, colId});
			continue;
		}

		if (!hasValues) { // forceSparse with pattern matrix -> SPARSE of 1.0s
			auto cols = make_shared<IntVector>();
			copyAndValidateRowSlice(*cols, rowIndices, begin, len, nRows, colId, error);
			auto vals = make_shared<std::vector<RealType>>(len, static_cast<RealType>(1));
			pending.push_back({std::move(cols), std::move(vals), SPARSE, colId});
			continue;
		}

		// Single pass over rows + values on the column slice: validate row indices
		// while counting kept entries and detecting non-binary values.
		size_t kept = 0;
		bool allOne = !forceSparse;
		for (size_t k = begin; k < end; ++k) {
			const int64_t r = static_cast<int64_t>(rowIndices[k]);
			if (r < 0 || r >= nRows64) {
				std::ostringstream stream;
				stream << "Row index out of range";
				error->throwError(stream);
			}
			if (k > begin) {
				const int64_t prev = static_cast<int64_t>(rowIndices[k - 1]);
				if (r == prev) {
					std::ostringstream stream;
					stream << "Repeated row-column entry at " << r << " - " << colId;
					error->throwError(stream);
				}
				if (r < prev) {
					std::ostringstream stream;
					stream << "row_indices must be sorted in ascending order within each column";
					error->throwError(stream);
				}
			}
			const double v = readScalar<ValueT>(valuesData, k);
			if (v != 0.0) {
				++kept;
				if (!isBoolValues && allOne && v != 1.0) {
					allOne = false;
				}
			}
		}

		if (allOne) {
			auto cols = make_shared<IntVector>();
			if (kept == len) {
				copyAndValidateRowSlice(*cols, rowIndices, begin, len, nRows, colId, error);
			} else {
				cols->reserve(kept);
				for (size_t k = begin; k < end; ++k) {
					if (readScalar<ValueT>(valuesData, k) != 0.0) {
						const int64_t r = static_cast<int64_t>(rowIndices[k]);
						if (r < 0 || r >= nRows64) {
							std::ostringstream stream;
							stream << "Row index out of range";
							error->throwError(stream);
						}
						cols->push_back(static_cast<int>(r));
					}
				}
				validatePrivateRows(*cols, colId);
			}
			pending.push_back({std::move(cols), nullptr, INDICATOR, colId});
		} else {
			auto cols = make_shared<IntVector>();
			auto vals = make_shared<std::vector<RealType>>();
			if (kept == len) {
				copyAndValidateRowSlice(*cols, rowIndices, begin, len, nRows, colId, error);
				SliceCopier<RealType, ValueT>::copy(*vals, valuesData, begin, len);
			} else {
				cols->reserve(kept);
				vals->reserve(kept);
				for (size_t k = begin; k < end; ++k) {
					const double v = readScalar<ValueT>(valuesData, k);
					if (v != 0.0) {
						const int64_t r = static_cast<int64_t>(rowIndices[k]);
						if (r < 0 || r >= nRows64) {
							std::ostringstream stream;
							stream << "Row index out of range";
							error->throwError(stream);
						}
						cols->push_back(static_cast<int>(r));
						vals->push_back(static_cast<RealType>(v));
					}
				}
				validatePrivateRows(*cols, colId);
			}
			pending.push_back({std::move(cols), std::move(vals), SPARSE, colId});
		}
	}

	// Commit all validated columns atomically.
	const bool hadCols = X.getNumberOfColumns() > 0;
	int64_t maxId = maxCovariateId;
	X.reserveColumns(X.getNumberOfColumns() + nCols);
	for (size_t j = 0; j < nCols; ++j) {
		auto& col = pending[j];
		maxId = (hadCols || j > 0) ? std::max(maxId, col.id) : col.id;
		X.push_back(std::move(col.cols), std::move(col.vals), col.format);
		X.getColumn(X.getNumberOfColumns() - 1).add_label(col.id);
	}

	maxCovariateId = maxId;
	touchedX = true;
}

template <typename RealType, typename IndexT>
void dispatchPositionalCSCByValue(
		CompressedDataMatrix<RealType>& X,
		const PositionalIndexView& indptr,
		const IndexT* rowIndices,
		const size_t nnz,
		const PositionalValueView& values,
		const PositionalIndexView& covariateIds,
		const size_t nRows,
		const bool forceSparse,
		const uint8_t* isIndicator,
		const loggers::ErrorHandlerPtr& error,
		int64_t& maxCovariateId,
		std::atomic<bool>& touchedX) {

	if (values.kind == PositionalValueView::NONE || values.size == 0) {
		loadPositionalCSCImpl<RealType, IndexT, NoValueTag>(
			X, indptr, rowIndices, nnz, nullptr, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		return;
	}
	switch (values.kind) {
	case PositionalValueView::FLOAT64:
		loadPositionalCSCImpl<RealType, IndexT, double>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::FLOAT32:
		loadPositionalCSCImpl<RealType, IndexT, float>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::BOOL:
		loadPositionalCSCImpl<RealType, IndexT, BoolTag>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT8:
		loadPositionalCSCImpl<RealType, IndexT, int8_t>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::UINT8:
		loadPositionalCSCImpl<RealType, IndexT, uint8_t>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT32:
		loadPositionalCSCImpl<RealType, IndexT, int32_t>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT64:
		loadPositionalCSCImpl<RealType, IndexT, int64_t>(
			X, indptr, rowIndices, nnz, values.data, covariateIds, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::NONE:
		break;
	}
}

template <typename RealType, typename IndexT, typename ValueT>
void loadPositionalCSRImpl(
		CompressedDataMatrix<RealType>& X,
		const PositionalIndexView& indptr,
		const IndexT* colIndices,
		const size_t nnz,
		const void* valuesData,
		const PositionalIndexView& covariateIds,
		const PositionalIndexView* rowOrder,
		const size_t nRows,
		const bool forceSparse,
		const uint8_t* isIndicator,
		const loggers::ErrorHandlerPtr& error,
		int64_t& maxCovariateId,
		std::atomic<bool>& touchedX) {

	const size_t nCols = covariateIds.size;
	const size_t nCsrRows = indptr.size - 1;
	const int64_t nnzSigned = static_cast<int64_t>(nnz);
	const bool hasValues = !std::is_same<ValueT, NoValueTag>::value;
	const bool isBoolValues = std::is_same<ValueT, BoolTag>::value;

	std::vector<size_t> colNnz(nCols, 0);
	std::vector<uint8_t> colSparse(nCols, forceSparse ? 1 : 0);
	std::vector<int> lastSeenRow(nCols, -1);

	// Snapshot indicator flags so concurrent mutation cannot diverge Pass 1 vs Pass 2.
	std::vector<uint8_t> indicatorMask;
	if (!forceSparse && hasValues && isIndicator != nullptr) {
		indicatorMask.assign(isIndicator, isIndicator + nCols);
	}
	const uint8_t* maskPtr = indicatorMask.empty() ? nullptr : indicatorMask.data();

	auto throwConcurrent = [&]() {
		std::ostringstream stream;
		stream << "Input buffer was modified concurrently during load";
		error->throwError(stream);
	};

	// Pass 1: validate column indices and duplicate (row, col) entries atomically before mutating X,
	// while computing exact per-column entry counts and INDICATOR vs SPARSE formats.
	for (size_t r = 0; r < nRows; ++r) {
		const int64_t srcRaw = (rowOrder != nullptr) ? rowOrder->at(r) : static_cast<int64_t>(r);
		if (srcRaw < 0 || static_cast<uint64_t>(srcRaw) >= static_cast<uint64_t>(nCsrRows)) {
			throwConcurrent();
		}
		const size_t src = static_cast<size_t>(srcRaw);
		const int64_t beginRaw = indptr.at(src);
		const int64_t endRaw = indptr.at(src + 1);
		if (beginRaw < 0 || endRaw < beginRaw || endRaw > nnzSigned) {
			throwConcurrent();
		}
		const size_t begin = static_cast<size_t>(beginRaw);
		const size_t end = static_cast<size_t>(endRaw);
		const int rInt = static_cast<int>(r);
		for (size_t k = begin; k < end; ++k) {
			const int64_t colRaw = static_cast<int64_t>(colIndices[k]);
			if (colRaw < 0 || static_cast<uint64_t>(colRaw) >= static_cast<uint64_t>(nCols)) {
				std::ostringstream stream;
				stream << "Column index out of range";
				error->throwError(stream);
			}
			const size_t col = static_cast<size_t>(colRaw);
			if (lastSeenRow[col] == rInt) {
				std::ostringstream stream;
				stream << "Repeated row-column entry at " << r << " - " << covariateIds.at(col);
				error->throwError(stream);
			}
			lastSeenRow[col] = rInt;

			if (!hasValues || (maskPtr != nullptr && maskPtr[col] != 0)) {
				++colNnz[col];
			} else {
				const double v = readScalar<ValueT>(valuesData, k);
				if (v != 0.0) {
					++colNnz[col];
					if (!isBoolValues && !colSparse[col] && v != 1.0) {
						colSparse[col] = 1;
					}
				}
			}
		}
	}

	if (nCols == 0) return;

	lastSeenRow.clear();
	lastSeenRow.shrink_to_fit();

	// Allocate all columns locally before mutating X (strong exception safety on bad_alloc or concurrent change).
	std::vector<IntVectorPtr> colRows(nCols);
	std::vector<std::shared_ptr<std::vector<RealType>>> colVals(nCols);
	std::vector<int*> rowWrite(nCols, nullptr);
	std::vector<int*> rowEnd(nCols, nullptr);
	std::vector<RealType*> valWrite(nCols, nullptr);
	std::vector<RealType*> valEnd(nCols, nullptr);

	for (size_t j = 0; j < nCols; ++j) {
		colRows[j] = make_shared<IntVector>(colNnz[j]);
		rowWrite[j] = colRows[j]->data();
		rowEnd[j] = rowWrite[j] + colNnz[j];
		if (colSparse[j]) {
			colVals[j] = make_shared<std::vector<RealType>>(colNnz[j]);
			valWrite[j] = colVals[j]->data();
			valEnd[j] = valWrite[j] + colNnz[j];
		}
	}

	// Pass 2: scatter rows r = 0..nRows-1 directly into pre-sized local columns in ascending row order,
	// with defensive bounds checks against concurrent mutation of input buffers.
	for (size_t r = 0; r < nRows; ++r) {
		const int64_t srcRaw = (rowOrder != nullptr) ? rowOrder->at(r) : static_cast<int64_t>(r);
		if (srcRaw < 0 || static_cast<uint64_t>(srcRaw) >= static_cast<uint64_t>(nCsrRows)) {
			throwConcurrent();
		}
		const size_t src = static_cast<size_t>(srcRaw);
		const int64_t beginRaw = indptr.at(src);
		const int64_t endRaw = indptr.at(src + 1);
		if (beginRaw < 0 || endRaw < beginRaw || endRaw > nnzSigned) {
			throwConcurrent();
		}
		const size_t begin = static_cast<size_t>(beginRaw);
		const size_t end = static_cast<size_t>(endRaw);
		const int rInt = static_cast<int>(r);
		for (size_t k = begin; k < end; ++k) {
			const int64_t colRaw = static_cast<int64_t>(colIndices[k]);
			if (colRaw < 0 || static_cast<uint64_t>(colRaw) >= static_cast<uint64_t>(nCols)) {
				throwConcurrent();
			}
			const size_t col = static_cast<size_t>(colRaw);
			if (!hasValues) {
				if (rowWrite[col] >= rowEnd[col]) throwConcurrent();
				*(rowWrite[col]++) = rInt;
				if (valWrite[col] != nullptr) {
					if (valWrite[col] >= valEnd[col]) throwConcurrent();
					*(valWrite[col]++) = static_cast<RealType>(1);
				}
			} else if (maskPtr != nullptr && maskPtr[col] != 0) {
				if (rowWrite[col] >= rowEnd[col]) throwConcurrent();
				*(rowWrite[col]++) = rInt;
			} else {
				const double v = readScalar<ValueT>(valuesData, k);
				if (v != 0.0) {
					if (rowWrite[col] >= rowEnd[col]) throwConcurrent();
					*(rowWrite[col]++) = rInt;
					if (valWrite[col] != nullptr) {
						if (valWrite[col] >= valEnd[col]) throwConcurrent();
						*(valWrite[col]++) = static_cast<RealType>(v);
					}
				}
			}
		}
	}

	// Verify exact fill count (catches concurrent drops or row-order divergence).
	for (size_t j = 0; j < nCols; ++j) {
		if (rowWrite[j] != rowEnd[j] || valWrite[j] != valEnd[j]) {
			throwConcurrent();
		}
	}

	// Commit all columns atomically to X.
	const bool hadCols = X.getNumberOfColumns() > 0;
	int64_t maxId = maxCovariateId;
	X.reserveColumns(X.getNumberOfColumns() + nCols);
	for (size_t j = 0; j < nCols; ++j) {
		const int64_t colId = covariateIds.at(j);
		maxId = (hadCols || j > 0) ? std::max(maxId, colId) : colId;
		X.push_back(std::move(colRows[j]), std::move(colVals[j]), colSparse[j] ? SPARSE : INDICATOR);
		X.getColumn(X.getNumberOfColumns() - 1).add_label(colId);
	}

	maxCovariateId = maxId;
	touchedX = true;
}

template <typename RealType, typename IndexT>
void dispatchPositionalCSRByValue(
		CompressedDataMatrix<RealType>& X,
		const PositionalIndexView& indptr,
		const IndexT* colIndices,
		const size_t nnz,
		const PositionalValueView& values,
		const PositionalIndexView& covariateIds,
		const PositionalIndexView* rowOrder,
		const size_t nRows,
		const bool forceSparse,
		const uint8_t* isIndicator,
		const loggers::ErrorHandlerPtr& error,
		int64_t& maxCovariateId,
		std::atomic<bool>& touchedX) {

	if (values.kind == PositionalValueView::NONE || values.size == 0) {
		loadPositionalCSRImpl<RealType, IndexT, NoValueTag>(
			X, indptr, colIndices, nnz, nullptr, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		return;
	}
	switch (values.kind) {
	case PositionalValueView::FLOAT64:
		loadPositionalCSRImpl<RealType, IndexT, double>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::FLOAT32:
		loadPositionalCSRImpl<RealType, IndexT, float>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::BOOL:
		loadPositionalCSRImpl<RealType, IndexT, BoolTag>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT8:
		loadPositionalCSRImpl<RealType, IndexT, int8_t>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::UINT8:
		loadPositionalCSRImpl<RealType, IndexT, uint8_t>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT32:
		loadPositionalCSRImpl<RealType, IndexT, int32_t>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::INT64:
		loadPositionalCSRImpl<RealType, IndexT, int64_t>(
			X, indptr, colIndices, nnz, values.data, covariateIds, rowOrder, nRows, forceSparse, isIndicator,
			error, maxCovariateId, touchedX);
		break;
	case PositionalValueView::NONE:
		break;
	}
}

inline ptrdiff_t naturalStride(PositionalValueView::Kind kind) {
	switch (kind) {
	case PositionalValueView::FLOAT64: return sizeof(double);
	case PositionalValueView::FLOAT32: return sizeof(float);
	case PositionalValueView::BOOL:    return sizeof(uint8_t);
	case PositionalValueView::INT8:    return sizeof(int8_t);
	case PositionalValueView::UINT8:   return sizeof(uint8_t);
	case PositionalValueView::INT32:   return sizeof(int32_t);
	case PositionalValueView::INT64:   return sizeof(int64_t);
	case PositionalValueView::NONE:    return 0;
	}
	return 0;
}

} // namespace

template <typename RealType>
void ModelData<RealType>::loadYPositional(
		const PositionalValueView& oY,
		const PositionalValueView& oTime,
		const PositionalIndexView& oStratumId,
		const PositionalIndexView& oRowId) {

	const bool previouslyLoaded = y.size() > 0;
	const size_t n = oY.size;
	const bool hasTime = (oTime.kind != PositionalValueView::NONE && oTime.size > 0);

	if (   (oStratumId.size > 0 && oStratumId.size != n)
		|| (oRowId.size > 0 && oRowId.size != n)
		|| (hasTime && oTime.size != n)
		|| (previouslyLoaded && y.size() != n)
	) {
		std::ostringstream stream;
		stream << "Mismatched outcome column dimensions";
		error->throwError(stream);
	}
	if (previouslyLoaded && (oStratumId.size > 0 || oRowId.size > 0)) {
		std::ostringstream stream;
		stream << "Strata or row IDs are already loaded";
		error->throwError(stream);
	}

	copyPositionalValues(y, oY, naturalStride(oY.kind));
	if (hasTime) {
		copyPositionalValues(offs, oTime, naturalStride(oTime.kind));
	}
	touchedY = true;

	if (!previouslyLoaded) {
		const bool processStrata = oStratumId.size > 0;
		if (processStrata) {
			pid.reserve(n);
			for (size_t i = 0; i < n; ++i) {
				const IdType cInStratum = oStratumId.at(i);
				if (i == 0) {
					lastStratumMap.first = cInStratum;
					lastStratumMap.second = 0;
					nPatients++;
				} else if (cInStratum != lastStratumMap.first) {
					lastStratumMap.first = cInStratum;
					lastStratumMap.second++;
					nPatients++;
				}
				pid.push_back(lastStratumMap.second);
			}
		} else {
			nPatients = n;
		}

		bool consecutive = oRowId.size > 0;
		for (size_t i = 1; i < oRowId.size && consecutive; ++i) {
			const IdType prev = oRowId.at(i - 1);
			consecutive = (prev < std::numeric_limits<IdType>::max()) && (oRowId.at(i) == prev + 1);
		}
		consecutiveRowIds = consecutive;
		firstRowId = consecutive ? oRowId.at(0) : 0;

		if (oRowId.size > 0 && !consecutive) {
			rowIdMap.reserve(oRowId.size);
			labels.reserve(oRowId.size);
			for (size_t i = 0; i < oRowId.size; ++i) {
				const IdType currentRowId = oRowId.at(i);
				rowIdMap[currentRowId] = i;
				labels.push_back(std::to_string(currentRowId));
			}
		}

		getX().nRows = n;
	}
}

template <typename RealType>
void ModelData<RealType>::loadDenseColumn(
		const IdType covariateId,
		const PositionalValueView& values,
		const ptrdiff_t byteStride,
		const bool checkExisting) {

	if (checkExisting && getColumnIndexByName(covariateId) >= 0) {
		std::ostringstream stream;
		stream << "Variable " << covariateId << " already exists";
		error->throwError(stream);
	}
	const bool hadCols = X.getNumberOfColumns() > 0;
	auto colData = make_shared<RealVector>();
	copyPositionalValues(*colData, values, byteStride);
	X.push_back(nullptr, colData, DENSE);
	X.getColumn(getNumberOfColumns() - 1).add_label(covariateId);
	maxCovariateId = hadCols ? std::max(maxCovariateId, covariateId) : covariateId;
	touchedX = true;
}

template <typename RealType>
void ModelData<RealType>::loadPositionalCSC(
		const PositionalIndexView& indptr,
		const PositionalIndexView& rowIndices,
		const PositionalValueView& values,
		const PositionalIndexView& covariateIds,
		const bool forceSparse,
		const uint8_t* isIndicator) {

	const size_t nRows = getNumberOfRows();
	if (rowIndices.kind == PositionalIndexView::INT32) {
		dispatchPositionalCSCByValue(
			X, indptr, static_cast<const int32_t*>(rowIndices.data), rowIndices.size, values,
			covariateIds, nRows, forceSparse, isIndicator, error, maxCovariateId, touchedX);
	} else {
		dispatchPositionalCSCByValue(
			X, indptr, static_cast<const int64_t*>(rowIndices.data), rowIndices.size, values,
			covariateIds, nRows, forceSparse, isIndicator, error, maxCovariateId, touchedX);
	}
}

template <typename RealType>
void ModelData<RealType>::loadPositionalCSR(
		const PositionalIndexView& indptr,
		const PositionalIndexView& colIndices,
		const PositionalValueView& values,
		const PositionalIndexView& covariateIds,
		const PositionalIndexView* rowOrder,
		const bool forceSparse,
		const uint8_t* isIndicator) {

	const size_t nRows = getNumberOfRows();
	if (colIndices.kind == PositionalIndexView::INT32) {
		dispatchPositionalCSRByValue(
			X, indptr, static_cast<const int32_t*>(colIndices.data), colIndices.size, values,
			covariateIds, rowOrder, nRows, forceSparse, isIndicator, error, maxCovariateId, touchedX);
	} else {
		dispatchPositionalCSRByValue(
			X, indptr, static_cast<const int64_t*>(colIndices.data), colIndices.size, values,
			covariateIds, rowOrder, nRows, forceSparse, isIndicator, error, maxCovariateId, touchedX);
	}
}


template <typename RealType>
int ModelData<RealType>::loadX(
		const IdType covariateId,
		const std::vector<IdType>& rowId,
		const std::vector<double>& covariateValue,
		const bool reload,
		const bool append,
		const bool forceSparse,
		const bool checkExisting) {

    const bool hasCovariateValues = covariateValue.size() > 0;
    const bool useRowMap = consecutiveRowIds || rowIdMap.size() > 0;

    // Determine covariate type
    FormatType newType = rowId.size() == 0 ?
            (hasCovariateValues ? DENSE : INTERCEPT) :
            (hasCovariateValues &&
                (!(covariateValue[0] == 1.0 || covariateValue[0] == 0.0) || forceSparse) ?
                SPARSE : INDICATOR
            );

// 	std::function<size_t(IdType)>
//     xform = [this](IdType id) {
//         return rowIdMap[id];
//     };

    // Determine if covariate already exists (a linear search over the columns, which batch loaders do once instead)
    int index = checkExisting ? getColumnIndexByName(covariateId) : -1;
    if (index >= 0) {
        // already exists
        if (!reload) {
            std::ostringstream stream;
            stream << "Variable " << covariateId << " already exists";
            error->throwError(stream);
        }
//        if (newType == INTERCEPT && getHasInterceptCovariate()) {
//            std::ostringstream stream;
//            stream << "Cyclops data object already has an intercept";
//            error->throwError(stream);
//        }
        if (append) {
            // TODO Check that new and old type are compatible
            std::ostringstream stream;
            stream << "Variable appending is not yet supported";
            error->throwError(stream);
        }
//         if (rowIdMap.size() > 0) {
//         	// make deep copy and destruct old column
//   			replace(index,
//   				boost::make_transform_iterator(begin(rowId), xform),
//   				boost::make_transform_iterator(end(rowId), xform),
//   				begin(covariateValue), end(covariateValue),
//   				newType);
//         } else {
//             if (newType == SPARSE || newType == INDICATOR) {
//                 std::ostringstream stream;
//                 stream << "Must use row IDs with sparse/indicator columns";
//                 error->throwError(stream);
//             }
// 	        replace(index,
//     	        begin(rowId), end(rowId),
// 	            begin(covariateValue), end(covariateValue),
//     	        newType);
//     	}
//     	getColumn(index).add_label(covariateId);
        std::ostringstream stream;
        stream << "Replacng variables is not yet supported";
        error->throwError(stream);
    } else {

        // brand new, make deep copy
        if (newType == DENSE || newType == INTERCEPT) {
            X.push_back(begin(rowId), end(rowId),
                      begin(covariateValue), end(covariateValue),
                      newType);
        } else { // SPARSE or INDICATOR
            X.push_back(newType);
            CompressedDataColumn<RealType>& column = X.getColumn(getNumberOfColumns() - 1);

            auto rowIdItr = std::begin(rowId);
            auto covariateValueItr = std::begin(covariateValue);
            const auto rowIdEnd = std::end(rowId);

            while (rowIdItr != rowIdEnd) {
                if (rowIdItr != std::begin(rowId) && *rowIdItr == *(rowIdItr - 1)) {
                    std::ostringstream stream;
                    stream << "Repeated row-column entry at ";
                    stream << *rowIdItr << " - " << covariateId;
                    throw std::range_error(stream.str());
                }
                const auto mappedRow = (useRowMap) ? mapRowId(*rowIdItr) : *rowIdItr;
                if (hasCovariateValues) {
                    if (*covariateValueItr != 0.0) {
                        if (newType == INDICATOR && *covariateValueItr != 1.0) {
                            // Upcast
                            column.convertColumnToSparse();
                            newType = SPARSE;
                        }
                        if (newType == SPARSE) {
                            column.getDataVector().push_back(*covariateValueItr);
                        }
                        column.getColumnsVector().push_back(mappedRow);
                    }
                    ++covariateValueItr;
                } else {
                    column.getColumnsVector().push_back(mappedRow);
                }
                ++rowIdItr;
            }
        }

        index = getNumberOfColumns() - 1;
        X.getColumn(index).add_label(covariateId);
    }

    if (newType == INTERCEPT) {
        setHasInterceptCovariate(true);
        if (index != 0) {
            X.moveToFront(index);
        }
    }
    touchedX = true;
    return index;
}

template <typename RealType>
std::vector<std::string> ModelData<RealType>::loadStratTimeEffects(
        const std::vector<IdType>& oStratumId,
        const std::vector<IdType>& oRowId,
        const std::vector<IdType>& oSubjectId,
        const std::vector<IdType>& timeEffectCovariateIds) {

    materializeRowIds(); // looks rows up through rowIdMap and adds rows
    int numOfFixedCov = getNumberOfColumns();

    // Valid subjects in current stratum
    // Subject-to-row by stratum
    std::vector<vector<IdType>> validSubjects;
    std::vector<std::unordered_map<IdType,IdType>> subjectToRowByStratum;

    IdType prevStrata = -1;
    for (size_t i = 0; i < oRowId.size(); ++i) {

        if (oStratumId[i] != prevStrata) {
            validSubjects.push_back({});
            subjectToRowByStratum.push_back({});
        }
        validSubjects.back().push_back(oSubjectId[i]);
        subjectToRowByStratum.back()[oSubjectId[i]] = oRowId[i];

        prevStrata = oStratumId[i];
    }

    // Transpose of X
    bsccs::shared_ptr<CompressedDataMatrix<RealType>> Xt;
    Xt = X.transpose(); // TODO figure out formatType of Xt

    // Name for new covariates
    std::vector<std::pair<IdType, IdType>> timeEffectCovariates; // (stratum, timeEffectCovariateName)
    std::vector<std::string> timeEffectCovariatesName; // timeEffectCovariateName_stratum

    // Start from the 2nd stratum
    for (int st = 1; st <= pid.back(); st++) {

        bool skip = true; // Skip this stratum if no subject has time varying cov
        for (IdType sub : validSubjects[st]) {

            IdType fixedRow = rowIdMap[subjectToRowByStratum[0][sub]]; // mappedRow of sub at 1st stratum

            if (Xt->getColumn(fixedRow).getNumberOfEntries() > 0) { // Loop over existed (time-indept) covariates of sub
                size_t i = 0, j = 0;
                while (i < Xt->getColumn(fixedRow).getNumberOfEntries()
                           && Xt->getCompressedColumnVectorSTL(fixedRow)[i] < numOfFixedCov
                           && j < timeEffectCovariateIds.size()) {

                    IdType coefIdx = Xt->getCompressedColumnVectorSTL(fixedRow)[i]; // Column index of sub
                    int timeIdx = getColumnIndexByName(timeEffectCovariateIds[j]); // Column index of time-varying cov
                    if (coefIdx == timeIdx) {
                        skip = false; break; // The stratum contains subject with time-varying cov
                    } else if (coefIdx < timeIdx) {
                        i++;
                    } else {
                        j++;
                    }
                }
            }
        }
        if (skip) continue;

        for (IdType sub : validSubjects[st]) {

            IdType fixedRow = rowIdMap[subjectToRowByStratum[0][sub]]; // mappedRow of sub at 1st stratum

            if (Xt->getColumn(fixedRow).getNumberOfEntries() > 0) { // Loop over existed (time-indept) covariates of sub
                size_t i = 0, j = 0;

                IdType newRow = rowIdMap[subjectToRowByStratum[st][sub]]; // mappedRow of sub at current stratum
                while (i < Xt->getColumn(fixedRow).getNumberOfEntries()
                           && Xt->getCompressedColumnVectorSTL(fixedRow)[i] < numOfFixedCov
                           && j < timeEffectCovariateIds.size()) {

                    IdType coefIdx = Xt->getCompressedColumnVectorSTL(fixedRow)[i]; // Column index of sub
                    int timeIdx = getColumnIndexByName(timeEffectCovariateIds[j]); // Column index of time-varying cov
                    auto index = coefIdx; // Column index to append, will be updated below if append to a time-varying coef
                    bool append = true;

                    if (coefIdx == timeIdx) {
                        // coefIdx is a time-varying coef, append data, move both pointers

                        if (timeEffectCovariateIdMap.find(timeEffectCovariateIds[j]) != timeEffectCovariateIdMap.end()) {
                            // The time-varying coefficient was already created at this stratum
                            index = timeEffectCovariateIdMap[timeEffectCovariateIds[j]];
                        } else {
                            // Create a new column for the time-varying coefficient at this stratum
                            index = getNumberOfColumns();
                            timeEffectCovariateIdMap[timeEffectCovariateIds[j]] = index;
                            X.push_back(X.getFormatType(coefIdx));
                            IdType timeCovId = ++maxCovariateId;
                            X.getColumn(index).add_label(timeCovId); // TODO return label to user or automatically exclude this column from L1 regularization
                            timeEffectCovariates.push_back({st+1, timeEffectCovariateIds[j]}); // (stratum, timeEffectCovariateName)
                            // std::cout << "Create a new column with label [" << timeCovId << "] at stratum [" << st+1 << "] from cov [" << timeEffectCovariateIds[j] << "]\n";
                            std::ostringstream stream;
                            stream << "Created a new column with label " << timeCovId
                                   << " at stratum " << st+1
                                   << " from cov " << timeEffectCovariateIds[j];
                            log->writeLine(stream);
                        }

                        i++;
                        if (j < timeEffectCovariateIds.size()-1) j++;

                    } else if (coefIdx < timeIdx
                                   || (coefIdx > timeIdx && j == timeEffectCovariateIds.size()-1)) {
                        // coefIdx is a time-indept coef, append data, move the pointer to the next coefIdx
                        i++;
                    } else {
                        // Only move the pointer for timeEffectCovariateIds
                        append = false;
                        j++;
                    }

                    if (append) {
                        FormatType formatType = X.getColumn(coefIdx).getFormatType(); // formatType for getting data
                        if (formatType == SPARSE || formatType == DENSE) {
                            // TODO better way to get data from (coefIdx, fixedRow)?
                            auto it = find(X.getCompressedColumnVectorSTL(coefIdx).begin(), X.getCompressedColumnVectorSTL(coefIdx).end(), fixedRow);
                            int rowIdx = it - X.getCompressedColumnVectorSTL(coefIdx).begin();
                            X.getColumn(index).add_data(newRow, X.getDataVectorSTL(coefIdx)[rowIdx]);
                        } else {
                            X.getColumn(index).add_data(newRow, static_cast<int>(1));
                        }
                    }
                }
            }
        }
        timeEffectCovariateIdMap.clear(); // clear map for next stratum
    }
/*
    // TODO need to sort mappedRow in ascending order?
    for (int index = 0; index < getNumberOfColumns(); index++) {
        X.getColumn(index).sortRows();
    }
*/
    // Sort columns
    getX().sortColumns(CompressedDataColumn<RealType>::sortNumerically);

    // Sort timeEffectCovariatesName
    std::sort(timeEffectCovariates.begin(), timeEffectCovariates.end());
    for (auto p : timeEffectCovariates) {
        timeEffectCovariatesName.push_back(std::to_string(p.second) + '_' + std::to_string(p.first));
    }

    return timeEffectCovariatesName;
}

template <typename RealType>
size_t ModelData<RealType>::append(
        const std::vector<IdType>& oStratumId,
        const std::vector<IdType>& oRowId,
        const std::vector<double>& oY,
        const std::vector<double>& oTime,
        const std::vector<IdType>& cRowId,
        const std::vector<IdType>& cCovariateId,
        const std::vector<double>& cCovariateValue) {

    materializeRowIds(); // adds rows with their labels
    // Check covariate dimensions
    if ((cRowId.size() != cCovariateId.size()) ||
        (cRowId.size() != cCovariateValue.size())) {
        std::ostringstream stream;
        stream << "Mismatched covariate column dimensions";
        error->throwError(stream);
    }

    // TODO Check model-specific outcome dimensions
    if ((oStratumId.size() != oY.size()) ||   // Causing crashes
        (oStratumId.size() != oRowId.size())) {
        std::ostringstream stream;
        stream << "Mismatched outcome column dimensions";
        error->throwError(stream);
    }

    // TODO Check model-specific outcome dimensions
  //  if (requiresStratumID(

    bool hasTime = oTime.size() == oY.size();

    const size_t nOutcomes = oStratumId.size();
    const size_t nCovariates = cCovariateId.size();

//#define DEBUG_64BIT
#ifdef DEBUG_64BIT
    std::cout << "sizeof(long) = " << sizeof(long) << std::endl;
    std::cout << "sizeof(int64_t) = " << sizeof(uint64_t) << std::endl;
    std::cout << "sizeof(long long) = " << sizeof(long long) << std::endl;
#endif

    size_t cOffset = 0;

    for (size_t i = 0; i < nOutcomes; ++i) {

    	// TODO Begin code duplication with 'loadY'
        IdType cInStratum = oStratumId[i];
        if (X.nRows == 0) {
        	lastStratumMap.first = cInStratum;
        	lastStratumMap.second = 0;
        	nPatients++;
        } else {
        	if (cInStratum != lastStratumMap.first) {
          	    lastStratumMap.first = cInStratum;
            	lastStratumMap.second++;
            	nPatients++;
        	}
        }
        pid.push_back(lastStratumMap.second);
        y.push_back(oY[i]);
        if (hasTime) {
            offs.push_back(oTime[i]);
        }

        IdType currentRowId = oRowId[i];
        // TODO Check timing on adding label as string
        std::stringstream ss;
        ss << currentRowId;
        labels.push_back(ss.str());
        // TODO End code duplication with 'loadY;

#ifdef DEBUG_64BIT
        std::cout << currentRowId << std::endl;
#endif
        while (cOffset < nCovariates && cRowId[cOffset] == currentRowId) {
            auto covariate = cCovariateId[cOffset];
            auto value = cCovariateValue[cOffset];

			if (!sparseIndexer.hasColumn(covariate)) {
				// Add new column
				sparseIndexer.addColumn(covariate, INDICATOR);
			}

			auto& column = sparseIndexer.getColumn(covariate);
			if (value != static_cast<RealType>(1) && value != static_cast<RealType>(0)) {
				if (column.getFormatType() == INDICATOR) {
					std::ostringstream stream;
					stream << "Up-casting covariate " << column.getLabel() << " to sparse!";
					log->writeLine(stream);
					column.convertColumnToSparse();
				}
			}

			// Add to storage
			bool valid = column.add_data(X.nRows, value);
			if (!valid) {
				std::ostringstream stream;
				stream << "Warning: repeated sparse entries in data row: "
						<< cRowId[cOffset]
						<< ", column: " << column.getLabel();
				log->writeLine(stream);
			}
			++cOffset;
        }
        ++X.nRows;
    }
    return nOutcomes;
}

template <typename RealType>
std::vector<double> ModelData<RealType>::normalizeCovariates(const NormalizationType type) {
    std::vector<double> normalizations;
    normalizations.reserve(getNumberOfColumns());

    const auto nRows = getNumberOfRows();

    size_t index = hasOffsetCovariate ? 1 : 0;
    if (hasInterceptCovariate) {
        normalizations.push_back(1.0);
        ++index;
    }

    for ( ; index < getNumberOfColumns(); ++index) {
        auto& column = X.getColumn(index);
        FormatType format = column.getFormatType();
        if (format == DENSE || format == SPARSE) {

            double scale = 1.0;
            if (type == NormalizationType::STANDARD_DEVIATION) {
                auto sumOp = [](double x, double y) {
                    return x + y;
                };

                auto squaredSumOp = [](double x, double y) {
                    return x + y * y;
                };

                auto mean = column.accumulate(sumOp, 0.0) / nRows;
                auto newSS = column.accumulate(squaredSumOp, 0.0);
                auto variance = (newSS - (mean * mean * nRows)) / nRows;
                scale = 1.0 / std::sqrt(variance);

            } else if (type == NormalizationType::MAX) {
                auto maxOp = [](double max, double x) {
                    auto abs = std::abs(x);
                    if (abs > max) {
                        max = abs;
                    }
                    return max;
                };
                scale = 1.0 / column.accumulate(maxOp, 0.0);

            } else if ( type == NormalizationType::MEDIAN) {
                auto data = column.copyData();
                std::transform(data.begin(), data.end(), data.begin(), [](double x) {
                    return std::abs(x);
                }); // TODO Copy and transform in single loop
                scale = 1.0 / median(data.begin(), data.end());
            } else {  // type == NormalizationType::Q95
                auto data = column.copyData();
                std::transform(data.begin(), data.end(), data.begin(), [](double x) {
                    return std::abs(x);
                }); // TODO Copy and transform in single loop
                scale = 1.0 / quantile(data.begin(), data.end(), 0.95);
            }

            auto scaleOp = [scale](double x) {
                return x * scale;
            };

            column.transform(scaleOp);
            normalizations.push_back(scale);
        } else {
            normalizations.push_back(1.0);
        }
    }
    return normalizations;
}

template <typename RealType>
size_t ModelData<RealType>::getNumberOfPatients() const {
    if (nPatients == 0) {
        nPatients = getNumberOfStrata();
    }
	return nPatients;
}

template <typename RealType>
int ModelData<RealType>::getNumberOfTypes() const {
	return nTypes;
}

template <typename RealType>
const std::string ModelData<RealType>::getConditionId() const {
	return conditionId;
}

template <typename RealType>
ModelData<RealType>::~ModelData() {
	// Do nothing
}

// template <typename RealType>
// ModelData<RealType>::getPrecisionType() const;

template <>
PrecisionType ModelData<double>::getPrecisionType() const {
    return PrecisionType::FP64;
}

template <>
PrecisionType ModelData<float>::getPrecisionType() const {
    return PrecisionType::FP32;
}

template <typename RealType>
const int*  ModelData<RealType>::getPidVector() const { // TODO deprecated
//	return makeDeepCopy(&pid[0], pid.size());
    return (pid.size() == 0) ? nullptr : pid.data();
}

// std::vector<int>* ModelData::getPidVectorSTL() { // TODO deprecated
// 	return new std::vector<int>(pid);
// }

template <typename RealType>
IntVector ModelData<RealType>::getPidVectorSTL() const {
    return getStratumIndexRef(); // copy
}

template <typename RealType>
void ModelData<RealType>::addIntercept() {
    // The column loadX() makes for an intercept: INTERCEPT storage (1 in every row, no stored values) at column 0.
    // Code that finds the intercept by its storage, such as the adaptive intercept, needs it; a DENSE column of ones
    // fits the same but is not recognised. Callers add it before an offset covariate, which must stay in column 0.
    X.insert(0, INTERCEPT);
    setHasInterceptCovariate(true);
}

// The intercept of data loaded with its columns, such as R's formula data: the column after any offset, a dense column
// of ones. It gets the storage addIntercept() gives an intercept (see there). A column that is not all ones (formula
// data with several outcome types has one intercept column per type) stays dense and is still the intercept
template <typename RealType>
void ModelData<RealType>::setInterceptCovariate() {
    const size_t index = hasOffsetCovariate ? 1 : 0;
    if (index >= getNumberOfColumns()) {
        std::ostringstream stream;
        stream << "No intercept column";
        error->throwError(stream);
    }
    X.getColumn(index).convertColumnToIntercept(getNumberOfRows());
    setHasInterceptCovariate(true);
}

template <typename RealType>
void ModelData<RealType>::logTransformCovariate(const IdType covariate) {
    X.getColumn(covariate).transform([](RealType x) {
        return std::log(x);
    });
}

template <typename RealType>
void ModelData<RealType>::convertCovariateToDense(const IdType covariate) {
    IdType index = getColumnIndex(covariate);
    X.getColumn(index).convertColumnToDense(getNumberOfRows());
}

template <typename RealType>
void ModelData<RealType>::convertAllCovariatesToDense(int length) {
    for (int index = 0; index < getNumberOfColumns(); ++index) {
        X.getColumn(index).convertColumnToDense(length);
    }
}

template <typename RealType>
void ModelData<RealType>::setOffsetCovariate(const IdType covariate) {
    int index;
    if (covariate == -1) { // TODO  Bad, magic number
        moveTimeToCovariate(true);
        index = getNumberOfCovariates() - 1;
    } else {
        index = getColumnIndexByName(covariate);
    }
    if (hasInterceptCovariate && index == (hasOffsetCovariate ? 1 : 0)) {
        // Code finds the intercept after any offset: the column behind it would be taken for the intercept (and an
        // intercept column has no values to log-transform)
        std::ostringstream stream;
        stream << "The intercept cannot be the offset covariate";
        error->throwError(stream);
    }
    X.moveToFront(index);
    X.getColumn(0).add_label(-1); // TODO Generic label for offset?
    setHasOffsetCovariate(true);
}

template <typename RealType>
const RealType* ModelData<RealType>::getYVector() const { // TODO deprecated
//	return makeDeepCopy(&y[0], y.size());
	return &y[0];
}

// template <typename RealType>
// void ModelData<RealType>::setYVector(Vector<RealType> y_){
// 	y = y_;
// }

//int* ModelData::getNEventVector() { // TODO deprecated
////	return makeDeepCopy(&nevents[0], nevents.size());
//	return &nevents[0];
//}

template <typename RealType>
RealType* ModelData<RealType>::getOffsetVector() { // TODO deprecated
//	return makeDeepCopy(&offs[0], offs.size());
	return &offs[0];
}

// void ModelData::sortDataColumns(vector<int> sortedInds){
// 	reindexVector(allColumns,sortedInds);
// }

template <typename RealType>
double ModelData<RealType>::getSquaredNorm() const {

	int startIndex = 0;
	if (hasInterceptCovariate) ++startIndex;
	if (hasOffsetCovariate) ++startIndex;

	std::vector<double> squaredNorm;

	for (size_t index = startIndex; index < getNumberOfColumns(); ++index) {
		squaredNorm.push_back(X.getColumn(index).squaredSumColumn(getNumberOfRows()));
	}

	return std::accumulate(squaredNorm.begin(), squaredNorm.end(), 0.0);
}

template <typename RealType>
size_t ModelData<RealType>::getNumberOfStrata() const {
    const auto nRows = getNumberOfRows();
    if (nRows == 0) {
        return 0;
    }

    if (nStrata == 0) {
        if (pid.empty()) {
            return (nStrata = nRows);
        }
        nStrata = 1;
        int cLabel = pid[0];
        for (size_t i = 1; i < pid.size(); ++i) {
            if (cLabel != pid[i]) {
                cLabel = pid[i];
                nStrata++;
            }
        }
    }
    return nStrata;
}

template <typename RealType>
double ModelData<RealType>::getNormalBasedDefaultVar() const {
// 	return getNumberOfVariableColumns() * getNumberOfRows() / getSquaredNorm();
	// Reciprocal of what is reported in Genkins et al.
	if (pooledRows.loaded && pooledRows.rows() > 0) {
		// plr: the norm and the rows of the person-interval design, where column j enters row r of subject i as
		// x_ij tau_s(j)(t_r): sum_i x_ij^2 q_i,s(j), q_is = sum over the subject's rows of tau_s(t_r)^2
		const size_t K = getNumberOfRows();
		const int S = pooledRows.basisColumns + 1;
		std::vector<double> q(K * S, 0.0);
		for (size_t i = 0; i < K; ++i) {
			for (size_t r = pooledRows.rowStart[i]; r < pooledRows.rowStart[i + 1]; ++r) {
				const int t = pooledRows.intervalOf(i, r);
				q[i * S] += 1.0;
				for (int b = 0; b < pooledRows.basisColumns; ++b) {
					const double v = pooledRows.basis[static_cast<size_t>(t) * pooledRows.basisColumns + b];
					q[i * S + b + 1] += v * v;
				}
			}
		}
		std::vector<int> slot(getNumberOfColumns(), 0);
		for (const auto& column : pooledRows.timeColumns) {
			slot[getColumnIndex(column.first)] = column.second + 1;
		}
		int startIndex = 0;
		if (hasInterceptCovariate) ++startIndex;
		if (hasOffsetCovariate) ++startIndex;
		double norm = 0.0;
		for (size_t index = startIndex; index < getNumberOfColumns(); ++index) {
			for (GenericIterator<RealType> it(X, static_cast<int>(index)); it; ++it) {
				const double x = static_cast<double>(it.value());
				norm += x * x * q[it.index() * S + slot[index]];
			}
		}
		return norm / getNumberOfVariableColumns() / static_cast<double>(pooledRows.rows());
	}
	return getSquaredNorm() / getNumberOfVariableColumns() / getNumberOfRows();
}

template <typename RealType>
int ModelData<RealType>::getNumberOfVariableColumns() const {
	int dim = getNumberOfColumns();
	if (hasInterceptCovariate) --dim;
	if (hasOffsetCovariate) --dim;
	return dim;
}

template <typename RealType>
double ModelData<RealType>::sum(const IdType covariate, const int power) const {

    size_t index = getColumnIndex(covariate);
    if (power == 0) {
		return reduce(index, ZeroPower());
	} else if (power == 1) {
		return reduce(index, FirstPower());
	} else {
		return reduce(index, SecondPower());
	}
}

template <typename RealType>
void ModelData<RealType>::sumByGroup(std::vector<double>& out, const IdType covariate, const IdType groupBy, const int power) const {
    size_t covariateIndex = getColumnIndex(covariate);
    size_t groupByIndex = getColumnIndex(groupBy);
    out.resize(2);
    if (power == 0) {
        reduceByGroup(out, covariateIndex, groupByIndex, ZeroPower());
    } else if (power == 1) {
        reduceByGroup(out, covariateIndex, groupByIndex, FirstPower());
    } else {
        reduceByGroup(out, covariateIndex, groupByIndex, SecondPower());
    }
}

template <typename RealType>
void ModelData<RealType>::sumByPid(std::vector<double>& out, const IdType covariate, const int power) const {
    size_t covariateIndex = getColumnIndex(covariate);
    // Use getStratumIndexRef(), not raw `pid`: unstratified data leave `pid`
    // empty (each row is its own stratum), matching binaryReductionByStratum.
    const auto& strata = getStratumIndexRef();
    out.assign(getNumberOfPatients(), 0.0);
    if (power == 0) {
        reduceByGroup(out, covariateIndex, strata, ZeroPower());
    } else if (power == 1) {
        reduceByGroup(out, covariateIndex, strata, FirstPower());
    } else {
        reduceByGroup(out, covariateIndex, strata, SecondPower());
    }
}

template <typename RealType>
std::string ModelData<RealType>::getRowLabel(const size_t i) const {
    if (consecutiveRowIds && i < getX().nRows) {
        return std::to_string(firstRowId + static_cast<IdType>(i));
    }
    if (i >= labels.size()) {
        return missing;
    } else {
        return labels[i];
    }
}

template <>
const std::string ModelData<double>::missing = "NA";

template <>
const std::string ModelData<float>::missing = "NA";

// Instantiate classes
template class ModelData<double>;
template class ModelData<float>;

} // namespace
