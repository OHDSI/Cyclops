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

		for (size_t i = 0; i < oRowId.size(); ++i) { // ignored if oRowId.size() == 0
			IdType currentRowId = oRowId[i];
			rowIdMap[currentRowId] = i;

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

			// TODO Check timing on adding label as string
			std::stringstream ss;
			ss << currentRowId;
			labels.push_back(ss.str());
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
int ModelData<RealType>::loadMultipleX(
		const std::vector<int64_t>& covariateIds,
		const std::vector<int64_t>& rowIds,
		const std::vector<double>& covariateValues,
		const bool checkCovariateIds,
		const bool checkCovariateBounds,
		const bool append,
		const bool forceSparse) {

	auto columnIdItr = std::begin(covariateIds);
	const auto columnIdEnd = std::end(covariateIds);
	auto rowIdItr = std::begin(rowIds);
	auto covariateValueItr = std::begin(covariateValues);

	int firstColumnIndex = getNumberOfColumns();

	auto previousColumnId = *columnIdItr;

	auto index = getColumnIndexByName(previousColumnId);
	if (index >= 0) {
		if (!append) {
            std::ostringstream stream;
            stream << "Variable " << previousColumnId << " already exists";
            error->throwError(stream);
		}
		firstColumnIndex = index;
	} else {
		--previousColumnId; // not found
	}

	const bool hasCovariateValues = covariateValues.size() > 0;
	const bool useRowMap = rowIdMap.size() > 0;

// 	std::function<size_t(IdType)>
//     xform = [this](IdType id) {
//         return rowIdMap[id];
//     };


	while (columnIdItr != columnIdEnd) {

		if (index < 0) { // Create new column if necessary
		    const auto format =
		        (hasCovariateValues &&
		            (!(*covariateValueItr == 1.0 || *covariateValueItr == 0.0) // not 0 or 1
                    || forceSparse)) ?
		                SPARSE : INDICATOR;
	        X.push_back(format);
	        index = getNumberOfColumns() - 1;
	        X.getColumn(index).add_label(*columnIdItr);
        }

		// Append data into CompressedDataColumn
		CompressedDataColumn<RealType>& column = X.getColumn(index);
		FormatType format = column.getFormatType();

		// Grab whole column worth of data
		const auto columnId = *columnIdItr;
		auto lastRowId = *rowIdItr - 1;

		while (columnIdItr != columnIdEnd && *columnIdItr == columnId) {
		    if (*rowIdItr == lastRowId) {
		        std::ostringstream stream;
		        stream << "Repeated row-column entry at ";
		        stream << *rowIdItr << " - " << *columnIdItr;
		        throw std::range_error(stream.str());
		    }
		    const auto mappedRow = (useRowMap) ? rowIdMap[*rowIdItr] : *rowIdItr;
		    if (hasCovariateValues) {
                if (*covariateValueItr != 0.0) {
                    if (format == INDICATOR && *covariateValueItr != 1.0) {
                        // Upcast
                        column.convertColumnToSparse();
                        format = SPARSE;
                    }
                    if (format == SPARSE) {
                        column.getDataVector().push_back(*covariateValueItr);
                    }
                    column.getColumnsVector().push_back(mappedRow);
                }
		        ++covariateValueItr;
		    } else {
		        column.getColumnsVector().push_back(mappedRow);
		    }

		    lastRowId = *rowIdItr;

			++columnIdItr;
			++rowIdItr;
		}
		index = -1; // new column
	}

	maxCovariateId = *max_element(covariateIds.begin(), covariateIds.end());

	touchedX = true;
	return firstColumnIndex;
}

template <typename RealType>
int ModelData<RealType>::loadX(
		const IdType covariateId,
		const std::vector<IdType>& rowId,
		const std::vector<double>& covariateValue,
		const bool reload,
		const bool append,
		const bool forceSparse) {

    const bool hasCovariateValues = covariateValue.size() > 0;
    const bool useRowMap = rowIdMap.size() > 0;

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

    // Determine if covariate already exists
    int index = getColumnIndexByName(covariateId);
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

            auto lastRowId = *rowIdItr - 1;

            while (rowIdItr != rowIdEnd) {
                if (*rowIdItr == lastRowId) {
                    std::ostringstream stream;
                    stream << "Repeated row-column entry at";
                    stream << *rowIdItr << " - " << covariateId;
                    throw std::range_error(stream.str());
                }
                const auto mappedRow = (useRowMap) ? rowIdMap[*rowIdItr] : *rowIdItr;
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
                lastRowId = *rowIdItr;
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
    if (pid.size() == 0) {
        std::vector<int> tPid(getNumberOfRows());
        std::iota (std::begin(tPid), std::end(tPid), 0);
        return tPid;
    } else {
    	return pid;
    }
}

template <typename RealType>
void ModelData<RealType>::addIntercept() {
    // TODO Use INTERCEPT
    X.insert(0, DENSE); // add to front, TODO fix if offset
    setHasInterceptCovariate(true);
    const size_t numRows = getNumberOfRows();
    for (size_t i = 0; i < numRows; ++i) {
        X.getColumn(0).add_data(i, static_cast<RealType>(1));
    }
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
    out.resize(nPatients);
    if (power == 0) {
        reduceByGroup(out, covariateIndex, pid, ZeroPower());
    } else if (power == 1) {
        reduceByGroup(out, covariateIndex, pid, FirstPower());
    } else {
        reduceByGroup(out, covariateIndex, pid, SecondPower());
    }
}

template <typename RealType>
const std::string& ModelData<RealType>::getRowLabel(const size_t i) const {
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
