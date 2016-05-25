/*
 * ModelData.h
 *
 *  Created on: August, 2012
 *      Author: msuchard
 */

#ifndef MODELDATA_H_
#define MODELDATA_H_

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

// using std::map;
// using std::string;
// using std::vector;
// using std::stringstream;

#include "CompressedDataMatrix.h"
#include "io/ProgressLogger.h"
#include "io/SparseIndexer.h"

//#define USE_DRUG_STRING

namespace bsccs {

// template <class T> void reindexVector(std::vector<T>& vec, std::vector<int> ind) {
// 	int n = (int) vec.size();
// 	std::vector<T> temp = vec;
// 	for(int i = 0; i < n; i++){
// 		vec[i] = temp[ind[i]];
// 	}
// }

class ModelData : public CompressedDataMatrix {
public:
//	ModelData();

	ModelData(
    	ModelType modelType,
        loggers::ProgressLoggerPtr log,
        loggers::ErrorHandlerPtr error
    );

//	pid.begin(), pid.end(),
//	y.begin(), y.end(),
//	z.begin(), z.end(),
//	offs.begin(), offs.end(),
//	xip.begin(), xii.end()

	template <typename IntegerVector, typename RealVector>
	ModelData(
	        ModelType _modelType,
			const IntegerVector& _pid,
			const RealVector& _y,
			const RealVector& _z,
			const RealVector& _offs,
            loggers::ProgressLoggerPtr _log,
            loggers::ErrorHandlerPtr _error
			) :
		modelType(_modelType), nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false), isFinalized(false)
		, pid(_pid.begin(), _pid.end()) // copy
		, y(_y.begin(), _y.end()) // copy
		, z(_z.begin(), _z.end()) // copy
		, offs(_offs.begin(), _offs.end()) // copy
		, sparseIndexer(*this)
		, log(_log), error(_error)
		, touchedY(true), touchedX(true)
		{

	}

	virtual ~ModelData();

	size_t append(
        const std::vector<IdType>& oStratumId,
        const std::vector<IdType>& oRowId,
        const std::vector<double>& oY,
        const std::vector<double>& oTime,
        const std::vector<IdType>& cRowId,
        const std::vector<IdType>& cCovariateId,
        const std::vector<double>& cCovariateValue
    );


	void loadY(
		const std::vector<IdType>& stratumId,
		const std::vector<IdType>& rowId,
		const std::vector<double>& y,
		const std::vector<double>& time
	);

	int loadX(
		const IdType covariateId,
		const std::vector<IdType>& rowId,
		const std::vector<double>& covariateValue,
		const bool reload,
		const bool append,
		const bool forceSparse
	);


	int loadMultipleX(
		const std::vector<int64_t>& covariateId,
		const std::vector<int64_t>& rowId,
		const std::vector<double>& covariateValue,
		const bool checkCovariateIds,
		const bool checkCovariateBounds,
		const bool append,
		const bool forceSparse
	);

	const int* getPidVector() const;
	const real* getYVector() const;
	void setYVector(std::vector<real> y_);
	int* getNEventVector();
	real* getOffsetVector();
//	map<int, IdType> getDrugNameMap();
	int getNumberOfPatients() const;
	const std::string getConditionId() const;
	std::vector<int> getPidVectorSTL() const;

	const std::vector<real>& getZVectorRef() const {
		return z;
	}

	const std::vector<real>& getTimeVectorRef() const {
		return offs;
	}

	const std::vector<real>& getYVectorRef() const {
		return y;
	}

	const std::vector<int>& getPidVectorRef() const { // Not const because PIDs can get renumbered
		return pid;
	}

	std::vector<real>& getZVectorRef() {
		return z;
	}

	std::vector<real>& getYVectorRef() {
		return y;
	}

	std::vector<int>& getPidVectorRef() { // Not const because PIDs can get renumbered
		return pid;
	}

//	const std::vector<int>& getNEventsVectorRef() const {
//		return nevents;
//	}

	bool getHasOffsetCovariate() const {
		return hasOffsetCovariate;
	}

	void setHasOffsetCovariate(bool b) {
		hasOffsetCovariate = b;
	}

	bool getHasInterceptCovariate() const {
		return hasInterceptCovariate;
	}

	void setHasInterceptCovariate(bool b) {
		hasInterceptCovariate = b;
	}

	bool getHasRowLabels() const {
		return (labels.size() == getNumberOfRows());
	}

	bool getIsFinalized() const {
	    return isFinalized;
	}

	void setIsFinalized(bool b) {
	    isFinalized = b;
	}

	void sortDataColumns(std::vector<int> sortedInds);

	double getSquaredNorm() const;

	double getNormalBasedDefaultVar() const;

	int getNumberOfVariableColumns() const;

	int getNumberOfTypes() const;

	ModelType getModelType() const { return modelType; }

	size_t getNumberOfStrata() const;

    size_t getColumnIndex(const IdType covariate) const;

    void moveTimeToCovariate(bool takeLog);

    std::vector<double> normalizeCovariates(const NormalizationType type);

	const std::string& getRowLabel(size_t i) const {
		if (i >= labels.size()) {
			return missing;
		} else {
			return labels[i];
		}
	}

	void clean() const { touchedY = false; touchedX = false; }

	const bool getTouchedY() const { return touchedY; }

	const bool getTouchedX() const { return touchedX; }

	// TODO Improve encapsulation
	friend class SCCSInputReader;
	friend class CLRInputReader;
	friend class RTestInputReader;
	friend class CoxInputReader;
	friend class CCTestInputReader;
	friend class GenericSparseReader;

	template <class FormatType, class MissingPolicy> friend class BaseInputReader;
	template <class ImputationPolicy> friend class BBRInputReader;
	template <class ImputationPolicy> friend class CSVInputReader;

protected:
    ModelType modelType;

	mutable int nPatients;
	mutable size_t nStrata;

	bool hasOffsetCovariate;
	bool hasInterceptCovariate;
	bool isFinalized;

	IntVector pid;
	RealVector y;
	RealVector z; // TODO Remove
	RealVector offs; // TODO Rename to 'time'
	IntVector nevents; // TODO Where are these used?
	std::string conditionId;
	std::vector<std::string> labels; // TODO Change back to 'long'

	int nTypes;

private:
	// Disable copy-constructors and copy-assignment
	ModelData(const ModelData&);
	ModelData& operator = (const ModelData&);

	static const std::string missing;

    std::pair<IdType,int> lastStratumMap;

    SparseIndexer sparseIndexer;

protected:
    loggers::ProgressLoggerPtr log;
    loggers::ErrorHandlerPtr error;

    typedef bsccs::unordered_map<IdType,size_t> RowIdMap;
    RowIdMap rowIdMap;


    mutable bool touchedY;
    mutable bool touchedX;
};


template <typename Itr>
auto median(Itr begin, Itr end) -> typename Itr::value_type {
    const auto size = std::distance(begin, end);
    auto target = begin + size / 2;

    std::nth_element(begin, target, end);
    if (size % 2 != 0) { // Odd number of elements
        return *target;
    } else { // Even number of elements
        auto targetNeighbor = std::max_element(begin, target);
        return (*target + *targetNeighbor) / 2.0;
    }
}

template <typename Itr>
auto quantile(Itr begin, Itr end, double q) -> typename Itr::value_type {
    const auto size = std::distance(begin, end);

    auto fraction = (size - 1) * q;
    const auto lo = std::floor(fraction);
    const auto hi = std::ceil(fraction);
    fraction -= lo;

    auto high = begin + hi;
    std::nth_element(begin, high, end);

    if (hi == lo) { // whole number
        return *high;
    } else {
        auto low = std::max_element(begin, high);
        return (1.0 - fraction) * (*low) + fraction * (*high);
    }
}

} // namespace

#endif /* MODELDATA_H_ */
