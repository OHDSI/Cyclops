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
#include <limits>

// using std::map;
// using std::string;
// using std::vector;
// using std::stringstream;

#include "CompressedDataMatrix.h"
#include "Iterators.h"
#include "io/ProgressLogger.h"
#include "io/SparseIndexer.h"

//#define USE_DRUG_STRING

namespace bsccs {

 template <class T> void reindexVector(std::vector<T>& vec, std::vector<int> ind) {
 	int n = (int) vec.size();
 	std::vector<T> temp = vec;
 	for(int i = 0; i < n; i++){
 		vec[i] = temp[ind[i]];
 	}
 }

class AbstractModelData {
    // fp-agnostic interface to model data
public:

    virtual PrecisionType getPrecisionType() const = 0;

    virtual void clean() const = 0;

    virtual int getColumnIndexByName(IdType name) const = 0;

    virtual std::string getColumnLabel(const IdType& covariate) const = 0;

    virtual IdType getColumnNumericalLabel(const IdType& covariate) const = 0;

    virtual size_t getColumnIndex(const IdType covariate) const = 0;

    virtual FormatType getColumnType(const IdType& covariate) const = 0;

    virtual std::string getColumnTypeString(const IdType& covariate) const = 0;

    virtual ModelType getModelType() const = 0;

    virtual const IntVector& getPidVectorRef() const = 0;

    virtual IntVector getPidVectorSTL() const = 0;

    virtual size_t getNumberOfPatients() const = 0;

    virtual size_t getNumberOfRows() const = 0;

    virtual size_t getNumberOfEntries(const IdType& covariate) const = 0;

    virtual size_t getNumberOfCovariates() const = 0;

    virtual bool getHasOffsetCovariate() const = 0;

    virtual bool getHasInterceptCovariate() const = 0;

    virtual bool getHasRowLabels() const = 0;

    virtual const std::string& getRowLabel(const size_t covariate) const = 0;

    virtual bool getTouchedY() const = 0;

    virtual bool getTouchedX() const = 0;

    virtual double getNormalBasedDefaultVar() const = 0;

    virtual int getNumberOfTypes() const = 0;

    virtual std::vector<double> copyYVector() const = 0;

    virtual std::vector<double> copyTimeVector() const = 0;

    virtual std::vector<double> copyZVector() const = 0;

    virtual std::vector<double> univariableCorrelation(
            const std::vector<IdType>& covariateLabel) const = 0;

    virtual void sumByPid(
            std::vector<double>& result, const IdType covariate, const int power) const = 0;

    virtual void sumByGroup(
            std::vector<double>& result, const IdType covariate, const IdType groupBy,
            const int power = 1) const = 0;

    virtual double sum(const IdType covariate, const int power) const = 0;

    virtual std::vector<double> normalizeCovariates(NormalizationType normalizationType) = 0;

    virtual void setHasInterceptCovariate(bool hasIntercept) = 0;

    virtual bool getIsFinalized() const = 0;

    virtual void setIsFinalized(bool finalized) = 0;

    virtual void addIntercept() = 0;

    virtual void setOffsetCovariate(const IdType covariate) = 0;

    virtual void logTransformCovariate(const IdType covariate) = 0;

    virtual void convertCovariateToDense(const IdType covariate) = 0;

	virtual double innerProductWithOutcome(const size_t index) const = 0;

    virtual void loadY(
            const std::vector<IdType>& stratumId,
            const std::vector<IdType>& rowId,
            const std::vector<double>& y,
            const std::vector<double>& time
    ) = 0;

    virtual int loadX(
            const IdType covariateId,
            const std::vector<IdType>& rowId,
            const std::vector<double>& covariateValue,
            const bool reload,
            const bool append,
            const bool forceSparse
    ) = 0;

    virtual int loadMultipleX(
            const std::vector<int64_t>& covariateId,
            const std::vector<int64_t>& rowId,
            const std::vector<double>& covariateValue,
            const bool checkCovariateIds,
            const bool checkCovariateBounds,
            const bool append,
            const bool forceSparse
    ) = 0;

    virtual std::vector<std::string> loadStratTimeEffects(
            const std::vector<IdType>& oStratumId,
            const std::vector<IdType>& oRowId,
            const std::vector<IdType>& oSubjectId,
            const std::vector<IdType>& timeEffectCovariateId
    ) = 0;

    virtual size_t append(
            const std::vector<IdType>& oStratumId,
            const std::vector<IdType>& oRowId,
            const std::vector<double>& oY,
            const std::vector<double>& oTime,
            const std::vector<IdType>& cRowId,
            const std::vector<IdType>& cCovariateId,
            const std::vector<double>& cCovariateValue
    ) = 0;

    virtual void printMatrixMarketFormat(std::ostream& stream) const = 0;

    virtual int getFloatingPointSize() const = 0;

    virtual ~AbstractModelData() { }

// private:
//     AbstractModelData(const AbstractModelData&);
//     AbstractModelData& operator = (const AbstractModelData&);

};


template <typename RealType>
class ModelData : public AbstractModelData {
public:
//	ModelData();

    typedef typename CompressedDataColumn<RealType>::RealVector RealVector;
    typedef typename CompressedDataColumn<RealType>::RealVectorPtr RealVectorPtr;

    int getFloatingPointSize() const { return sizeof(RealType) * 8; }

    size_t getNumberOfColumns() const { return X.getNumberOfColumns(); }

    size_t getNumberOfRows() const { return X.getNumberOfRows(); }

  	size_t getNumberOfEntries(const IdType& covariate) const { return X.getNumberOfEntries(covariate); }

    int getColumnIndexByName(IdType name) const { return X.getColumnIndexByName(name); }

    void printMatrixMarketFormat(std::ostream& stream) const { X.printMatrixMarketFormat(stream); }


    // using CompressedDataMatrix<RealType>::getNumberOfColumns;
    // using CompressedDataMatrix<RealType>::getNumberOfRows;
    // using CompressedDataMatrix<RealType>::getFormatType;
    // using CompressedDataMatrix<RealType>::getColumn;
    // using CompressedDataMatrix<RealType>::push_back;
    // using CompressedDataMatrix<RealType>::moveToFront;
    // using CompressedDataMatrix<RealType>::insert;

    CompressedDataMatrix<RealType> X;

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

	template <typename InputIntegerVector, typename InputRealVector>
	ModelData(
	        ModelType _modelType,
			const InputIntegerVector& _pid,
			const InputRealVector& _y,
			const InputRealVector& _z,
			const InputRealVector& _offs,
            loggers::ProgressLoggerPtr _log,
            loggers::ErrorHandlerPtr _error
			) :
		modelType(_modelType), nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false), isFinalized(false)
		, pid(_pid.begin(), _pid.end()) // copy
		, y(_y.begin(), _y.end()) // copy
		, z(_z.begin(), _z.end()) // copy
		, offs(_offs.begin(), _offs.end()) // copy
		, sparseIndexer(X)
		, log(_log), error(_error)
		, touchedY(true), touchedX(true)
		{

	}

	virtual ~ModelData();

	AbstractModelData* castToFloat() {
	    auto* floatModelData = new ModelData<float>(modelType, pid, y, z, offs, log, error);



	    return floatModelData;
	}

	const CompressedDataMatrix<RealType>& getX() const {
	    return X;
	}

	CompressedDataMatrix<RealType>& getX() {
	    return X;
	}

	PrecisionType getPrecisionType() const;

	std::string getColumnLabel(const IdType& covariate) const {
	    return X.getColumn(covariate).getLabel();
	}

	std::string getColumnTypeString(const IdType& covariate) const {
	    return X.getColumn(covariate).getTypeString();
	}

	FormatType getColumnType(const IdType& covariate) const {
		return X.getColumn(covariate).getFormatType();
	}

	IdType getColumnNumericalLabel(const IdType& covariate) const {
	    return X.getColumn(covariate).getNumericalLabel();
	}

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

	std::vector<std::string> loadStratTimeEffects(
		const std::vector<IdType>& oStratumId,
		const std::vector<IdType>& oRowId,
		const std::vector<IdType>& oSubjectId,
		const std::vector<IdType>& timeEffectCovariateId
	);

	const int* getPidVector() const;
	const RealType* getYVector() const;
	void setYVector(std::vector<double> y_);
	int* getNEventVector();
	RealType* getOffsetVector();
//	map<int, IdType> getDrugNameMap();
	size_t getNumberOfPatients() const;
	const std::string getConditionId() const;
	IntVector getPidVectorSTL() const;

	const RealVector& getZVectorRef() const {
		return z;
	}

	const RealVector& getTimeVectorRef() const {
		return offs;
	}

	const RealVector& getYVectorRef() const {
		return y;
	}

	const IntVector& getPidVectorRef() const { // Not const because PIDs can get renumbered
		return pid;
	}

	std::vector<double> copyYVector() const {
        std::vector<double> copy(y.size());
	    std::copy(std::begin(y), std::end(y), std::begin(copy));
	    return copy;
	}

	std::vector<double> copyZVector() const {
	    std::vector<double> copy(z.size());
	    std::copy(std::begin(z), std::end(z), std::begin(copy));
	    return copy;
	}

	std::vector<double> copyTimeVector() const {
	    std::vector<double> copy(offs.size());
	    std::copy(std::begin(offs), std::end(offs), std::begin(copy));
	    return copy;
	}

	RealVector& getZVectorRef() {
		return z;
	}

    RealVector& getYVectorRef() {
		return y;
	}

	IntVector& getPidVectorRef() { // Not const because PIDs can get renumbered
		return pid;
	}

//	const std::vector<int>& getNEventsVectorRef() const {
//		return nevents;
//	}

    void addIntercept();

	void setOffsetCovariate(const IdType covariate);

	void logTransformCovariate(const IdType covariate);

	void convertAllCovariatesToDense(int length);

	void convertCovariateToDense(const IdType covariate);

    size_t getNumberOfCovariates() const {
        return getNumberOfColumns();
    }

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

	const std::string& getRowLabel(const size_t i) const;

	void clean() const { touchedY = false; touchedX = false; }

	bool getTouchedY() const { return touchedY; }

	bool getTouchedX() const { return touchedX; }

	double sum(const IdType covariate, const int power = 1) const;

	void standardize(const IdType covariate);

	void sumByGroup(std::vector<double>& out, const IdType covariate, const IdType groupBy, const int power = 1) const;

	void sumByPid(std::vector<double>& out, const IdType covariate, const int power = 1) const;

	template <typename F>
	void transform(const size_t index, F func) {
	    switch (X.getFormatType(index)) {
	    case INDICATOR :
	        transformImpl<IndicatorIterator<RealType>>(index, func);
	        break;
	    case SPARSE :
	        transformImpl<SparseIterator<RealType>>(index, func);
	        break;
	    case DENSE :
	        transformImpl<DenseIterator<RealType>>(index, func);
	        break;
	    case INTERCEPT :
	        transformImpl<InterceptIterator<RealType>>(index, func);
	        break;
	    }
	}

	template <typename F>
	double reduce(const long index, F func) const {
	    if (index < 0) { // reduce outcome
	        return reduceOutcomeImpl(func);
	    }
	    double sum = 0.0;
	    switch (X.getFormatType(index)) {
	    case INDICATOR :
	        sum = reduceImpl<IndicatorIterator<RealType>>(index, func);
	        break;
	    case SPARSE :
	        sum = reduceImpl<SparseIterator<RealType>>(index, func);
	        break;
	    case DENSE :
	        sum = reduceImpl<DenseIterator<RealType>>(index, func);
	        break;
	    case INTERCEPT :
	        sum = reduceImpl<InterceptIterator<RealType>>(index, func);
	        break;
	    }
	    return sum;
	}

	template <typename F>
	double innerProductWithOutcome(const size_t index, F func) const {
	    double sum = 0.0;
	    switch (X.getFormatType(index)) {
	    case INDICATOR :
	        sum = innerProductWithOutcomeImpl<IndicatorIterator<RealType>>(index, func);
	        break;
	    case SPARSE :
	        sum = innerProductWithOutcomeImpl<SparseIterator<RealType>>(index, func);
	        break;
	    case DENSE :
	        sum = innerProductWithOutcomeImpl<DenseIterator<RealType>>(index, func);
	        break;
	    case INTERCEPT :
	        sum = innerProductWithOutcomeImpl<InterceptIterator<RealType>>(index, func);
	        break;
	    }
	    return sum;
	}

	double innerProductWithOutcome(const size_t index) const {
		return innerProductWithOutcome(index, InnerProduct());
	}

	template <typename T, typename F>
	void reduceByGroup(T& out, const size_t reductionIndex, const size_t groupByIndex, F func) const {
	    if (X.getFormatType(groupByIndex) != INDICATOR) {
	        std::ostringstream stream;
	        stream << "Grouping by non-indicators is not yet supported.";
	        error->throwError(stream);
	    }
	    switch (X.getFormatType(reductionIndex)) {
	    case INDICATOR :
	        reduceByGroupImpl<IndicatorIterator<RealType>>(out, reductionIndex, groupByIndex, func);
	        break;
	    case SPARSE :
	        reduceByGroupImpl<SparseIterator<RealType>>(out, reductionIndex, groupByIndex, func);
	        break;
	    case DENSE :
	        reduceByGroupImpl<DenseIterator<RealType>>(out, reductionIndex, groupByIndex, func);
	        break;
	    case INTERCEPT :
	        reduceByGroupImpl<InterceptIterator<RealType>>(out, reductionIndex, groupByIndex, func);
	        break;
	    }
	}

	struct Sum {
	    inline RealType operator()(RealType x, RealType y) {
	        return x + y;
	    }
	};

	struct ZeroPower {
	    inline RealType operator()(RealType x) {
	        return x == 0.0 ? 0.0 : 1.0;
	    }
	};

	struct FirstPower {
	    inline RealType operator()(RealType x) {
	        return x;
	    }
	};

	struct SecondPower {
	    inline RealType operator()(RealType x) {
	        return x * x;
	    }
	};

	struct InnerProduct {
	    inline RealType operator()(RealType x, RealType y) {
	        return x * y;
	    }
	};


	std::vector<double> univariableCorrelation(const std::vector<IdType>& covariateLabel) const {

	    const double Ey1 = reduce(-1, FirstPower()) / getNumberOfRows();
	    const double Ey2 = reduce(-1, SecondPower()) / getNumberOfRows();
	    const double Vy = Ey2 - Ey1 * Ey1;

	    std::vector<double> result;

	    auto oneVariable = [this, &result, Ey1, Vy](const size_t index) {
	        const double Ex1 = this->reduce(index, FirstPower()) / this->getNumberOfRows();
	        const double Ex2 = this->reduce(index, SecondPower()) / this->getNumberOfRows();
	        const double Exy = this->innerProductWithOutcome(index, InnerProduct()) / this->getNumberOfRows();

	        const double Vx = Ex2 - Ex1 * Ex1;
	        const double cov = Exy - Ex1 * Ey1;
	        const double cor = (Vx > 0.0 && Vy > 0.0) ?
	        cov / std::sqrt(Vx) / std::sqrt(Vy) : std::numeric_limits<double>::quiet_NaN();

	        // Rcpp::Rcout << index << " " << Ey1 << " " << Ey2 << " " << Ex1 << " " << Ex2 << std::endl;
	        // Rcpp::Rcout << index << " " << ySquared << " " << xSquared <<  " " << crossProduct << std::endl;
	        result.push_back(cor);
	    };

	    if (covariateLabel.size() == 0) {
	        result.reserve(getNumberOfCovariates());
	        size_t index = (getHasOffsetCovariate()) ? 1 : 0;
	        for (; index <  getNumberOfCovariates(); ++index) {
	            oneVariable(index);
	        }
	    } else {
	        result.reserve(covariateLabel.size());
	        for(auto it = covariateLabel.begin(); it != covariateLabel.end(); ++it) {
	            oneVariable(getColumnIndex(*it));
	        }
	    }

	    return result;
	}


	template <typename T, typename F>
	void binaryReductionByStratum(T& out, const size_t reductionIndex, F func) const {
	    binaryReductionByGroup(out, reductionIndex, pid, func);
	}

	// TODO Improve encapsulation
	// friend class SCCSInputReader;
	// friend class CLRInputReader;
	// friend class RTestInputReader;
	// friend class CoxInputReader;
	// friend class CCTestInputReader;
	// friend class GenericSparseReader;
	// friend class InputReader;
	//
	// template <class FormatType, class MissingPolicy> friend class BaseInputReader;
	template <class ImputationPolicy> friend class BBRInputReader;
	// template <class ImputationPolicy> friend class CSVInputReader;

	friend void push_back_label(ModelData<double>& modeData, const std::string& label);
	friend void push_back_pid(ModelData<double>& modeData, const int cases);
	friend void push_back_y(ModelData<double>& modelData, const double value);
	friend void push_back_nevents(ModelData<double>& modelData, const int num);
	friend void push_back_z(ModelData<double>& modelData, const double value);
	friend void push_back_offs(ModelData<double>& modelData, const double value);
	friend void setConditionId(ModelData<double>& modelData, const std::string& id);
	friend void setNumberPatients(ModelData<double>& modelData, const int cases);
	friend void setNumberRows(ModelData<double>& modelData, const int nrows);

    friend void push_back_label(ModelData<float>& modeData, const std::string& label);
    friend void push_back_pid(ModelData<float>& modeData, const int cases);
    friend void push_back_y(ModelData<float>& modelData, const float value);
    friend void push_back_nevents(ModelData<float>& modelData, const int num);
    friend void push_back_z(ModelData<float>& modelData, const float value);
    friend void push_back_offs(ModelData<float>& modelData, const float value);
    friend void setConditionId(ModelData<float>& modelData, const std::string& id);
    friend void setNumberPatients(ModelData<float>& modelData, const int cases);
    friend void setNumberRows(ModelData<float>& modelData, const int nrows);

protected:

    template <typename T, typename F>
    void binaryReductionByGroup(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) const {
        switch (X.getFormatType(reductionIndex)) {
        case INDICATOR :
            binaryReductionByGroup<IndicatorIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case SPARSE :
            binaryReductionByGroup<SparseIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case DENSE :
            binaryReductionByGroup<DenseIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case INTERCEPT :
            binaryReductionByGroup<InterceptIterator<RealType>>(out, reductionIndex, groups, func);
            break;

        }
    }

    template <typename IteratorType, typename T, typename F>
    void binaryReductionByGroup(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) const {
        IteratorType it(X, reductionIndex);

        for (; it; ++it) {
            out[groups[it.index()]] = func(out[groups[it.index()]], it.value());
        }
    }

    template <typename T, typename F>
    void reduceByGroup(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) const {
        switch (X.getFormatType(reductionIndex)) {
        case INDICATOR :
            reduceByGroupImpl<IndicatorIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case SPARSE :
            reduceByGroupImpl<SparseIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case DENSE :
            reduceByGroupImpl<DenseIterator<RealType>>(out, reductionIndex, groups, func);
            break;
        case INTERCEPT :
            reduceByGroupImpl<InterceptIterator<RealType>>(out, reductionIndex, groups, func);
            break;

        }
    }

    template <typename IteratorType, typename T, typename F>
    void reduceByGroupImpl(T& out, const size_t reductionIndex, const size_t groupByIndex, F func) const {
        IteratorType reduceIt(X, reductionIndex);
        IndicatorIterator<RealType> groupByIt(X, groupByIndex);

        GroupByIterator<IteratorType,RealType> it(reduceIt, groupByIt);
        for (; it; ++it) {
            out[it.group()] += func(it.value()); // TODO compute reduction in registers
        }
    }

    template <typename IteratorType, typename T, typename F>
    void reduceByGroupImpl(T& out, const size_t reductionIndex, const std::vector<int>& groups, F func) const {
        IteratorType it(X, reductionIndex);

        for (; it; ++it) {
            out[groups[it.index()]] += func(it.value()); // TODO compute reduction in registers
        }
    }

    template <typename IteratorType, typename F>
    void transformImpl(const size_t index, F func) {
        IteratorType it(X, index);
        for (; it; ++it) {
            it.ref() = func(it.value()); // TODO No yet implemented
        }
    }

    template <typename F>
    double reduceOutcomeImpl(F func) const {
        double sum = 0.0;
        for(auto it = std::begin(y); it != std::end(y); ++it) {
            sum += func(*it);
        }
        return sum;
    }

    template <typename IteratorType, typename F>
    double reduceImpl(const size_t index, F func) const {
        double sum = 0.0;
        IteratorType it(X, index);
        for (; it; ++it) {
            sum += func(it.value());
        }
        return sum;
    }

    template <typename IteratorType, typename F>
    double innerProductWithOutcomeImpl(const size_t index, F func) const {
        double sum = 0.0;
        IteratorType it(X, index);
        for (; it; ++it) {
            sum += func(y[it.index()], it.value());
        }
        return sum;
    }

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
	int64_t maxCovariateId;
	unordered_map<IdType, IdType> timeEffectCovariateIdMap; // (timeEffectCovariateName, index)

private:
	// Disable copy-constructors and copy-assignment
	ModelData(const ModelData&);
	ModelData& operator = (const ModelData&);

	static const std::string missing;

    std::pair<IdType,int> lastStratumMap;

    SparseIndexer<RealType> sparseIndexer;

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

// #include "ModelData.cpp"

#endif /* MODELDATA_H_ */
