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

#include "ModelData.h"

namespace bsccs {

using std::string;
using std::vector;

ModelData::ModelData(
    ModelType _modelType,
    loggers::ProgressLoggerPtr _log,
    loggers::ErrorHandlerPtr _error
    ) : modelType(_modelType), nPatients(0), nStrata(0), hasOffsetCovariate(false), hasInterceptCovariate(false), isFinalized(false),
        lastStratumMap(0,0), sparseIndexer(*this), log(_log), error(_error) {
	// Do nothing
}


size_t ModelData::getColumnIndex(const IdType covariate) const {
    int index = getColumnIndexByName(covariate);
    if (index == -1) {
        std::ostringstream stream;
        stream << "Variable " << covariate << " is unknown";
        error->throwError(stream);
    }
    return index;
}

void ModelData::moveTimeToCovariate(bool takeLog) {
    push_back(NULL, make_shared<RealVector>(offs.begin(), offs.end()), DENSE); // TODO Remove copy?
}

//#define DEBUG_64BIT

size_t ModelData::append(
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
        IdType cInStratum = oStratumId[i];
        if (nRows == 0) {
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

#ifdef DEBUG_64BIT
        std::cout << currentRowId << std::endl;        
#endif        
        while (cOffset < nCovariates && cRowId[cOffset] == currentRowId) {          
            IdType covariate = cCovariateId[cOffset];
            real value = cCovariateValue[cOffset];
            
			if (!sparseIndexer.hasColumn(covariate)) {
				// Add new column
				sparseIndexer.addColumn(covariate, INDICATOR);			
			}

			CompressedDataColumn& column = sparseIndexer.getColumn(covariate);
			if (value != static_cast<real>(1) && value != static_cast<real>(0)) {
				if (column.getFormatType() == INDICATOR) {
					std::ostringstream stream;
					stream << "Up-casting covariate " << column.getLabel() << " to sparse!";
					log->writeLine(stream);
					column.convertColumnToSparse();
				}
			}

			// Add to storage
			bool valid = column.add_data(nRows, value);
			if (!valid) {
				std::ostringstream stream;
				stream << "Warning: repeated sparse entries in data row: "
						<< cRowId[cOffset]
						<< ", column: " << column.getLabel();
				log->writeLine(stream);
			}
			++cOffset;		                        
        }
        ++nRows;
    }        
    return nOutcomes;   
}

int ModelData::getNumberOfPatients() const {
    if (nPatients == 0) {
        nPatients = getNumberOfStrata();
    }
	return nPatients;
}

int ModelData::getNumberOfTypes() const {
	return nTypes;
}

const string ModelData::getConditionId() const {
	return conditionId;
}

ModelData::~ModelData() {
	// Do nothing
}

const int* ModelData::getPidVector() const { // TODO deprecated
//	return makeDeepCopy(&pid[0], pid.size());
	return &pid[0];
}

// std::vector<int>* ModelData::getPidVectorSTL() { // TODO deprecated
// 	return new std::vector<int>(pid);
// }

std::vector<int> ModelData::getPidVectorSTL() const {
	return pid;
}

const real* ModelData::getYVector() const { // TODO deprecated
//	return makeDeepCopy(&y[0], y.size());
	return &y[0];
}

void ModelData::setYVector(vector<real> y_){
	y = y_;
}

//int* ModelData::getNEventVector() { // TODO deprecated
////	return makeDeepCopy(&nevents[0], nevents.size());
//	return &nevents[0];
//}

real* ModelData::getOffsetVector() { // TODO deprecated
//	return makeDeepCopy(&offs[0], offs.size());
	return &offs[0];
}

// void ModelData::sortDataColumns(vector<int> sortedInds){
// 	reindexVector(allColumns,sortedInds);
// }

double ModelData::getSquaredNorm() const {

	int startIndex = 0;
	if (hasInterceptCovariate) ++startIndex;
	if (hasOffsetCovariate) ++startIndex;

	std::vector<double> squaredNorm;

	for (size_t index = startIndex; index < getNumberOfColumns(); ++index) {
		squaredNorm.push_back(getColumn(index).squaredSumColumn());
	}

	return std::accumulate(squaredNorm.begin(), squaredNorm.end(), 0.0);
}

size_t ModelData::getNumberOfStrata() const {     
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

double ModelData::getNormalBasedDefaultVar() const {
	return getNumberOfVariableColumns() * getNumberOfRows() / getSquaredNorm();
}

int ModelData::getNumberOfVariableColumns() const {
	int dim = getNumberOfColumns();
	if (hasInterceptCovariate) --dim;
	if (hasOffsetCovariate) --dim;
	return dim;
}

const string ModelData::missing = "NA";

} // namespace
