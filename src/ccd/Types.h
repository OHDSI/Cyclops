/*
 * Types.h
 *
 *  Created on: Aug 20, 2013
 *      Author: msuchard
 */

#ifndef CCD_TYPES_H_
#define CCD_TYPES_H_

#include <vector>
#include <map>

#ifdef R_BUILD
    #include "boost/smart_ptr.hpp"
#endif

namespace bsccs {

#ifdef R_BUILD
    using boost::shared_ptr;
#else
    using std::shared_ptr;    
#endif

// Output types

typedef std::pair<std::string,double> ExtraInformation;
typedef std::vector<ExtraInformation> ExtraInformationVector;

struct ProfileInformation {
	bool defined;
	double lower95Bound;
	double upper95Bound;

	ProfileInformation() : defined(false), lower95Bound(0.0), upper95Bound(0.0) { }
	ProfileInformation(double lower, double upper) : defined(true), lower95Bound(lower),
			upper95Bound(upper) { }
};

typedef std::map<int, ProfileInformation> ProfileInformationMap;
typedef std::vector<ProfileInformation> ProfileInformationList;


#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif 

enum PriorType {
	NONE = 0,
	LAPLACE,
	NORMAL
};

enum ConvergenceType {
	GRADIENT,
	LANGE,
	MITTAL,
	ZHANG_OLES
};

enum NoiseLevels {
	SILENT = 0,
	QUIET,
	NOISY
};

enum UpdateReturnFlags {
	SUCCESS = 0,
	FAIL,
	MAX_ITERATIONS,
	ILLCONDITIONED,
	MISSING_COVARIATES
};

#ifdef USE_DRUG_STRING
	typedef string DrugIdType; // TODO Strings do not get sorted in numerical order
#else
	typedef int DrugIdType;
#endif

typedef std::vector<DrugIdType> ProfileVector;

namespace Models {

enum ModelType {
	NONE = 0,
	NORMAL,
	POISSON,
	LOGISTIC,
	CONDITIONAL_LOGISTIC,
	SELF_CONTROLLED_MODEL,
	COX
};

static bool requiresStratumID(const ModelType modelType) {
	return (modelType == CONDITIONAL_LOGISTIC || modelType == SELF_CONTROLLED_MODEL);
}

static bool requiresCensoredData(const ModelType modelType) {
	return (modelType == COX);
}

static bool requiresOffset(const ModelType modelType) {
	return (modelType == SELF_CONTROLLED_MODEL);
}

} // namespace Models

// Hierarchical prior types

typedef std::map<int, int> HierarchicalParentMap;
typedef std::map<int, std::vector<int> > HierarchicalChildMap;

} // namespace bsccs

#endif /* CCD_TYPES_H_ */
