/*
 * Types.h
 *
 *  Created on: Aug 20, 2013
 *      Author: msuchard
 */

#ifndef CCD_TYPES_H_
#define CCD_TYPES_H_

#include <cstdint>
#include <vector>
#include <map>

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
// C++11
    #include <memory>
    namespace bsccs {
        using std::shared_ptr;
        using std::make_shared;
        using std::unique_ptr;
    }
#else
// C++98 (R build)
    #include "boost/smart_ptr.hpp"
    namespace bsccs {
        using boost::shared_ptr;
        using boost::make_shared;
        using boost::unique_ptr;
    }
#endif

#ifdef WIN_BUILD
    #include <tr1/unordered_map>
    namespace bsccs {
        using std::tr1::unordered_map;
    }
#else
    #include <unordered_map>
    namespace bsccs {
        using std::unordered_map;
    }
#endif

// #ifdef R_BUILD  // old alternative -DR_BUILD
// #endif

namespace bsccs {

template<typename T, typename ...Args>
bsccs::unique_ptr<T> make_unique( Args&& ...args ) {
    return bsccs::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
}

// Internal types

#ifdef DOUBLE_PRECISION
	typedef double real;
#else
	typedef float real;
#endif

typedef std::vector<int> IntVector;
typedef std::vector<real> RealVector;
typedef bsccs::shared_ptr<IntVector> IntVectorPtr;
typedef bsccs::shared_ptr<RealVector> RealVectorPtr;
typedef int64_t IdType;

// Output types

typedef std::pair<std::string,double> ExtraInformation;
typedef std::vector<ExtraInformation> ExtraInformationVector;

struct ProfileInformation {
	bool defined;
	double lower95Bound;
	double upper95Bound;
	int evaluations;

	ProfileInformation() : defined(false), lower95Bound(0.0), upper95Bound(0.0), evaluations(0) { }
	ProfileInformation(double lower, double upper) : defined(true), lower95Bound(lower),
			upper95Bound(upper), evaluations(0) { }
	ProfileInformation(double lower, double upper, int evals) : defined(true), lower95Bound(lower),
			upper95Bound(upper), evaluations(evals) { }
};

typedef std::map<IdType, ProfileInformation> ProfileInformationMap;
typedef std::vector<ProfileInformation> ProfileInformationList;

namespace priors {

enum PriorType {
	NONE = 0,
	LAPLACE,
	NORMAL
};

} // namespace priors

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

typedef std::vector<IdType> ProfileVector;

enum class ModelType {
	NONE = 0,
	NORMAL,
	POISSON,
	LOGISTIC,
	CONDITIONAL_LOGISTIC,
	TIED_CONDITIONAL_LOGISTIC,
	CONDITIONAL_POISSON,
	SELF_CONTROLLED_MODEL,
	COX,
	COX_RAW,
	SIZE_OF_ENUM // Keep at end
};

enum class SelectorType {
	DEFAULT,
	BY_PID,
	BY_ROW,
	SIZE_OF_ENUM // Keep at end
};

enum class NormalizationType {
    STANDARD_DEVIATION,
    MAX,
    MEDIAN,
    Q95,
    SIZE_OF_ENUM // Keep at end
};

namespace Models {

inline bool removeIntercept(const ModelType modelType) {
	return (modelType == ModelType::CONDITIONAL_LOGISTIC ||
			modelType == ModelType::TIED_CONDITIONAL_LOGISTIC ||
			modelType == ModelType::CONDITIONAL_POISSON ||
			modelType == ModelType::SELF_CONTROLLED_MODEL);
}

inline bool requiresStratumID(const ModelType modelType) {
	return (modelType == ModelType::CONDITIONAL_LOGISTIC ||
			modelType == ModelType::TIED_CONDITIONAL_LOGISTIC ||
            modelType == ModelType::CONDITIONAL_POISSON ||
            modelType == ModelType::SELF_CONTROLLED_MODEL);
}

inline bool requiresCensoredData(const ModelType modelType) {
	return (modelType == ModelType::COX ||
			modelType == ModelType::COX_RAW);
}

inline bool requiresOffset(const ModelType modelType) {
	return (modelType == ModelType::SELF_CONTROLLED_MODEL);
}

//#define UNUSED(x) ((void)(x))
//UNUSED(requiresStratumID);

} // namespace Models

// Hierarchical prior types

// typedef std::map<int, int> HierarchicalParentMap;
// typedef std::map<int, std::vector<int> > HierarchicalChildMap;
typedef std::vector<int> HierarchicalParentMap;
typedef std::vector<std::vector<int> > HierarchicalChildMap;
typedef std::map<IdType, ProfileVector> NeighborhoodMap;

} // namespace bsccs

#endif /* CCD_TYPES_H_ */
