/*
 * AbstractModelSpecifics.cpp
 *
 *  Created on: Jul 13, 2012
 *      Author: msuchard
 */

#include <stdexcept>
#include <set>

#include "AbstractModelSpecifics.h"
#include "ModelData.h"
#include "engine/ModelSpecifics.h"

#ifdef HAVE_OPENCL
#include "engine/GpuModelSpecifics.hpp"
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
#include "engine/GpuModelSpecificsCox.hpp"
#endif //HAVE_CUDA

namespace bsccs {

template <class Model, typename RealType>
AbstractModelSpecifics* deviceFactory(
        const ModelData<RealType>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {
    AbstractModelSpecifics* model = nullptr;

    switch (deviceType) {
    case DeviceType::CPU :
        model = new ModelSpecifics<Model,RealType>(modelData);
        break;
#ifdef HAVE_CUDA
    case DeviceType::GPU :
        model = new GpuModelSpecificsCox<Model,RealType>(modelData, deviceName);
        break;
#endif //HAVE_CUDA
    default:
        break; // nullptr
    }
    return model;
}

template <class Model, typename RealType, class ModelG>
AbstractModelSpecifics* deviceFactory(
        const ModelData<RealType>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {
    AbstractModelSpecifics* model = nullptr;

    switch (deviceType) {
#ifdef HAVE_OPENCL
    case DeviceType::GPU :
        model = new GpuModelSpecifics<Model,RealType,ModelG>(modelData, deviceName);
        break;
#endif // HAVE_OPENCL
    default:
        break; // nullptr
    }
    return model;
}

template <typename RealType>
AbstractModelSpecifics* precisionFactory(
        const ModelType modelType,
        const ModelData<RealType>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName);

template <>
AbstractModelSpecifics* precisionFactory<float>(
        const ModelType modelType,
        const ModelData<float>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {

    AbstractModelSpecifics* model = nullptr;

    switch (modelType) {
    case ModelType::SELF_CONTROLLED_MODEL :
    	model =  deviceFactory<SelfControlledCaseSeries<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::CONDITIONAL_LOGISTIC :
    	model =  deviceFactory<ConditionalLogisticRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::TIED_CONDITIONAL_LOGISTIC :
    	model =  deviceFactory<TiedConditionalLogisticRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::EFRON_CONDITIONAL_LOGISTIC :
    	model =  deviceFactory<EfronConditionalLogisticRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::LOGISTIC :
    	model = deviceFactory<LogisticRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::NORMAL :
    	model = deviceFactory<LeastSquares<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::POISSON :
    	model = deviceFactory<PoissonRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::CONDITIONAL_POISSON :
    	model = deviceFactory<ConditionalPoissonRegression<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::COX_RAW :
    	model = deviceFactory<CoxProportionalHazards<float>,float>(modelData, deviceType, deviceName);
    	break;
    case ModelType::COX :
        model = deviceFactory<BreslowTiedCoxProportionalHazards<float>,float>(modelData, deviceType, deviceName);
        break;
    case ModelType::TIME_VARYING_COX :
	model = deviceFactory<TimeVaryingCoxProportionalHazards<float>,float>(modelData, deviceType, deviceName);
	break;
    case ModelType::FINE_GRAY:
        model = deviceFactory<BreslowTiedFineGray<float>,float>(modelData, deviceType, deviceName);
        break;
    default:
    	break;
    }

#ifdef HAVE_OPENCL
    if (deviceType == DeviceType::GPU) {
    	 switch (modelType) {
    	 case ModelType::SELF_CONTROLLED_MODEL :
//    	     	model =  deviceFactory<SelfControlledCaseSeries<float>,float,SelfControlledCaseSeriesG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<ConditionalLogisticRegression<float>,float,ConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::TIED_CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<TiedConditionalLogisticRegression<float>,float,TiedConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::EFRON_CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<EfronConditionalLogisticRegression<float>,float,EfronConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::LOGISTIC :
    	     	model = deviceFactory<LogisticRegression<float>,float,LogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::NORMAL :
//    	     	model = deviceFactory<LeastSquares<float>,float,LeastSquaresG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::POISSON :
//    	     	model = deviceFactory<PoissonRegression<float>,float,PoissonRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::CONDITIONAL_POISSON :
//    	     	model = deviceFactory<ConditionalPoissonRegression<float>,float,ConditionalPoissonRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::COX_RAW :
//    	     	model = deviceFactory<CoxProportionalHazards<float>,float,CoxProportionalHazardsG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::COX :
//    	     	model = deviceFactory<BreslowTiedCoxProportionalHazards<float>,float,BreslowTiedCoxProportionalHazardsG>(modelData, deviceType, deviceName);
    	     	break;
    	    default:
    	        break;
    	    }
    }
#endif // HAVE_OPENCL
    return model;
}

template <>
AbstractModelSpecifics* precisionFactory<double>(
        const ModelType modelType,
        const ModelData<double>& modelData,
        const DeviceType deviceType,
        const std::string& deviceName) {

    AbstractModelSpecifics* model = nullptr;

    switch (modelType) {
    case ModelType::SELF_CONTROLLED_MODEL :
        model =  deviceFactory<SelfControlledCaseSeries<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_LOGISTIC :
        model =  deviceFactory<ConditionalLogisticRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::TIED_CONDITIONAL_LOGISTIC :
        model =  deviceFactory<TiedConditionalLogisticRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::EFRON_CONDITIONAL_LOGISTIC :
        model =  deviceFactory<EfronConditionalLogisticRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::LOGISTIC :
        model = deviceFactory<LogisticRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::NORMAL :
        model = deviceFactory<LeastSquares<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::POISSON :
        model = deviceFactory<PoissonRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_POISSON :
        model = deviceFactory<ConditionalPoissonRegression<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX_RAW :
        model = deviceFactory<CoxProportionalHazards<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX :
        model = deviceFactory<BreslowTiedCoxProportionalHazards<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::TIME_VARYING_COX :
        model = deviceFactory<TimeVaryingCoxProportionalHazards<double>,double>(modelData, deviceType, deviceName);
        break;
    case ModelType::FINE_GRAY:
        model = deviceFactory<BreslowTiedFineGray<double>,double>(modelData, deviceType, deviceName);
        break;
    default:
        break;
    }

#ifdef HAVE_OPENCL
    if (deviceType == DeviceType::GPU) {
    	 switch (modelType) {
    	 case ModelType::SELF_CONTROLLED_MODEL :
//    	     	model =  deviceFactory<SelfControlledCaseSeries<double>,double,SelfControlledCaseSeriesG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<ConditionalLogisticRegression<double>,double,ConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::TIED_CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<TiedConditionalLogisticRegression<double>,double,TiedConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::EFRON_CONDITIONAL_LOGISTIC :
//    	     	model =  deviceFactory<EfronConditionalLogisticRegression<double>,double,EfronConditionalLogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::LOGISTIC :
    	     	model = deviceFactory<LogisticRegression<double>,double,LogisticRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::NORMAL :
//    	     	model = deviceFactory<LeastSquares<double>,double,LeastSquaresG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::POISSON :
//    	     	model = deviceFactory<PoissonRegression<double>,double,PoissonRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::CONDITIONAL_POISSON :
//    	     	model = deviceFactory<ConditionalPoissonRegression<double>,double,ConditionalPoissonRegressionG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::COX_RAW :
//    	     	model = deviceFactory<CoxProportionalHazards<double>,double,CoxProportionalHazardsG>(modelData, deviceType, deviceName);
    	     	break;
    	     case ModelType::COX :
//    	     	model = deviceFactory<BreslowTiedCoxProportionalHazards<double>,double,BreslowTiedCoxProportionalHazardsG>(modelData, deviceType, deviceName);
    	     	break;
    	    default:
    	        break;
    	    }
    }
#endif // HAVE_OPENCL
    return model;
}

AbstractModelSpecifics* AbstractModelSpecifics::factory(const ModelType modelType,
                                                        const AbstractModelData& abstractModelData,
                                                        const DeviceType deviceType,
                                                        const std::string& deviceName) {

    AbstractModelSpecifics* model = nullptr;

    if (modelType != ModelType::LOGISTIC && modelType != ModelType::COX && modelType != ModelType::TIME_VARYING_COX && modelType != ModelType::FINE_GRAY && deviceType == DeviceType::GPU) {
        return model; // Implementing lr and cox first on GPU.
    }

    switch(abstractModelData.getPrecisionType()) {
    case PrecisionType::FP64 :
        model = precisionFactory<double>(modelType,
                                         static_cast<const ModelData<double>&>(abstractModelData),
                                         deviceType, deviceName);
        break;
    case PrecisionType::FP32 :
        model = precisionFactory<float>(modelType,
                                        static_cast<const ModelData<float>&>(abstractModelData),
                                        deviceType, deviceName);
        break;
    default :
        break;
    }

    return model;
}

AbstractModelSpecifics::AbstractModelSpecifics(const AbstractModelData& input)
	: hPidOriginal(input.getPidVectorRef()), hPid(const_cast<int*>(hPidOriginal.data())),
      hPidSize(hPidOriginal.size()),
      boundType(MmBoundType::METHOD_2) {

	// Do nothing
}

AbstractModelSpecifics::~AbstractModelSpecifics() { }

void AbstractModelSpecifics::makeDirty(void) {
	hessianCrossTerms.erase(hessianCrossTerms.begin(), hessianCrossTerms.end());

//	for (HessianSparseMap::iterator it = hessianSparseCrossTerms.begin();
//			it != hessianSparseCrossTerms.end(); ++it) {
//		delete it->second;
//	}
}

int AbstractModelSpecifics::getAlignedLength(int N) {
	return (N / 16) * 16 + (N % 16 == 0 ? 0 : 16);
}

} // namespace
