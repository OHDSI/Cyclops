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
#ifdef HAVE_OPENCL
    case DeviceType::GPU :
        model = new GpuModelSpecifics<Model,RealType>(modelData, deviceName);
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
    case ModelType::LOGISTIC :
        model = deviceFactory<LogisticRegression<float>,float>(modelData, deviceType, deviceName);
        break;
    case ModelType::POISSON :
        model = deviceFactory<PoissonRegression<float>,float>(modelData, deviceType, deviceName);
        break;
    case ModelType::CONDITIONAL_POISSON :
        model = deviceFactory<ConditionalPoissonRegression<float>,float>(modelData, deviceType, deviceName);
        break;
    case ModelType::COX :
        model = deviceFactory<BreslowTiedCoxProportionalHazards<float>,float>(modelData, deviceType, deviceName);
        break;
    default:
        break;
    }

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
    default:
        break;
    }
    return model;
}

AbstractModelSpecifics* AbstractModelSpecifics::factory(const ModelType modelType,
                                                        const AbstractModelData& abstractModelData,
                                                        const DeviceType deviceType,
                                                        const std::string& deviceName) {
    AbstractModelSpecifics* model = nullptr;

    if (modelType != ModelType::LOGISTIC && deviceType == DeviceType::GPU) {
        return model; // Implementing lr first on GPU.
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
