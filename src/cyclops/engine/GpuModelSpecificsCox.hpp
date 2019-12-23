//
// Created by Jianxiao Yang on 2019-12-19.
//

#ifndef GPUMODELSPECIFICSCOX_HPP
#define GPUMODELSPECIFICSCOX_HPP


// #define USE_VECTOR
#undef USE_VECTOR

// #define GPU_DEBUG
#undef GPU_DEBUG
//#define USE_LOG_SUM
#define TIME_DEBUG

//#include <Rcpp.h>

#include "BaseGpuModelSpecifics.hpp"
#include "Iterators.h"

namespace bsccs{

    namespace compute = boost::compute;

    template <class BaseModel, typename RealType>
    class GpuModelSpecificsCox :
            public BaseGpuModelSpecifics<BaseModel, RealType> {
    public:

        using BaseGpuModelSpecifics<BaseModel, RealType>::device;

        GpuModelSpecificsCox(const ModelData<RealType>& input,
                             const std::string& deviceName)
        : BaseGpuModelSpecifics<BaseModel, RealType>(input, deviceName){

            std::cerr << "ctor GpuModelSpecificsCox" << std::endl;

            // Get device ready to compute
            std::cerr << "Using: " << device.name() << std::endl;
        }

        virtual ~GpuModelSpecificsCox() {
            std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
        }

        bool isGPU() {return true;};

    };
} // namespace bsccs

#endif //GPUMODELSPECIFICSCOX_HPP
