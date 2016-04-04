/*
 * GpuModelSpecifics.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef GPUMODELSPECIFICS_HPP_
#define GPUMODELSPECIFICS_HPP_

#include "ModelSpecifics.h"

namespace bsccs {

template <class BaseModel, typename WeightType>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, WeightType> {
public:
    GpuModelSpecifics(const ModelData& input) : ModelSpecifics<BaseModel,WeightType>(input) { }

    virtual ~GpuModelSpecifics() { }
};

} // namespace bsccs

#endif /* GPUMODELSPECIFICS_HPP_ */
//
