/*
 * GpuModelSpecifics.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef GPUMODELSPECIFICS_HPP_
#define GPUMODELSPECIFICS_HPP_

#include "ModelSpecifics.h"

#include <boost/compute/algorithm/reduce.hpp>

namespace bsccs {

template <class BaseModel, typename WeightType>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, WeightType> {
public:
    GpuModelSpecifics(const ModelData& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,WeightType>(input) {

        std::cerr << "ctor GpuModelSpecifics" << std::endl;

        // Get device ready to compute
        device = boost::compute::system::find_device(deviceName);
        // device = boost::compute::system::default_device(); // TODO Generalize
        std::cerr << "Using: " << device.name() << std::endl;

        ctx = boost::compute::context{device};
        queue = boost::compute::command_queue{ctx, device
            , boost::compute::command_queue::enable_profiling
        };
    }

    virtual ~GpuModelSpecifics() {
        std::cerr << "dtor GpuModelSpecifics" << std::endl;
    }

private:
    // boost::compute objects
    boost::compute::device device;
    boost::compute::context ctx;
    boost::compute::command_queue queue;
    boost::compute::program program;
};

} // namespace bsccs

#endif /* GPUMODELSPECIFICS_HPP_ */
//
