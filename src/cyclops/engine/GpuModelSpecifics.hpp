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

namespace compute = boost::compute;

namespace detail {

template <typename DeviceVec, typename HostVec>
DeviceVec allocateAndCopyToDevice(const HostVec& hostVec, compute::context& context, compute::command_queue& queue) {
    DeviceVec deviceVec(hostVec.size(), context);
    compute::copy(std::begin(hostVec), std::end(hostVec), std::begin(deviceVec), queue);
    return std::move(deviceVec);
}

}; // namespace detail

template <typename RealType>
class GpuColumn {
public:
    typedef compute::vector<RealType> DataVector;
    typedef compute::vector<int> IndicesVector;

    //GpuColumn(const GpuColumn<RealType>& copy);

    GpuColumn(const CompressedDataColumn& column,
              compute::context& context,
              compute::command_queue& queue)
        : format(column.getFormatType()) {

            // Data vector
            if (format == FormatType::SPARSE ||
                format == FormatType::DENSE) {
                const auto& columnData = column.getDataVector();
                data = detail::allocateAndCopyToDevice<DataVector>(columnData, context, queue);
            }

            // Indices vector
            if (format == FormatType::INDICATOR ||
                format == FormatType::SPARSE) {
                const auto& columnIndices = column.getColumnsVector();
                indices = detail::allocateAndCopyToDevice<IndicesVector>(columnIndices, context, queue);
            }
        }

    virtual ~GpuColumn() { }

    const IndicesVector& getIndicesVector() const { return indices; }
    const DataVector& getDataVector() const { return data; }

private:
    FormatType format;
    IndicesVector indices;
    DataVector data;
};


template <class BaseModel, typename WeightType>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, WeightType> {
public:
    GpuModelSpecifics(const ModelData& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,WeightType>(input) {

        std::cerr << "ctor GpuModelSpecifics" << std::endl;

        // Get device ready to compute
        device = boost::compute::system::find_device(deviceName);
        std::cerr << "Using: " << device.name() << std::endl;

        ctx = boost::compute::context{device};
        queue = boost::compute::command_queue{ctx, device
            , boost::compute::command_queue::enable_profiling
        };

        // Copy data
        for (size_t i = 0; i < input.getNumberOfColumns(); ++i) {
            columns.emplace_back(GpuColumn<real>(input.getColumn(i), ctx, queue));
        }
        auto& inputY = input.getYVectorRef();
        y = detail::allocateAndCopyToDevice<compute::vector<real> >(inputY, ctx, queue);

        buildAllKernels();

        printAllKernels();
    }

    virtual ~GpuModelSpecifics() {
        std::cerr << "dtor GpuModelSpecifics" << std::endl;
    }

private:

    void buildGradientHessianKernel() {

        std::stringstream options;
        options << "-DTILE_DIM=" << 16 << " -DREAL_VECTOR=float -DREAL=float";

        std::stringstream code;
        // code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code <<
            " __kernel void computeSSR(__global const REAL_VECTOR *locations,  \n" <<
            "  						   __global const REAL *observations,      \n" <<
            "						   __global REAL *squaredResiduals,        \n" <<
            "						   const uint locationCount) {             \n" <<
            " }                                                                \n";

        compute::program program = compute::program::build_with_source(code.str(), ctx, options.str());
        kernelGradientHessian = compute::kernel(program, "computeSSR");

    }

    void printKernel(compute::kernel& kernel) {
        auto program = kernel.get_program();
        auto buildOptions = program.get_build_info<std::string>(CL_PROGRAM_BUILD_OPTIONS, device);

        std::cerr // TODO Change to R
            << "Options: " << buildOptions << std::endl
            << program.source()
            << std::endl;
    }

    void buildAllKernels() {
        buildGradientHessianKernel();
    }

    void printAllKernels() {
        printKernel(kernelGradientHessian);
    }

    // boost::compute objects
    compute::device device;
    compute::context ctx;
    compute::command_queue queue;
    compute::program program;

    compute::kernel kernelGradientHessian;

    // vectors of columns
    std::vector<GpuColumn<real> > columns;
    compute::vector<real> y;
};

} // namespace bsccs

#endif /* GPUMODELSPECIFICS_HPP_ */
