/*
 * GpuModelSpecifics.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef GPUMODELSPECIFICS_HPP_
#define GPUMODELSPECIFICS_HPP_


// #define GPU_DEBUG
#undef GPU_DEBUG

#include <Rcpp.h>

#include "ModelSpecifics.h"

#include <boost/compute/algorithm/reduce.hpp>

namespace bsccs {

namespace compute = boost::compute;

namespace detail {

namespace constant {
    static const int updateXBetaBlockSize = 16;
}; // namespace constant

template <typename DeviceVec, typename HostVec>
DeviceVec allocateAndCopyToDevice(const HostVec& hostVec, const compute::context& context, compute::command_queue& queue) {
    DeviceVec deviceVec(hostVec.size(), context);
    compute::copy(std::begin(hostVec), std::end(hostVec), std::begin(deviceVec), queue);
    return std::move(deviceVec);
}

template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDevice(const HostVec& hostVec, DeviceVec& deviceVec, compute::command_queue& queue) {
    deviceVec.resize(hostVec.size());
    compute::copy(std::begin(hostVec), std::end(hostVec), std::begin(deviceVec), queue);
}

template <typename HostVec, typename DeviceVec>
void compare(const HostVec& host, const DeviceVec& device, const std::string& error, double tolerance = 1E-10) {
    bool valid = true;

    for (size_t i = 0; i < host.size(); ++i) {
        auto h = host[i];
        auto d = device[i];
        if (std::abs(h - d) > tolerance) {
            std::cerr << "@ " << i << " : " << h << " - " << d << " = " << (h - d) << std::endl;
            valid = false;
        }
    }
    if (!valid) {
        //forward_exception_to_r(error);
        Rcpp::stop(error);
        // throw new std::logic_error(error);
    }
}

}; // namespace detail

struct SourceCode {
    std::string body;
    std::string name;

    SourceCode(std::string body, std::string name) : body(body), name(name) { }
};

template <typename RealType>
class GpuColumn {
public:
    typedef compute::vector<RealType> DataVector;
    typedef compute::vector<int> IndicesVector;
    typedef compute::uint_ UInt;

    //GpuColumn(const GpuColumn<RealType>& copy);

    GpuColumn(const CompressedDataColumn& column,
              const compute::context& context,
              compute::command_queue& queue,
              size_t denseLength)
        : format(column.getFormatType()), indices(context), data(context) {

            // Data vector
            if (format == FormatType::SPARSE ||
                format == FormatType::DENSE) {
                const auto& columnData = column.getDataVector();
                detail::resizeAndCopyToDevice(columnData, data, queue);
            }

            // Indices vector
            if (format == FormatType::INDICATOR ||
                format == FormatType::SPARSE) {
                const auto& columnIndices = column.getColumnsVector();
                detail::resizeAndCopyToDevice(columnIndices, indices, queue);
            }

            // Task count
            if (format == FormatType::DENSE ||
                format == FormatType::INTERCEPT) {
                tasks = static_cast<UInt>(denseLength);
            } else { // INDICATOR, SPARSE
                tasks = static_cast<UInt>(column.getNumberOfEntries());
            }
        }

    virtual ~GpuColumn() { }

    const IndicesVector& getIndicesVector() const { return indices; }
    const DataVector& getDataVector() const { return data; }
    UInt getTaskCount() const { return tasks; }

private:
    FormatType format;
    IndicesVector indices;
    DataVector data;
    UInt tasks;
};


template <class BaseModel, typename WeightType>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, WeightType> {
public:

    using ModelSpecifics<BaseModel, WeightType>::modelData;
    using ModelSpecifics<BaseModel, WeightType>::offsExpXBeta;
    using ModelSpecifics<BaseModel, WeightType>::hXBeta;
    using ModelSpecifics<BaseModel, WeightType>::hY;
    using ModelSpecifics<BaseModel, WeightType>::hKWeight;
    using ModelSpecifics<BaseModel, WeightType>::hPid;
    using ModelSpecifics<BaseModel, WeightType>::hPidInternal;
    using ModelSpecifics<BaseModel, WeightType>::hOffs;
    using ModelSpecifics<BaseModel, WeightType>::denomPid;
    using ModelSpecifics<BaseModel, WeightType>::hXjY;
    using ModelSpecifics<BaseModel, WeightType>::hXjX;
    using ModelSpecifics<BaseModel, WeightType>::K;
    using ModelSpecifics<BaseModel, WeightType>::J;
    using ModelSpecifics<BaseModel, WeightType>::N;
    using ModelSpecifics<BaseModel, WeightType>::duration;

    const static int tpb = 32; // threads-per-block
    const static int wgs = 4;  // work-group-size

    const static int globalWorkSize = tpb * wgs;

    GpuModelSpecifics(const ModelData& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,WeightType>(input),
      device(compute::system::find_device(deviceName)),
      ctx(device),
      queue(ctx, device
          , compute::command_queue::enable_profiling
      ),
      dY(ctx), dXBeta(ctx), dExpXBeta(ctx), dDenominator(ctx), dBuffer(ctx), dKWeight(ctx),
      dId(ctx) {

        std::cerr << "ctor GpuModelSpecifics" << std::endl;

        // Get device ready to compute
        std::cerr << "Using: " << device.name() << std::endl;
    }

    virtual ~GpuModelSpecifics() {
        std::cerr << "dtor GpuModelSpecifics" << std::endl;
    }

    virtual void deviceInitialization() {

        int need = 0;

        // Copy data
        for (size_t j = 0; j < J /*modelData.getNumberOfColumns()*/; ++j) {
            const auto& column = modelData.getColumn(j);
            columns.emplace_back(GpuColumn<real>(column, ctx, queue, K));
            need |= (1 << column.getFormatType());
        }
        std::vector<FormatType> neededFormatTypes;
        for (int t = 0; t < 4; ++t) {
            if (need & (1 << t)) {
                neededFormatTypes.push_back(static_cast<FormatType>(t));
            }
        }

        auto& inputY = modelData.getYVectorRef();
        detail::resizeAndCopyToDevice(inputY, dY, queue);

        // Internal buffers
        detail::resizeAndCopyToDevice(hXBeta, dXBeta, queue);
        detail::resizeAndCopyToDevice(offsExpXBeta, dExpXBeta, queue);
        detail::resizeAndCopyToDevice(denomPid, dDenominator, queue);
        detail::resizeAndCopyToDevice(hPidInternal, dId, queue);

        std::cerr << "Format types required: " << need << std::endl;

        buildAllKernels(neededFormatTypes);

        printAllKernels(std::cerr);
    }

    virtual void computeRemainingStatistics(bool useWeights) {

        // Currently RS only computed on CPU and then copied
        ModelSpecifics<BaseModel, WeightType>::computeRemainingStatistics(useWeights);

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        if (BaseModel::likelihoodHasDenominator) {
            compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue);
            compute::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator), queue);
        }

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

    }

    virtual void computeGradientAndHessian(int index, double *ogradient,
                                           double *ohessian, bool useWeights) {

#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::computeGradientAndHessian(index, ogradient, ohessian, useWeights);
        std::cerr << *ogradient << " & " << *ohessian << std::endl;
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        FormatType formatType = modelData.getFormatType(index);
        auto& kernel = (useWeights) ? // Double-dispatch
                            kernelGradientHessianWeighted[formatType] :
                            kernelGradientHessianNoWeight[formatType];

        auto& column = columns[index];
        const auto taskCount = column.getTaskCount();

        size_t loops = taskCount / globalWorkSize;
        if (taskCount % globalWorkSize != 0) {
            ++loops;
        }

        // std::cerr << dBuffer.get_buffer() << std::endl;

        if (dBuffer.size() < 2 * wgs) {
            dBuffer.resize(2 * wgs, queue);
            //compute::fill(std::begin(dBuffer), std::end(dBuffer), 0.0, queue); // TODO Not needed
            kernel.set_arg(8, dBuffer); // Can get reallocated.
            hBuffer.resize(2 * wgs);
        }

        // std::cerr << dBuffer.get_buffer() << std::endl << std::endl;

        if (dKWeight.size() == 0) {
            kernel.set_arg(10, 0);
        } else {
            kernel.set_arg(10, dKWeight); // TODO Only when dKWeight gets reallocated
        }

        kernel.set_arg(0, column.getDataVector());
        kernel.set_arg(1, column.getIndicesVector());
        kernel.set_arg(2, taskCount);
        kernel.set_arg(3, 0.0); // TODO remove

//         std::cerr << "loop= " << loops << std::endl;
//         std::cerr << "n   = " << taskCount << std::endl;
//         std::cerr << "gWS = " << globalWorkSize << std::endl;
//         std::cerr << "tpb = " << tpb << std::endl;
//
        // std::cerr << kernel.get_program().source() << std::endl;


//         compute::vector<real> tmpR(taskCount, ctx);
//         compute::vector<int> tmpI(taskCount, ctx);

        // kernel.set_arg(0, tmpR);
        // kernel.set_arg(1, tmpI);

//         kernel.set_arg(4, tmpR);
//         kernel.set_arg(5, tmpR);
//         kernel.set_arg(6, tmpR);
//         kernel.set_arg(7, tmpR);
        // kernel.set_arg(8, tmpR);
        kernel.set_arg(8, dBuffer); // TODO Why is this necessary?
//         kernel.set_arg(9, tmpI);
//         kernel.set_arg(10, tmpR);

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, tpb);
        queue.finish();

//         for (int i = 0; i < wgs; ++i) {
//             std::cerr << ", " << dBuffer[i];
//         }
//         std::cerr << std::endl;

        // Get result
        compute::copy(std::begin(dBuffer), std::end(dBuffer), std::begin(hBuffer), queue);
        double gradient = 0.0;
        double hessian = 0.0;

        for (int i = 0; i < wgs; ++i) { // TODO Use SSE
            gradient += hBuffer[i];
            hessian  += hBuffer[i + wgs];
        }

        if (BaseModel::precomputeGradient) { // Compile-time switch
            gradient -= hXjY[index];
        }

        if (BaseModel::precomputeHessian) { // Compile-time switch
            hessian += static_cast<real>(2.0) * hXjX[index];
        }

#ifdef GPU_DEBUG
        std::cerr << gradient << " & " << hessian << std::endl << std::endl;
#endif // GPU_DEBUG

        *ogradient = gradient;
        *ohessian = hessian;

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "compGradHessG" + getFormatTypeExtension(formatType) + " ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

    }

    virtual void updateXBeta(real realDelta, int index, bool useWeights) {

#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::updateXBeta(realDelta, index, useWeights);
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        auto& kernel = kernelUpdateXBeta[modelData.getFormatType(index)];
        auto& column = columns[index];
        const auto taskCount = column.getTaskCount();

        kernel.set_arg(0, column.getDataVector());
        kernel.set_arg(1, column.getIndicesVector());
        kernel.set_arg(2, taskCount);
        kernel.set_arg(3, realDelta);

        size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
        if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
            ++workGroups;
        }
        const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
        queue.finish();

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "updateXBetaG" + getFormatTypeExtension(modelData.getFormatType(index)) + "  ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

#ifdef GPU_DEBUG
        // Compare results:
        detail::compare(hXBeta, dXBeta, "xBeta not equal");
        detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
        detail::compare(denomPid, dDenominator, "denominator not equal");
#endif // GPU_DEBUG
    }

    virtual void setWeights(real* inWeights, bool useCrossValidation) {
        // Currently only computed on CPU and then copied to GPU
        ModelSpecifics<BaseModel, WeightType>::setWeights(inWeights, useCrossValidation);

        detail::resizeAndCopyToDevice(hKWeight, dKWeight, queue);
    }

private:

    void buildAllUpdateXBetaKernels(const std::vector<FormatType>& neededFormatTypes) {
        for (FormatType formatType : neededFormatTypes) {
            buildUpdateXBetaKernel(formatType);
        }
    }

    void buildAllGradientHessianKernels(const std::vector<FormatType>& neededFormatTypes) {
        int b = 0;
        for (FormatType formatType : neededFormatTypes) {
            buildGradientHessianKernel(formatType, true); ++b;
            buildGradientHessianKernel(formatType, false); ++b;
        }
    }

    std::string getFormatTypeExtension(FormatType formatType) {
        switch (formatType) {
        case DENSE:
            return "Den";
        case SPARSE:
            return "Spa";
        case INDICATOR:
            return "Ind";
        case INTERCEPT:
            return "Icp";
        default: return "";
        }
    }

    SourceCode writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights);

    SourceCode writeCodeForUpdateXBetaKernel(FormatType formatType);

    void buildGradientHessianKernel(FormatType formatType, bool useWeights) {

        std::stringstream options;
        options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;

        auto source = writeCodeForGradientHessianKernel(formatType, useWeights);

        auto program = compute::program::build_with_source(source.body, ctx, options.str());
        auto kernel = compute::kernel(program, source.name);

        // Run-time constant arguments.
        kernel.set_arg(4, dY);
        kernel.set_arg(5, dXBeta);
        kernel.set_arg(6, dExpXBeta);
        kernel.set_arg(7, dDenominator);
        kernel.set_arg(8, dBuffer);  // TODO Does not seem to stick
        kernel.set_arg(9, dId);
        kernel.set_arg(10, dKWeight); // TODO Does not seem to stick

        if (useWeights) {
            kernelGradientHessianWeighted[formatType] = std::move(kernel);
        } else {
            kernelGradientHessianNoWeight[formatType] = std::move(kernel);
        }
    }

    void buildUpdateXBetaKernel(FormatType formatType) {

        std::stringstream options;
        options << "-DREAL=double";

        auto source = writeCodeForUpdateXBetaKernel(formatType);
        auto program = compute::program::build_with_source(source.body, ctx, options.str());
        auto kernel = compute::kernel(program, source.name);

        // Run-time constant arguments.
        kernel.set_arg(4, dY);
        kernel.set_arg(5, dXBeta);
        kernel.set_arg(6, dExpXBeta);
        kernel.set_arg(7, dDenominator);
        kernel.set_arg(8, dId);

        kernelUpdateXBeta[formatType] = std::move(kernel);
    }

    void printKernel(compute::kernel& kernel, std::ostream& stream) {

        auto program = kernel.get_program();
        auto buildOptions = program.get_build_info<std::string>(CL_PROGRAM_BUILD_OPTIONS, device);

        stream // TODO Change to R
            << "Options: " << buildOptions << std::endl
            << program.source()
            << std::endl;
    }

    void buildAllKernels(const std::vector<FormatType>& neededFormatTypes) {
        buildAllGradientHessianKernels(neededFormatTypes);
        buildAllUpdateXBetaKernels(neededFormatTypes);
    }

    void printAllKernels(std::ostream& stream) {
        for (auto& entry : kernelGradientHessianWeighted) {
            printKernel(entry.second, stream);
        }

        for (auto& entry : kernelGradientHessianNoWeight) {
            printKernel(entry.second, stream);
        }

        for (auto& entry : kernelUpdateXBeta) {
            printKernel(entry.second, stream);
        }
    }

    // boost::compute objects
    const compute::device device;
    const compute::context ctx;
    compute::command_queue queue;
    compute::program program;

    std::map<FormatType, compute::kernel> kernelGradientHessianWeighted;
    std::map<FormatType, compute::kernel> kernelGradientHessianNoWeight;
    std::map<FormatType, compute::kernel> kernelUpdateXBeta;

    // vectors of columns
    std::vector<GpuColumn<real> > columns;

    std::vector<real> hBuffer;

    // Internal storage
    compute::vector<real> dY;
    compute::vector<real> dXBeta;
    compute::vector<real> dExpXBeta;
    compute::vector<real> dDenominator;
    compute::vector<real> dBuffer;
    compute::vector<real> dKWeight;
    compute::vector<int> dId;
};

} // namespace bsccs


#include "Kernels.hpp"

#endif /* GPUMODELSPECIFICS_HPP_ */
