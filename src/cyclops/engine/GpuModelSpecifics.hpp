/*
 * GpuModelSpecifics.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef GPUMODELSPECIFICS_HPP_
#define GPUMODELSPECIFICS_HPP_

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
void compare(const HostVec& host, const DeviceVec& device, const std::string& error) {
    bool valid = true;

    for (size_t i = 0; i < host.size(); ++i) {
        auto h = host[i];
        auto d = device[i];
        if (h != d) {
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
    using ModelSpecifics<BaseModel, WeightType>::hPid;
    using ModelSpecifics<BaseModel, WeightType>::hPidInternal;
    using ModelSpecifics<BaseModel, WeightType>::hOffs;
    using ModelSpecifics<BaseModel, WeightType>::denomPid;
    using ModelSpecifics<BaseModel, WeightType>::K;
    using ModelSpecifics<BaseModel, WeightType>::J;
    using ModelSpecifics<BaseModel, WeightType>::N;
    using ModelSpecifics<BaseModel, WeightType>::duration;

    GpuModelSpecifics(const ModelData& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,WeightType>(input),
      device(compute::system::find_device(deviceName)),
      ctx(device),
      queue(ctx, device
          , compute::command_queue::enable_profiling
      ),
      dY(ctx), dXBeta(ctx), dExpXBeta(ctx), dDenominator(ctx), dId(ctx) {

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

        printAllKernels();
    }

    virtual void computeRemainingStatistics(bool useWeights) {

        ModelSpecifics<BaseModel, WeightType>::computeRemainingStatistics(useWeights);

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue); // TODO is this necessary?
        compute::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator), queue);

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

    }

    virtual void updateXBeta(real realDelta, int index, bool useWeights) {

        // Modifies: xBeta, expXBeta, denominator
//         compute::copy(std::begin(hXBeta), std::end(hXBeta), std::begin(dXBeta), queue);
//         compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue);
//         compute::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator), queue);
        //

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
        auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

        // Run-time dispatch to implementation depending on covariate FormatType
        switch(modelData.getFormatType(index)) {
        case INDICATOR :
            updateXBetaImpl<IndicatorIterator>(realDelta, index, useWeights);
            break;
        case SPARSE :
            updateXBetaImpl<SparseIterator>(realDelta, index, useWeights);
            break;
        case DENSE :
            updateXBetaImpl<DenseIterator>(realDelta, index, useWeights);
            break;
        case INTERCEPT :
            updateXBetaImpl<InterceptIterator>(realDelta, index, useWeights);
            break;
        }

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

        // std::cerr << "executed: " << getFormatTypeExtension(modelData.getFormatType(index)) << std::endl;

        // Compare results:
        detail::compare(hXBeta, dXBeta, "xBeta not equal");
        detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
        detail::compare(denomPid, dDenominator, "denominator not equal");

        // std::cerr << "done compare" << std::endl;

#ifdef CYCLOPS_DEBUG_TIMING
#ifndef CYCLOPS_DEBUG_TIMING_LOW
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        duration["updateXBetaG     "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

    }

    template <class IteratorType>
    inline void updateXBetaImpl(real realDelta, int index, bool useWeights) {

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
        auto start = bsccs::chrono::steady_clock::now();
#endif
#endif

        auto range = helper::getRangeX(modelData, index, typename IteratorType::tag());

        auto kernel = UpdateXBetaKernel<BaseModel,IteratorType,real,int>(
            realDelta, begin(offsExpXBeta), begin(hXBeta),
            begin(hY),
            begin(hPid),
            begin(denomPid),
            begin(hOffs)
        );

        variants::for_each(
            range.begin(), range.end(),
            kernel,
            SerialOnly()
        );

       // computeAccumlatedDenominator(useWeights);

#ifdef CYCLOPS_DEBUG_TIMING
#ifdef CYCLOPS_DEBUG_TIMING_LOW
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "updateXBetaG" + IteratorType::name + "  ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
#endif

    }

private:

    void buildAllUpdateXBetaKernels(const std::vector<FormatType>& neededFormatTypes) {
        for (FormatType formatType : neededFormatTypes) {
            buildUpdateXBetaKernel(formatType);
        }
    }

    void buildAllGradientHessianKernels(const std::vector<FormatType>& neededFormatTypes) {
        for (FormatType formatType : neededFormatTypes) {
            buildGradientHessianKernel(formatType);
        }
    }

    void buildGradientHessianKernel(FormatType formatType) {

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
        kernelGradientHessian[formatType] = compute::kernel(program, "computeSSR");

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

    SourceCode writeCodeForUpdateXBetaKernel(FormatType formatType) {

        std::string name = "updateXBeta" + getFormatTypeExtension(formatType);

        std::stringstream code;
        code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        code << "__kernel void " << name << "(     \n" <<
                "       __global const REAL* X,    \n" <<
                "       __global const int* K,     \n" <<
                "       const uint N,              \n" <<
                "       const REAL delta,          \n" <<
                "       __global const REAL* Y,    \n" <<
                "       __global REAL* xBeta,      \n" <<
                "       __global REAL* expXBeta,   \n" <<
                "       __global REAL* denominator,\n" <<
                "       __global const int* id) {  \n" <<
                "   const uint task = get_global_id(0); \n";

        if (formatType == INDICATOR || formatType == SPARSE) {
            code << "   const uint k = K[task];         \n";
        } else { // DENSE, INTERCEPT
            code << "   const uint k = task;            \n";
        }

        if (formatType == SPARSE || formatType == DENSE) {
            code << "   const REAL inc = delta * X[task]; \n";
        } else { // INDICATOR, INTERCEPT
            code << "   const REAL inc = delta;           \n";
        }

        code << "   if (task < N) {      \n" <<
                "       xBeta[k] += inc; \n";

        if (BaseModel::likelihoodHasDenominator) {
            // TODO: The following is not YET thread-safe for multi-row observations

            //                     real oldEntry = expXBeta[k];
            //                     real newEntry = expXBeta[k] = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
            //                     denominator[BaseModel::getGroup(pid, k)] += (newEntry - oldEntry);
            code << "       const REAL oldEntry = expXBeta[k];                 \n" <<
                    "       const REAL newEntry = expXBeta[k] = exp(xBeta[k]); \n" <<
                    "       denominator[k] += (newEntry - oldEntry);           \n";

            // LOGISTIC MODEL ONLY
            //                     const real t = BaseModel::getOffsExpXBeta(offs, xBeta[k], y[k], k);
            //                     expXBeta[k] = t;
            //                     denominator[k] = static_cast<real>(1.0) + t;
            //             code << "    const REAL t = 0.0;               \n" <<
            //                     "   expXBeta[k] = exp(xBeta[k]);      \n" <<
            //                     "   denominator[k] = REAL(1.0) + tmp; \n";
        }

        code << "   } \n";
        code << "}    \n";

        return SourceCode(code.str(), name);
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

    void printKernel(compute::kernel& kernel) {
        auto program = kernel.get_program();
        auto buildOptions = program.get_build_info<std::string>(CL_PROGRAM_BUILD_OPTIONS, device);

        std::cerr // TODO Change to R
            << "Options: " << buildOptions << std::endl
            << program.source()
            << std::endl;
    }

    void buildAllKernels(const std::vector<FormatType>& neededFormatTypes) {
        // buildAllGradientHessianKernels(neededFormatTypes);
        buildAllUpdateXBetaKernels(neededFormatTypes);
    }

    void printAllKernels() {
        for (auto& entry : kernelGradientHessian) {
            printKernel(entry.second);
        }

        for (auto& entry : kernelUpdateXBeta) {
            printKernel(entry.second);
        }
    }

    // boost::compute objects
    const compute::device device;
    const compute::context ctx;
    compute::command_queue queue;
    compute::program program;

    std::map<FormatType, compute::kernel> kernelGradientHessian;
    std::map<FormatType, compute::kernel> kernelUpdateXBeta;

    // vectors of columns
    std::vector<GpuColumn<real> > columns;

    // Internal storage
    compute::vector<real> dY;
    compute::vector<real> dXBeta;
    compute::vector<real> dExpXBeta;
    compute::vector<real> dDenominator;
    compute::vector<int> dId;
};

} // namespace bsccs

#endif /* GPUMODELSPECIFICS_HPP_ */
