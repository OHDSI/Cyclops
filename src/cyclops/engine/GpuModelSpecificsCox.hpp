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

#ifdef CYCLOPS_DEBUG_TIMING
        using ModelSpecifics<BaseModel, RealType>::duration;
#endif
        using BaseGpuModelSpecifics<BaseModel, RealType>::ctx;
        using BaseGpuModelSpecifics<BaseModel, RealType>::device;
        using BaseGpuModelSpecifics<BaseModel, RealType>::queue;
        using BaseGpuModelSpecifics<BaseModel, RealType>::neededFormatTypes;
        using BaseGpuModelSpecifics<BaseModel, RealType>::double_precision;

        using BaseGpuModelSpecifics<BaseModel, RealType>::modelData;
        using BaseGpuModelSpecifics<BaseModel, RealType>::hX;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::K;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::J;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::N;
        using BaseGpuModelSpecifics<BaseModel, RealType>::offsExpXBeta;
        using BaseGpuModelSpecifics<BaseModel, RealType>::hXBeta;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hY;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hOffs;
        using BaseGpuModelSpecifics<BaseModel, RealType>::denomPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::denomPid2;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::numerPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::numerPid2;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hNWeight;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hKWeight;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::accDenomPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::accNumerPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::accNumerPid2;

        using BaseGpuModelSpecifics<BaseModel, RealType>::dKWeight;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::dNWeight;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dY;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::dOffs;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dId;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::dBeta;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dXBeta;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dExpXBeta;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dDenominator;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::dDenominator2;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::dAccDenominator;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dColumns;

        int tpb = 256; // threads-per-block  // Appears best on K40


        GpuModelSpecificsCox(const ModelData<RealType>& input,
                             const std::string& deviceName)
        : BaseGpuModelSpecifics<BaseModel, RealType>(input, deviceName){

            std::cerr << "ctor GpuModelSpecificsCox" << std::endl;

        }

        virtual ~GpuModelSpecificsCox() {
            std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
        }

        virtual void deviceInitialization() {
            BaseGpuModelSpecifics<BaseModel, RealType>::deviceInitialization();
            hXBetaKnown = true;
            dXBetaKnown = true;
            buildAllKernels(neededFormatTypes);
        }


//        virtual void computeRemainingStatistics(bool useWeights) {
//
//            std::cerr << "GPU::cRS called" << std::endl;
//
//            // Currently RS only computed on CPU and then copied
//            ModelSpecifics<BaseModel, RealType>::computeRemainingStatistics(useWeights);
//
//#ifdef CYCLOPS_DEBUG_TIMING
//            auto start = bsccs::chrono::steady_clock::now();
//#endif
//            /*
//            if (algorithmType == AlgorithmType::MM) {
//                compute::copy(std::begin(hBeta), std::end(hBeta), std::begin(dBeta), queue);
//            }
//            */
//
//            if (BaseModel::likelihoodHasDenominator) {
//                compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue);
//                compute::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator), queue);
//            }
//
//
//#ifdef CYCLOPS_DEBUG_TIMING
//            auto end = bsccs::chrono::steady_clock::now();
//            ///////////////////////////"
//            duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
//#endif
//
//        }

        virtual void updateXBeta(double realDelta, int index, bool useWeights) {

#ifdef GPU_DEBUG
            ModelSpecifics<BaseModel, WeightType>::updateXBeta(realDelta, index, useWeights);
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif

            FormatType formatType = hX.getFormatType(index);
            auto& kernel = kernelUpdateXBeta[formatType];
//         auto& column = columns[index];
//         const auto taskCount = column.getTaskCount();
            const auto taskCount = dColumns.getTaskCount(index);

            kernel.set_arg(0, dColumns.getDataOffset(index)); // offX
            kernel.set_arg(1, dColumns.getIndicesOffset(index)); // offK
            kernel.set_arg(2, taskCount); // N
            if (double_precision){
                kernel.set_arg(3, realDelta);
            } else {
                auto fDelta = (float)realDelta;
                kernel.set_arg(3, fDelta);
            }
            kernel.set_arg(4, dColumns.getData()); // *X
            kernel.set_arg(5, dColumns.getIndices()); // *K
//            kernel.set_arg(6, dY);
//            kernel.set_arg(7, dXBeta);
//            kernel.set_arg(8, dExpXBeta);
//            kernel.set_arg(9, dDenominator);
//            kernel.set_arg(10, dId);

            size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
            if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
                ++workGroups;
            }
            const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;

            queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
            queue.finish();

            hXBetaKnown = false; // dXBeta was just updated

            ModelSpecifics<BaseModel, RealType>::computeRemainingStatistics(useWeights);

//            // print results
////            std::vector<RealType> hxb;
//            hXBeta.resize(dExpXBeta.size());
//            compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
//            std::cout << "dXBeta: ";
//            for (auto x:hXBeta) {
//                std::cout << x << " ";
//            }
//            std::cout << "\n";
//
////            std::vector<RealType> hexb;
//            offsExpXBeta.resize(dExpXBeta.size());
//            compute::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(offsExpXBeta), queue);
//            std::cout << "dExpXBeta: ";
//            for (auto x:offsExpXBeta) {
//                std::cout << x << " ";
//            }
//            std::cout << "\n";
//
//            denomPid.resize(dDenominator.size());
//            compute::copy(std::begin(dDenominator), std::end(dDenominator), std::begin(denomPid), queue);
//            std::cout << "dDenominator: ";
//            for (auto x:denomPid) {
//                std::cout << x << " ";
//            }
//            std::cout << "\n";

#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name = "updateXBetaG" + getFormatTypeExtension(formatType) + "  ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

#ifdef GPU_DEBUG
            // Compare results:
        detail::compare(hXBeta, dXBeta, "xBeta not equal");
        detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
        detail::compare(denomPid, dDenominator, "denominator not equal");
#endif // GPU_DEBUG
        }

        virtual const std::vector<double> getXBeta() {
            if (!hXBetaKnown) {
                compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
                hXBetaKnown = true;
            }
            return ModelSpecifics<BaseModel, RealType>::getXBeta();
        }

        virtual const std::vector<double> getXBetaSave() {
            return ModelSpecifics<BaseModel, RealType>::getXBetaSave();
        }

        virtual void saveXBeta() {
            if (!hXBetaKnown) {
                compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
                hXBetaKnown = true;
            }
            ModelSpecifics<BaseModel, RealType>::saveXBeta();
        }

        virtual void zeroXBeta() {

            //std::cerr << "GPU::zXB called" << std::endl;

            ModelSpecifics<BaseModel, RealType>::zeroXBeta(); // touches hXBeta

            dXBetaKnown = false;
        }

        virtual void axpyXBeta(const double beta, const int j) {

            //std::cerr << "GPU::aXB called" << std::endl;

            ModelSpecifics<BaseModel, RealType>::axpyXBeta(beta, j); // touches hXBeta

            dXBetaKnown = false;
        }

    private:
        void buildAllUpdateXBetaKernels(const std::vector<FormatType>& neededFormatTypes) {
            for (FormatType formatType : neededFormatTypes) {
                buildUpdateXBetaKernel(formatType);
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

        SourceCode writeCodeForUpdateXBetaKernel(FormatType formatType);

        void buildUpdateXBetaKernel(FormatType formatType) {

            std::stringstream options;
//            options << "-DREAL=" << (sizeof(real) == 8 ? "double" : "float");
            if (double_precision) {
                options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;
            } else {
                options << "-DREAL=float -DTMP_REAL=float -DTPB=" << tpb;
            }
            options << " -cl-mad-enable";

            auto source = writeCodeForUpdateXBetaKernel(formatType);

            std::cerr << source.body << std::endl;

            auto program = compute::program::build_with_source(source.body, ctx, options.str());
            std::cout << "built updateXBeta program \n";
            auto kernel = compute::kernel(program, source.name);

            // Rcpp::stop("uXB");

            // Run-time constant arguments.
            kernel.set_arg(6, dY);
            kernel.set_arg(7, dXBeta);
            kernel.set_arg(8, dExpXBeta);
            kernel.set_arg(9, dDenominator);
            kernel.set_arg(10, dId);

            kernelUpdateXBeta[formatType] = std::move(kernel);
        }

        void buildAllKernels(const std::vector<FormatType>& neededFormatTypes) {
//            buildAllGradientHessianKernels(neededFormatTypes);
//            std::cout << "built gradhessian kernels \n";
            buildAllUpdateXBetaKernels(neededFormatTypes);
            std::cout << "built updateXBeta kernels \n";
        }

        std::map<FormatType, compute::kernel> kernelUpdateXBeta;

        bool hXBetaKnown;
        bool dXBetaKnown;
    };
} // namespace bsccs

#include "KernelsCox.hpp"

#endif //GPUMODELSPECIFICSCOX_HPP
