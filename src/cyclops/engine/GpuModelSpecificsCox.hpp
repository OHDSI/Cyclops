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
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "BaseGpuModelSpecifics.hpp"
#include "Iterators.h"
#include "CudaKernel.h"
//#include "CudaDetail.h"

namespace bsccs{

    namespace compute = boost::compute;

//    namespace detail {
//
//        namespace constant {
//            static const int updateXBetaBlockSize = 256; // 512; // Appears best on K40
//            static const int updateAllXBetaBlockSize = 32;
//            int exactCLRBlockSize = 32;
//            int exactCLRSyncBlockSize = 32;
//            static const int maxBlockSize = 256;
//        }; // namespace constant
//    }; // namespace detail
 
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
        using BaseGpuModelSpecifics<BaseModel, RealType>::hXjY;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hPid;
       	using BaseGpuModelSpecifics<BaseModel, RealType>::K;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::J;
        using BaseGpuModelSpecifics<BaseModel, RealType>::N;
        using BaseGpuModelSpecifics<BaseModel, RealType>::offsExpXBeta;
        using BaseGpuModelSpecifics<BaseModel, RealType>::hXBeta;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hY;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hOffs;
        using BaseGpuModelSpecifics<BaseModel, RealType>::denomPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::denomPid2;
        using BaseGpuModelSpecifics<BaseModel, RealType>::numerPid;
        using BaseGpuModelSpecifics<BaseModel, RealType>::numerPid2;
        using BaseGpuModelSpecifics<BaseModel, RealType>::hNWeight;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::hKWeight;
        using BaseGpuModelSpecifics<BaseModel, RealType>::accDenomPid;
        using BaseGpuModelSpecifics<BaseModel, RealType>::accNumerPid;
        using BaseGpuModelSpecifics<BaseModel, RealType>::accNumerPid2;

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
        using BaseGpuModelSpecifics<BaseModel, RealType>::dAccDenominator;
        using BaseGpuModelSpecifics<BaseModel, RealType>::dCudaColumns;

        std::vector<double> hBuffer;
        std::vector<double> hBuffer1;

        int tpb = 256; // threads-per-block  // Appears best on K40
        int PSC_K = 32;
        int PSC_WG_SIZE = 256;

//        AllGpuColumns<RealType> dCudaColumns;
        CudaKernel<RealType> CudaData;
	
        GpuModelSpecificsCox(const ModelData<RealType>& input,
                             const std::string& deviceName)
        : BaseGpuModelSpecifics<BaseModel, RealType>(input, deviceName)
          // , dCudaColumns()
          , CudaData()
		    {
       

            std::cerr << "ctor GpuModelSpecificsCox" << std::endl;

        }

        virtual ~GpuModelSpecificsCox() {
            std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
        }

        virtual void deviceInitialization() {
            BaseGpuModelSpecifics<BaseModel, RealType>::deviceInitialization();
	        CudaData.initialize(K, N);
            hXBetaKnown = true;
            dXBetaKnown = true;
        }


        virtual void computeRemainingStatistics(bool useWeights) {

            std::cerr << "GPU::cRS called" << std::endl;

            // Currently RS only computed on CPU and then copied
            ModelSpecifics<BaseModel, RealType>::computeRemainingStatistics(useWeights);

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
            /*
            if (algorithmType == AlgorithmType::MM) {
                compute::copy(std::begin(hBeta), std::end(hBeta), std::begin(dBeta), queue);
            }
            */

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

//            std::cout << "GPU computeGradientAndHessian \n";
		    std::vector<RealType> gradient(1, static_cast<RealType>(0));
		    std::vector<RealType> hessian(1, static_cast<RealType>(0));
		
		    FormatType formatType = hX.getFormatType(index);
		    const auto taskCount = dCudaColumns.getTaskCount(index);

#ifdef CYCLOPS_DEBUG_TIMING
            auto start0 = bsccs::chrono::steady_clock::now();
#endif
            // cuda class
//            CudaKernel<RealType> CudaData(N);

#ifdef CYCLOPS_DEBUG_TIMING
            auto end0 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name0 = "compGradHessG" + getFormatTypeExtension(formatType) + "  cudaMalloc";
            duration[name0] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end0 - start0).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start1 = bsccs::chrono::steady_clock::now();
#endif
	        // copy from host to device
            cudaMemcpy(CudaData.d_Numer, &numerPid[0], sizeof(RealType) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(CudaData.d_Numer2, &numerPid2[0], sizeof(RealType) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(CudaData.d_AccDenom, &accDenomPid[0], sizeof(RealType) * N, cudaMemcpyHostToDevice);
            cudaMemcpy(CudaData.d_NWeight, &hNWeight[0], sizeof(RealType) * N, cudaMemcpyHostToDevice);

#ifdef CYCLOPS_DEBUG_TIMING
            auto end1 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name1 = "compGradHessG" + getFormatTypeExtension(formatType) + " cudaMemcpy1";
            duration[name1] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end1 - start1).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start2 = bsccs::chrono::steady_clock::now();
#endif	
            // scan (computeAccumlatedNumerator)
            CudaData.CubScan(CudaData.d_Numer, CudaData.d_AccNumer, N);
            CudaData.CubScan(CudaData.d_Numer2, CudaData.d_AccNumer2, N);

#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name2 = "compGradHessG" + getFormatTypeExtension(formatType) + "    accNumer";
            duration[name2] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start3 = bsccs::chrono::steady_clock::now();
#endif
	        // kernel
            int gridSize, blockSize;
            blockSize = 256;
            gridSize = (int)ceil((double)N/blockSize);
	        CudaData.computeGradientAndHessian(N, gridSize, blockSize);
    
#ifdef CYCLOPS_DEBUG_TIMING
            auto end3 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name3 = "compGradHessG" + getFormatTypeExtension(formatType) + " transReduce";
            duration[name3] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end3 - start3).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start4 = bsccs::chrono::steady_clock::now();
#endif
	        // copy the results from device to host
            cudaMemcpy(&gradient[0], CudaData.d_Gradeint, sizeof(RealType), cudaMemcpyDeviceToHost);
            cudaMemcpy(&hessian[0], CudaData.d_Hessian, sizeof(RealType), cudaMemcpyDeviceToHost);
            
#ifdef CYCLOPS_DEBUG_TIMING
            auto end4 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name4 = "compGradHessG" + getFormatTypeExtension(formatType) + " cudaMemcpy2";
            duration[name4] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start4).count();
#endif	    
	    
//	        std::cout << "g: " << gradient[0] << " h: " << hessian[0] << " xjy: " << hXjY[index] << '\n';

            gradient[0] -= hXjY[index];
            *ogradient = static_cast<double>(gradient[0]);
            *ohessian = static_cast<double>(hessian[0]);

	 
	}


        virtual void updateXBeta(double delta, int index, bool useWeights) {

#ifdef GPU_DEBUG
            ModelSpecifics<BaseModel, WeightType>::updateXBeta(delta, index, useWeights);
#endif // GPU_DEBUG

//	    ModelSpecifics<BaseModel, RealType>::updateXBeta(delta, index, useWeights);
	    //std::cout << "GPU updateXBeta \n";
/*
            // FOR TEST: check data
            std::cout << "delta: " << delta << '\n';
            std::cout << "K: " << K  << " N: " << N << " TaskCount: " << dColumns.getTaskCount(index) << '\n';
            std::cout << "offX: " << dColumns.getDataOffset(index) << " offK: " << dColumns.getIndicesOffset(index) << '\n';

            // FOR TEST: check data
            std::cout << "delta: " << delta << '\n';
            std::cout << "old hXBeta: ";
            for (auto x:hXBeta) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
            std::cout << "old offsExpXBeta: ";
            for (auto x:offsExpXBeta) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
            std::cout << "old accDenomPid: ";
            for (auto x:accDenomPid) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
*/
            FormatType formatType = hX.getFormatType(index);
	        const auto taskCount = dCudaColumns.getTaskCount(index);

            // cuda class
//            CudaKernel<RealType> CudaData(K);

	        // copy from host to device
            cudaMemcpy(CudaData.d_XBeta, &hXBeta[0], sizeof(RealType) * K, cudaMemcpyHostToDevice);
            cudaMemcpy(CudaData.d_ExpXBeta, &offsExpXBeta[0], sizeof(RealType) * K, cudaMemcpyHostToDevice);

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif

            // updateXBeta kernel
            int gridSize, blockSize;
            blockSize = 256;
            gridSize = (int)ceil((double)taskCount/blockSize);
            CudaData.updateXBeta(dCudaColumns.getData(), dCudaColumns.getIndices(), dCudaColumns.getDataOffset(index), dCudaColumns.getIndicesOffset(index), taskCount, static_cast<RealType>(delta), gridSize, blockSize);

            // scan (computeAccumlatedDenominator)
	        CudaData.CubScan(CudaData.d_ExpXBeta, CudaData.d_AccDenom, K);

#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name = "updateXBetaG" + getFormatTypeExtension(formatType) + "  ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
            // copy the results from host to host
            cudaMemcpy(&hXBeta[0], CudaData.d_XBeta, sizeof(RealType) * K, cudaMemcpyDeviceToHost);
            cudaMemcpy(&offsExpXBeta[0], CudaData.d_ExpXBeta, sizeof(RealType) * K, cudaMemcpyDeviceToHost);
            cudaMemcpy(&accDenomPid[0], CudaData.d_AccDenom, sizeof(RealType) * K, cudaMemcpyDeviceToHost);
/*
            std::cout << "new hXBeta: ";
            for (auto x:hXBeta) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
            std::cout << "new offsExpXBeta: ";
            for (auto x:offsExpXBeta) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
            std::cout << "new accDenomPid: ";
            for (auto x:accDenomPid) {
                    std::cout << x << " ";
            }
            std::cout << "\n";
*/

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

        bool hXBetaKnown;
        bool dXBetaKnown;
    };
} // namespace bsccs

//#include "KernelsCox.hpp"

#endif //GPUMODELSPECIFICSCOX_HPP
