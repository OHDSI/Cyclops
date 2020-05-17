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

#include "ModelSpecifics.hpp"
#include "Iterators.h"
#include "CudaKernel.h"
#include "CudaDetail.h"

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

    template <typename RealType>
    class CudaAllGpuColumns {
    public:

        typedef thrust::device_vector<RealType> DataVector;
        typedef thrust::device_vector<int> IndicesVector;
        typedef unsigned int UInt;
        typedef thrust::device_vector<UInt> dStartsVector;
        typedef std::vector<UInt> hStartsVector;

        CudaAllGpuColumns() {
            // Do nothing
//		std::cout << "ctor AGC \n";
        }

        virtual ~CudaAllGpuColumns() {
//		std::cout << "dtor AGC \n";       
	}

        void initialize(const CompressedDataMatrix<RealType>& mat,
                        size_t K, bool pad) {
            std::vector<RealType> flatData;
            std::vector<int> flatIndices;

            std::cerr << "Cuda AGC start" << std::endl;

            UInt dataStart = 0;
            UInt indicesStart = 0;

            for (int j = 0; j < mat.getNumberOfColumns(); ++j) {
                const auto& column = mat.getColumn(j);
                const auto format = column.getFormatType();

                dataStarts.push_back(dataStart);
                indicesStarts.push_back(indicesStart);
                formats.push_back(format);

                // Data vector
                if (format == FormatType::SPARSE ||
                    format == FormatType::DENSE) {
                    appendAndPad(column.getDataVector(), flatData, dataStart, pad);
                }

                // Indices vector
                if (format == FormatType::INDICATOR ||
                    format == FormatType::SPARSE) {
                    appendAndPad(column.getColumnsVector(), flatIndices, indicesStart, pad);
                }

                // Task count
                if (format == FormatType::DENSE ||
                    format == FormatType::INTERCEPT) {
                    taskCounts.push_back(K);
                } else { // INDICATOR, SPARSE
                    taskCounts.push_back(column.getNumberOfEntries());
                }
            }
            
	    resizeAndCopyToDeviceCuda(flatData, data);
            resizeAndCopyToDeviceCuda(flatIndices, indices);
            resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
            resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
            resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
/*
            CudaDetail<RealType> rdetail;
            CudaDetail<int> idetail;
            CudaDetail<UInt> udetail;
            rdetail.resizeAndCopyToDeviceCuda(flatData, data);
            idetail.resizeAndCopyToDeviceCuda(flatIndices, indices);
            udetail.resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
            udetail.resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
            udetail.resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
*/
            std::cerr << "cuda AGC end " << flatData.size() << " " << flatIndices.size() << std::endl;
        }

        UInt getDataOffset(int column) const {
            return dataStarts[column];
        }

        UInt getIndicesOffset(int column) const {
            return indicesStarts[column];
        }

        UInt getTaskCount(int column) const {
            return taskCounts[column];
        }

        const DataVector& getData() const {
            return data;
        }

	    const IndicesVector& getIndices() const {
            return indices;
        }
	
	    const dStartsVector& getDataStarts() const {
            return ddataStarts;
        }

        const dStartsVector& getIndicesStarts() const {
            return dindicesStarts;
        }

        const dStartsVector& getTaskCounts() const {
            return dtaskCounts;
        }

        const std::vector<FormatType> getFormat() const{
            return formats;
        }


    private:

        template <class T>
        void appendAndPad(const T& source, T& destination, UInt& length, bool pad) {
            for (auto x : source) {
                destination.push_back(x);
            }
            if (pad) {
                auto i = source.size();
                const auto end = detail::getAlignedLength<16>(i);
                for (; i < end; ++i) {
                    destination.push_back(typename T::value_type());
                }
                length += end;
            } else {
                length += source.size();
            }
        }


        IndicesVector indices;
        DataVector data;

        hStartsVector taskCounts;
        hStartsVector dataStarts;
        hStartsVector indicesStarts;

        dStartsVector dtaskCounts;
        dStartsVector ddataStarts;
        dStartsVector dindicesStarts;

        //std::vector<UInt> taskCounts;
        //std::vector<UInt> dataStarts;
        //std::vector<UInt> indicesStarts;
        std::vector<FormatType> formats;
    };

    template <class BaseModel, typename RealType>
    class GpuModelSpecificsCox :
            public ModelSpecifics<BaseModel, RealType> {
    public:

#ifdef CYCLOPS_DEBUG_TIMING
        using ModelSpecifics<BaseModel, RealType>::duration;
#endif
        using ModelSpecifics<BaseModel, RealType>::modelData;
        using ModelSpecifics<BaseModel, RealType>::hX;
        using ModelSpecifics<BaseModel, RealType>::hXjY;
//        using ModelSpecifics<BaseModel, RealType>::hPid;
       	using ModelSpecifics<BaseModel, RealType>::K;
        using ModelSpecifics<BaseModel, RealType>::J;
        using ModelSpecifics<BaseModel, RealType>::N;
        using ModelSpecifics<BaseModel, RealType>::offsExpXBeta;
        using ModelSpecifics<BaseModel, RealType>::hXBeta;
//        using ModelSpecifics<BaseModel, RealType>::hY;
//        using ModelSpecifics<BaseModel, RealType>::hOffs;
        using ModelSpecifics<BaseModel, RealType>::denomPid;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::denomPid2;
        using ModelSpecifics<BaseModel, RealType>::numerPid;
        using ModelSpecifics<BaseModel, RealType>::numerPid2;
        using ModelSpecifics<BaseModel, RealType>::hNWeight;
        using ModelSpecifics<BaseModel, RealType>::hKWeight;
        using ModelSpecifics<BaseModel, RealType>::accDenomPid;
        using ModelSpecifics<BaseModel, RealType>::accNumerPid;
        using ModelSpecifics<BaseModel, RealType>::accNumerPid2;
/*
	// device 
//        using BaseGpuModelSpecifics<BaseModel, RealType>::ctx;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::device;
//        using BaseGpuModelSpecifics<BaseModel, RealType>::queue;
        using BaseGpuModelSpecifics<BaseModel, RealType>::neededFormatTypes;
        using BaseGpuModelSpecifics<BaseModel, RealType>::double_precision;
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
*/
        std::vector<double> hBuffer;
        std::vector<double> hBuffer1;

        int tpb = 256; // threads-per-block  // Appears best on K40
        int PSC_K = 32;
        int PSC_WG_SIZE = 256;

        CudaAllGpuColumns<RealType> dCudaColumns;
        CudaKernel<RealType> CudaData;
	
        GpuModelSpecificsCox(const ModelData<RealType>& input,
                             const std::string& deviceName)
        : ModelSpecifics<BaseModel,RealType>(input),
          dCudaColumns(),
	  dXBeta(), dExpXBeta(),
	  dDenominator(), dAccDenominator(),
          dKWeight(), dNWeight()
          , CudaData()
	{
       
            std::cerr << "ctor GpuModelSpecificsCox" << std::endl;

        }

        virtual ~GpuModelSpecificsCox() {
            std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
        }

        virtual void deviceInitialization() {
//            BaseGpuModelSpecifics<BaseModel, RealType>::deviceInitialization();
            RealType blah = 0;
            if (sizeof(blah)==8) {
                double_precision = true;
            }

#ifdef TIME_DEBUG
            std::cerr << "start dI" << std::endl;
#endif
            //isNvidia = compute::detail::is_nvidia_device(queue.get_device());

            //std::cout << "maxWgs: " << maxWgs << "\n";

            int need = 0;

            // Copy data
	        dCudaColumns.initialize(hX, K, true);
            //this->initializeMmXt();
            //dColumnsXt.initialize(*hXt, queue, K, true);
            formatList.resize(J);

            for (size_t j = 0; j < J /*modelData.getNumberOfColumns()*/; ++j) {

#ifdef TIME_DEBUG
                //  std::cerr << "dI " << j << std::endl;
#endif
                FormatType format = hX.getFormatType(j);
                //const auto& column = modelData.getColumn(j);
                // columns.emplace_back(GpuColumn<real>(column, ctx, queue, K));
                need |= (1 << format);

                indicesFormats[format].push_back(j);
                formatList[j] = format;
            }
            // Rcpp::stop("done");

//            std::vector<FormatType> neededFormatTypes;
            for (int t = 0; t < 4; ++t) {
                if (need & (1 << t)) {
                    neededFormatTypes.push_back(static_cast<FormatType>(t));
                }
            }
/*          
	    // Internal buffers
            resizeAndCopyToDeviceCuda(hXBeta, dXBeta);
            hXBetaKnown = true; dXBetaKnown = true;
            resizeAndCopyToDeviceCuda(offsExpXBeta, dExpXBeta);
            resizeAndCopyToDeviceCuda(denomPid, dDenominator);
            resizeAndCopyToDeviceCuda(accDenomPid, dAccDenominator);
*/

            resizeAndCopyToDeviceCuda(hKWeight, dKWeight);
            resizeAndCopyToDeviceCuda(hNWeight, dNWeight);

            std::cerr << "Format types required: " << need << std::endl;

        }
        
	virtual void setWeights(double* inWeights, bool useCrossValidation) {
            // Currently only computed on CPU and then copied to GPU
            ModelSpecifics<BaseModel, RealType>::setWeights(inWeights, useCrossValidation);

            resizeAndCopyToDeviceCuda(hKWeight, dKWeight);
            resizeAndCopyToDeviceCuda(hNWeight, dNWeight);
            // std::cout << "GPUMS hKWeight" ;
            // for (auto x : hKWeight ){
            //     std::cout << x ;
            // }
            // std::cout << '\n';
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
/*
            if (BaseModel::likelihoodHasDenominator) {
                thrust::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta));
                thrust::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator));
            }

*/
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

        }

/*
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
            cudaMemcpy(&gradient[0], CudaData.d_Gradient, sizeof(RealType), cudaMemcpyDeviceToHost);
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
*/

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
            resizeAndCopyToDeviceCuda(hXBeta, dXBeta);
            resizeAndCopyToDeviceCuda(offsExpXBeta, dExpXBeta);
            resizeAndCopyToDeviceCuda(accDenomPid, dAccDenominator);

	        // copy from host to device
//            cudaMemcpy(CudaData.d_XBeta, &hXBeta[0], sizeof(RealType) * K, cudaMemcpyHostToDevice);
//            cudaMemcpy(CudaData.d_ExpXBeta, &offsExpXBeta[0], sizeof(RealType) * K, cudaMemcpyHostToDevice);

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif

            // updateXBeta kernel
            int gridSize, blockSize;
            blockSize = 256;
            gridSize = (int)ceil((double)taskCount/blockSize);
            CudaData.updateXBeta(dCudaColumns.getData(), dCudaColumns.getIndices(), dCudaColumns.getDataOffset(index), dCudaColumns.getIndicesOffset(index), taskCount, static_cast<RealType>(delta), dXBeta, dExpXBeta, gridSize, blockSize);
            
            // scan (computeAccumlatedDenominator)
	        CudaData.CubScan(thrust::raw_pointer_cast(&dExpXBeta[0]), thrust::raw_pointer_cast(&dAccDenominator[0]), K);

#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name = "updateXBetaG" + getFormatTypeExtension(formatType) + "  ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

	    thrust::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta));
	    thrust::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(offsExpXBeta));
	    thrust::copy(std::begin(dAccDenominator), std::end(dAccDenominator), std::begin(accDenomPid));

/*
	    // copy the results from host to host
            cudaMemcpy(&hXBeta[0], CudaData.d_XBeta, sizeof(RealType) * K, cudaMemcpyDeviceToHost);
            cudaMemcpy(&offsExpXBeta[0], CudaData.d_ExpXBeta, sizeof(RealType) * K, cudaMemcpyDeviceToHost);
            cudaMemcpy(&accDenomPid[0], CudaData.d_AccDenom, sizeof(RealType) * K, cudaMemcpyDeviceToHost);

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
                thrust::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta));
                hXBetaKnown = true;
            }
            return ModelSpecifics<BaseModel, RealType>::getXBeta();
        }

        virtual const std::vector<double> getXBetaSave() {
            return ModelSpecifics<BaseModel, RealType>::getXBetaSave();
        }

        virtual void saveXBeta() {
            if (!hXBetaKnown) {
                thrust::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta));
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
	bool double_precision = false;

        std::map<FormatType, std::vector<int>> indicesFormats;
        std::vector<FormatType> formatList;
        std::vector<FormatType> neededFormatTypes;

	// device storage
	thrust::device_vector<RealType> dXBeta;
        thrust::device_vector<RealType> dExpXBeta;
        thrust::device_vector<RealType> dDenominator;
        thrust::device_vector<RealType> dAccDenominator;
        thrust::device_vector<RealType> dKWeight;
        thrust::device_vector<RealType> dNWeight;


    };
} // namespace bsccs

//#include "KernelsCox.hpp"

#endif //GPUMODELSPECIFICSCOX_HPP
