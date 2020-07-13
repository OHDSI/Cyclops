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
        }

        virtual ~CudaAllGpuColumns() {
        }

        void initialize(const CompressedDataMatrix<RealType>& mat,
                        size_t K, bool pad) {
            //std::vector<RealType> flatData;
            //std::vector<int> flatIndices;

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
/*
			resizeAndCopyToDeviceCuda(flatData, data);
			resizeAndCopyToDeviceCuda(flatIndices, indices);
			resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
			resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
			resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
*/
/*
            std::cout << "flatIndices: ";
            for (auto x:flatIndices) {
             std::cout << x << " ";
            }
            std::cout << "\n";	

            CudaDetail<RealType> rdetail;
            CudaDetail<int> idetail;
            CudaDetail<UInt> udetail;
            rdetail.resizeAndCopyToDeviceCuda(flatData, data);
            idetail.resizeAndCopyToDeviceCuda(flatIndices, indices);
            udetail.resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
            udetail.resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
            udetail.resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
*/
            std::cerr << "cuda AGC end " << flatData.size() << " " << flatIndices.size() << " " << dataStarts.size() << " " << indicesStarts.size() << " " << taskCounts.size() << std::endl;
        }

	void resizeAndCopyColumns () {
                        resizeAndCopyToDeviceCuda(flatData, data);
                        resizeAndCopyToDeviceCuda(flatIndices, indices);
                        resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
                        resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
                        resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
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

        const std::vector<int>& getHIndices() const {
            return flatIndices;
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

        std::vector<RealType> flatData;
        std::vector<int> flatIndices;

        dStartsVector dtaskCounts;
        dStartsVector ddataStarts;
        dStartsVector dindicesStarts;

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
          dBeta(), dXBeta(), dExpXBeta(),
          dDenominator(), dAccDenominator(),
          dKWeight(), dNWeight(),
          CudaData()
          {
            std::cerr << "ctor GpuModelSpecificsCox" << std::endl;
          }

        virtual ~GpuModelSpecificsCox() {
			cudaFree(dGH);
//			free(pGH);
			cudaFreeHost(pGH);
			std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
        }

        virtual void deviceInitialization() {
#ifdef TIME_DEBUG
            std::cerr << "start dI" << std::endl;
#endif

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif	    
            // Copy data
            dCudaColumns.initialize(hX, K, true);
/*
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z dColumnInitialize  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

	    formatList.resize(J);
	    int need = 0;

            for (size_t j = 0; j < J ; ++j) {

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
#ifdef CYCLOPS_DEBUG_TIMING
            auto start1 = bsccs::chrono::steady_clock::now();
#endif
*/
            dCudaColumns.resizeAndCopyColumns();
/*	    
#ifdef CYCLOPS_DEBUG_TIMING
            auto end1 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z resizeAndCopyColumn"] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end1 - start1).count();;
#endif

#ifdef CYCLOPS_DEBUG_TIMING
            auto start2 = bsccs::chrono::steady_clock::now();
#endif
*/        
	    	RealHBeta.resize(J);
		DoubleHBeta.resize(J);

	    	// Internal buffers
		resizeCudaVec(hXBeta, dXBeta); // K
		resizeCudaVec(offsExpXBeta, dExpXBeta);

		resizeAndZeroCudaVec(numerPid, dNumerator); // getAlignedLength(N + 1)
		if (BaseModel::hasTwoNumeratorTerms) {
		    resizeAndZeroCudaVec(numerPid2, dNumerator2);
		}
		resizeCudaVec(numerPid, dAccNumer);
		resizeCudaVec(numerPid2, dAccNumer2);

		resizeCudaVecSize(dAccDenominator, N); // N+1?
//		resizeCudaVecSize(dDeltaVector, J);

		cudaMalloc((void**)&dGH, sizeof(double2));
//		pGH = (double2 *)malloc(sizeof(double2));
		cudaMallocHost((void **) &pGH, sizeof(double2));
/*
		resizeCudaVec(numerPid, dBuffer1);
		resizeCudaVec(numerPid2, dBuffer2);
		resizeCudaVecSize(dBuffer3, N+1);
*/    
//		resizeCudaVecSize(indicesN, N);
/*		
#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z resizeAndCopyCuda  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();;
#endif
	
#ifdef CYCLOPS_DEBUG_TIMING
            auto start3 = bsccs::chrono::steady_clock::now();
#endif
*/	    
		// Allocate temporary storage for scan and reduction
		CudaData.allocTempStorage(dExpXBeta,
		        dNumerator,
		        dNumerator2,
		        dAccDenominator,
		        dAccNumer,
		        dAccNumer2,
		        dNWeight,
		        dGH,
		        N,
		        indicesN);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z cudaDevInitialization "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
//        std::cerr << "Format types required: " << need << std::endl;

        }

        virtual void setWeights(double* inWeights, bool useCrossValidation) {
			// Currently only computed on CPU and then copied to GPU
			ModelSpecifics<BaseModel, RealType>::setWeights(inWeights, useCrossValidation);

			resizeAndCopyToDeviceCuda(hKWeight, dKWeight);
			resizeAndCopyToDeviceCuda(hNWeight, dNWeight);
//			std::cout << "GPU::setWeights called \n";
        }

	virtual void computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
		
		ModelSpecifics<BaseModel,RealType>::computeFixedTermsInGradientAndHessian(useCrossValidation);
#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
		resizeAndCopyToDeviceCuda(hXjY, dXjY);	
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z compFixedGHG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

        }

        virtual void computeRemainingStatistics(bool useWeights) {

                        std::cerr << "GPU::cRS called" << std::endl;
                        hXBetaKnown = true;
#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
                        // Currently RS only computed on CPU and then copied
			ModelSpecifics<BaseModel, RealType>::computeRemainingStatistics(useWeights);

/*
            if (algorithmType == AlgorithmType::MM) {
                compute::copy(std::begin(hBeta), std::end(hBeta), std::begin(dBeta), queue);
            }
*/

			thrust::copy(std::begin(hXBeta), std::end(hXBeta), std::begin(dXBeta));
			thrust::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta));
			thrust::copy(std::begin(accDenomPid), std::end(accDenomPid)-1, std::begin(dAccDenominator));

#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z compRSG               "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

        }

        virtual double getGradientObjective(bool useCrossValidation) {
                        
                        // TODO write gpu version to avoid D-H copying
			return ModelSpecifics<BaseModel, RealType>::getGradientObjective(useCrossValidation);
      
      	}

        virtual double getLogLikelihood(bool useCrossValidation) {
//			std::cout << "GPU::cLL called" << std::endl;

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
			// TODO write gpu version to avoid D-H copying
			thrust::copy(std::begin(dAccDenominator), std::end(dAccDenominator), std::begin(accDenomPid));
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z compLogLikeG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
			// Currently LL only computed on CPU and then copied
			return ModelSpecifics<BaseModel, RealType>::getLogLikelihood(useCrossValidation); 
        }
        
	virtual void computeNumeratorForGradient(int index, bool useWeights) {

                        FormatType formatType = hX.getFormatType(index);
			int formatInt = getFormatTypeInt(formatType);
                        const auto taskCount = dCudaColumns.getTaskCount(index);

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
                        // sparse transformation
                        int gridSize, blockSize;
                        blockSize = 256;
                        gridSize = (int)ceil((double)taskCount/blockSize);
                        CudaData.computeNumeratorForGradient(dCudaColumns.getData(),
                                dCudaColumns.getIndices(),
                                dCudaColumns.getDataOffset(index),
                                dCudaColumns.getIndicesOffset(index),
                                taskCount,
                                dExpXBeta,
                                dNumerator,
                                dNumerator2,
				formatInt,
                                gridSize, blockSize);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name = "compNumForG" + getFormatTypeExtension(formatType) + "   ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

        }

        virtual void computeGradientAndHessian(int index, double *ogradient,
                                           double *ohessian, bool useWeights) {

//			std::cout << "GPU::cGAH \n";
//			std::vector<RealType> gradient(1, static_cast<RealType>(0));
//			std::vector<RealType> hessian(1, static_cast<RealType>(0));
			double2 GH;

			FormatType formatType = hX.getFormatType(index);
			const auto taskCount = dCudaColumns.getTaskCount(index);
/*
#ifdef CYCLOPS_DEBUG_TIMING
            auto start1 = bsccs::chrono::steady_clock::now();
#endif
                        // dense scan
//                        CudaData.CubScan(thrust::raw_pointer_cast(&dExpXBeta[0]), thrust::raw_pointer_cast(&dAccDenominator[0]), K);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end1 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name1 = "compGradHessG" + getFormatTypeExtension(formatType) + "    accDenom";
            duration[name1] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end1 - start1).count();
#endif

#ifdef CYCLOPS_DEBUG_TIMING
            auto start2 = bsccs::chrono::steady_clock::now();
#endif	
			// dense scan on tuple
			CudaData.computeAccumulatedNumerator(dNumerator,
			        dNumerator2,
			        dAccNumer,
			        dAccNumer2,
			        N);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name2 = "compGradHessG" + getFormatTypeExtension(formatType) + "    accNumer";
            duration[name2] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif
*/
#ifdef CYCLOPS_DEBUG_TIMING
            auto start2 = bsccs::chrono::steady_clock::now();
#endif
			// dense scan on tuple
			CudaData.computeAccumulatedNumerAndDenom(dExpXBeta,
			    			     dNumerator,
						     dNumerator2,
						     dAccDenominator,
						     dAccNumer,
						     dAccNumer2,
						     N);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name2 = "compGradHessG" + getFormatTypeExtension(formatType) + "    accNAndD";
            duration[name2] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start3 = bsccs::chrono::steady_clock::now();
#endif
			// dense transform reduction
			CudaData.computeGradientAndHessian(dAccNumer,
			        dAccNumer2,
			        dAccDenominator,
			        dNWeight,
			        dGH,
			        N
//			        , dCudaColumns.getHIndices(),
//			        dCudaColumns.getIndicesOffset(index),
//			        indicesN
			        );
#ifdef CYCLOPS_DEBUG_TIMING
            auto end3 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name3 = "compGradHessG" + getFormatTypeExtension(formatType) + " transReduce";
            duration[name3] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end3 - start3).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
            auto start4 = bsccs::chrono::steady_clock::now();
#endif    
			cudaMemcpy(pGH, dGH, sizeof(double2), cudaMemcpyDeviceToHost);
			GH = *pGH;
			std::cout << "index: " << index << " g: " << GH.x << " h: " << GH.y << " XjY: " << hXjY[index] << '\n';
#ifdef CYCLOPS_DEBUG_TIMING
            auto end4 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name4 = "compGradHessG" + getFormatTypeExtension(formatType) + " cudaMemcpy2";
            duration[name4] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start4).count();
/*
            double timerG = 0;
            timerG = std::chrono::duration<double, std::milli>(end4 - start4).count();
            std::cout << "timerG: " << timerG << '\n';
*/	    
            auto name = "compGradHessG" + getFormatTypeExtension(formatType) + " ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start2).count();
#endif	    
	    
			GH.x -= hXjY[index];
			*ogradient = static_cast<double>(GH.x);
			*ohessian = static_cast<double>(GH.y);
/*
#ifdef CYCLOPS_DEBUG_TIMING
            auto start1 = bsccs::chrono::steady_clock::now();
#endif
			// empty kernel
			CudaData.empty4(dAccNumer,
                            dAccNumer2,
                            dBuffer1,
                            dBuffer2);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end1 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name1 = "compGradHessG" + getFormatTypeExtension(formatType) + "      empty4";
            duration[name1] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end1 - start1).count();
#endif
*/
        }


        virtual void updateXBeta(double delta, int index, bool useWeights) {

#ifdef GPU_DEBUG
            ModelSpecifics<BaseModel, WeightType>::updateXBeta(delta, index, useWeights);
#endif // GPU_DEBUG

/*
            // FOR TEST: check data
            std::cout << "delta: " << delta << '\n';
            std::cout << "K: " << K  << " N: " << N << " TaskCount: " << dCudaColumns.getTaskCount(index) << '\n';
            std::cout << "index: " << index << " TaskCount: " << dCudaColumns.getTaskCount(index) << " offX: " << dCudaColumns.getDataOffset(index) << " offK: " << dCudaColumns.getIndicesOffset(index) << '\n';
*/
			FormatType formatType = hX.getFormatType(index);
			const auto taskCount = dCudaColumns.getTaskCount(index);

#ifdef CYCLOPS_DEBUG_TIMING
         auto start2 = bsccs::chrono::steady_clock::now();
#endif
			// sparse transformation
			int gridSize, blockSize;
			blockSize = 256;
			gridSize = (int)ceil((double)taskCount/blockSize);
			CudaData.updateXBeta(dCudaColumns.getData(),
			        dCudaColumns.getIndices(),
			        dCudaColumns.getDataOffset(index),
			        dCudaColumns.getIndicesOffset(index),
			        taskCount,
			        static_cast<RealType>(delta),
			        dXBeta,
			        dExpXBeta,
			        dNumerator,
			        dNumerator2,
			        gridSize, blockSize);
			hXBetaKnown = false;
#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
//            auto name2 = "updateXBetaG" + getFormatTypeExtension(formatType) + "    transform";
//            duration[name2] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
            auto name = "updateXBetaG" + getFormatTypeExtension(formatType) + "  ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif

/*
#ifdef CYCLOPS_DEBUG_TIMING
         auto start3 = bsccs::chrono::steady_clock::now();
#endif
			// dense scan
			CudaData.CubScan(thrust::raw_pointer_cast(&dExpXBeta[0]), thrust::raw_pointer_cast(&dAccDenominator[0]), K);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end3 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name3 = "updateXBetaG" + getFormatTypeExtension(formatType) + "     accDenom";
            duration[name3] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end3 - start3).count();

            auto name = "updateXBetaG" + getFormatTypeExtension(formatType) + "  ";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end3 - start2).count();
#endif
*/
/*
#ifdef CYCLOPS_DEBUG_TIMING
           auto start4 = bsccs::chrono::steady_clock::now();
#endif
            // empty kernel
            CudaData.empty2(dAccDenominator,
                            dBuffer3);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end4 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name4 = "updateXBetaG" + getFormatTypeExtension(formatType) + "       empty2";
            duration[name4] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start4).count();
#endif
*/

#ifdef GPU_DEBUG
            // Compare results:
            detail::compare(hXBeta, dXBeta, "xBeta not equal");
            detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
            detail::compare(denomPid, dDenominator, "denominator not equal");
#endif // GPU_DEBUG
        }

	virtual void updateBetaAndDelta(int index, bool useWeights) {

			FormatType formatType = hX.getFormatType(index);
			int formatInt = getFormatTypeInt(formatType);
			const auto taskCount = dCudaColumns.getTaskCount(index);

			int gridSize, blockSize;
			blockSize = 256;
			gridSize = (int)ceil((double)taskCount/blockSize);

            ////////////////////////// computeGradientAndHessian
#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
			// sparse transformation
			CudaData.computeNumeratorForGradient(dCudaColumns.getData(),
			        dCudaColumns.getIndices(),
			        dCudaColumns.getDataOffset(index),
			        dCudaColumns.getIndicesOffset(index),
			        taskCount,
			        dExpXBeta,
			        dNumerator,
			        dNumerator2,
				formatInt,
			        gridSize, blockSize);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + "    compNumer";
            duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
/*
            std::vector<RealType> tempNumer(16, 0);
            thrust::copy(std::begin(dNumerator), std::end(dNumerator), std::begin(tempNumer));
            std::cout << "Numer: ";
            for (auto x:tempNumer) {
                std::cout << x << " ";
            }
            std::cout << "\n";
	    */
#ifdef CYCLOPS_DEBUG_TIMING
            auto start1 = bsccs::chrono::steady_clock::now();
#endif
			// dense scan on tuple
			CudaData.computeAccumulatedNumerator(dNumerator,
			        dNumerator2,
			        dAccNumer,
			        dAccNumer2,
			        N);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end1 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name1 = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + "     accNumer";
            duration[name1] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end1 - start1).count();
#endif
            /*
	    std::vector<RealType> tempANumer(16, 0);
            thrust::copy(std::begin(dAccNumer), std::end(dAccNumer), std::begin(tempANumer));
            std::cout << "AccNumer: ";
            for (auto x:tempANumer) {
                std::cout << x << " ";
            }
            std::cout << "\n";
	    std::vector<RealType> tempADenom(16, 0);
            thrust::copy(std::begin(dAccDenominator), std::end(dAccDenominator), std::begin(tempADenom));
            std::cout << "AccDenom: ";
            for (auto x:tempADenom) {
                std::cout << x << " ";
            }
            std::cout << "\n";
	    */
#ifdef CYCLOPS_DEBUG_TIMING
            auto start2 = bsccs::chrono::steady_clock::now();
#endif
			// dense transform reduction
			CudaData.computeGradientAndHessian(dAccNumer,
			        dAccNumer2,
			        dAccDenominator,
			        dNWeight,
			        dGH,
			        N);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end2 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name2 = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + "  transReduce";
            duration[name2] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif
/*
	    double2 temp;
	    cudaMemcpy(pGH, dGH, sizeof(double2), cudaMemcpyDeviceToHost);
	    temp = *pGH;
	    std::cout << "index: " << index << " g: " << temp.x << " h: " << temp.y << '\n';
            
    	    std::vector<RealType> tempXjY(16, 0);
            thrust::copy(std::begin(dXjY), std::end(dXjY), std::begin(tempXjY));
            std::cout << "XjY: ";
            for (auto x:tempXjY) {
                std::cout << x << " ";
            }
            std::cout << "\n";
 
            ////////////////////////// processDelta
#ifdef CYCLOPS_DEBUG_TIMING
            auto start3 = bsccs::chrono::steady_clock::now();
#endif
            CudaData.processDelta(dDeltaVector,
                    dBound,
                    dBeta,
                    dXjY,
                    dGH,
                    dPriorParams,
                    priorTypes,
                    index,
		    gridSize, blockSize);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end3 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name3 = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + " processDelta";
            duration[name3] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end3 - start3).count();
#endif
	    
            double2 temp;
            cudaMemcpy(pGH, dGH, sizeof(double2), cudaMemcpyDeviceToHost);
            temp = *pGH;
	    std::cout << "index: " << index << " g: " << temp.x << " h: " << temp.y << '\n';
	    
	    std::vector<RealType> tempDelta;
	    tempDelta.resize(J);
            thrust::copy(std::begin(dDeltaVector), std::end(dDeltaVector), std::begin(tempDelta));

            std::cout << "index: " << index << " g: " << temp.x << " h: " << temp.y << " delta: " << tempDelta[index] << '\n';
*/
            ////////////////////////// updateXBeta
#ifdef CYCLOPS_DEBUG_TIMING
            auto start4 = bsccs::chrono::steady_clock::now();
#endif
			// sparse transformation
			CudaData.updateXBeta1(dCudaColumns.getData(),
			        dCudaColumns.getIndices(),
			        dCudaColumns.getDataOffset(index),
			        dCudaColumns.getIndicesOffset(index),
			        taskCount,
			        dGH,
				dXjY,
				dBound,
				dBeta,
			        dXBeta,
			        dExpXBeta,
			        dNumerator,
			        dNumerator2,
			        index, formatInt,
			        gridSize, blockSize);
			hXBetaKnown = false;
#ifdef CYCLOPS_DEBUG_TIMING
            auto end4 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name4 = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + "  updateXBeta";
            duration[name4] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start4).count();
#endif
/*	    
        std::vector<RealType> tempEXB(K, 0);
        thrust::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(tempEXB));
        std::cout << "exb: ";
        for (auto x:tempEXB) {
            std::cout << x << " ";
        }
        std::cout << "\n";
*/
#ifdef CYCLOPS_DEBUG_TIMING
         auto start5 = bsccs::chrono::steady_clock::now();
#endif
			// dense scan
			CudaData.CubScan(thrust::raw_pointer_cast(&dExpXBeta[0]), thrust::raw_pointer_cast(&dAccDenominator[0]), K);
#ifdef CYCLOPS_DEBUG_TIMING
            auto end5 = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            auto name5 = "updateBetaAndDelta" + getFormatTypeExtension(formatType) + "     accDenom";
            duration[name5] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end5 - start5).count();
#endif
        }

	virtual const std::vector<double> getXBeta() {

#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
			if (!hXBetaKnown) {
//	    			std::cout << "GPU::getXBeta called \n";
				thrust::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta));
				hXBetaKnown = true;
			}
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z getXBetaG             "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
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
//			std::cerr << "GPU::zXB called" << std::endl;
			ModelSpecifics<BaseModel, RealType>::zeroXBeta(); // touches hXBeta
			dXBetaKnown = false;
        }

        virtual void axpyXBeta(const double beta, const int j) {
//			std::cerr << "GPU::aXB called" << std::endl;
			ModelSpecifics<BaseModel, RealType>::axpyXBeta(beta, j); // touches hXBeta
			dXBetaKnown = false;
        }

	virtual std::vector<double> getBeta() {
#ifdef CYCLOPS_DEBUG_TIMING
            auto start = bsccs::chrono::steady_clock::now();
#endif
//		std::cout << "GPU::getBeta called \n";
		thrust::copy(std::begin(dBeta), std::end(dBeta), std::begin(RealHBeta));
		std::copy(RealHBeta.begin(), RealHBeta.end(), DoubleHBeta.begin());
#ifdef CYCLOPS_DEBUG_TIMING
            auto end = bsccs::chrono::steady_clock::now();
            ///////////////////////////"
            duration["z getBetaG              "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
		return DoubleHBeta;
        }
	
	virtual void resetBeta() {
		std::vector<RealType> tempHBeta;
		tempHBeta.resize(J, 0.0);
		resizeAndCopyToDeviceCuda(tempHBeta, dBeta);
        }

	bool isCUDA() {return true;};
	
	void setBounds(double initialBound) {
		std::vector<RealType> temp;
		temp.resize(J, initialBound);
		resizeAndCopyToDeviceCuda(temp, dBound);
	}
	
	void setPriorTypes(std::vector<int>& inTypes) {
		priorTypes.resize(J);
		for (int i=0; i<J; i++) {
			priorTypes[i] = inTypes[i];
		}
	}
	
	void setPriorParams(std::vector<double>& inParams) {
		std::vector<RealType> temp;
		temp.resize(J, 0.0);
		for (int i=0; i<J; i++) {
			temp[i] = inParams[i];
		}
		resizeAndCopyToDeviceCuda(temp, dPriorParams);
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

        int getFormatTypeInt(FormatType formatType) {
            switch (formatType) {
                case DENSE:
                    return 1;
                case SPARSE:
                    return 2;
                case INDICATOR:
                    return 3;
                case INTERCEPT:
                    return 4;
                default: return 0;
            }
        }
		bool hXBetaKnown;
		bool dXBetaKnown;
/*
		bool double_precision = false;

		std::map<FormatType, std::vector<int>> indicesFormats;
		std::vector<FormatType> formatList;
		std::vector<FormatType> neededFormatTypes;
*/

		std::vector<RealType> priorTypes;
		std::vector<RealType> RealHBeta;
		std::vector<double> DoubleHBeta;

		// device storage
		thrust::device_vector<RealType> dKWeight;
		thrust::device_vector<RealType> dNWeight;

		thrust::device_vector<RealType> dXjY;
		thrust::device_vector<RealType> dBeta;
		thrust::device_vector<RealType> dXBeta;
		thrust::device_vector<RealType> dExpXBeta;
		thrust::device_vector<RealType> dDenominator;
		thrust::device_vector<RealType> dAccDenominator;

		thrust::device_vector<RealType> dNumerator;
		thrust::device_vector<RealType> dNumerator2;

		thrust::device_vector<int> indicesN;
		thrust::device_vector<RealType> dBound;
//		thrust::device_vector<RealType> dDeltaVector;
		thrust::device_vector<RealType> dPriorParams;

		// buffer
		double2 *dGH; // device GH
		double2 *pGH; // host GH
		thrust::device_vector<RealType> dAccNumer;
		thrust::device_vector<RealType> dAccNumer2;
		thrust::device_vector<RealType> dGradient;
		thrust::device_vector<RealType> dHessian;

		thrust::device_vector<RealType> dBuffer1;
		thrust::device_vector<RealType> dBuffer2;
		thrust::device_vector<RealType> dBuffer3;
    };
} // namespace bsccs

//#include "KernelsCox.hpp"

#endif //GPUMODELSPECIFICSCOX_HPP
