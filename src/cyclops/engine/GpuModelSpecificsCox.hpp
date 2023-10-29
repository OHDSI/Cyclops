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
#include "ModelSpecifics.hpp"
#include "Iterators.h"
#include "CudaKernel.h"
#include "CudaDetail.h"

namespace bsccs{
	
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
#ifdef DEBUG_GPU_COX
		std::cerr << "ctor CudaAllGpuColumns" << std::endl;
#endif
	}

	virtual ~CudaAllGpuColumns() {
#ifdef DEBUG_GPU_COX
		std::cerr << "dtor CudaAllGpuColumns" << std::endl;
#endif
	}

	void initialize(const CompressedDataMatrix<RealType>& mat,
			size_t K, bool pad) {
			
//		std::vector<RealType> flatData;
//		std::vector<int> flatIndices;
#ifdef DEBUG_GPU_COX
		std::cerr << "Cuda AGC start" << std::endl;
#endif

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
#ifdef DEBUG_GPU_COX
		std::cerr << "cuda AGC end " << flatData.size() << " " << flatIndices.size() << " " << dataStarts.size() << " " << indicesStarts.size() << " " << taskCounts.size() << std::endl;
#endif
	}

	void resizeAndCopyColumns (cudaStream_t* stream) {
#ifdef DEBUG_GPU_COX
		std::cout << "resizeAndCopyColumns \n";
#endif
		resizeAndCopyToDeviceCuda(flatData, data, stream);
		resizeAndCopyToDeviceCuda(flatIndices, indices, stream);
		resizeAndCopyToDeviceCuda(dataStarts, ddataStarts, stream);
		resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts, stream);
		resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts, stream);
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


template <typename RealType> struct MakePair;
template <> struct MakePair<float>
{
	typedef float2 type;
};
template <> struct MakePair<double>
{
	typedef double2 type;
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
	using ModelSpecifics<BaseModel, RealType>::hPidInternal;
	using ModelSpecifics<BaseModel, RealType>::accReset;
	using ModelSpecifics<BaseModel, RealType>::K;
	using ModelSpecifics<BaseModel, RealType>::J;
	using ModelSpecifics<BaseModel, RealType>::N;
	using ModelSpecifics<BaseModel, RealType>::offsExpXBeta;
	using ModelSpecifics<BaseModel, RealType>::hXBeta;
	using ModelSpecifics<BaseModel, RealType>::hY;
//	using ModelSpecifics<BaseModel, RealType>::hOffs;
	using ModelSpecifics<BaseModel, RealType>::denomPid;
	using ModelSpecifics<BaseModel, RealType>::numerPid;
	using ModelSpecifics<BaseModel, RealType>::numerPid2;
	using ModelSpecifics<BaseModel, RealType>::hNWeight;
	using ModelSpecifics<BaseModel, RealType>::hKWeight;
	using ModelSpecifics<BaseModel, RealType>::hYWeight;
	using ModelSpecifics<BaseModel, RealType>::hYWeightDouble;
	using ModelSpecifics<BaseModel, RealType>::accDenomPid;

	int tpb = 256; // threads-per-block  // Appears best on K40
	int PSC_K = 32;
	int PSC_WG_SIZE = 256;

	typedef typename MakePair<RealType>::type RealType2;

//	CudaAllGpuColumns<RealType> dCudaColumns;
	CudaKernel<RealType, RealType2> CoxKernels;
	CudaAllGpuColumns<RealType> dCudaColumns;

	GpuModelSpecificsCox(const ModelData<RealType>& input,
			const std::string deviceName)
		: ModelSpecifics<BaseModel,RealType>(input),
//		dCudaColumns(),
		dXjY(), dPid(), dY(),
		priorTypes(), RealHBeta(), dBetaBuffer(),
		dNumerator(), dNumerator2(),
		dGH(),
		dBound(), dPriorParams(),
		dBeta(), dXBeta(), dExpXBeta(),
		dDenominator(), dAccDenom(),
		dAccNumer(), dAccNumer2(), dDecDenom(), dDecNumer(), dDecNumer2(),
		dKWeight(), dNWeight(), dYWeight(),
		CoxKernels(deviceName), dCudaColumns(){
#ifdef DEBUG_GPU_COX
		std::cerr << "ctor GpuModelSpecificsCox" << std::endl;
#endif
	}

	virtual ~GpuModelSpecificsCox() {
		cudaFree(dGH);
//		cudaFreeHost(pGH);
#ifdef DEBUG_GPU_COX
		std::cerr << "dtor GpuModelSpecificsCox" << std::endl;
#endif
	}

virtual AbstractModelSpecifics* clone(ComputeDeviceArguments computeDevice) const {
	return new GpuModelSpecificsCox<BaseModel,RealType>(modelData, computeDevice.name);
}

virtual void setPidForAccumulation(const double* weights) {

	if (BaseModel::isScanByKey) {

		// get original stratumId
		ModelSpecifics<BaseModel,RealType>::getOriginalPid();

		// set accReset
		// TODO consider weights here?
		int lastPid = hPidInternal[0];
		for (size_t k = 1; k < K; ++k) {
			int nextPid = hPidInternal[k];
			if (nextPid != lastPid) { // start new strata
				accReset.push_back(k);
				lastPid = nextPid;
			}
		}
		accReset.push_back(K);
#ifdef DEBUG_GPU_COX
		std::cerr << "Num of strata: " << accReset.size() << std::endl;
#endif
		// copy stratumId from host to device
		CoxKernels.resizeAndCopyToDeviceInt(hPidInternal, dPid);
	}

	N = K;
}

virtual void deviceInitialization() {
#ifdef TIME_DEBUG
	std::cerr << "start dI" << std::endl;
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
	// Initialize columns
	dCudaColumns.initialize(hX, K, true);
/*
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
*/
	// Copy columns to device
	dCudaColumns.resizeAndCopyColumns(CoxKernels.getStream());

	// Allocate host storage	    
	RealHBeta.resize(J);

	// Allocate device storage
	CoxKernels.resizeAndCopyToDevice(hY, dY);

//	resizeCudaVec(hXBeta, dXBeta); // K
//	resizeCudaVec(offsExpXBeta, dExpXBeta); // K
//	resizeCudaVec(denomPid, dDenominator); // K

	CoxKernels.resizeAndFillToDevice(dXBeta, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dExpXBeta, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dDenominator, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dNumerator, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dNumerator2, static_cast<RealType>(0.0), K);

	CoxKernels.resizeAndFillToDevice(dAccNumer, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dAccNumer2, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dDecNumer, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dDecNumer2, static_cast<RealType>(0.0), K);
	CoxKernels.resizeAndFillToDevice(dDecDenom, static_cast<RealType>(0.0), K);

	cudaMalloc((void**)&dGH, sizeof(RealType2));
//	cudaMallocHost((void **) &pGH, sizeof(RealType2));

	// Allocate temporary storage for scan and reduction
	if (BaseModel::isTwoWayScan) {
		CoxKernels.allocTempStorageFG(dDenominator,
				dNumerator,
				dNumerator2,
				dAccDenom,
				dAccNumer,
				dAccNumer2,
				dDecDenom,
				dDecNumer,
				dDecNumer2,
				dNWeight,
				dYWeight,
				dY,
				dGH,
				N);
	} else if (BaseModel::isScanByKey) {
		CoxKernels.allocTempStorageByKey(dPid,
				dExpXBeta,
				dNumerator,
				dNumerator2,
				dAccDenom,
				dNWeight,
				dGH,
				N);
	} else {
		CoxKernels.allocTempStorage(dExpXBeta,
				dNumerator,
				dNumerator2,
				dAccDenom,
				dNWeight,
				dGH,
				N);
	}
#ifdef DEBUG_GPU_COX
	std::cout << "K: " << K << " N: " << N << '\n';
#endif

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z cudaDevInit    "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
//        std::cerr << "Format types required: " << need << std::endl;

}

virtual void setWeights(double* inWeights, double *cenWeights, bool useCrossValidation) {
	// Currently only computed on CPU and then copied to GPU
//	ModelSpecifics<BaseModel, RealType>::setWeights(inWeights, useCrossValidation);

	// Host

	// Set K weights
	offCV = 0;
	if (hKWeight.size() != K) {
		hKWeight.resize(K);
	}
	if (useCrossValidation) {
		for (size_t k = 0; k < K; ++k) {
			hKWeight[k] = static_cast<RealType>(inWeights[k]);
		}
		// Find first non-zero weight
		while(inWeights != nullptr && inWeights[offCV] == 0.0 && offCV < K) {
			offCV++;
		}
	} else {
		std::fill(hKWeight.begin(), hKWeight.end(), static_cast<RealType>(1));
	}

	// Set N weights (these are the same for independent data models
	if (hNWeight.size() != K) {
		hNWeight.resize(K);
	}

	std::fill(hNWeight.begin(), hNWeight.end(), static_cast<RealType>(0));
	if (BaseModel::isTwoWayScan) {
		for (size_t k = 0; k < K; ++k) {
			hNWeight[k] = hKWeight[k] * ((hY[k] != static_cast<RealType>(1)) ? static_cast<RealType>(0) : static_cast<RealType>(1));
		}
	} else {
		for (size_t k = 0; k < K; ++k) {
			hNWeight[k] = hY[k] * hKWeight[k];
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
	// Device
	CoxKernels.resizeAndCopyToDevice(hKWeight, dKWeight);
	CoxKernels.resizeAndCopyToDevice(hNWeight, dNWeight);
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
	// Fine-gray
	if (BaseModel::isTwoWayScan) {
		if (hYWeight.size() != K) {
			hYWeight.resize(K);
		}
		if (hYWeightDouble.size() != K) {
			hYWeightDouble.resize(K);
		}
		for (size_t k = 0; k < K; ++k) {
			hYWeight[k] = cenWeights[k];
			hYWeightDouble[k] = cenWeights[k];
		}
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
		// Device
		CoxKernels.resizeAndCopyToDevice(hYWeight, dYWeight);
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
	}

}

virtual void computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
			
	ModelSpecifics<BaseModel,RealType>::computeFixedTermsInGradientAndHessian(useCrossValidation);
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
//	resizeAndCopyToDeviceCuda(hXjY, dXjY);	
	CoxKernels.resizeAndCopyToDevice(hXjY, dXjY);
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
}

virtual void computeRemainingStatistics(bool useWeights) {
			
	hXBetaKnown = true;
	// Currently RS only computed on CPU and then copied
//	ModelSpecifics<BaseModel, RealType>::computeRemainingStatistics(useWeights);

	// Host

	auto& xBeta = getXBeta();
	
	if (denomPid.size() != K) {
		denomPid.resize(K, static_cast<RealType>(0));
	}

	if (accDenomPid.size() != K) {
		accDenomPid.resize(K, static_cast<RealType>(0));
	}

	// Update exb, denom, and accDenom

	RealType totalDenom = static_cast<RealType>(0);
	auto reset = begin(accReset);

	for (size_t k = 0; k < K; ++k) {

		offsExpXBeta[k] = std::exp(xBeta[k]);
		denomPid[k] =  hKWeight[k] * std::exp(xBeta[k]);

		if (BaseModel::isScanByKey) {
			if (static_cast<unsigned int>(*reset) == k) {
				totalDenom = static_cast<RealType>(0);
				++reset;
			}
		}

		totalDenom += denomPid[k];
		accDenomPid[k] = totalDenom;
	}

	// Fine-gray
	if (BaseModel::isTwoWayScan) {
		totalDenom = static_cast<RealType>(0);
		for (int i = (K - 1); i >= 0; i--) {
			totalDenom += (hY[i] > static_cast<RealType>(1)) ? denomPid[i] / hYWeight[i] : 0;
			accDenomPid[i] += (hY[i] == static_cast<RealType>(1)) ? hYWeight[i] * totalDenom : 0;
		}
	}

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
        auto start = bsccs::chrono::steady_clock::now();
#endif
	// Device
	if (dAccDenom.size() != K) {
		CoxKernels.resizeAndCopyToDevice(accDenomPid, dAccDenom);
	} else {
		CoxKernels.copyFromHostToDevice(accDenomPid, dAccDenom);
	}

	CoxKernels.copyFromHostToDevice(hXBeta, dXBeta);
	CoxKernels.copyFromHostToDevice(offsExpXBeta, dExpXBeta);
	CoxKernels.copyFromHostToDevice(denomPid, dDenominator);
//	CoxKernels.copyFromHostToDevice(accDenomPid, dAccDenom);

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
}
		
virtual double getLogLikelihood(bool useCrossValidation) {

	// Device
	// TODO write gpu version to avoid D-H copying
	if (BaseModel::isTwoWayScan) {
		CoxKernels.computeTwoWayAccumlatedDenominator(dDenominator,
				dAccDenom,
				dDecDenom,
				dYWeight,
				dY,
				K);
	} else if (BaseModel::isScanByKey) {
		CoxKernels.computeAccumlatedDenominatorByKey(dPid, dDenominator, dAccDenom, K);
	} else {
		CoxKernels.computeAccumlatedDenominator(dDenominator, dAccDenom, K);
	}
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start0 = bsccs::chrono::steady_clock::now();
#endif
	CoxKernels.copyFromDeviceToHost(dAccDenom, accDenomPid);	
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end0 = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end0 - start0).count();;
#endif
	// Host
	RealType logLikelihood = static_cast<RealType>(0.0);

	for (size_t i = 0; i < K; i++) {
		logLikelihood += hKWeight[i] * ((hY[i] != 1) ? 0 : hXBeta[i]);
	}

	for (size_t i = 0; i < K; i++) {
		logLikelihood -= hNWeight[i] * std::log(accDenomPid[i]);
        }

	return static_cast<double>(logLikelihood);
//	return ModelSpecifics<BaseModel, RealType>::getLogLikelihood(useCrossValidation); 
}
		
virtual double getPredictiveLogLikelihood(double* weights) {
//	std::cout << "GPUMS::getPredictiveLogLikelihood called \n";

	// Save old weights
	std::vector<double> saveKWeight;
	if (saveKWeight.size() != K) {
		saveKWeight.resize(K);
	}
	for (size_t k = 0; k < K; ++k) {
		saveKWeight[k] = hKWeight[k]; // make copy to a double vector
	}

	// Set new weights
//	setPidForAccumulation(weights);
	setWeights(weights, BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true);
	computeRemainingStatistics(true); // compute accDenomPid

	// Compute predictive loglikelihood
	RealType logLikelihood = static_cast<RealType>(0.0);
	for (size_t k = 0; k < K; ++k) {
		logLikelihood += weights[k] == 0.0 ? 0.0
			: hY[k] * weights[k] * (hXBeta[k] - std::log(accDenomPid[k]));
	}
	
	// Set back old weights
//	setPidForAccumulation(&saveKWeight[0]);
	setWeights(saveKWeight.data(), BaseModel::isTwoWayScan ? hYWeightDouble.data() : nullptr, true);
	computeRemainingStatistics(true);
	
	return static_cast<double>(logLikelihood);
}

virtual void setHXBeta() { // for confint
	CoxKernels.copyFromDeviceToHost(dXBeta, hXBeta);
	hXBetaKnown = true;
}

virtual void updateXBeta(double delta, int index, bool useWeights) { // for confint
	FormatType formatType = hX.getFormatType(index);
	const auto taskCount = dCudaColumns.getTaskCount(index);
	
	int gridSize, blockSize;
	blockSize = 256;
	gridSize = (int)ceil((double)taskCount/blockSize);
	
	CoxKernels.updateXBeta(dCudaColumns.getData(),
			dCudaColumns.getIndices(),
			dCudaColumns.getDataOffset(index),
			dCudaColumns.getIndicesOffset(index),
			taskCount,
			static_cast<RealType>(delta),
			dKWeight,
			dBeta,
			dBetaBuffer,
			dXBeta,
			dExpXBeta,
			dDenominator,
			dAccDenom,
			dNumerator,
			dNumerator2,
			index,
			K,
			formatType,
			gridSize, blockSize);
	hXBetaKnown = false;
}

virtual void updateBetaAndDelta(int index, bool useWeights) {

	FormatType formatType = hX.getFormatType(index);
	const auto taskCount = dCudaColumns.getTaskCount(index);

	int gridSize, blockSize;
	blockSize = 256;
	gridSize = (int)ceil((double)taskCount/blockSize);


	////////////////////////// computeGradientAndHessian
#ifdef CYCLOPS_DEBUG_TIMING
	auto start = bsccs::chrono::steady_clock::now();
#endif
	// sparse transformation
	CoxKernels.computeNumeratorForGradient(dCudaColumns.getData(),
					dCudaColumns.getIndices(),
					dCudaColumns.getDataOffset(index),
					dCudaColumns.getIndicesOffset(index),
					taskCount,
					dKWeight,
					dExpXBeta,
					dNumerator,
					dNumerator2,
					formatType,
					gridSize, blockSize);
#ifdef CYCLOPS_DEBUG_TIMING
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compNumForGradG  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif


#ifdef CYCLOPS_DEBUG_TIMING
	auto start2 = bsccs::chrono::steady_clock::now();
#endif
	
	if (BaseModel::isTwoWayScan) {

		CoxKernels.computeTwoWayGradientAndHessian(dNumerator,
				dNumerator2,
				dDenominator,
				dAccNumer,
				dAccNumer2,
				dAccDenom,
				dDecNumer,
				dDecNumer2,
				dDecDenom,
				dNWeight,
				dYWeight,
				dY,
				dGH,
				formatType,
				offCV,
				K);
	} else if (BaseModel::isScanByKey) {

		CoxKernels.computeGradientAndHessianByKey(dPid,
				dNumerator,
				dNumerator2,
				dDenominator,
				dAccDenom,
				dNWeight,
				dGH,
				formatType,
				offCV,
				K);

	} else {

	// dense scan
	CoxKernels.computeAccumlatedDenominator(dDenominator, dAccDenom, K);

	// dense scan with transform reduction
	CoxKernels.computeGradientAndHessian(dNumerator,
					dNumerator2,
					dAccDenom,
					dNWeight,
					dGH,
					formatType,
					offCV,
					K);
/*
	// dense scan with transform reduction (including Denom -> accDenom)
	CoxKernels.computeGradientAndHessian1(dNumerator,
			dNumerator2,
			dDenominator,
			dAccDenom,
			dNWeight,
			dGH,
			formatType,
			offCV,
			K);
*/
	}

#ifdef CYCLOPS_DEBUG_TIMING
	auto end2 = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compGradAndHessG "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif

/*
#ifdef CYCLOPS_DEBUG_TIMING
	auto start2 = bsccs::chrono::steady_clock::now();
#endif
	// dense scan with transform reduction (including Denom -> accDenom)
	CoxKernels.computeGradientAndHessian1(dNumerator,
					dNumerator2,
					dDenominator,
					dAccDenom,
					dNWeight,
					dGH,
					formatType,
					offCV,
					K);
#ifdef CYCLOPS_DEBUG_TIMING
	auto end2 = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["compGradAndHessG "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end2 - start2).count();
#endif
*/	

	////////////////////////// updateXBetaAndDelta
#ifdef CYCLOPS_DEBUG_TIMING
	auto start4 = bsccs::chrono::steady_clock::now();
#endif
	// sparse transformation
	CoxKernels.updateXBetaAndDelta(dCudaColumns.getData(),
				dCudaColumns.getIndices(),
				dCudaColumns.getDataOffset(index),
				dCudaColumns.getIndicesOffset(index),
				taskCount,
				dGH,
				dXjY,
				dBound,
				dBoundBuffer,
				dKWeight,
				dBeta,
				dBetaBuffer,
				dXBeta,
				dExpXBeta,
				dDenominator,
				dNumerator,
				dNumerator2,
				dPriorParams,
				getPriorTypes(index),
				index, 
				formatType,
				gridSize, blockSize);
	hXBetaKnown = false;
#ifdef CYCLOPS_DEBUG_TIMING
	auto end4 = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["updateXBetaG     "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end4 - start4).count();
#endif
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	duration["GPU GH           "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start + end2 - start2 + end4 - start4).count();
#endif
}

virtual const std::vector<double> getXBeta() {

#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
	if (!hXBetaKnown) {
		CoxKernels.copyFromDeviceToHost(dXBeta, hXBeta);
		hXBetaKnown = true;
	}
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
	return ModelSpecifics<BaseModel, RealType>::getXBeta();
}
		
virtual const std::vector<double> getXBetaSave() {
	return ModelSpecifics<BaseModel, RealType>::getXBetaSave();
}

virtual void saveXBeta() {
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
	if (!hXBetaKnown) {
		CoxKernels.copyFromDeviceToHost(dXBeta, hXBeta);
		hXBetaKnown = true;
	}
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
	ModelSpecifics<BaseModel, RealType>::saveXBeta();
}

virtual void zeroXBeta() {
//	std::cerr << "GPU::zXB called" << std::endl;
	ModelSpecifics<BaseModel, RealType>::zeroXBeta(); // touches hXBeta
	dXBetaKnown = false;
}

virtual void axpyXBeta(const double beta, const int j) {
//	std::cerr << "GPU::aXB called" << std::endl;
	ModelSpecifics<BaseModel, RealType>::axpyXBeta(beta, j); // touches hXBeta
	dXBetaKnown = false;
}

virtual std::vector<double> getBeta() {
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto start = bsccs::chrono::steady_clock::now();
#endif
	CoxKernels.copyFromDeviceToDevice(dBound, dBoundBuffer);
	CoxKernels.copyFromDeviceToDevice(dBeta, dBetaBuffer);
	CoxKernels.copyFromDeviceToHost(dBeta, RealHBeta);
//	CoxKernels.getBeta(RealHBeta);
//	CoxKernels.getBound();
#ifdef CYCLOPS_DEBUG_TIMING_GPU_COX
	auto end = bsccs::chrono::steady_clock::now();
	///////////////////////////"
	duration["z Data transfer  "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
	return std::vector<double>(std::begin(RealHBeta), std::end(RealHBeta));
}
		
virtual void resetBeta() {
	CoxKernels.resizeAndFillToDevice(dBeta, static_cast<RealType>(0.0), J);
	CoxKernels.resizeAndFillToDevice(dBetaBuffer, static_cast<RealType>(0.0), J);
//	CoxKernels.resetBeta(dBeta, dBetaBuffer, J);
}

bool isCUDA() {return true;};
	
void setBounds(double initialBound) {
	CoxKernels.resizeAndFillToDevice(dBound, static_cast<RealType>(initialBound), J);
	CoxKernels.resizeAndFillToDevice(dBoundBuffer, static_cast<RealType>(initialBound), J);
//	CoxKernels.setBounds(dBound, dBoundBuffer, static_cast<RealType>(initialBound), J);
}
	
const int getPriorTypes(int index) const {
	return priorTypes[index];
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
		temp[i] = static_cast<RealType>(inParams[i]);
	}
	CoxKernels.resizeAndCopyToDevice(temp, dPriorParams);
}

void turnOnStreamCV(int foldToCompute) {
	streamCV = true;
	streamCVFolds = foldToCompute;
	CoxKernels.allocStreams(streamCVFolds);
#ifdef DEBUG_GPU_COX
	std::cout << "GPUMS streamCVFolds: " << streamCVFolds << '\n';
#endif
}

void setFold(int inFold){
	fold = inFold;
	CoxKernels.setFold(inFold);
#ifdef DEBUG_GPU_COX
	std::cout << "GPUMS current fold: " << fold << '\n';
#endif
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
	
//	const std::string currentDevice = CoxKernels.getDeviceName();
	const std::string currentDevice;
	bool streamCV;
	int streamCVFolds;
	int fold;

	bool hXBetaKnown;
	bool dXBetaKnown;
/*
	bool double_precision = false;

	std::map<FormatType, std::vector<int>> indicesFormats;
	std::vector<FormatType> formatList;
	std::vector<FormatType> neededFormatTypes;
*/

	size_t offCV;
	std::vector<int> priorTypes;
	std::vector<RealType> RealHBeta;

	thrust::device_vector<int> dPid;

	// device storage
	thrust::device_vector<RealType> dKWeight;
	thrust::device_vector<RealType> dNWeight;
	thrust::device_vector<RealType> dYWeight;

	thrust::device_vector<RealType> dXjY;
	thrust::device_vector<RealType> dY;
	thrust::device_vector<RealType> dBeta;
	thrust::device_vector<RealType> dBetaBuffer;
	thrust::device_vector<RealType> dXBeta;
	thrust::device_vector<RealType> dExpXBeta;
	thrust::device_vector<RealType> dDenominator;
	thrust::device_vector<RealType> dAccDenom;

	thrust::device_vector<RealType> dNumerator;
	thrust::device_vector<RealType> dNumerator2;

	thrust::device_vector<RealType> dAccNumer;
	thrust::device_vector<RealType> dAccNumer2;
	thrust::device_vector<RealType> dDecNumer;
	thrust::device_vector<RealType> dDecNumer2;
	thrust::device_vector<RealType> dDecDenom;

	thrust::device_vector<RealType> dBound;
	thrust::device_vector<RealType> dBoundBuffer;
	thrust::device_vector<RealType> dPriorParams;

	RealType2 *dGH; // device GH
//	RealType2 *pGH; // host GH

}; // GpuModelSpecificsCox
} // namespace bsccs

//#include "KernelsCox.hpp"

#endif //GPUMODELSPECIFICSCOX_HPP
