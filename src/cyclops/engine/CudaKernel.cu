#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector_types.h>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/for_each.h>

#include "CudaKernel.h"

using namespace cub;


template <typename RealType, FormatTypeCuda formatType>
__global__ void kernelComputeNumeratorForGradient(int offX,
						  int offK,
						  const int taskCount,
						  const RealType* d_X,
						  const int* d_K,
						  RealType* d_KWeight,
						  RealType* d_ExpXBeta,
						  RealType* d_Numerator,
						  RealType* d_Numerator2)
{
	int task = blockIdx.x * blockDim.x + threadIdx.x;

	int k;
	if (formatType == INDICATOR || formatType == SPARSE) {
		k = d_K[offK + task];
	} else { // DENSE, INTERCEPT
		k = task;
	}

	if (task < taskCount) {
		if (formatType == SPARSE || formatType == DENSE) {
			d_Numerator[k] = d_X[offX + task] * d_ExpXBeta[k] * d_KWeight[k];
			d_Numerator2[k] = d_X[offX + task] * d_Numerator[k];
		} else { // INDICATOR, INTERCEPT
			d_Numerator[k] = d_ExpXBeta[k] * d_KWeight[k];
		}
	}
}

template <typename RealType, FormatTypeCuda formatType>
__global__ void kernelUpdateXBeta(int offX,
				  int offK,
				  const int taskCount,
				  RealType delta,
				  const RealType* d_X,
				  const int* d_K,
				  RealType* d_KWeight,
				  RealType* d_XBeta,
				  RealType* d_ExpXBeta,
				  RealType* d_Denominator,
				  RealType* d_Numerator,
				  RealType* d_Numerator2)
{
	// update xb, exb, and denom if needed
	// zero numer and numer2

	int task = blockIdx.x * blockDim.x + threadIdx.x;

	int k;
	if (formatType == INDICATOR || formatType == SPARSE) {
		k = d_K[offK + task];
	} else { // DENSE, INTERCEPT
		k = task;
	}

	if (delta != 0.0) { // update xb and exb, zero numer

		RealType inc;
		if (formatType == SPARSE || formatType == DENSE) {
			inc = delta * d_X[offX + task];
		} else { // INDICATOR, INTERCEPT
			inc = delta;
		}

		if (task < taskCount) {
			RealType xb = d_XBeta[k] + inc;
			d_XBeta[k] = xb;
			d_ExpXBeta[k] = exp(xb);
			d_Denominator[k] = exp(xb) * d_KWeight[k];
			d_Numerator[k] = 0;
			if (formatType != INDICATOR) {
				d_Numerator2[k] = 0;
			}
		}

	} else { // only zero numer

		if (task < taskCount) {
			d_Numerator[k] = 0;
			if (formatType != INDICATOR) {
				d_Numerator2[k] = 0;
			}
		}
	}
}

template <typename RealType, typename RealType2, FormatTypeCuda formatType, PriorTypeCuda priorType>
__global__ void kernelUpdateXBetaAndDelta(int offX,
					  int offK,
					  const int taskCount,
					  int index,
					  const RealType* d_X,
					  const int* d_K,
					  RealType2* d_GH,
					  RealType* d_XjY,
					  RealType* d_Bound,
					  RealType* d_BoundBuffer,
					  RealType* d_KWeight,
					  RealType* d_Beta,
					  RealType* d_BetaBuffer,
					  RealType* d_XBeta,
					  RealType* d_ExpXBeta,
					  RealType* d_Denominator,
					  RealType* d_Numerator,
					  RealType* d_Numerator2,
					  RealType* d_PriorParams)
{
	// get gradient, hessian, and old beta
	RealType2 GH = *d_GH;
	RealType g = GH.x - d_XjY[index];
	RealType h = GH.y;
	if (h < 0.0) {
		g = 0.0;
		h = 0.0;
	}
	RealType beta = d_BetaBuffer[index];

	// process delta
	RealType delta;
	if (priorType == NOPRIOR) {
		delta = -g/h;
	}
	if (priorType == LAPLACE) {
		RealType lambda = d_PriorParams[index];
		RealType neg_update = - (g - lambda) / h;
		RealType pos_update = - (g + lambda) / h;
		if (beta == 0) {
			if (neg_update < 0) {
				delta = neg_update;
			} else if (pos_update > 0) {
				delta = pos_update;
			} else {
				delta = 0;
			}
		} else {
			if (beta < 0) {
				delta = neg_update;
				if (beta+delta > 0) delta = -beta;
			} else {
				delta = pos_update;
				if (beta+delta < 0) delta = -beta;
			}
		}
	}
	if (priorType == NORMAL) {
		RealType variance = d_PriorParams[index];
		delta = - (g + (beta / variance)) / (h + (1.0 / variance));
	}

	// update beta
	RealType bound = d_BoundBuffer[index];
	if (delta < -bound) {
		delta = -bound;
	} else if (delta > bound) {
		delta = bound;
	}
	d_Beta[index] = delta + beta;

	// update bound
	auto intermediate = max(2*abs(delta), bound/2);
	intermediate = max(intermediate, 0.001);
	d_Bound[index] = intermediate;


	// update xb, exb, and denom if needed
	// zero numer and numer2

	int task = blockIdx.x * blockDim.x + threadIdx.x;
                
	int k;
	if (formatType == INDICATOR || formatType == SPARSE) {
		k = d_K[offK + task];
	} else { // DENSE, INTERCEPT
		k = task;
	}

	if (delta != 0.0) { // update xb and exb, zero numer
		
		RealType inc;
		if (formatType == SPARSE || formatType == DENSE) {
			inc = delta * d_X[offX + task];
		} else { // INDICATOR, INTERCEPT
			inc = delta;
		}
		
		if (task < taskCount) {
			RealType xb = d_XBeta[k] + inc;
			d_XBeta[k] = xb;
			d_ExpXBeta[k] = exp(xb);
			d_Denominator[k] = exp(xb) * d_KWeight[k];
			d_Numerator[k] = 0;
			if (formatType != INDICATOR) {
				d_Numerator2[k] = 0;
			}
		}

	} else { // only zero numer

		if (task < taskCount) {
			d_Numerator[k] = 0;
			if (formatType != INDICATOR) {
				d_Numerator2[k] = 0;
			}
		}
	}
}


struct TuplePlus
{
	template<typename L, typename R>
	__host__ __device__
	thrust::tuple<L, L> operator()(thrust::tuple<L, L>& lhs, thrust::tuple<R, R>& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs), 
					  thrust::get<1>(lhs) + thrust::get<1>(rhs));
	}
};

struct TuplePlus3
{
	template<typename L, typename R>
	__host__ __device__
	thrust::tuple<L, L, L> operator()(thrust::tuple<L, L, L>& lhs, thrust::tuple<R, R, R>& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
					  thrust::get<1>(lhs) + thrust::get<1>(rhs),
					  thrust::get<2>(lhs) + thrust::get<2>(rhs));
	}
};

struct TuplePlus6
{
	template<typename L, typename R>
	__host__ __device__
	thrust::tuple<L, L, L, L, L, L> operator()(thrust::tuple<L, L, L, L, L, L>& lhs, thrust::tuple<R, R, R, R, R, R>& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
				thrust::get<1>(lhs) + thrust::get<1>(rhs),
				thrust::get<2>(lhs) + thrust::get<2>(rhs),
				thrust::get<3>(lhs) + thrust::get<3>(rhs),
				thrust::get<4>(lhs) + thrust::get<4>(rhs),
				thrust::get<5>(lhs) + thrust::get<5>(rhs));
	}
};

struct Double2Plus
{
	__host__ __device__
	double2 operator()(double2& a, double2& b)
	{
		double2 out;
		out.x = a.x + b.x;
		out.y = a.y + b.y;
		return out;
	}
};

struct Float2Plus
{
	__host__ __device__
	float2 operator()(float2& a, float2& b)
	{
		float2 out;
		out.x = a.x + b.x;
		out.y = a.y + b.y;
		return out;
	}
};

struct RealType2Plus
{
	template<typename RealType2>
	__host__ __device__
	RealType2 operator()(RealType2& a, RealType2& b)
	{
		RealType2 out;
		out.x = a.x + b.x;
		out.y = a.y + b.y;
		return out;
	}
};


template <typename RealType, typename RealType2>
CudaKernel<RealType, RealType2>::CudaKernel(const std::string& deviceName)
{
	// set device
//	cudaSetDevice(1);
	bool success = false;	
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);
	desiredDeviceName = deviceName;
	for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, deviceIndex);
		if (deviceProperties.name == desiredDeviceName) {
			cudaSetDevice(deviceIndex);
			success = true;
		}
	}

	// create stream
	stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
	cudaStreamCreate(&stream[0]);
	
	if (success) {
		std::cout << "ctor CudaKernel on " << deviceName << '\n';
	} else {
		std::cout << "ctor CudaKernel on default device \n";
	}
}

template <typename RealType, typename RealType2>
CudaKernel<RealType, RealType2>::~CudaKernel()
{
	cudaStreamDestroy(stream[0]);
	free(stream);
	cudaFree(d_temp_storage_accd); // accDenom
	cudaFree(d_temp_storage_gh); // cGAH
	cudaFree(d_temp_storage_faccd); // FG: two-way scan for accDenom
	cudaFree(d_temp_storage_fs); // FG: forward scans
	cudaFree(d_temp_storage_bs); // FG: backward scans
	cudaFree(d_temp_storage_fgh); // FG: cGAH

//	cudaDeviceReset();
	std::cout << "dtor CudaKernel \n";
}


template <typename RealType, typename RealType2>
const std::string CudaKernel<RealType, RealType2>::getDeviceName() {
	std::cout << "getDeviceName: " << desiredDeviceName << '\n';
	return desiredDeviceName;
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::allocStreams(int streamCVFolds)
{
	CVFolds = streamCVFolds;
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::setFold(int currentFold)
{
	fold = currentFold;
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::resizeAndCopyToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec)
{
	deviceVec.resize(hostVec.size());
	cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVec.data()),
			thrust::raw_pointer_cast(hostVec.data()),
			deviceVec.size()*sizeof(RealType),
			cudaMemcpyHostToDevice, stream[0]);
	cudaStreamSynchronize(stream[0]);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::resizeAndFillToDevice(thrust::device_vector<RealType>& deviceVec, RealType val, int num_items)
{
	deviceVec.resize(num_items);
	thrust::fill(deviceVec.begin(), deviceVec.end(), val);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::copyFromHostToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec)
{
	cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVec.data()),
			thrust::raw_pointer_cast(hostVec.data()),
			deviceVec.size()*sizeof(RealType),
			cudaMemcpyHostToDevice, stream[0]);
	cudaStreamSynchronize(stream[0]);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::copyFromDeviceToHost(const thrust::device_vector<RealType>& deviceVec, std::vector<RealType>& hostVec)
{
	cudaMemcpyAsync(thrust::raw_pointer_cast(hostVec.data()),
			thrust::raw_pointer_cast(deviceVec.data()),
			deviceVec.size()*sizeof(RealType),
			cudaMemcpyDeviceToHost, stream[0]);
	cudaStreamSynchronize(stream[0]);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::copyFromDeviceToDevice(const thrust::device_vector<RealType>& source, thrust::device_vector<RealType>& destination)
{
	thrust::copy(thrust::cuda::par.on(stream[0]), source.begin(), source.end(), destination.begin());
	cudaStreamSynchronize(stream[0]);
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
						       thrust::device_vector<RealType>& d_Numerator,
						       thrust::device_vector<RealType>& d_Numerator2,
						       thrust::device_vector<RealType>& d_AccDenom,
						       thrust::device_vector<RealType>& d_NWeight,
						       RealType2* d_GH,
						       size_t& N)
{
	d_init.x = d_init.y = 0.0;

	// for scan in accDenom
	DeviceScan::InclusiveSum(d_temp_storage_accd, temp_storage_bytes_accd, &d_Denominator[0], &d_AccDenom[0], N, stream[0]);
	cudaMalloc(&d_temp_storage_accd, temp_storage_bytes_accd);

	// for fused scan reduction (double scan)
	auto begin0 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(), 
								   d_Numerator2.begin()));
	auto begin1 = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin(), 
								   d_NWeight.begin()));
	DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
			begin0, begin1, d_GH,
			TuplePlus(), RealType2Plus(), compGradHessInd, N, stream[0]);

/*
	auto begin2 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(),
								   d_Numerator2.begin(),
								   d_Denominator.begin()));
	
	// triple scan without storing accDenom
	DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
			begin2, thrust::raw_pointer_cast(&d_NWeight[0]), d_GH,
			TuplePlus3(), RealType2Plus(), compGradHessInd1, N, stream[0]);
*/
	cudaMalloc(&d_temp_storage_gh, temp_storage_bytes_gh);

	cudaStreamSynchronize(stream[0]);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::allocTempStorageFG(thrust::device_vector<RealType>& d_Denominator,
		thrust::device_vector<RealType>& d_Numerator,
		thrust::device_vector<RealType>& d_Numerator2,
		thrust::device_vector<RealType>& d_AccDenom,
		thrust::device_vector<RealType>& d_AccNumer,
		thrust::device_vector<RealType>& d_AccNumer2,
		thrust::device_vector<RealType>& d_DecDenom,
		thrust::device_vector<RealType>& d_DecNumer,
		thrust::device_vector<RealType>& d_DecNumer2,
		thrust::device_vector<RealType>& d_NWeight,
		thrust::device_vector<RealType>& d_YWeight,
		thrust::device_vector<RealType>& d_Y,
		RealType2* d_GH,
		size_t& N)
{
	// two-way scan for accDenom
	auto begin_denoms = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.begin(),
				d_Denominator.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto begin_accDenoms = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(),
				d_DecDenom.rbegin()));
	
	TransformInputIterator<Tup2, TwoWayScan<RealType>, NRZipVec4> itr_scan(begin_denoms, twoWayScan);
	DeviceScan::InclusiveScan(d_temp_storage_faccd, temp_storage_bytes_faccd,
			itr_scan, begin_accDenoms,
			TuplePlus(), N, stream[0]);
	cudaMalloc(&d_temp_storage_faccd, temp_storage_bytes_faccd);
	
	// forward scans
	auto begin_forward = thrust::make_zip_iterator(thrust::make_tuple(
				d_Numerator.begin(),
				d_Numerator2.begin(),
				d_Denominator.begin()));
	auto begin_acc = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccNumer.begin(),
				d_AccNumer2.begin(),
				d_AccDenom.begin()));
	DeviceScan::InclusiveScan(d_temp_storage_fs, temp_storage_bytes_fs,
			begin_forward, begin_acc,
			TuplePlus3(), N, stream[0]);
	cudaMalloc(&d_temp_storage_fs, temp_storage_bytes_fs);
	
	// backward scans
	auto rbegin_backward = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.rbegin(),
				d_Numerator.rbegin(),
				d_Numerator2.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto rbegin_dec = thrust::make_zip_iterator(thrust::make_tuple(
				d_DecDenom.rbegin(),
				d_DecNumer.rbegin(),
				d_DecNumer2.rbegin()));
	TransformInputIterator<Tup3, BackwardScans<RealType>, RZipVec5> itr_scans(rbegin_backward, backwardScans);
	DeviceScan::InclusiveScan(d_temp_storage_bs, temp_storage_bytes_bs, 
			itr_scans, rbegin_dec, 
			TuplePlus3(), N, stream[0]);
	cudaMalloc(&d_temp_storage_bs, temp_storage_bytes_bs);
/*
	// two-way scans
	auto in_2way = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.begin(),
				d_Denominator.rbegin(),
				d_Numerator.begin(),
				d_Numerator.rbegin(),
				d_Numerator2.begin(),
				d_Numerator2.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto out_2way = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(),
				d_DecDenom.rbegin(),
				d_AccNumer.begin(),
				d_DecNumer.rbegin(),
				d_AccNumer2.begin(),
				d_DecNumer2.rbegin()));
	TransformInputIterator<Tup6, TwoWayScans<RealType>, NRZipVec8> itr_scans(in_2way, twoWayScans);
	DeviceScan::InclusiveScan(d_temp_storage_bs, temp_storage_bytes_bs,
			itr_scans, out_2way,
			TuplePlus6(), N, stream[0]);
	cudaMalloc(&d_temp_storage_bs, temp_storage_bytes_bs);
*/
	// transform reduction
	d_init.x = d_init.y = 0.0;
	auto begin_gh = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(),
				d_AccNumer.begin(),
				d_AccNumer2.begin(),
				d_DecDenom.begin(),
				d_DecNumer.begin(),
				d_DecNumer2.begin(),
				d_NWeight.begin(),
				d_Y.begin(),
				d_YWeight.begin()));
	TransformInputIterator<RealType2, TwoWayReduce<RealType, RealType2, true>, ZipVec9> itr_gh(begin_gh, fineGrayInd);
	DeviceReduce::Reduce(d_temp_storage_fgh, temp_storage_bytes_fgh,
			itr_gh, d_GH,
			N, RealType2Plus(), d_init, stream[0]);
	cudaMalloc(&d_temp_storage_fgh, temp_storage_bytes_fgh);
	
	cudaStreamSynchronize(stream[0]);
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeNumeratorForGradient(const thrust::device_vector<RealType>& d_X,
							  	  const thrust::device_vector<int>& d_K,
								  unsigned int offX,
								  unsigned int offK,
								  const unsigned int taskCount,
								  thrust::device_vector<RealType>& d_KWeight,
								  thrust::device_vector<RealType>& d_ExpXBeta,
								  thrust::device_vector<RealType>& d_Numerator,
								  thrust::device_vector<RealType>& d_Numerator2,
								  FormatType& formatType,
								  int gridSize, int blockSize)
{
        switch (formatType) {
                case DENSE :
                        kernelComputeNumeratorForGradient<RealType, DENSE><<<gridSize, blockSize, 0, stream[0]>>>(offX,
                                                                                               offK,
                                                                                               taskCount,
                                                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator2[0]));
                break;	
                case SPARSE :
                        kernelComputeNumeratorForGradient<RealType, SPARSE><<<gridSize, blockSize, 0, stream[0]>>>(offX,
                                                                                               offK,
                                                                                               taskCount,
                                                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator2[0]));
                break;
                case INDICATOR :
                        kernelComputeNumeratorForGradient<RealType, INDICATOR><<<gridSize, blockSize, 0, stream[0]>>>(offX,
                                                                                               offK,
                                                                                               taskCount,
                                                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator2[0]));
                break;
                case INTERCEPT :
                        kernelComputeNumeratorForGradient<RealType, INTERCEPT><<<gridSize, blockSize, 0, stream[0]>>>(offX,
                                                                                               offK,
                                                                                               taskCount,
                                                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                                                               thrust::raw_pointer_cast(&d_Numerator2[0]));
                break;
	}
	
	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeGradientAndHessian(thrust::device_vector<RealType>& d_Numerator,
								thrust::device_vector<RealType>& d_Numerator2,
								thrust::device_vector<RealType>& d_AccDenom,
								thrust::device_vector<RealType>& d_NWeight,
								RealType2* d_GH,
								FormatType& formatType,
								size_t& offCV,
								size_t& N)
{
	// fused scan reduction
	auto begin0 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin() + offCV, 
								   d_Numerator2.begin() + offCV));
	auto begin1 = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin() + offCV, 
								   d_NWeight.begin() + offCV));
	if (formatType == INDICATOR) {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
				begin0, begin1, d_GH,
				TuplePlus(), RealType2Plus(), compGradHessInd, N - offCV, stream[0]);
	} else {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
				begin0, begin1, d_GH,
				TuplePlus(), RealType2Plus(), compGradHessNInd, N - offCV, stream[0]);
	}

	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeGradientAndHessian1(thrust::device_vector<RealType>& d_Numerator,
								 thrust::device_vector<RealType>& d_Numerator2,
								 thrust::device_vector<RealType>& d_Denominator,
								 thrust::device_vector<RealType>& d_AccDenom,
								 thrust::device_vector<RealType>& d_NWeight,
								 RealType2* d_GH,
								 FormatType& formatType,
								 size_t& offCV,
								 size_t& N)
{
	// fused scan reduction (including Denom -> accDenom)
	auto begin2 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin() + offCV,
								   d_Numerator2.begin() + offCV,
								   d_Denominator.begin() + offCV));
	if (formatType == INDICATOR) {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh,
				begin2, thrust::raw_pointer_cast(&d_NWeight[0]), d_GH,
				TuplePlus3(), RealType2Plus(), compGradHessInd1, N - offCV, stream[0]);
	} else {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh,
				begin2, thrust::raw_pointer_cast(&d_NWeight[0]), d_GH,
				TuplePlus3(), RealType2Plus(), compGradHessNInd1, N - offCV, stream[0]);
	}

	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeTwoWayGradientAndHessian(thrust::device_vector<RealType>& d_Numerator,
								thrust::device_vector<RealType>& d_Numerator2,
								thrust::device_vector<RealType>& d_Denominator,
								thrust::device_vector<RealType>& d_AccNumer,
								thrust::device_vector<RealType>& d_AccNumer2,
								thrust::device_vector<RealType>& d_AccDenom,
								thrust::device_vector<RealType>& d_DecNumer,
								thrust::device_vector<RealType>& d_DecNumer2,
								thrust::device_vector<RealType>& d_DecDenom,
								thrust::device_vector<RealType>& d_NWeight,
								thrust::device_vector<RealType>& d_YWeight,
								thrust::device_vector<RealType>& d_Y,
								RealType2* d_GH,
								FormatType& formatType,
								size_t& offCV,
								size_t& N)
{

	// forward scan
	auto begin_forward = thrust::make_zip_iterator(thrust::make_tuple(
				d_Numerator.begin(), 
				d_Numerator2.begin(), 
				d_Denominator.begin()));
	auto begin_acc = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccNumer.begin(), 
				d_AccNumer2.begin(), 
				d_AccDenom.begin()));
	DeviceScan::InclusiveScan(d_temp_storage_fs, temp_storage_bytes_fs,
			begin_forward, begin_acc,
			TuplePlus3(), N, stream[0]);
//	cudaStreamSynchronize(stream[0]);
	
	// backward scan
	auto rbegin_backward = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.rbegin(),
				d_Numerator.rbegin(),
				d_Numerator2.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto rbegin_dec = thrust::make_zip_iterator(thrust::make_tuple(
				d_DecDenom.rbegin(),
				d_DecNumer.rbegin(),
				d_DecNumer2.rbegin()));
	TransformInputIterator<Tup3, BackwardScans<RealType>, RZipVec5> itr_scans(rbegin_backward, backwardScans);
	DeviceScan::InclusiveScan(d_temp_storage_bs, temp_storage_bytes_bs, 
			itr_scans, rbegin_dec, 
			TuplePlus3(), N, stream[0]);
/*
	// two-way scans
	auto in_2way = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.begin(),
				d_Denominator.rbegin(),
				d_Numerator.begin(),
				d_Numerator.rbegin(),
				d_Numerator2.begin(),
				d_Numerator2.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto out_2way = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(),
				d_DecDenom.rbegin(),
				d_AccNumer.begin(),
				d_DecNumer.rbegin(),
				d_AccNumer2.begin(),
				d_DecNumer2.rbegin()));
	TransformInputIterator<Tup6, TwoWayScans<RealType>, NRZipVec8> itr_scans(in_2way, twoWayScans);
	DeviceScan::InclusiveScan(d_temp_storage_bs, temp_storage_bytes_bs,
			itr_scans, out_2way,
			TuplePlus6(), N, stream[0]);
*/
	cudaStreamSynchronize(stream[0]);

	// transform reduction
	auto begin_gh = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(),
				d_AccNumer.begin(),
				d_AccNumer2.begin(),
				d_DecDenom.begin(),
				d_DecNumer.begin(),
				d_DecNumer2.begin(),
				d_NWeight.begin(),
				d_Y.begin(),
				d_YWeight.begin()));
	if (formatType == INDICATOR) {
		TransformInputIterator<RealType2, TwoWayReduce<RealType, RealType2, true>, ZipVec9> itr_gh(begin_gh, fineGrayInd);
		DeviceReduce::Reduce(d_temp_storage_fgh, temp_storage_bytes_fgh,
				itr_gh, d_GH,
				N, RealType2Plus(), d_init, stream[0]);
	} else {
		TransformInputIterator<RealType2, TwoWayReduce<RealType, RealType2, false>, ZipVec9> itr_gh(begin_gh, fineGrayNInd);
		DeviceReduce::Reduce(d_temp_storage_fgh, temp_storage_bytes_fgh,
				itr_gh, d_GH,
				N, RealType2Plus(), d_init, stream[0]);
	}

	cudaStreamSynchronize(stream[0]);
}


template <typename RealType, typename RealType2, FormatTypeCuda formatType>
void dispatchPriorType(const thrust::device_vector<RealType>& d_X,
		       const thrust::device_vector<int>& d_K,
		       unsigned int offX,
		       unsigned int offK,
		       const unsigned int taskCount,
		       RealType2* d_GH,
		       thrust::device_vector<RealType>& d_XjY,
		       thrust::device_vector<RealType>& d_Bound,
		       thrust::device_vector<RealType>& d_BoundBuffer,
		       thrust::device_vector<RealType>& d_KWeight,
		       thrust::device_vector<RealType>& d_Beta,
		       thrust::device_vector<RealType>& d_BetaBuffer,			
		       thrust::device_vector<RealType>& d_XBeta,
		       thrust::device_vector<RealType>& d_ExpXBeta,
		       thrust::device_vector<RealType>& d_Denominator,
		       thrust::device_vector<RealType>& d_Numerator,
		       thrust::device_vector<RealType>& d_Numerator2,
		       thrust::device_vector<RealType>& d_PriorParams,
		       const int priorTypes,
		       int index, cudaStream_t stream,
		       int gridSize, int blockSize)
{
	switch (priorTypes) {
		case 0 :
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, NOPRIOR><<<gridSize, blockSize, 0, stream>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
                                                               thrust::raw_pointer_cast(&d_BoundBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                               thrust::raw_pointer_cast(&d_Beta[0]),
                                                               thrust::raw_pointer_cast(&d_BetaBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_XBeta[0]),
                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                               thrust::raw_pointer_cast(&d_Denominator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator2[0]),
                                                               thrust::raw_pointer_cast(&d_PriorParams[0]));
			break;
		case 1 :
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, LAPLACE><<<gridSize, blockSize, 0, stream>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
                                                               thrust::raw_pointer_cast(&d_BoundBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                               thrust::raw_pointer_cast(&d_Beta[0]),
                                                               thrust::raw_pointer_cast(&d_BetaBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_XBeta[0]),
                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                               thrust::raw_pointer_cast(&d_Denominator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator2[0]),
                                                               thrust::raw_pointer_cast(&d_PriorParams[0]));
			break;
		case 2 :
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, NORMAL><<<gridSize, blockSize, 0, stream>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
                                                               thrust::raw_pointer_cast(&d_BoundBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_KWeight[0]),
                                                               thrust::raw_pointer_cast(&d_Beta[0]),
                                                               thrust::raw_pointer_cast(&d_BetaBuffer[0]),
                                                               thrust::raw_pointer_cast(&d_XBeta[0]),
                                                               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
                                                               thrust::raw_pointer_cast(&d_Denominator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator[0]),
                                                               thrust::raw_pointer_cast(&d_Numerator2[0]),
                                                               thrust::raw_pointer_cast(&d_PriorParams[0]));
			break;
	}
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::updateXBetaAndDelta(const thrust::device_vector<RealType>& d_X,
							  const thrust::device_vector<int>& d_K,
							  unsigned int offX,
							  unsigned int offK,
							  const unsigned int taskCount,
							  RealType2* d_GH,
							  thrust::device_vector<RealType>& d_XjY,
							  thrust::device_vector<RealType>& d_Bound,
							  thrust::device_vector<RealType>& d_BoundBuffer,
							  thrust::device_vector<RealType>& d_KWeight,
							  thrust::device_vector<RealType>& d_Beta,
							  thrust::device_vector<RealType>& d_BetaBuffer,
							  thrust::device_vector<RealType>& d_XBeta,
							  thrust::device_vector<RealType>& d_ExpXBeta,
							  thrust::device_vector<RealType>& d_Denominator,
							  thrust::device_vector<RealType>& d_Numerator,
							  thrust::device_vector<RealType>& d_Numerator2,
							  thrust::device_vector<RealType>& d_PriorParams,
							  const int priorTypes,
							  int index,
							  FormatType& formatType,
							  int gridSize, int blockSize)
{
	switch (formatType) {
		case DENSE :
			dispatchPriorType<RealType, RealType2, DENSE>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_BoundBuffer, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, stream[0], gridSize, blockSize);
			break;
		case SPARSE :
			dispatchPriorType<RealType, RealType2, SPARSE>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_BoundBuffer, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, stream[0], gridSize, blockSize);
			break;
		case INDICATOR :
			dispatchPriorType<RealType, RealType2, INDICATOR>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_BoundBuffer, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, stream[0], gridSize, blockSize);
			break;
		case INTERCEPT :
			dispatchPriorType<RealType, RealType2, INTERCEPT>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_BoundBuffer, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, stream[0], gridSize, blockSize);
			break;
	}

	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeAccumlatedDenominator(thrust::device_vector<RealType>& d_Denominator, 
		thrust::device_vector<RealType>& d_AccDenom, 
		int N)
{
	// Launch kernel
	DeviceScan::InclusiveSum(d_temp_storage_accd, temp_storage_bytes_accd, 
			d_Denominator.begin(), d_AccDenom.begin(), 
			N, stream[0]);

	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeTwoWayAccumlatedDenominator(thrust::device_vector<RealType>& d_Denominator,
		thrust::device_vector<RealType>& d_AccDenom, 
		thrust::device_vector<RealType>& d_DecDenom, 
		thrust::device_vector<RealType>& d_YWeight, 
		thrust::device_vector<RealType>& d_Y,
		int N)
{
	// two-way scan
	auto begin_denoms = thrust::make_zip_iterator(thrust::make_tuple(
				d_Denominator.begin(),
				d_Denominator.rbegin(),
				d_Y.rbegin(),
				d_YWeight.rbegin()));
	auto begin_accDenoms = thrust::make_zip_iterator(thrust::make_tuple(
				d_AccDenom.begin(), 
				d_DecDenom.rbegin()));

	TransformInputIterator<Tup2, TwoWayScan<RealType>, NRZipVec4> itr_scan(begin_denoms, twoWayScan);
	DeviceScan::InclusiveScan(d_temp_storage_faccd, temp_storage_bytes_faccd, 
			itr_scan, begin_accDenoms, 
			TuplePlus(), N, stream[0]);
	cudaStreamSynchronize(stream[0]);

	// add two scans together
	auto begin_denom = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin(), d_DecDenom.begin()));
	auto end_denom = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.end(), d_DecDenom.end()));
	auto begin_conditions = thrust::make_zip_iterator(thrust::make_tuple(d_Y.begin(), d_YWeight.begin()));
	thrust::transform(begin_denom, end_denom, begin_conditions, d_AccDenom.begin(), scansAddition);
	cudaStreamSynchronize(stream[0]);
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::updateXBeta(const thrust::device_vector<RealType>& d_X,
		const thrust::device_vector<int>& d_K,
		unsigned int offX,
		unsigned int offK,
		const unsigned int taskCount,
		RealType d_Delta,
		thrust::device_vector<RealType>& d_KWeight,
		thrust::device_vector<RealType>& d_Beta,
		thrust::device_vector<RealType>& d_BetaBuffer,
		thrust::device_vector<RealType>& d_XBeta,
		thrust::device_vector<RealType>& d_ExpXBeta,
		thrust::device_vector<RealType>& d_Denominator,
		thrust::device_vector<RealType>& d_AccDenom,
		thrust::device_vector<RealType>& d_Numerator,
		thrust::device_vector<RealType>& d_Numerator2,
		int index, size_t& N,
		FormatType& formatType,
		int gridSize, int blockSize)
{
	d_Beta[index] += d_Delta;
	d_BetaBuffer[index] = d_Beta[index];

	switch (formatType) {
		case DENSE :
			kernelUpdateXBeta<RealType, DENSE><<<gridSize, blockSize, 0, stream[0]>>>(offX, offK, taskCount, d_Delta,
			        thrust::raw_pointer_cast(&d_X[0]),
			        thrust::raw_pointer_cast(&d_K[0]),
			        thrust::raw_pointer_cast(&d_KWeight[0]),
			        thrust::raw_pointer_cast(&d_XBeta[0]),
			        thrust::raw_pointer_cast(&d_ExpXBeta[0]),
			        thrust::raw_pointer_cast(&d_Denominator[0]),
			        thrust::raw_pointer_cast(&d_Numerator[0]),
			        thrust::raw_pointer_cast(&d_Numerator2[0]));
			break;
		case SPARSE :
			kernelUpdateXBeta<RealType, SPARSE><<<gridSize, blockSize, 0, stream[0]>>>(offX, offK, taskCount, d_Delta,
			        thrust::raw_pointer_cast(&d_X[0]),
			        thrust::raw_pointer_cast(&d_K[0]),
			        thrust::raw_pointer_cast(&d_KWeight[0]),
			        thrust::raw_pointer_cast(&d_XBeta[0]),
			        thrust::raw_pointer_cast(&d_ExpXBeta[0]),
			        thrust::raw_pointer_cast(&d_Denominator[0]),
			        thrust::raw_pointer_cast(&d_Numerator[0]),
			        thrust::raw_pointer_cast(&d_Numerator2[0]));
			break;
		case INDICATOR :
			kernelUpdateXBeta<RealType, INDICATOR><<<gridSize, blockSize, 0, stream[0]>>>(offX, offK, taskCount, d_Delta,
			        thrust::raw_pointer_cast(&d_X[0]),
			        thrust::raw_pointer_cast(&d_K[0]),
			        thrust::raw_pointer_cast(&d_KWeight[0]),
			        thrust::raw_pointer_cast(&d_XBeta[0]),
			        thrust::raw_pointer_cast(&d_ExpXBeta[0]),
			        thrust::raw_pointer_cast(&d_Denominator[0]),
			        thrust::raw_pointer_cast(&d_Numerator[0]),
			        thrust::raw_pointer_cast(&d_Numerator2[0]));
			break;
		case INTERCEPT :
			kernelUpdateXBeta<RealType, INTERCEPT><<<gridSize, blockSize, 0, stream[0]>>>(offX, offK, taskCount, d_Delta,
			        thrust::raw_pointer_cast(&d_X[0]),
			        thrust::raw_pointer_cast(&d_K[0]),
			        thrust::raw_pointer_cast(&d_KWeight[0]),
			        thrust::raw_pointer_cast(&d_XBeta[0]),
			        thrust::raw_pointer_cast(&d_ExpXBeta[0]),
			        thrust::raw_pointer_cast(&d_Denominator[0]),
			        thrust::raw_pointer_cast(&d_Numerator[0]),
			        thrust::raw_pointer_cast(&d_Numerator2[0]));
			break;
	}

	cudaStreamSynchronize(stream[0]);
//	cudaDeviceSynchronize();
}

/* currently not using
template <typename RealType>
void CudaKernel<RealType>::processDelta(double2* d_GH,
                thrust::device_vector<RealType>& d_XjY,
                thrust::device_vector<RealType>& d_Delta,
                thrust::device_vector<RealType>& d_Beta,
                thrust::device_vector<RealType>& d_BetaBuffer,
                thrust::device_vector<RealType>& d_Bound,
                thrust::device_vector<RealType>& d_PriorParams,
                const int priorType,
                int index,
                int gridSize, int blockSize)
{
        switch (priorType) {
                case 0 :
                        kernelProcessDelta<RealType, NOPRIOR><<<gridSize, blockSize>>>(d_GH,
                                thrust::raw_pointer_cast(&d_XjY[0]),
                                thrust::raw_pointer_cast(&d_Delta[0]),
                                thrust::raw_pointer_cast(&d_Beta[0]),
                                thrust::raw_pointer_cast(&d_Bound[0]),
                                thrust::raw_pointer_cast(&d_PriorParams[0]),
                                index);
                        break;
                case 1 :
                        kernelProcessDelta<RealType, LAPLACE><<<gridSize, blockSize>>>(d_GH,
                                thrust::raw_pointer_cast(&d_XjY[0]),
                                thrust::raw_pointer_cast(&d_Delta[0]),
                                thrust::raw_pointer_cast(&d_Beta[0]),
                                thrust::raw_pointer_cast(&d_Bound[0]),
                                thrust::raw_pointer_cast(&d_PriorParams[0]),
                                index);
                        break;
                case 2 :
                        kernelProcessDelta<RealType, NORMAL><<<gridSize, blockSize>>>(d_GH,
                                thrust::raw_pointer_cast(&d_XjY[0]),
                                thrust::raw_pointer_cast(&d_Delta[0]),
                                thrust::raw_pointer_cast(&d_Beta[0]),
                                thrust::raw_pointer_cast(&d_Bound[0]),
                                thrust::raw_pointer_cast(&d_PriorParams[0]),
                                index);
                        break;
        }

    cudaDeviceSynchronize();
}

template <typename RealType>
void CudaKernel<RealType>::empty4(thrust::device_vector<RealType>& d_AccNumer,
				  thrust::device_vector<RealType>& d_AccNumer2,
				  thrust::device_vector<RealType>& d_Buffer1,
				  thrust::device_vector<RealType>& d_Buffer2)
{
	d_Buffer1 = d_AccNumer;
	d_Buffer2 = d_AccNumer2;
}

template <typename RealType>
void CudaKernel<RealType>::empty2(thrust::device_vector<RealType>& d_AccDenom,
                                  thrust::device_vector<RealType>& d_Buffer3)
{
	d_Buffer3 = d_AccDenom;
}

template <typename RealType>
void CudaKernel<RealType>::computeAccumulatedNumerator(thrust::device_vector<RealType>& d_Numerator,
                                                       thrust::device_vector<RealType>& d_Numerator2,
                                                       thrust::device_vector<RealType>& d_AccNumer,
                                                       thrust::device_vector<RealType>& d_AccNumer2,
                                                       size_t& N)
{
        auto results = thrust::make_zip_iterator(thrust::make_tuple(d_AccNumer.begin(), d_AccNumer2.begin()));
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(), d_Numerator2.begin()));

        // Launch kernel
        DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, results, TuplePlus(), N);
        cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
}

template <typename RealType>
void CudaKernel<RealType>::computeAccumulatedNumerAndDenom(thrust::device_vector<RealType>& d_Denominator,
                                                           thrust::device_vector<RealType>& d_Numerator,
                                                           thrust::device_vector<RealType>& d_Numerator2,
                                                           thrust::device_vector<RealType>& d_AccDenom,
                                                           thrust::device_vector<RealType>& d_AccNumer,
                                                           thrust::device_vector<RealType>& d_AccNumer2,
                                                           size_t& N)
{
        auto results_acc = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin(), d_AccNumer.begin(), d_AccNumer2.begin()));
        auto begin_acc = thrust::make_zip_iterator(thrust::make_tuple(d_Denominator.begin(), d_Numerator.begin(), d_Numerator2.begin()));

        // Launch kernel
        DeviceScan::InclusiveScan(d_temp_storage_acc, temp_storage_bytes_acc, begin_acc, results_acc, TuplePlus3(), N);
        cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
}

*/

template class CudaKernel<float, float2>;
template class CudaKernel<double, double2>;

