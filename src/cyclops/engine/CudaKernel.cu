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
	RealType bound = d_Bound[index];
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
CudaKernel<RealType, RealType2>::CudaKernel()
{
	std::cout << "ctor CudaKernel \n";
}

template <typename RealType, typename RealType2>
CudaKernel<RealType, RealType2>::~CudaKernel()
{
	cudaFree(d_temp_storage0); // accDenom
	cudaFree(d_temp_storage_gh); // cGAH

	std::cout << "dtor CudaKernel \n";
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
						       thrust::device_vector<RealType>& d_Numerator,
						       thrust::device_vector<RealType>& d_Numerator2,
						       thrust::device_vector<RealType>& d_AccDenom,
						       thrust::device_vector<RealType>& d_AccNumer,
						       thrust::device_vector<RealType>& d_AccNumer2,
						       thrust::device_vector<RealType>& d_NWeight,
						       RealType2* d_GH,
						       RealType2* d_BlockGH,
						       size_t& N)
{
	// for scan in accDenom
	DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, &d_Denominator[0], &d_AccDenom[0], N);
	cudaMalloc(&d_temp_storage0, temp_storage_bytes0);
/*
	// for fused scan reduction (double scan)
	auto begin0 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(), 
								   d_Numerator2.begin()));
	auto begin1 = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin(), 
								   d_NWeight.begin()));
	DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
			begin0, begin1, 
			d_BlockGH, d_GH,
			TuplePlus(), Double2Plus(), compGradHessInd, N);
*/

	auto begin2 = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(),
								   d_Numerator2.begin(),
								   d_Denominator.begin()));
	
	// triple scan without storing accDenom
	DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
			begin2, thrust::raw_pointer_cast(&d_NWeight[0]), // input
			d_BlockGH, d_GH, // output
			TuplePlus3(), RealType2Plus(), compGradHessInd1, N);

	cudaMalloc(&d_temp_storage_gh, temp_storage_bytes_gh);
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
                        kernelComputeNumeratorForGradient<RealType, DENSE><<<gridSize, blockSize>>>(offX,
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
                        kernelComputeNumeratorForGradient<RealType, SPARSE><<<gridSize, blockSize>>>(offX,
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
                        kernelComputeNumeratorForGradient<RealType, INDICATOR><<<gridSize, blockSize>>>(offX,
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
                        kernelComputeNumeratorForGradient<RealType, INTERCEPT><<<gridSize, blockSize>>>(offX,
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
	
	cudaDeviceSynchronize();
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeGradientAndHessian(thrust::device_vector<RealType>& d_Numerator,
								thrust::device_vector<RealType>& d_Numerator2,
								thrust::device_vector<RealType>& d_AccNumer,
						 		thrust::device_vector<RealType>& d_AccNumer2,
						 		thrust::device_vector<RealType>& d_AccDenom,
						 		thrust::device_vector<RealType>& d_NWeight,
						 		RealType2* d_GH,
						 		RealType2* d_BlockGH,
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
				begin0, begin1, 
				d_BlockGH, d_GH,
				TuplePlus(), RealType2Plus(), compGradHessInd, N - offCV);
	} else {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh, 
				begin0, begin1, 
				d_BlockGH, d_GH,
				TuplePlus(), RealType2Plus(), compGradHessNInd, N - offCV);
	}
	cudaDeviceSynchronize();
}


template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::computeGradientAndHessian1(thrust::device_vector<RealType>& d_Numerator,
								 thrust::device_vector<RealType>& d_Numerator2,
								 thrust::device_vector<RealType>& d_Denominator,
								 thrust::device_vector<RealType>& d_AccNumer,
								 thrust::device_vector<RealType>& d_AccNumer2,
								 thrust::device_vector<RealType>& d_AccDenom,
								 thrust::device_vector<RealType>& d_NWeight,
								 RealType2* d_GH,
								 RealType2* d_BlockGH,
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
				begin2, thrust::raw_pointer_cast(&d_NWeight[0]),
				d_BlockGH, d_GH,
				TuplePlus3(), RealType2Plus(), compGradHessInd1, N - offCV);
	} else {
		DeviceFuse::ScanReduce(d_temp_storage_gh, temp_storage_bytes_gh,
				begin2, thrust::raw_pointer_cast(&d_NWeight[0]),
				d_BlockGH, d_GH,
				TuplePlus3(), RealType2Plus(), compGradHessNInd1, N - offCV);
	}

	cudaDeviceSynchronize();
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
		       int gridSize, int blockSize)
{
	switch (priorTypes) {
		case 0 :
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, NOPRIOR><<<gridSize, blockSize>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
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
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, LAPLACE><<<gridSize, blockSize>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
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
			kernelUpdateXBetaAndDelta<RealType, RealType2, formatType, NORMAL><<<gridSize, blockSize>>>(offX, offK, taskCount, index,
                                                               thrust::raw_pointer_cast(&d_X[0]),
                                                               thrust::raw_pointer_cast(&d_K[0]),
                                                               d_GH,
                                                               thrust::raw_pointer_cast(&d_XjY[0]),
                                                               thrust::raw_pointer_cast(&d_Bound[0]),
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
                                                          taskCount, d_GH, d_XjY, d_Bound, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, gridSize, blockSize);
			break;
		case SPARSE :
			dispatchPriorType<RealType, RealType2, SPARSE>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, gridSize, blockSize);
			break;
		case INDICATOR :
			dispatchPriorType<RealType, RealType2, INDICATOR>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, gridSize, blockSize);
			break;
		case INTERCEPT :
			dispatchPriorType<RealType, RealType2, INTERCEPT>(d_X, d_K, offX, offK,
                                                          taskCount, d_GH, d_XjY, d_Bound, d_KWeight,
                                                          d_Beta, d_BetaBuffer, d_XBeta, d_ExpXBeta, d_Denominator,
                                                          d_Numerator, d_Numerator2,
                                                          d_PriorParams, priorTypes,
                                                          index, gridSize, blockSize);
			break;
	}

        cudaDeviceSynchronize();
}

template <typename RealType, typename RealType2>
void CudaKernel<RealType, RealType2>::CubScan(RealType* d_in, RealType* d_out, int num_items)
{
	// Launch kernel
	DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);
	cudaDeviceSynchronize();
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
void CudaKernel<RealType>::updateXBeta(const thrust::device_vector<RealType>& d_X,
		const thrust::device_vector<int>& d_K,
		unsigned int offX,
		unsigned int offK,
		const unsigned int taskCount,
		thrust::device_vector<RealType>& d_Delta,
		thrust::device_vector<RealType>& d_KWeight,
		thrust::device_vector<RealType>& d_XBeta,
		thrust::device_vector<RealType>& d_ExpXBeta,
		thrust::device_vector<RealType>& d_Denominator,
		thrust::device_vector<RealType>& d_Numerator,
		thrust::device_vector<RealType>& d_Numerator2,
		int index,
		FormatType& formatType,
		int gridSize, int blockSize)
{
	switch (formatType) {
		case DENSE :
			kernelUpdateXBeta<RealType, DENSE><<<gridSize, blockSize>>>(offX, offK, taskCount, d_Delta[index],
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
			kernelUpdateXBeta<RealType, SPARSE><<<gridSize, blockSize>>>(offX, offK, taskCount, d_Delta[index],
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
			kernelUpdateXBeta<RealType, INDICATOR><<<gridSize, blockSize>>>(offX, offK, taskCount, d_Delta[index],
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
			kernelUpdateXBeta<RealType, INTERCEPT><<<gridSize, blockSize>>>(offX, offK, taskCount, d_Delta[index],
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

