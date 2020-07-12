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

template <typename RealType>
__global__ void kernelUpdateXBeta(int offX,
				  int offK,
				  const int taskCount,
				  RealType delta,
				  const RealType* d_X,
				  const int* d_K,
				  RealType* d_XBeta,
				  RealType* d_ExpXBeta,
				  RealType* d_Numerator,
				  RealType* d_Numerator2)
{
	int task = blockIdx.x * blockDim.x + threadIdx.x;

//	if (formatType == INDICATOR || formatType == SPARSE) {
	    int k = d_K[offK + task];
//	} else { // DENSE, INTERCEPT
//	    int k = task;
//	}

//	if (formatType == SPARSE || formatType == DENSE) {
//	    RealType inc = delta * d_X[offX + task];
//	} else { // INDICATOR, INTERCEPT
	    RealType inc = delta;
//	}

	if (task < taskCount) {
	    RealType xb = d_XBeta[k] + inc;
	    d_XBeta[k] = xb;
	    d_ExpXBeta[k] = exp(xb);
	    d_Numerator[k] = 0;
	    d_Numerator2[k] = 0;
	}
}

template <typename RealType>
__global__ void kernelUpdateXBeta1(int offX,
				  int offK,
				  const int taskCount,
				  int index,
				  const RealType* d_X,
				  const int* d_K,
				  double2* d_GH,
				  RealType* d_XjY,
				  RealType* d_Bound,
				  RealType* d_Beta,
				  RealType* d_XBeta,
				  RealType* d_ExpXBeta,
				  RealType* d_Numerator,
				  RealType* d_Numerator2)
{
	// process delta, update beta and bound
	double2 GH = *d_GH;
	RealType g = GH.x - d_XjY[index];
	RealType h = GH.y;
	RealType beta = d_Beta[index];

	RealType delta = -g/h; // no prior

	RealType bound = d_Bound[index];
	if (delta < -bound) {
		delta = -bound;
	} else if (delta > bound) {
		delta = bound;
	}
	d_Beta[index] = delta + beta;

	auto intermediate = max(2*abs(delta), bound/2);
	intermediate = max(intermediate, 0.001);
	d_Bound[index] = intermediate;


	// update xb and exb
	int task = blockIdx.x * blockDim.x + threadIdx.x;

//	if (formatType == INDICATOR || formatType == SPARSE) {
	    int k = d_K[offK + task];
//	} else { // DENSE, INTERCEPT
//	    int k = task;
//	}

//	if (formatType == SPARSE || formatType == DENSE) {
//	    RealType inc = delta * d_X[offX + task];
//	} else { // INDICATOR, INTERCEPT
	    RealType inc = delta;
//	}

	if (task < taskCount) {
	    RealType xb = d_XBeta[k] + inc;
	    d_XBeta[k] = xb;
	    d_ExpXBeta[k] = exp(xb);
	    d_Numerator[k] = 0;
	    d_Numerator2[k] = 0;
	}
}

template <typename RealType>
__global__ void kernelComputeNumeratorForGradient(int offX,
                                                  int offK,
                                                  const int taskCount,
                                                  const RealType* d_X,
                                                  const int* d_K,
                                                  RealType* d_ExpXBeta,
                                                  RealType* d_Numerator,
                                                  RealType* d_Numerator2)
{
	int task = blockIdx.x * blockDim.x + threadIdx.x;

//	if (formatType == INDICATOR || formatType == SPARSE) {
	    int k = d_K[offK + task];
//	} else { // DENSE, INTERCEPT
//	    int k = task;
//	}

	if (task < taskCount) {
//	    if (formatType == SPARSE || formatType == DENSE) {
//	        d_Numerator[k] = d_X[offX + task] * d_ExpXBeta[k];
//	        d_Numerator2[k] = d_X[offX + task] * d_Numerator[k];
//	    } else { // INDICATOR, INTERCEPT
	        d_Numerator[k] = d_ExpXBeta[k];
//	    }
	}
}

template <typename RealType>
__global__ void kernelProcessDelta(RealType* d_DeltaVector,
                                   RealType* d_Bound,
                                   RealType* d_Beta,
                                   RealType* d_XjY,
                                   double2* d_GH,
                                   int index)
{
	double2 GH = *d_GH;
	RealType g = GH.x - d_XjY[index];
	RealType h = GH.y;
	RealType beta = d_Beta[index];
	RealType delta = -g/h;
	RealType bound = d_Bound[index];

	if (delta < -bound) {
		delta = -bound;
	} else if (delta > bound) {
		delta = bound;
	}
	d_DeltaVector[index] = delta;
	d_Beta[index] = delta + beta;

	auto intermediate = max(2*abs(delta), bound/2);
	intermediate = max(intermediate, 0.001);
	d_Bound[index] = intermediate;
}


struct TuplePlus
{
	template<typename L, typename R>
	__host__ __device__
	thrust::tuple<L, L> operator()(thrust::tuple<L, L> lhs, thrust::tuple<R, R> rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs), thrust::get<1>(lhs) + thrust::get<1>(rhs));
	}
};

struct TuplePlus3
{
	template<typename L, typename R>
	__host__ __device__
	thrust::tuple<L, L, L> operator()(thrust::tuple<L, L, L> lhs, thrust::tuple<R, R, R> rhs)
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


template <typename RealType>
CudaKernel<RealType>::CudaKernel()
{
	std::cout << "CUDA class Created \n";
}

template <typename RealType>
CudaKernel<RealType>::~CudaKernel()
{
	cudaFree(d_temp_storage0); // accDenom
	cudaFree(d_temp_storage); // accNumer
	cudaFree(d_temp_storage_acc); // accNAndD
	cudaFree(d_temp_storage_gh); // cGAH
//	cudaFree(d_init);
	std::cout << "CUDA class Destroyed \n";
}

template <typename RealType>
void CudaKernel<RealType>::allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
					    thrust::device_vector<RealType>& d_Numerator,
					    thrust::device_vector<RealType>& d_Numerator2,
					    thrust::device_vector<RealType>& d_AccDenom,
					    thrust::device_vector<RealType>& d_AccNumer,
					    thrust::device_vector<RealType>& d_AccNumer2,
					    thrust::device_vector<RealType>& d_NWeight,
//					    thrust::device_vector<RealType>& d_Gradient,
//					    thrust::device_vector<RealType>& d_Hessian,
					    double2* dGH,
					    size_t& N,
					    thrust::device_vector<int>& indicesN)
{
//	thrust::sequence(indicesN.begin(), indicesN.end());

	// for scan in accDenom
	DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, &d_Denominator[0], &d_AccDenom[0], N);
	cudaMalloc(&d_temp_storage0, temp_storage_bytes0);

	// for scan in accNumer
	auto results = thrust::make_zip_iterator(thrust::make_tuple(d_AccNumer.begin(), d_AccNumer2.begin()));
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_Numerator.begin(), d_Numerator2.begin()));
	DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, results, TuplePlus(), N);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // for scan in compDAndN
        auto results_acc = thrust::make_zip_iterator(thrust::make_tuple(d_AccDenom.begin(), d_AccNumer.begin(), d_AccNumer2.begin()));
        auto begin_acc = thrust::make_zip_iterator(thrust::make_tuple(d_Denominator.begin(), d_Numerator.begin(), d_Numerator2.begin()));

        DeviceScan::InclusiveScan(d_temp_storage_acc, temp_storage_bytes_acc, begin_acc, results_acc, TuplePlus3(), N);
        cudaMalloc(&d_temp_storage_acc, temp_storage_bytes_acc);

	// for reduction in compGAndH
	auto begin_gh = thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(),
											  d_AccNumer.begin(),
											  d_AccDenom.begin(),
											  d_AccNumer2.begin()));

	TransformInputIterator<double2, functorCGH<RealType>, ZipVec4> itr(begin_gh, cGAH);

	d_init.x = d_init.y = 0.0;

	DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh, itr, dGH, N, Double2Plus(), d_init);
	cudaMalloc(&d_temp_storage_gh, temp_storage_bytes_gh);

}

template <typename RealType>
void CudaKernel<RealType>::updateXBeta(const thrust::device_vector<RealType>& d_X,
				       const thrust::device_vector<int>& d_K,
				       unsigned int offX,
				       unsigned int offK,
				       const unsigned int taskCount,
				       RealType delta,
				       thrust::device_vector<RealType>& d_XBeta,
				       thrust::device_vector<RealType>& d_ExpXBeta,
				       thrust::device_vector<RealType>& d_Numerator,
				       thrust::device_vector<RealType>& d_Numerator2,
				       int gridSize, int blockSize)
{
	kernelUpdateXBeta<<<gridSize, blockSize>>>(offX,
		    			                       offK,
		    			                       taskCount,
		    			                       delta,
		    			                       thrust::raw_pointer_cast(&d_X[0]),
		    			                       thrust::raw_pointer_cast(&d_K[0]),
		    			                       thrust::raw_pointer_cast(&d_XBeta[0]),
		    			                       thrust::raw_pointer_cast(&d_ExpXBeta[0]),
		    			                       thrust::raw_pointer_cast(&d_Numerator[0]),
		    			                       thrust::raw_pointer_cast(&d_Numerator2[0]));
	cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
}

template <typename RealType>
void CudaKernel<RealType>::updateXBeta1(const thrust::device_vector<RealType>& d_X,
				       const thrust::device_vector<int>& d_K,
				       unsigned int offX,
				       unsigned int offK,
				       const unsigned int taskCount,
				       double2* d_GH,
				       thrust::device_vector<RealType>& d_XjY,
				       thrust::device_vector<RealType>& d_Bound,
				       thrust::device_vector<RealType>& d_Beta,
				       thrust::device_vector<RealType>& d_XBeta,
				       thrust::device_vector<RealType>& d_ExpXBeta,
				       thrust::device_vector<RealType>& d_Numerator,
				       thrust::device_vector<RealType>& d_Numerator2,
				       int index,
				       int gridSize, int blockSize)
{
	kernelUpdateXBeta1<<<gridSize, blockSize>>>(offX,
		    			                       offK,
		    			                       taskCount,
		    			                       index,
		    			                       thrust::raw_pointer_cast(&d_X[0]),
		    			                       thrust::raw_pointer_cast(&d_K[0]),
							       d_GH,
							       thrust::raw_pointer_cast(&d_XjY[0]),
							       thrust::raw_pointer_cast(&d_Bound[0]),
							       thrust::raw_pointer_cast(&d_Beta[0]),
		    			                       thrust::raw_pointer_cast(&d_XBeta[0]),
		    			                       thrust::raw_pointer_cast(&d_ExpXBeta[0]),
		    			                       thrust::raw_pointer_cast(&d_Numerator[0]),
		    			                       thrust::raw_pointer_cast(&d_Numerator2[0]));
	cudaDeviceSynchronize();
}

template <typename RealType>
void CudaKernel<RealType>::computeNumeratorForGradient(const thrust::device_vector<RealType>& d_X,
		    				    				       const thrust::device_vector<int>& d_K,
		    				    				       unsigned int offX,
		    				    				       unsigned int offK,
		    				    				       const unsigned int taskCount,
		    				    				       thrust::device_vector<RealType>& d_ExpXBeta,
		    				    				       thrust::device_vector<RealType>& d_Numerator,
		    				    				       thrust::device_vector<RealType>& d_Numerator2,
		    				    				       int gridSize, int blockSize)
{
	kernelComputeNumeratorForGradient<<<gridSize, blockSize>>>(offX,
		    				    				               offK,
		    				    				               taskCount,
		    				    				               thrust::raw_pointer_cast(&d_X[0]),
		    				    				               thrust::raw_pointer_cast(&d_K[0]),
		    				    				               thrust::raw_pointer_cast(&d_ExpXBeta[0]),
		    				    				               thrust::raw_pointer_cast(&d_Numerator[0]),
		    				    				               thrust::raw_pointer_cast(&d_Numerator2[0]));
	cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
}


template <typename RealType>
void CudaKernel<RealType>::processDelta(thrust::device_vector<RealType>& d_DeltaVector,
		    				    		thrust::device_vector<RealType>& d_Bound,
		    				    		thrust::device_vector<RealType>& d_Beta,
		    				    		thrust::device_vector<RealType>& d_XjY,
		    				    		double2* d_GH,
		    				    		thrust::device_vector<RealType>& d_PriorParams,
		    				    		std::vector<RealType>& priorTypes,
		    				    		int index,
		    				    		int gridSize, int blockSize)
{
//	std::cout << "processDelta kernel \n";
	kernelProcessDelta<<<1, 1>>>(thrust::raw_pointer_cast(&d_DeltaVector[0]),
	                                            thrust::raw_pointer_cast(&d_Bound[0]),
	                                            thrust::raw_pointer_cast(&d_Beta[0]),
	                                            thrust::raw_pointer_cast(&d_XjY[0]),
	                                            d_GH,
	                                            index);
	cudaDeviceSynchronize();
	d_DeltaVector[index];	
}

template <typename RealType>
void CudaKernel<RealType>::computeGradientAndHessian(thrust::device_vector<RealType>& d_AccNumer,
						     thrust::device_vector<RealType>& d_AccNumer2,
						     thrust::device_vector<RealType>& d_AccDenom,
						     thrust::device_vector<RealType>& d_NWeight,
//						     thrust::device_vector<RealType>& d_Gradient,
//						     thrust::device_vector<RealType>& d_Hessian,
						     double2* dGH,
						     size_t& N
//						     ,const std::vector<int>& K,
//                                                     unsigned int offK,
//                                                     thrust::device_vector<int>& indicesN
						     )
{
//	int start = K[offK];
/*
	for (int i = K[offK]; i < N; i++) {
	    std::cout << indicesN[i] << '\n';
	}
*/
	// cub transfrom reduction
	auto begin_gh = thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(),
                                            	                 d_AccNumer.begin(),
                                            	                 d_AccDenom.begin(),
                                            	                 d_AccNumer2.begin()));
	// transform iterator
	TransformInputIterator<double2, functorCGH<RealType>, ZipVec4> itr(begin_gh, cGAH);

	// Launch kernel
	DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh, itr, dGH, N, Double2Plus(), d_init);
	cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
/*
	// thrust::transform_reduce
	GH = thrust::transform_reduce(
                    thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(), d_AccNumer.begin(), d_AccDenom.begin(), d_AccNumer2.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.end(), d_AccNumer.end(), d_AccDenom.end(), d_AccNumer2.end())),
                    cGAH,
                    d_init,
                    Double2Plus());

	// start from the first non-zero entry

	// Determine temporary device storage requirements and allocate temporary storage
	DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh,
	    thrust::make_permutation_iterator(itr, indicesN.begin() + start),
	    results_gh, N, TuplePlus(), init);
	cudaMalloc(&d_temp_storage_gh, temp_storage_bytes_gh);

	// Launch kernel
	DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh,
	    thrust::make_permutation_iterator(itr, indicesN.begin() + start),
	    results_gh, N, TuplePlus(), init);
*/
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
void CudaKernel<RealType>::CubScan(RealType* d_in, RealType* d_out, int num_items)
{
	// Launch kernel
	DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);
	cudaDeviceSynchronize(); // MAS Wait until kernel completes; may be important for timing
}


/* currently not using
template <typename RealType>
__global__ void kernelComputeGradientAndHessian(RealType* d_BufferG,
                                                RealType* d_BufferH,
                                                const RealType* d_AccNumer,
                                                const RealType* d_AccNumer2,
                                                const RealType* d_AccDenom,
                                                const RealType* d_NWeight,
                                                int N)
{
        int task = blockIdx.x * blockDim.x + threadIdx.x;

        if (task < N) {
                RealType t = d_AccNumer[task] / d_AccDenom[task];
                RealType g = d_NWeight[task] * t;
                d_BufferG[task] = g;
//          if (IteratorType::isIndicator) {
                        d_BufferH[task] = g * (1.0 - t);
//          } else {
//              d_BufferH[task] = d_NWeight[task] * (d_AccNumer2[task] / d_AccDenom[task] - t * t);
//          }
        }
}

template <typename RealType>
void CudaKernel<RealType>::CubReduce(RealType* d_in, RealType* d_out, int num_items)
{

    	// Declare temporary storage
    	void *d_temp_storage0 = NULL;
    	size_t temp_storage_bytes0 = 0;

    	// Determine temporary device storage requirements and allocate temporary storage
    	DeviceReduce::Sum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);
    	cudaMalloc(&d_temp_storage0, temp_storage_bytes0); // MAS Why?

    	// Launch kernel
    	DeviceReduce::Sum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    	cudaFree(d_temp_storage0);
}
*/

template class CudaKernel<float>;
template class CudaKernel<double>;

