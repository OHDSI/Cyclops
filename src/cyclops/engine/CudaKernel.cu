#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
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
				  const int* K, 
				  RealType* d_XBeta, 
				  RealType* d_ExpXBeta,
				  RealType* d_Numerator,
                  RealType* d_Numerator2)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    //if (formatType == INDICATOR || formatType == SPARSE) {
	int k = K[offK + task];
    //} else { // DENSE, INTERCEPT
//	int k = task;
    //}

    //if (formatType == SPARSE || formatType == DENSE) {
//	RealType inc = delta * d_X[offX + task];
    //} else { // INDICATOR, INTERCEPT
	RealType inc = delta;
    //}

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
                                                  const RealType* dX,
                                                  const int* K,
                                                  RealType* dExpXBeta,
                                                  RealType* dNumerator,
                                                  RealType* dNumerator2)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    //if (formatType == INDICATOR || formatType == SPARSE) {
	    int k = K[offK + task];
    //} else { // DENSE, INTERCEPT
//	    int k = task;
    //}

    if (task < taskCount) {
        //if (formatType == SPARSE || formatType == DENSE) {
//        	  dNumerator[k] = dX[offX + task] * dExpXBeta[k];
//            dNumerator2[k] = dX[offX + task] * dNumerator[k];
        //} else { // INDICATOR, INTERCEPT
              dNumerator[k] = dExpXBeta[k];
        //}
    }
}

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
        //if (IteratorType::isIndicator) {
            d_BufferH[task] = g * (1.0 - t);
        //} else {
//	    d_BufferH[task] = d_NWeight[task] * (d_AccNumer2[task] / d_AccDenom[task] - t * t);
        //}
    }
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
/*
template <typename RealType>
struct functorCGH :
        public thrust::unary_function<thrust::tuple<RealType, RealType, RealType, RealType>,
                                      thrust::tuple<RealType, RealType>>
{
        typedef typename thrust::tuple<RealType, RealType, RealType, RealType> InputTuple;
        typedef typename thrust::tuple<RealType, RealType>       OutputTuple;

	__host__ __device__
                OutputTuple operator()(const InputTuple& t) const
                {
			auto temp = thrust::get<0>(t) * thrust::get<1>(t) / thrust::get<2>(t);
                        return OutputTuple(temp, temp * (1 - thrust::get<1>(t) / thrust::get<2>(t)));
                }
};
*/
template <typename RealType>
CudaKernel<RealType>::CudaKernel()
{ 
	std::cout << "CUDA class Created \n";
}

template <typename RealType>
CudaKernel<RealType>::~CudaKernel()
{
//    cudaFree(d_temp_storage0);
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
					    thrust::device_vector<RealType>& d_Gradient,
					    thrust::device_vector<RealType>& d_Hessian,
					    size_t& N)
{
/*	
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
*/
    // for reduction in compGAndH
}

template <typename RealType>
void CudaKernel<RealType>::updateXBeta(const thrust::device_vector<RealType>& X, 
				       const thrust::device_vector<int>& K, 
				       unsigned int offX, 
				       unsigned int offK, 
				       const unsigned int taskCount, 
				       RealType delta, 
				       thrust::device_vector<RealType>& dXBeta, 
				       thrust::device_vector<RealType>& dExpXBeta, 
				       thrust::device_vector<RealType>& dNumerator,
				       thrust::device_vector<RealType>& dNumerator2,
				       int gridSize, int blockSize)
{
    kernelUpdateXBeta<<<gridSize, blockSize>>>(offX, 
		    			       offK, 
					       taskCount, 
					       delta, 
					       thrust::raw_pointer_cast(&X[0]), 
					       thrust::raw_pointer_cast(&K[0]), 
					       thrust::raw_pointer_cast(&dXBeta[0]), 
					       thrust::raw_pointer_cast(&dExpXBeta[0]),
					       thrust::raw_pointer_cast(&dNumerator[0]),
					       thrust::raw_pointer_cast(&dNumerator2[0]));
}

template <typename RealType>
void CudaKernel<RealType>::computeNumeratorForGradient(const thrust::device_vector<RealType>& X,
                                                       const thrust::device_vector<int>& K,
                                                       unsigned int offX,
                                                       unsigned int offK,
                                                       const unsigned int taskCount,
                                                       thrust::device_vector<RealType>& dExpXBeta,
                                                       thrust::device_vector<RealType>& dNumerator,
                                                       thrust::device_vector<RealType>& dNumerator2,
                                                       int gridSize, int blockSize)
{
    kernelComputeNumeratorForGradient<<<gridSize, blockSize>>>(offX,
                                                               offK,
                                                               taskCount,
                                                               thrust::raw_pointer_cast(&X[0]),
                                                               thrust::raw_pointer_cast(&K[0]),
                                                               thrust::raw_pointer_cast(&dExpXBeta[0]),
                                                               thrust::raw_pointer_cast(&dNumerator[0]),
                                                               thrust::raw_pointer_cast(&dNumerator2[0]));
}

template <typename RealType>
void CudaKernel<RealType>::computeGradientAndHessian(thrust::device_vector<RealType>& d_AccNumer, 
						     thrust::device_vector<RealType>& d_AccNumer2, 
						     thrust::device_vector<RealType>& d_AccDenom, 
						     thrust::device_vector<RealType>& d_NWeight, 
						     thrust::device_vector<RealType>& d_Gradient, 
						     thrust::device_vector<RealType>& d_Hessian, 
						     size_t& N)
{
    d_AccDenom[N] = static_cast<RealType>(1); // avoid nan

    // cub transfrom reduction
    auto begin_gh = thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(),
                                                              d_AccNumer.begin(),
                                                              d_AccDenom.begin(),
                                                              d_AccNumer2.begin()));
    auto results_gh = thrust::make_zip_iterator(thrust::make_tuple(d_Gradient.begin(), d_Hessian.begin()));

    // transform iterator
    TransformInputIterator<Tup2, functorCGH<RealType>, ZipVec4> itr(begin_gh, cGAH);

    // reduction
    void *d_temp_storage_gh = NULL;
    size_t temp_storage_bytes_gh = 0;

    DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh, itr, results_gh, N, TuplePlus(), init);
    cudaMalloc(&d_temp_storage_gh, temp_storage_bytes_gh);
    DeviceReduce::Reduce(d_temp_storage_gh, temp_storage_bytes_gh, itr, results_gh, N, TuplePlus(), init);

    cudaFree(d_temp_storage_gh);

//    std::cout << "G: " << d_Gradient[0] << " H: " << d_Hessian[0] << '\n';
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

    void *d_temp_storage_acc = NULL;
    size_t temp_storage_bytes_acc = 0;

    DeviceScan::InclusiveScan(d_temp_storage_acc, temp_storage_bytes_acc, begin_acc, results_acc, TuplePlus3(), N);
    cudaMalloc(&d_temp_storage_acc, temp_storage_bytes_acc);

    DeviceScan::InclusiveScan(d_temp_storage_acc, temp_storage_bytes_acc, begin_acc, results_acc, TuplePlus3(), N);

    cudaFree(d_temp_storage_acc);
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

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, results, TuplePlus(), N);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
   
    DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, begin, results, TuplePlus(), N);

    cudaFree(d_temp_storage);
}


template <typename RealType>
void CudaKernel<RealType>::CubReduce(RealType* d_in, RealType* d_out, int num_items)
{
    // Allocate temporary storage
    void *d_temp_storage0 = NULL;
    size_t temp_storage_bytes0 = 0;

    // Determine temporary device storage requirements
    DeviceReduce::Sum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage0, temp_storage_bytes0);

    // Launch kernel
    DeviceReduce::Sum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    cudaFree(d_temp_storage0);
}

template <typename RealType>
void CudaKernel<RealType>::CubScan(RealType* d_in, RealType* d_out, int num_items)
{
    // Allocate temporary storage
    void *d_temp_storage0 = NULL;
    size_t temp_storage_bytes0 = 0;

    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage0, temp_storage_bytes0);

    // Launch kernel
    DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

//    cudaFree(d_temp_storage0);

}


/*
template <class RealType>
void CudaKernel<RealType>::computeAccDenomMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class RealType>
void CudaKernel<RealType>::computeAccDenom(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);
}

template <class RealType>
void CudaKernel<RealType>::computeAccNumerMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer, d_AccNumer, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class RealType>
void CudaKernel<RealType>::computeAccNumer(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer, d_AccNumer, num_items);
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer2, d_AccNumer2, num_items);
}

template <class RealType>
void CudaKernel<RealType>::CubExpScanMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_XBeta, d_AccDenom, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class RealType>
void CudaKernel<RealType>::CubExpScan(int num_items)
{
//    auto start = std::chrono::steady_clock::now();

    TransformInputIterator<RealType, CustomExp, RealType*> d_itr(d_XBeta, exp_op);
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_itr, d_AccDenom, num_items);
    
//    auto end = std::chrono::steady_clock::now();
//    timerG += std::chrono::duration<double, std::milli>(end - start).count();
//    std::cout << "GPU takes " << timerG << " ms" << '\n';
}
*/

template class CudaKernel<float>;
template class CudaKernel<double>;

