#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
//#include <thrust/for_each.h>

#include "CudaKernel.h"

using namespace cub;
	
template <typename RealType>
__global__ void kernelUpdateXBeta(int offX, int offK, const int taskCount, RealType delta,
                const RealType* d_X, const int* K, RealType* d_XBeta, RealType* d_ExpXBeta)
//__global__ void kernelUpdateXBeta(RealType* d_X, RealType* d_XBeta, RealType* d_ExpXBeta, RealType delta, int N)
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
	d_ExpXBeta[k] = expf(xb);
    }
}

template <typename RealType>
__global__ void kernelComputeGradientAndHessian(RealType* d_BufferG, RealType* d_BufferH, const RealType* d_AccNumer, const RealType* d_AccNumer2, const RealType* d_AccDenom, const RealType* d_NWeight, int N)
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

template <typename RealType>
CudaKernel<RealType>::CudaKernel()
{ 
	std::cout << "CUDA class Created \n";
}

template <typename RealType>
CudaKernel<RealType>::~CudaKernel()
{
    std::cout << "CUDA class Destroyed \n";
}

template <typename RealType>
void CudaKernel<RealType>::initialize(int K, int N)
{
/*
	//TODO use thrust	
    cudaMalloc(&d_XBeta,  sizeof(RealType) * K);
    cudaMalloc(&d_ExpXBeta,  sizeof(RealType) * K);

    cudaMalloc(&d_Numer,  sizeof(RealType) * N);
    cudaMalloc(&d_Numer2,  sizeof(RealType) * N);

    cudaMalloc(&d_AccDenom, sizeof(RealType) * N);
    cudaMalloc(&d_AccNumer, sizeof(RealType) * N);
    cudaMalloc(&d_AccNumer2, sizeof(RealType) * N);

    cudaMalloc(&d_BufferG, sizeof(RealType) * N);
    cudaMalloc(&d_BufferH, sizeof(RealType) * N);
    cudaMalloc(&d_Gradient, sizeof(RealType));
    cudaMalloc(&d_Hessian, sizeof(RealType));

    cudaMalloc(&d_NWeight, sizeof(RealType) * N);    
    
    std::cout << "Initialize CUDA vector \n";
    */
}

template <typename RealType>
void CudaKernel<RealType>::updateXBeta(const thrust::device_vector<RealType>& X, const thrust::device_vector<int>& K, unsigned int offX, unsigned int offK, const unsigned int taskCount, RealType delta, thrust::device_vector<RealType>& dXBeta, thrust::device_vector<RealType>& dExpXBeta, int gridSize, int blockSize)
{
    kernelUpdateXBeta<<<gridSize, blockSize>>>(offX, offK, taskCount, delta, thrust::raw_pointer_cast(&X[0]), thrust::raw_pointer_cast(&K[0]), thrust::raw_pointer_cast(&dXBeta[0]), thrust::raw_pointer_cast(&dExpXBeta[0]));
}

template <typename RealType>
void CudaKernel<RealType>::computeGradientAndHessian(thrust::device_vector<RealType>& d_AccNumer, thrust::device_vector<RealType>& d_AccNumer2, thrust::device_vector<RealType>& d_AccDenom, thrust::device_vector<RealType>& d_NWeight, thrust::device_vector<RealType>& d_Gradient, thrust::device_vector<RealType>& d_Hessian, size_t& N, int& gridSize, int& blockSize)
{
typedef thrust::tuple<RealType,RealType> Tup2;
typedef typename thrust::device_vector<RealType>::iterator VecItr;
typedef thrust::tuple<VecItr,VecItr,VecItr,VecItr> TupVec4;
typedef thrust::zip_iterator<TupVec4> ZipVec4;

    d_AccDenom[N] = static_cast<RealType>(1); // avoid nan
/*
    // thrust transform reduction
    functorCGH<RealType> cGAH;
    thrust::tuple<RealType, RealType> init = thrust::make_tuple<RealType, RealType>(0, 0);
    thrust::tuple<RealType, RealType> gh = thrust::transform_reduce(
                     thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(), 
                                                                  d_AccNumer.begin(), 
                                                                  d_AccDenom.begin(), 
                                                                  d_AccNumer2.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.end(), 
                                                                  d_AccNumer.end(), 
                                                                  d_AccDenom.end(), 
                                                                  d_AccNumer2.end())),
                     cGAH,
                     init,
                     TuplePlus());
 
    d_Gradient[0] = thrust::get<0>(gh);
    d_Hessian[0] = thrust::get<1>(gh);
*/
    // cub transfrom reduction

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.begin(),
                                                              d_AccNumer.begin(),
                                                              d_AccDenom.begin(),
                                                              d_AccNumer2.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_NWeight.end(),
                                                            d_AccNumer.end(),
                                                            d_AccDenom.end(),
                                                            d_AccNumer2.end()));
    thrust::tuple<RealType, RealType> init = thrust::make_tuple<RealType, RealType>(0, 0);
    auto gh = thrust::make_zip_iterator(thrust::make_tuple(d_Gradient.begin(), d_Hessian.begin()));

    // transform iterator
    functorCGH<RealType> cGAH;
    TransformInputIterator<Tup2, functorCGH<RealType>, ZipVec4> itr(begin, cGAH);

    // reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, itr, gh, N, TuplePlus(), init);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, itr, gh, N, TuplePlus(), init);

    cudaFree(d_temp_storage);

//    std::cout << "G: " << d_Gradient[0] << " H: " << d_Hessian[0] << '\n';
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
//void CudaKernel<RealType>::CubScan(thrust::device_vector<RealType>& d_in, thrust::device_vector<RealType>& d_out, int num_items)
void CudaKernel<RealType>::CubScan(RealType* d_in, RealType* d_out, int num_items)
{
    // Allocate temporary storage
    void *d_temp_storage0 = NULL;
    size_t temp_storage_bytes0 = 0;

    // Determine temporary device storage requirements
    //DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, thrust::raw_pointer_cast(&d_in[0]), thrust::raw_pointer_cast(&d_out[0]), num_items);
    DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage0, temp_storage_bytes0);

    // Launch kernel
    DeviceScan::InclusiveSum(d_temp_storage0, temp_storage_bytes0, d_in, d_out, num_items);

    cudaFree(d_temp_storage0);

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

