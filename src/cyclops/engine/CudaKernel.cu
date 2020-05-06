#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "CudaKernel.h"

using namespace cub;
	
template <typename T>
__global__ void kernelUpdateXBeta(int offX, int offK, const int taskCount, T delta,
                T* d_X, int* K, T* d_XBeta, T* d_ExpXBeta)
//__global__ void kernelUpdateXBeta(T* d_X, T* d_XBeta, T* d_ExpXBeta, T delta, int N)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    //if (formatType == INDICATOR || formatType == SPARSE) {
	int k = K[offK + task];
    //} else { // DENSE, INTERCEPT
//	int k = task;
    //}

    //if (formatType == SPARSE || formatType == DENSE) {
//	T inc = delta * d_X[offX + task];
    //} else { // INDICATOR, INTERCEPT
	T inc = delta;
    //}

    if (task < taskCount) {
	T xb = d_XBeta[k] + inc;
        d_XBeta[k] = xb;
	d_ExpXBeta[k] = expf(xb);
    }
}

template <typename T>
__global__ void kernelComputeGradientAndHessian(T* d_Gradient, T* d_Hessian, T* d_AccNumer, T* d_AccNumer2, T* d_AccDenom, T* d_NWeight, int N)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    if (task < N) {
        T t = d_AccNumer[task] / d_AccDenom[task];
        T g = d_NWeight[task] * t;
        d_Gradient[task] = g;
        //if (IteratorType::isIndicator) {
            d_Hessian[task] = g * (1.0 - t);
        //} else {
//	    d_Hessian[task] = d_NWeight[task] * (d_AccNumer2[task] / d_AccDenom[task] - t * t);
        //}
    }
}

template <class T>
CudaKernel<T>::CudaKernel(thrust::device_vector<T>& X, thrust::device_vector<int>& K, T* h_XBeta, T* h_ExpXBeta, int num_items)
{
//    std::cout << "X size: " << sizeof(X) << " T size: " << sizeof(T) << '\n';
//    std::cout << "K size: " << sizeof(K) << " int size: " << sizeof(int) << '\n';
    
    // Allocate device arrays
    cudaMalloc(&d_XBeta,  sizeof(T) * num_items);
    cudaMalloc(&d_ExpXBeta,  sizeof(T) * num_items);
    cudaMalloc(&d_AccDenom, sizeof(T) * num_items);

    // Copy input from host to device
    d_X = thrust::raw_pointer_cast(&X[0]);
    d_K = thrust::raw_pointer_cast(&K[0]);
    cudaMemcpy(d_XBeta, h_XBeta, sizeof(T) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ExpXBeta, h_ExpXBeta, sizeof(T) * num_items, cudaMemcpyHostToDevice);
//    std::cout << "CUDA class Created \n";
}

template <class T>
CudaKernel<T>::CudaKernel(T* h_Numer, T* h_Numer2, T* h_AccDenom, T* h_NWeight, int num_items)
{
    // Allocate device arrays
    cudaMalloc(&d_Numer,  sizeof(T) * num_items);
    cudaMalloc(&d_Numer2,  sizeof(T) * num_items);
    cudaMalloc(&d_AccDenom, sizeof(T) * num_items);
    cudaMalloc(&d_NWeight, sizeof(T) * num_items);

    cudaMalloc(&d_AccNumer, sizeof(T) * num_items);
    cudaMalloc(&d_AccNumer2, sizeof(T) * num_items);
    cudaMalloc(&d_Gradient, sizeof(T) * num_items);
    cudaMalloc(&d_Hessian, sizeof(T) * num_items);
    cudaMalloc(&d_G, sizeof(T));
    cudaMalloc(&d_H, sizeof(T));


    // Copy input from host to device
    cudaMemcpy(d_Numer, h_Numer, sizeof(T) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Numer2, h_Numer2, sizeof(T) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_AccDenom, h_AccDenom, sizeof(T) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_NWeight, h_NWeight, sizeof(T) * num_items, cudaMemcpyHostToDevice);

//    std::cout << "CUDA class Created \n";
}

template <class T>
CudaKernel<T>::~CudaKernel()
{
//    std::cout << "CUDA class Destroyed \n";
}

template <class T>
void CudaKernel<T>::updateXBeta(unsigned int offX, unsigned int offK, const unsigned int taskCount, T delta, int gridSize, int blockSize)
{
//    auto start1 = std::chrono::steady_clock::now();

    kernelUpdateXBeta<<<gridSize, blockSize>>>(offX, offK, taskCount, delta, d_X, d_K, d_XBeta, d_ExpXBeta);

//    auto end1 = std::chrono::steady_clock::now();
//    timerG1 += std::chrono::duration<double, std::milli>(end1 - start1).count();
}

template <class T>
void CudaKernel<T>::computeGradientAndHessian(size_t& N, int& gridSize, int& blockSize)
{
//    auto start1 = std::chrono::steady_clock::now();

    kernelComputeGradientAndHessian<<<gridSize, blockSize>>>(d_Gradient, d_Hessian, d_AccNumer, d_AccNumer2, d_AccDenom, d_NWeight, N);

    CudaKernel<T>::CubReduce(d_Gradient, d_G, N);
    CudaKernel<T>::CubReduce(d_Hessian, d_H, N);

//    auto end1 = std::chrono::steady_clock::now();
//    timerG1 += std::chrono::duration<double, std::milli>(end1 - start1).count();
}

template <class T>
void CudaKernel<T>::CubReduce(T* d_in, T* d_out, int num_items)
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

template <class T>
void CudaKernel<T>::CubScan(T* d_in, T* d_out, int num_items)
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
    cudaFree(d_temp_storage0);
}

template <class T>
void CudaKernel<T>::computeAccDenomMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class T>
void CudaKernel<T>::computeAccDenom(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);
}

template <class T>
void CudaKernel<T>::computeAccNumerMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer, d_AccNumer, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class T>
void CudaKernel<T>::computeAccNumer(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer, d_AccNumer, num_items);
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_Numer2, d_AccNumer2, num_items);
}

template <class T>
void CudaKernel<T>::CubExpScanMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_XBeta, d_AccDenom, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class T>
void CudaKernel<T>::CubExpScan(int num_items)
{
//    auto start = std::chrono::steady_clock::now();

    TransformInputIterator<T, CustomExp, T*> d_itr(d_XBeta, exp_op);
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_itr, d_AccDenom, num_items);
    
//    auto end = std::chrono::steady_clock::now();
//    timerG += std::chrono::duration<double, std::milli>(end - start).count();
//    std::cout << "GPU takes " << timerG << " ms" << '\n';
}

template class CudaKernel<float>;
template class CudaKernel<double>;

