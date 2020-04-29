#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "CudaKernel.h"

using namespace cub;
	
template <typename T>
__global__ void kernelUpdateXBeta(int offX, int offK, int N, T delta,
                T* d_X, int* K, T* d_XBeta, T* d_ExpXBeta)
//__global__ void kernelUpdateXBeta(T* d_X, T* d_XBeta, T* d_ExpXBeta, T delta, int N)
{
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    //if (formatType == INDICATOR || formatType == SPARSE) {
//	int k = K[offK + task];
    //} else { // DENSE, INTERCEPT
	int k = task;
    //}

    //if (formatType == SPARSE || formatType == DENSE) {
	T inc = delta * d_X[offX + task];
    //} else { // INDICATOR, INTERCEPT
//	T inc = delta;
    //}

    if (task < N) {
	T xb = d_XBeta[k] + inc;
        d_XBeta[k] = xb;
	d_ExpXBeta[k] = expf(xb);
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
CudaKernel<T>::~CudaKernel()
{
//    std::cout << "CUDA class Destroyed \n";
}

template <class T>
void CudaKernel<T>::updateXBeta(unsigned int offX, unsigned int offK, unsigned int N, T delta, int gridSize, int blockSize)
{
//    auto start1 = std::chrono::steady_clock::now();

    kernelUpdateXBeta<<<gridSize, blockSize>>>(offX, offK, N, delta, d_X, d_K, d_XBeta, d_ExpXBeta);

//    auto end1 = std::chrono::steady_clock::now();
//    timerG1 += std::chrono::duration<double, std::milli>(end1 - start1).count();
}

template <class T>
void CudaKernel<T>::CubScanMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class T>
void CudaKernel<T>::CubScan(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ExpXBeta, d_AccDenom, num_items);
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

