#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include "CudaKernel.h"

using namespace cub;

template <class T>
CudaKernel<T>::CudaKernel(T* h_in, int num_items)
{

    // Allocate device arrays
    cudaMalloc(&d_in,  sizeof(T) * num_items);
    cudaMalloc(&d_out, sizeof(T) * num_items);

    // Copy input from host to device
    cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice);

//    std::cout << "Created \n";
}

template <class T>
CudaKernel<T>::~CudaKernel()
{
//    std::cout << "Destroyed \n";
}

template <class T>
void CudaKernel<T>::CubScanMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

template <class T>
void CudaKernel<T>::CubScan(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
}

template class CudaKernel<float>;
template class CudaKernel<double>;

