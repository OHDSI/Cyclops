#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include "CudaKernel.h"

using namespace cub;

CudaKernel::CudaKernel(float* h_in, int num_items)
{

    // Allocate device arrays
    cudaMalloc(&d_in,  sizeof(float) * num_items);
    cudaMalloc(&d_out, sizeof(float) * num_items);

    // Copy input from host to device
    cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice);

//    std::cout << "Created \n";
}

CudaKernel::CudaKernel(double* h_in, int num_items)
{

    // Allocate device arrays
    cudaMalloc(&d_in,  sizeof(double) * num_items);
    cudaMalloc(&d_out, sizeof(double) * num_items);

    // Copy input from host to device
    cudaMemcpy(d_in, h_in, sizeof(double) * num_items, cudaMemcpyHostToDevice);

//    std::cout << "Created \n";
}

CudaKernel::~CudaKernel()
{
//    std::cout << "Destroyed \n";
}

void CudaKernel::CubScanMalloc(int num_items)
{
    // Determine temporary device storage requirements
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

void CudaKernel::CubScan(int num_items)
{
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
}

