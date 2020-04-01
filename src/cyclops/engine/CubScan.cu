//
// Created by Jianxiao Yang on 2020-03-30.
//

#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>

#include <cub/cub.cuh>
#include "CubScan.h"

using namespace cub;

void CubScan(float* h_in, float* h_out, int N)
{

    // Allocate device arrays
    float *d_in;
    float *d_out;
    cudaMalloc(&d_in,  sizeof(float) * N);
    cudaMalloc(&d_out, sizeof(float) * N);

    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // {{{  Launch kernel

    float timerG = 0;
    auto startG = std::chrono::steady_clock::now();

    // Run
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    auto durationG = std::chrono::steady_clock::now() - startG;
    timerG = std::chrono::duration<float, std::milli>(durationG).count();

    // }}}

    // Copy the results to host
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

//    std::cout << "GPU takes " << timerG << "ms" << '\n';

}

void CubScan(double* h_in, double* h_out, int N)
{

    // Allocate device arrays
    double *d_in;
    double *d_out;
    cudaMalloc(&d_in,  sizeof(double) * N);
    cudaMalloc(&d_out, sizeof(double) * N);

    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(double) * N, cudaMemcpyHostToDevice);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // {{{  Launch kernel

    double timerG = 0;
    auto startG = std::chrono::steady_clock::now();

    // Run
    DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    auto durationG = std::chrono::steady_clock::now() - startG;
    timerG = std::chrono::duration<double, std::milli>(durationG).count();

    // }}}

    // Copy the results to host
    cudaMemcpy(h_out, d_out, sizeof(double) * N, cudaMemcpyDeviceToHost);

//    std::cout << "GPU takes " << timerG << "ms" << '\n';

}
