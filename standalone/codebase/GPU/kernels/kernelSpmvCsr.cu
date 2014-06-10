/*
 * This code is adopted from Bell and Garland, NVIDIA Corporation
 * under the Apache License, Version 2.0:
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * @author Marc Suchard
 */

#pragma once

#include "GPU/GPUImplDefs.h"

#ifdef __cplusplus
extern "C" {
#endif

//#include "sparse_formats.h"
//#include "texture.h"
//#include "kernels/spmv_common_device.cu.h"

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is 
//   used for accessing the x vector.

//template <typename unsigned int, typename REAL, unsigned int CSR_BLOCK_SIZE, bool UseCache>
__global__ void
spmv_csr_vector_kernel(const unsigned int * Ap,
#ifndef NO_COLUMNS
                       const unsigned int * Aj,
#endif
#ifndef IS_INDICATOR_MATRIX
                       const REAL * Ax,
#endif
                       const REAL * x,
                             REAL * y,
                       const unsigned int num_rows)
{
    __shared__ REAL sdata[CSR_BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
    __shared__ unsigned int ptrs[CSR_BLOCK_SIZE/WARP_SIZE][2];
    
    const unsigned int thread_id   = CSR_BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    const unsigned int warp_id     = thread_id   / WARP_SIZE;                // global warp index
    const unsigned int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    const unsigned int num_warps   = (CSR_BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

    for(unsigned int row = warp_id; row < num_rows; row += num_warps){
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
        const unsigned int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
        const unsigned int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

        // compute local sum
        REAL sum = 0;
        for(unsigned int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
#ifdef IS_INDICATOR_MATRIX
#ifdef NO_COLUMNS
        	sum += x[jj];
#else
        	sum += x[Aj[jj]];
#endif
#else
#ifdef NO_COLUMNS
        	sum += Ax[jj] * x[jj];
#else
        	sum += Ax[jj] * x[Aj[jj]];
#endif
#endif

        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x] = sum;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; //EMUSYNC;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; //EMUSYNC;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; //EMUSYNC;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; //EMUSYNC;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; //EMUSYNC;
       
        // first thread writes warp result
        if (thread_lane == 0)
            y[row] = sdata[threadIdx.x];
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

