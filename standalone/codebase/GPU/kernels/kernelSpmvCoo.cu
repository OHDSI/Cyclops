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

// segmented reduction in shared memory
__device__ REAL segreduce_warp(
		const unsigned int thread_lane,
		unsigned int row,
		REAL val,
		unsigned int * rows,
		REAL * vals) {

    rows[threadIdx.x] = row;
    vals[threadIdx.x] = val;

    if( thread_lane >=  1 && row == rows[threadIdx.x -  1] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; } 
    if( thread_lane >=  2 && row == rows[threadIdx.x -  2] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
    if( thread_lane >=  4 && row == rows[threadIdx.x -  4] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
    if( thread_lane >=  8 && row == rows[threadIdx.x -  8] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
    if( thread_lane >= 16 && row == rows[threadIdx.x - 16] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

    return val;
}

__device__ void segreduce_block(
		const unsigned int * idx,
		REAL * val) {

    REAL left = 0;
    if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) { left = val[threadIdx.x -   1]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) { left = val[threadIdx.x -   2]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) { left = val[threadIdx.x -   4]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) { left = val[threadIdx.x -   8]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) { left = val[threadIdx.x -  16]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) { left = val[threadIdx.x -  32]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) { left = val[threadIdx.x -  64]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) { left = val[threadIdx.x - 128]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) { left = val[threadIdx.x - 256]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
}

//////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_coo_flat_kernel
//   The input coo_matrix must be sorted by row.  Columns within each row
//   may appear in any order and duplicate entries are also acceptable.
//   A segmented reduction is used to compute the per-row
//   sums.
//
// spmv_coo_serial_kernel
//	 The same as above, but for very small numbers of entries (<= WARPSIZE)
//
__global__ void
spmv_coo_serial_kernel(const unsigned int * I,
                       const unsigned int * J,
#ifndef IS_INDICATOR_MATRIX
                       const REAL * V,
#endif
                       const REAL * x,
                             REAL * y,
                       const unsigned int num_nonzeros)
{
    for(unsigned int n = 0; n < num_nonzeros; n++){
#ifdef IS_INDICATOR_MATRIX
    	y[I[n]] += x[J[n]];
#else
    	y[I[n]] += V[n] * x[J[n]];
#endif
    }
}

__global__ void
spmv_coo_flat_kernel(const unsigned int * I,
                     const unsigned int * J,
#ifndef IS_INDICATOR_MATRIX
                     const REAL * V,
#endif
                     const REAL * x,
                           REAL * y,
                           unsigned int * temp_rows,
                           REAL * temp_vals,
                     const unsigned int num_nonzeros,
                     const unsigned int interval_size)
{
    __shared__ unsigned int rows[COO_BLOCK_SIZE];
    __shared__ REAL vals[COO_BLOCK_SIZE];

    const unsigned int thread_id   = COO_BLOCK_SIZE * blockIdx.x + threadIdx.x;                 // global thread index
    const unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1);                           // thread index within the warp
    const unsigned int warp_id     = thread_id   / WARP_SIZE;                               // global warp index

    const unsigned int interval_begin = warp_id * interval_size;                            // warp's offset into I,J,V
    const unsigned int interval_end   = min(interval_begin + interval_size, num_nonzeros);  // end of warps's work

    if(interval_begin >= interval_end)                                                   // warp has no work to do 
        return;

    if (thread_lane == 31){
        // initialize the carry in values
        rows[threadIdx.x] = I[interval_begin]; 
        vals[threadIdx.x] = 0;
    }
  
    for(unsigned int n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        unsigned int row = I[n];                                         // row index (i)
#ifdef IS_INDICATOR_MATRIX
        REAL val = x[J[n]];
#else
        REAL val = V[n] * x[J[n]];        					  			// A(i,j) * x(j)
#endif
        
        if (thread_lane == 0)
        {
            if(row == rows[threadIdx.x + 31])
                val += vals[threadIdx.x + 31];                        // row continues
            else
                y[rows[threadIdx.x + 31]] += vals[threadIdx.x + 31];  // row terminated
        }
        
        val = segreduce_warp(thread_lane, row, val, rows, vals);      // segmented reduction in shared memory

        if(thread_lane < 31 && row != rows[threadIdx.x + 1])
            y[row] += val;                                            // row terminated
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_rows[warp_id] = rows[threadIdx.x];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}

// The second level of the segmented reduction operation
__global__ void
spmv_coo_reduce_update_kernel(const unsigned int * temp_rows,
                              const REAL * temp_vals,
                                    REAL * y,
                              const unsigned int num_warps)
{
    __shared__ unsigned int rows[COO_BLOCK_SIZE + 1];
    __shared__ REAL vals[COO_BLOCK_SIZE + 1];

    const unsigned int end = num_warps - (num_warps & (COO_BLOCK_SIZE - 1));

    if (threadIdx.x == 0){
        rows[COO_BLOCK_SIZE] = (unsigned int) -1;
        vals[COO_BLOCK_SIZE] = (REAL)  0;
    }
    
    __syncthreads();

    unsigned int i = threadIdx.x;

    while (i < end){
        // do full blocks
        rows[threadIdx.x] = temp_rows[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        segreduce_block(rows, vals);

        if (rows[threadIdx.x] != rows[threadIdx.x + 1])
            y[rows[threadIdx.x]] += vals[threadIdx.x];

        __syncthreads();

        i += COO_BLOCK_SIZE;
    }

    if (end < num_warps){
        if (i < num_warps){
            rows[threadIdx.x] = temp_rows[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
            rows[threadIdx.x] = (unsigned int) -1;
            vals[threadIdx.x] = (REAL)  0;
        }

        __syncthreads();
   
        segreduce_block(rows, vals);

        if (i < num_warps)
            if (rows[threadIdx.x] != rows[threadIdx.x + 1])
                y[rows[threadIdx.x]] += vals[threadIdx.x];
    }
}

#ifdef __cplusplus
} // extern "C"
#endif
