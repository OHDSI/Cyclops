/*
 * kernelReductions.h
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */


#ifndef _MVSPAGGREGATE_KERNEL_H_
#define _MVSPAGGREGATE_KERNEL_H_

__global__ void kernelReduceFast(
				REAL *x,
				const unsigned int *rowIndices, 
				const REAL *y,
				const unsigned int numRows
				) {
	
	unsigned int tid = threadIdx.y;
	unsigned int bid = blockIdx.y;
	
	unsigned int ind2Dx = tid & (HALFWARP-1);
	unsigned int ind2Dy = tid >> HALFWARP_LOG2;
	
	unsigned int ub, lb;
	unsigned int myblock = bid * (BLOCK_SIZE_ROW / HALFWARP);
	unsigned int myi = myblock + ind2Dy;
	
	__shared__ int rowInd[(BLOCK_SIZE_ROW/HALFWARP)+1];
	__shared__ REAL tempProd[(BLOCK_SIZE_ROW/HALFWARP)][HALFWARP+PAD];

	if ((tid <= ((BLOCK_SIZE_ROW/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock + tid];
	
	__syncthreads();
	
	REAL t = 0;
	lb = rowInd[ind2Dy] + ind2Dx;
	ub = rowInd[ind2Dy + 1];
	
	if (myi < numRows) {
		for (unsigned int j = lb; j < ub; j += HALFWARP) {
			t += y[j];
		}
		tempProd[ind2Dy][ind2Dx] = t;
	}
	
	__syncthreads();

	// Works for HALFWARP=16
    if ((ind2Dx == 0) && (myi < numRows)) {
        t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] +
        	tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +
        	tempProd[ind2Dy][4] + tempProd[ind2Dy][5] +
        	tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +
        	tempProd[ind2Dy][8] + tempProd[ind2Dy][9] +
        	tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+
        	tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+
        	tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
        x[myi] = t;
    }
}
    
__global__ void kernelComputeIntermediatesAndReduceFast(
				REAL *x,
				const unsigned int *rowIndices,
				const int *offs,
				const REAL *xBeta,
				REAL *offsXBeta,
				const unsigned int numRows
				) {

	unsigned int tid = threadIdx.y;
	unsigned int bid = blockIdx.y;

	unsigned int ind2Dx = tid & (HALFWARP-1);
	unsigned int ind2Dy = tid >> HALFWARP_LOG2;

	unsigned int ub, lb;
	unsigned int myblock = bid * (BLOCK_SIZE_ROW / HALFWARP);
	unsigned int myi = myblock + ind2Dy;

	__shared__ int rowInd[(BLOCK_SIZE_ROW/HALFWARP)+1];
	__shared__ REAL tempProd[(BLOCK_SIZE_ROW/HALFWARP)][HALFWARP+PAD];

	if ((tid <= ((BLOCK_SIZE_ROW/HALFWARP))) && (myi < numRows))
		rowInd[tid] = rowIndices[myblock + tid];

	__syncthreads();

	REAL t = 0;
	lb = rowInd[ind2Dy] + ind2Dx;
	ub = rowInd[ind2Dy + 1];

	if (myi < numRows) {
		for (unsigned int j = lb; j < ub; j += HALFWARP) {
			REAL value = offs[j] * exp(xBeta[j]);
			offsXBeta[j] = value;
			t += value;
		}
		tempProd[ind2Dy][ind2Dx] = t;
	}

	__syncthreads();

	// Works for HALFWARP=16
    if ((ind2Dx == 0) && (myi < numRows)) {
        t = tempProd[ind2Dy][0] + tempProd[ind2Dy][1] +
        	tempProd[ind2Dy][2] + tempProd[ind2Dy][3] +
        	tempProd[ind2Dy][4] + tempProd[ind2Dy][5] +
        	tempProd[ind2Dy][6] + tempProd[ind2Dy][7] +
        	tempProd[ind2Dy][8] + tempProd[ind2Dy][9] +
        	tempProd[ind2Dy][10]+ tempProd[ind2Dy][11]+
        	tempProd[ind2Dy][12]+ tempProd[ind2Dy][13]+
        	tempProd[ind2Dy][14]+ tempProd[ind2Dy][15];
        x[myi] = t;
    }
}

__global__ void kernelReduceSum(
				REAL * g_idata, 
				REAL * g_odata, 
				unsigned int n,
				unsigned int blockSize) {

    __shared__ REAL sdata[512];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    REAL mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {    
	mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile REAL* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; __syncthreads(); }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; __syncthreads(); }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; __syncthreads(); }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; __syncthreads(); }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; __syncthreads(); }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; __syncthreads(); }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


#endif // _MVSPAGGREGATE_KERNEL_H_

