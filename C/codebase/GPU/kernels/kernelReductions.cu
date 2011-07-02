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

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

__device__ inline void parallelReduction(REAL* sdata, REAL& mySum) {
	const int tid = threadIdx.x;
	
	// Store current value and then reduce
    sdata[tid] = mySum;
    __syncthreads();

    if (WORK_BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (WORK_BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (WORK_BLOCK_SIZE >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // Declare shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile REAL* smem = sdata;
        if (WORK_BLOCK_SIZE >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (WORK_BLOCK_SIZE >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (WORK_BLOCK_SIZE >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (WORK_BLOCK_SIZE >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (WORK_BLOCK_SIZE >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (WORK_BLOCK_SIZE >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }	
}

__device__ inline void transform(
        const REAL numer,
        const REAL denom,
        const int nevents,
        REAL& gradient,
        REAL& hessian
) {
    REAL t = numer / denom;
    REAL g = nevents * t;
    gradient = g;
    hessian = g * (REAL(1.0) - t);
}

__global__ void kernelComputeGradientAndHessianWithReduction(
        const REAL* iNumer,
        const REAL* iDenom,
        const int* iNEvents,
        REAL* oGradient,
        REAL* oHessian,
        unsigned int length,
        unsigned int blockSize_not_used
) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < length) {	

    __shared__ REAL sgradient[WORK_BLOCK_SIZE];
    __shared__ REAL shessian[WORK_BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * WORK_BLOCK_SIZE + threadIdx.x;
    unsigned int gridSize = WORK_BLOCK_SIZE * gridDim.x;
    
    REAL mySumGradient = REAL(0.0);
    REAL mySumHessian = REAL(0.0);
    
    while (i < length) {
	    // Do work of this entry
	    REAL gradient;
	    REAL hessian;
	    transform(iNumer[i], iDenom[i], iNEvents[i], gradient, hessian);
//	    oGradient[i] = gradient;
//	    oHessian[i] = hessian;
	    
	    // Add to local thread sum
	    mySumGradient += gradient;
	    mySumHessian += hessian;
	    i += gridSize;
	}
	
	// Reduce across threads in block
	parallelReduction(sgradient, mySumGradient);
	parallelReduction(shessian, mySumHessian);
	
 	if (tid == 0) {
 	    oGradient[blockIdx.x] = sgradient[0];
 	    oHessian[blockIdx.x] = shessian[0];
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

