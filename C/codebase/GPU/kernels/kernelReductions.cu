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

#endif // _MVSPAGGREGATE_KERNEL_H_

