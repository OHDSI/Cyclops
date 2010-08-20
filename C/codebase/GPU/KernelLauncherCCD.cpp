/*
 * @author Marc A. Suchard
 */

/**************INCLUDES***********/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "GPU/KernelLauncherCCD.h"
#include "GPU/kernels/CCDKernels.h"

using namespace std;

/**************CODE***********/

// ceil(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

KernelLauncherCCD::KernelLauncherCCD(GPUInterface* inGpu) :
	KernelLauncher(inGpu) {
}

KernelLauncherCCD::~KernelLauncherCCD() {
}

unsigned char* KernelLauncherCCD::getKernelsString() {
	return KERNELS_STRING;
}

void KernelLauncherCCD::SetupKernelBlocksAndGrids() {
	// Currently do nothing
}

void KernelLauncherCCD::LoadKernels() {
//	fDotProduct = gpu->GetFunction("kernelDotProduct");
	fUpdateXBeta = gpu->GetFunction("kernelUpdateXBeta");
	fComputeIntermediates = gpu->GetFunction("kernelComputeIntermediates");
//	fReduceAll = gpu->GetFunction("kernelReduceAll");
	fReduceFast = gpu->GetFunction("kernelReduceFast");
	fComputeAndReduceFast = gpu->GetFunction("kernelComputeIntermediatesAndReduceFast");
// 	fReduceRow = gpu->GetFunction("kernelReduceRow");

	fComputeRatio = gpu->GetFunction("kernelComputeRatio");
	fComputeGradientHessian = gpu->GetFunction("kernelComputeGradientHessian");

	fSpmvCooSerial = gpu->GetFunction("spmv_coo_serial_kernel");
	fSpmvCooFlat = gpu->GetFunction("spmv_coo_flat_kernel");
	fSpmvCooReduce = gpu->GetFunction("spmv_coo_reduce_update_kernel");

	fSpmvCsr = gpu->GetFunction("spmv_csr_vector_kernel");
	fReduceTwo = gpu->GetFunction("kernelReduceTwo");
	fReduceSum = gpu->GetFunction("kernelReduceSum");

	fClearMemory = gpu->GetFunction("kernelClear");
}

//void KernelLauncherCCD::dotProduct(GPUPtr oC, GPUPtr iA, GPUPtr iB, int length) {
//	Dim3Int block(16, 1);
//	Dim3Int grid(1, 1);
//	gpu->LaunchKernelIntParams(fDotProduct, block, grid, 4, oC, iA, iB, length);
//}

void KernelLauncherCCD::computeDerivatives(
		GPUPtr y, // numerPid
		GPUPtr rows, // row labels
		GPUPtr columns, // column labels
		GPUPtr x, // offsExpXBeta
    	GPUPtr denomPid,
    	GPUPtr t1,
#ifdef GRADIENT_HESSIAN_GPU
        GPUPtr gradient,
        GPUPtr hessian,
        GPUPtr nEvent,
#endif
    	int nElements,
    	int nRows,
    	GPUPtr tmpRows,
    	GPUPtr tmpVals) {

	clearMemory(y, nRows);
	computeSpmvCooIndicatorMatrix(y, rows, columns, x, nElements, tmpRows, tmpVals);

	int nBlocks = nRows / MAKE_RATIO_BLOCK_SIZE + // TODO Compute once
			(nRows % MAKE_RATIO_BLOCK_SIZE == 0 ? 0 : 1);
	Dim3Int block(MAKE_RATIO_BLOCK_SIZE);
	Dim3Int grid(nBlocks);
#ifdef GRADIENT_HESSIAN_GPU
	gpu->LaunchKernelParams(fComputeGradientHessian, block, grid, 5, 1, 0,
			y, denomPid, nEvent, gradient, hessian, nRows);
#else
	gpu->LaunchKernelIntParams(fComputeRatio, block, grid, 4,
			y, denomPid, t1, nRows);
#endif
#ifdef PROFILE_GPU
	gpu->Synchronize();
#endif
}

void KernelLauncherCCD::computeIntermediates(GPUPtr offsExpXBeta,
		GPUPtr denomPid, GPUPtr offs, GPUPtr xBeta, GPUPtr rowOffsets,
		int nRows, int nPatients) {
#ifdef MIN_GPU
	int nBlocksI = nRows / COMPUTE_INTERMEDIATES_BLOCK_SIZE + // TODO Compute once
	(nRows % COMPUTE_INTERMEDIATES_BLOCK_SIZE == 0 ? 0 : 1);
	Dim3Int blockI(COMPUTE_INTERMEDIATES_BLOCK_SIZE);
	Dim3Int gridI(nBlocksI);
	gpu->LaunchKernelIntParams(fComputeIntermediates, blockI, gridI, 6,
			offsExpXBeta, denomPid, offs, xBeta, rowOffsets, nRows);
#else
#if 1 // Fastest on 9400M
	int nBlocksI = nRows / COMPUTE_INTERMEDIATES_BLOCK_SIZE + // TODO Compute once
	(nRows % COMPUTE_INTERMEDIATES_BLOCK_SIZE == 0 ? 0 : 1);
	Dim3Int blockI(COMPUTE_INTERMEDIATES_BLOCK_SIZE);
	Dim3Int gridI(nBlocksI);
	gpu->LaunchKernelParams(fComputeIntermediates, blockI, gridI, 5, 1, 0,
			offsExpXBeta, denomPid, offs, xBeta, rowOffsets, nRows);
#if 0
	int nBlocksR = nPatients / BLOCK_SIZE_REDUCE_ALL + // TODO Compute once
	(nPatients % BLOCK_SIZE_REDUCE_ALL == 0 ? 0 : 1);
	Dim3Int blockR(BLOCK_SIZE_REDUCE_ALL);
	Dim3Int gridR(nBlocksR);
	gpu->LaunchKernelIntParams(fReduceAll, blockR, gridR, 4, offsExpXBeta, denomPid, rowOffsets, nPatients);
#else
#if 1 // Fastest on 9400M ???
	unsigned int gridParam;
	gridParam = (unsigned int) nPatients / (BLOCK_SIZE_ROW/HALFWARP);
	if ((gridParam * (BLOCK_SIZE_ROW/HALFWARP)) < nPatients) gridParam++;
	Dim3Int gridR(1, gridParam);
	Dim3Int blockR(1, BLOCK_SIZE_ROW);
	gpu->LaunchKernelParams(fReduceFast, blockR, gridR, 3, 1, 0,
			denomPid, rowOffsets, offsExpXBeta, nPatients);
#else
	computeSpmvCsrIndicatorMatrixNoColumns(denomPid, rowOffsets, offsExpXBeta, nPatients);
#endif
#endif
#else
	unsigned int gridParam;
	gridParam = (unsigned int) nPatients / BLOCK_SIZE_ROW * HALFWARP;
	if ((gridParam * BLOCK_SIZE_ROW / HALFWARP) < nPatients)
		gridParam++;
	Dim3Int gridR(1, gridParam);
	Dim3Int blockR(1, BLOCK_SIZE_ROW);
	gpu->LaunchKernelIntParams(fComputeAndReduceFast, blockR, gridR, 6,
			denomPid, rowOffsets, offs, xBeta, offsExpXBeta, nPatients);
#endif
#endif
#ifdef PROFILE_GPU
	gpu->Synchronize();
#endif
}

#define RT_BLOCK_SIZE	512

void KernelLauncherCCD::reduceTwo(
		GPUPtr oC,
		GPUPtr iX,
		unsigned int length) {
	Dim3Int block(RT_BLOCK_SIZE);
	Dim3Int grid(2);
	gpu->LaunchKernelParams(fReduceTwo, block, grid, 2, 1, 0,
			oC, iX, length);
}

void KernelLauncherCCD::reduceSum(
		GPUPtr d_idata,
		GPUPtr d_odata,
		unsigned int size,
		unsigned int blocks) {
	unsigned int threads = 256; // TODO Consider where this should be calculated
	Dim3Int block(threads);
	Dim3Int grid(blocks);
	gpu->LaunchKernelParams(fReduceSum, block, grid, 2, 2, 0, 
			d_idata, d_odata, size, threads);
}

void KernelLauncherCCD::updateXBeta(GPUPtr xBeta, GPUPtr xIColumn, int length,
		double delta) {

	int nBlocks = length / UPDATE_XBETA_BLOCK_SIZE + // TODO Compute once
			(length % UPDATE_XBETA_BLOCK_SIZE == 0 ? 0 : 1);
	Dim3Int block(UPDATE_XBETA_BLOCK_SIZE, 1);
	Dim3Int grid(nBlocks, 1);
	gpu->LaunchKernelParams(fUpdateXBeta, block, grid, 2, 1, 1, xBeta, xIColumn,
			length, delta);
}

void KernelLauncherCCD::clearMemory(
		GPUPtr dest,
		int length) {

	unsigned int nBlocks = length / CLEAR_MEMORY_BLOCK_SIZE +
			(length % CLEAR_MEMORY_BLOCK_SIZE == 0 ? 0 : 1);

	Dim3Int block(CLEAR_MEMORY_BLOCK_SIZE);
	Dim3Int grid(nBlocks);

	gpu->LaunchKernelParams(fClearMemory, block, grid, 1, 1, 0, dest, length);
}

void KernelLauncherCCD::computeSpmvCsrIndicatorMatrixNoColumns(
		GPUPtr y,
		GPUPtr rowOffsets,
		GPUPtr x,
		unsigned int nRows) {

    const unsigned int WARPS_PER_BLOCK = CSR_BLOCK_SIZE / WARP_SIZE;
    const unsigned int nBlocks = min((unsigned int)CSR_MAX_BLOCKS, DIVIDE_INTO(nRows, WARPS_PER_BLOCK));

    Dim3Int block(CSR_BLOCK_SIZE);
    Dim3Int grid(nBlocks);

    gpu->LaunchKernelParams(fSpmvCsr, block, grid, 3, 1, 0,
    		rowOffsets, x, y, nRows);
//
//    spmv_csr_vector_kernel<unsigned int, REAL, CSR_BLOCK_SIZE, false> <<<NUM_BLOCKS, CSR_BLOCK_SIZE>>>
//        (d_csr.num_rows, d_csr.Ap, d_csr.Aj, d_csr.Ax, d_x, d_y);
//
}

void KernelLauncherCCD::computeSpmvCooIndicatorMatrix(
    		GPUPtr y,
    		GPUPtr row,
    		GPUPtr column,
    		GPUPtr x,
    		unsigned int nNonZeros,
    		GPUPtr tmpRows,
    		GPUPtr tmpVals) {

	if (nNonZeros == 0) {
		// TODO Clear memory
		cerr << "Memory clear is not yet implemented!" << endl;
		exit(-1);
	} else if (nNonZeros < WARP_SIZE) { // Test if this is beneficial; very fast!
		Dim3Int blocks(1);
		Dim3Int grid(1);
		gpu->LaunchKernelParams(fSpmvCooSerial, blocks, grid, 4, 1, 0,
				row, column, x, y, nNonZeros);
		return;
	}

    const unsigned int num_units  = nNonZeros / WARP_SIZE;
    const unsigned int num_warps  = min(num_units, (unsigned int) COO_WARPS_PER_BLOCK * COO_MAX_BLOCKS);
    const unsigned int num_blocks = DIVIDE_INTO(num_warps, COO_WARPS_PER_BLOCK);
    const unsigned int num_iters  = DIVIDE_INTO(num_units, num_warps);

    const unsigned int interval_size = WARP_SIZE * num_iters;
    const unsigned int tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)
    const unsigned int active_warps = (interval_size == 0) ? 0 : DIVIDE_INTO(tail, interval_size);

    Dim3Int blockFlat(COO_BLOCK_SIZE);
    Dim3Int gridFlat(num_blocks);
 
	gpu->LaunchKernelParams(fSpmvCooFlat, blockFlat, gridFlat, 6, 2, 0,
			row, column, x, y, tmpRows, tmpVals, tail, interval_size);

	Dim3Int blockSerial(1);
	Dim3Int gridSerial(1);

	gpu->LaunchKernelParams(fSpmvCooSerial, blockSerial, gridSerial, 4, 1, 0,
			row + sizeof(int) * tail, column + sizeof(int) * tail, x, y, nNonZeros - tail);

	Dim3Int blockReduce(COO_BLOCK_SIZE);
	Dim3Int gridReduce(1);
	
	gpu->LaunchKernelParams(fSpmvCooReduce, blockReduce, gridReduce, 3, 1, 0,
			tmpRows, tmpVals, y, active_warps);
}

