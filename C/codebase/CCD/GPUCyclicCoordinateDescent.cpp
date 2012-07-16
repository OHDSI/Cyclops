/*
 * GPUCyclicCoordinateDescent.cpp
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <cmath>
#include <exception>    // for exception, bad_exception
#include <stdexcept>    // for std exception hierarchy

#include "GPUCyclicCoordinateDescent.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncherCCD.h"

//#define CONTIG_MEMORY

using namespace std;

GPUCyclicCoordinateDescent::GPUCyclicCoordinateDescent(int deviceNumber, InputReader* reader,
		AbstractModelSpecifics& specifics)
	: CyclicCoordinateDescent(reader, specifics), hReader(reader) {
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::constructor\n");
#endif 	
    
	cout << "Running GPU version" << endl;
}

bool GPUCyclicCoordinateDescent::initializeDevice(int deviceNumber) {

	try {
	gpu = new GPUInterface;
	int gpuDeviceCount = 0;
	if (gpu->Initialize()) {
		gpuDeviceCount = gpu->GetDeviceCount();
		cout << "Number of GPU devices found: " << gpuDeviceCount << endl;
	} else {
		cerr << "Unable to initialize CUDA driver library!" << endl;
		return false;
	}

	if (deviceNumber < 0 || deviceNumber >= gpuDeviceCount) {
		cerr << "Unknown device number: " << deviceNumber << endl;
		return false;
	}

	int numberChars = 80;
	std::vector<char> charVectorName(numberChars);
	gpu->GetDeviceName(deviceNumber, &charVectorName[0], numberChars);
	string deviceNameString(&charVectorName[0]);

	std::vector<char> charVectorDesc(numberChars);
	gpu->GetDeviceDescription(deviceNumber, &charVectorDesc[0]);
	string deviceDescrString(&charVectorDesc[0]);

	cout << "Using " << deviceNameString << ": " << deviceDescrString << endl;

	kernels = new KernelLauncherCCD(gpu);
	gpu->SetDevice(deviceNumber, kernels->getKernelsString());
	kernels->LoadKernels();
	// GPU device is already to go!
	
	size_t initial = gpu->GetAvailableMemory();
	cerr << "Available = " << initial << endl;

#ifdef CONTIG_MEMORY
	int nonZero = 0;
	dXI = (GPUPtr*) malloc(J * sizeof(GPUPtr));
	vector<int> columnLength(J);
	for (int j = 0; j < J; j++) {
		columnLength[j] = hXI->getNumberOfEntries(j);
		nonZero += hXI->getNumberOfEntries(j);
	}
	GPUPtr ptr = gpu->AllocateIntMemory(nonZero);
	for (int j = 0; j < J; j++) {
		if (columnLength[j] != 0) {
			dXI[j] = ptr;
			gpu->MemcpyHostToDevice(dXI[j], hXI->getCompressedColumnVector(j),
					sizeof(int) * columnLength[j]);
			ptr += sizeof(int) * columnLength[j];
		} else {
			dXI[j] = NULL;
		}
	}
#else
	int nonZero = 0;
	dXI = (GPUPtr*) malloc(J * sizeof(GPUPtr));
	vector<int> columnLength(J);
	for (int j = 0; j < J; j++) {
		columnLength[j] = hXI->getNumberOfEntries(j);
		nonZero += hXI->getNumberOfEntries(j);
		if (columnLength[j] != 0) {
			dXI[j] = gpu->AllocateIntMemory(columnLength[j]);
			gpu->MemcpyHostToDevice(dXI[j], hXI->getCompressedColumnVector(j),
				sizeof(int) * columnLength[j]);
		} else {
			dXI[j] = NULL;
		}
	}
#endif
	cerr << "Used = " << (initial - gpu->GetAvailableMemory()) << endl;
//	dXColumnLength = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dXColumnLength, &columnLength[0], sizeof(int) * K);

//	cerr << "Memory allocate 1" << endl;
	// Allocate GPU memory for X and beta
//#ifndef NO_BETA
//	dBeta = gpu->AllocateRealMemory(J);
//	gpu->MemcpyHostToDevice(dBeta, hBeta, sizeof(REAL) * J); // Beta is never actually used on GPU
//#endif
//	cerr << "Available = " << gpu->GetAvailableMemory() << endl;

	dXBeta = gpu->AllocateRealMemory(K);
	gpu->MemcpyHostToDevice(dXBeta, hXBeta, sizeof(REAL) * K);
//	cerr << "Available = " << gpu->GetAvailableMemory() << endl;
//	cerr << "K = " << K << endl;
//	exit(-1);

//	cerr << "Memory allocate 2" << endl;
	// Allocate GPU memory for integer vectors
	dOffs = gpu->AllocateIntMemory(K);
	gpu->MemcpyHostToDevice(dOffs, hOffs, sizeof(int) * K);
//	dEta = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dEta, hY, sizeof(int) * K);
	dNEvents = gpu->AllocateIntMemory(N);
//	gpu->MemcpyHostToDevice(dNEvents, hNEvents, sizeof(int) * N); // Moved to computeNEvents
//	dPid = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dPid, hPid, sizeof(int) * K);

//	cerr << "Memory allocate 3" << endl;
	// Allocate GPU memory for intermediate calculations
	dOffsExpXBeta = gpu->AllocateRealMemory(K);
//	dXOffsExpXBeta = gpu->AllocateRealMemory(K);

	alignedN = getAlignedLength(N);
	dNumerPid = gpu->AllocateRealMemory(2 * alignedN);
	dDenomPid = dNumerPid  + sizeof(real) * alignedN; // GPUPtr is void* not real*

#ifdef GPU_SPARSE_PRODUCT
	std::vector<int> hNI; // temporary
	int totalLength = 0;
	maxNISize = 0;
	for (int j = 0; j < J; ++j) {
		const int size = sparseIndices[j]->size();
		if (size > maxNISize) {
			maxNISize = size;
		}
		totalLength += size;
	}
	avgNISize = totalLength / J;

//	std::cerr << "Original N = " << N << std::endl;
	std::cout << "Max patients per drug = " << maxNISize << std::endl;
	std::cout << "Avg patients per drug = " << avgNISize << std::endl;

	GPUPtr head = gpu->AllocateIntMemory(totalLength);
	dNI = (GPUPtr*) malloc(J * sizeof(GPUPtr));
	for (int j = 0; j < J; ++j) {
		const int n = sparseIndices[j]->size();
		if (n > 0) {
			dNI[j] = head;
			typedef std::vector<int>::iterator Iterator;
			Iterator it = sparseIndices[j]->begin();
			const Iterator end = sparseIndices[j]->end();
			for (; it != end; ++it) {
				hNI.push_back(*it);
			}
		} else {
			dNI[j] = NULL;
		}
		head += sizeof(int) * n; // GPUPtr is void* not int*
	}
	gpu->MemcpyHostToDevice(dNI[0], &hNI[0], sizeof(int) * totalLength);
	cacheSizeGH = kernels->getGradientAndHessianBlocks(
			maxNISize
//			avgNISize
			);
#else
	cacheSizeGH = kernels->getGradientAndHessianBlocks(N);
#endif

	alignedGHCacheSize = getAlignedLength(cacheSizeGH);
	dGradient = gpu->AllocateRealMemory(2 * alignedGHCacheSize);
	dHessian = dGradient + sizeof(real) * alignedGHCacheSize; // GPUPtr is void* not real*
	hGradient = (real*) malloc(2 * sizeof(real) * alignedGHCacheSize);
	hHessian = hGradient + alignedGHCacheSize;

//	cerr << "Memory allocate 5" << endl;
	// Allocate computed indices for sparse matrix operations
//	dXFullRowOffsets = gpu->AllocateIntMemory(N+1);
//	vector<int> rowOffsets(N + 1);
//	int offset = 0;
//	int currentPid = -1;
//	for (int i = 0; i < K; i++) {
//		int thisPid = hPid[i];
//		if (thisPid != currentPid) {
//			rowOffsets[thisPid] = offset;
//			currentPid = thisPid;
//		}
//		offset++;
//	}
//	rowOffsets[N] = offset;
//	gpu->MemcpyHostToDevice(dXFullRowOffsets, &rowOffsets[0], sizeof(int) * (N + 1));

//	cerr << "Memory allocate 6" << endl;

	initial = gpu->GetAvailableMemory();
	cerr << "Available = " << initial << endl;

	dXColumnRowIndicators = (GPUPtr*) malloc(J * sizeof(GPUPtr));
#ifdef CONTIG_MEMORY
	GPUPtr crPtr = gpu->AllocateIntMemory(nonZero);
#endif
	unsigned int maxActiveWarps = 0;
	for (int j = 0; j < J; j++) {
		const int n = hXI->getNumberOfEntries(j);
		if (n > 0) {
			if (n / WARP_SIZE > maxActiveWarps) { // TODO May be reduce for very large entries
				maxActiveWarps = n / WARP_SIZE;
			}
			vector<int> columnRowIndicators(n);
			const int* indicators = hXI->getCompressedColumnVector(j);
			for (int i = 0; i < n; i++) { // Loop through non-zero entries only
				int thisPid = hPid[indicators[i]];
				columnRowIndicators[i] = thisPid;
			}
#ifdef CONTIG_MEMORY
			dXColumnRowIndicators[j] = crPtr;
			crPtr += sizeof(int) * n;
#else
			dXColumnRowIndicators[j] = gpu->AllocateIntMemory(n);
#endif
			gpu->MemcpyHostToDevice(dXColumnRowIndicators[j],
					&columnRowIndicators[0], sizeof(int) * n);
		} else {
			dXColumnRowIndicators[j] = NULL;
		}
	}
	
	cerr << "Used = " << (initial - gpu->GetAvailableMemory()) << endl;

//	cerr << "Memory allocate 7" << endl;
	maxActiveWarps++;
//	dTmpCooRows = gpu->AllocateIntMemory(maxActiveWarps);
//	dTmpCooVals = gpu->AllocateRealMemory(maxActiveWarps);
	
//	cout << "MaxActiveWarps = " << maxActiveWarps << endl;

//	cerr << "Done with GPU memory allocation." << endl;
	
	//delete hReader;
	
	//hReader = reader; // Keep a local copy
	} catch (std::bad_alloc) {
		return false;
	}
	
	computeRemainingStatistics(true, 0);  // TODO Check index?  Probably not right.
	
	return true;

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::constructor\n");
#endif 	
}

GPUCyclicCoordinateDescent::~GPUCyclicCoordinateDescent() {

//	cerr << "1" << endl;
#ifdef CONTIG_MEMORY
	gpu->FreeMemory(dXI[0]);
#else
	if (dXI) {
		for (int j = 0; j < J; j++) {
			if (hXI->getNumberOfEntries(j) > 0) {
				gpu->FreeMemory(dXI[j]);
			}
		}
	}
#endif
	free(dXI);

//#ifndef NO_BETA
//	gpu->FreeMemory(dBeta);
//#endif
	gpu->FreeMemory(dXBeta);

	gpu->FreeMemory(dOffs);
//	gpu->FreeMemory(dEta);
	gpu->FreeMemory(dNEvents);
//	gpu->FreeMemory(dPid);
//	gpu->FreeMemory(dXFullRowOffsets);
	gpu->FreeMemory(dOffsExpXBeta);
//	gpu->FreeMemory(dXOffsExpXBeta);

//#ifdef MERGE_TRANSFORMATION
	gpu->FreeMemory(dNumerPid);
	gpu->FreeMemory(dGradient);
	free(hGradient);
//#else
//	gpu->FreeMemory(dDenomPid);
//	gpu->FreeMemory(dNumerPid);
//	gpu->FreeMemory(dT1);
//	gpu->FreeMemory(dGradient);
//	gpu->FreeMemory(dReducedGradientHessian);
//#endif

#ifdef GPU_SPARSE_PRODUCT
	if (dNI) {
		gpu->FreeMemory(dNI[0]);
	}
#endif


//	cerr << "4" << endl;
	if (dXColumnRowIndicators) {
#ifdef CONTIG_MEMORY
		gpu->FreeMemory(dXColumnRowIndicators[0]);
#else
		for (int j = 0; j < J; j++) {
			if (dXColumnRowIndicators[j]) {
				gpu->FreeMemory(dXColumnRowIndicators[j]); // TODO Causing error under Linux
			}
		}
#endif
	free(dXColumnRowIndicators);
	}

//	cerr << "5" << endl;
//	gpu->FreeMemory(dTmpCooRows);
//	gpu->FreeMemory(dTmpCooVals);

//	cerr << "6" << endl;
	delete kernels;
	delete gpu;
//	cerr << "7" << endl;
}

void GPUCyclicCoordinateDescent::resetBeta(void) {
	CyclicCoordinateDescent::resetBeta();
#ifndef NO_BETA
	gpu->MemcpyHostToDevice(dBeta, hBeta, sizeof(REAL) * J);
#endif
	gpu->MemcpyHostToDevice(dXBeta, hXBeta, sizeof(REAL) * K);
}

void GPUCyclicCoordinateDescent::computeNEvents(void) {
	CyclicCoordinateDescent::computeNEvents();
	gpu->MemcpyHostToDevice(dNEvents, hNEvents, sizeof(int) * N);
}

double GPUCyclicCoordinateDescent::getObjectiveFunction(void) {
//	return getLogLikelihood() + getLogPrior();
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::getObjectiveFunction\n");
#endif 	
    
	gpu->MemcpyDeviceToHost(hXBeta, dXBeta, sizeof(real) * K);
	return CyclicCoordinateDescent::getObjectiveFunction();
//	double criterion = 0;
//	for (int i = 0; i < K; i++) {
//		criterion += hXBeta[i] * hY[i];
//	}

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::getObjectiveFunction\n");
#endif
    
//    return criterion;
}

double GPUCyclicCoordinateDescent::computeZhangOlesConvergenceCriterion(void) {
	gpu->MemcpyDeviceToHost(hXBeta, dXBeta, sizeof(real) * K);

	// TODO Could do reduction on GPU
	return CyclicCoordinateDescent::computeZhangOlesConvergenceCriterion();
}

void GPUCyclicCoordinateDescent::updateXBeta(double delta, int index) {
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::updateXBeta\n");
#endif   	
	// Separate function for benchmarking
	hBeta[index] += delta;
	REAL gpuBeta = hBeta[index];
#ifndef NO_BETA
	gpu->MemcpyHostToDevice(dBeta + sizeof(REAL) * index, &gpuBeta, sizeof(REAL));
#endif
	
	const int n = hXI->getNumberOfEntries(index);

#ifdef TEST_SPARSE	
	kernels->updateXBetaAndFriends(dXBeta,  dOffsExpXBeta,  dDenomPid, dXColumnRowIndicators[index], NULL, dOffs, dXI[index], n, delta);
#else
	kernels->updateXBeta(dXBeta, dXI[index], n, delta);
#endif

#ifdef PROFILE_GPU
	gpu->Synchronize();
#endif
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUCylicCoordinateDescent::updateXBeta\n");
#endif  	
}

void GPUCyclicCoordinateDescent::computeRemainingStatistics(bool allStats, int index) {

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering  GPUCylicCoordinateDescent::computeRemainingStatistics\n");
#endif  

    if (allStats) {
    	// NEW
//    	kernels->computeIntermediates(dOffsExpXBeta, dDenomPid, dOffs, dXBeta, dXFullRowOffsets, K, N, allStats);
    	CyclicCoordinateDescent::computeRemainingStatistics(true, index);
    	gpu->MemcpyHostToDevice(dDenomPid, denomPid, sizeof(real) * N);
    	gpu->MemcpyHostToDevice(dOffsExpXBeta, offsExpXBeta, sizeof(real) * K);
    }

	sufficientStatisticsKnown = true;
#ifdef PROFILE_GPU
	gpu->Synchronize();
#endif

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUCylicCoordinateDescent::computeRemainingStatistics\n");
#endif  
	
}

void GPUCyclicCoordinateDescent::computeRatiosForGradientAndHessian(int index) {	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::computeRatiosForGradientAndHessian\n");
#endif  	
    
	// NEW
	const int n = hXI->getNumberOfEntries(index);
	kernels->computeDerivatives(
			dNumerPid,
			dXColumnRowIndicators[index],
			dXI[index],
			dOffsExpXBeta,
			dDenomPid,
			dT1,
			dGradient,
			dHessian,
			dNEvents,
			n,
			N,
			dTmpCooRows, dTmpCooVals);

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::computeRatiosForGradientAndHessian\n");
#endif  		
}

void GPUCyclicCoordinateDescent::getDenominators(void) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::getDenominators\n");
#endif  		
	gpu->MemcpyDeviceToHost(denomPid, dDenomPid, sizeof(real) * N);
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::getDenominators\n");
#endif 	
}

unsigned int nextPow2( unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

void GPUCyclicCoordinateDescent::computeGradientAndHession(int index, double *ogradient,
		double *ohessian) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::computeGradientAndHessian\n");
#endif 	
	real gradient = 0;
	real hessian = 0;

#ifdef GPU_SPARSE_PRODUCT
	int blockUsed = kernels->computeGradientAndHessianWithReductionSparse(dNumerPid, dDenomPid, dNEvents, dNI[index],
			dGradient, dHessian,
			sparseIndices[index]->size(),
//			N,
			1, SPARSE_WORK_BLOCK_SIZE);
	gpu->MemcpyDeviceToHost(hGradient, dGradient, sizeof(real) * 2 * alignedGHCacheSize);

	real* gradientCache = hGradient;
	const real* end = gradientCache + cacheSizeGH;
	real* hessianCache = hHessian;

	// TODO Remove code duplication with CPU version from here below
	for (; gradientCache != end; ++gradientCache, ++hessianCache) {
		gradient += *gradientCache;
		hessian += *hessianCache;
	}

#else
	// TODO dynamically determine threads/blocks.
	int blockUsed = kernels->computeGradientAndHessianWithReduction(dNumerPid, dDenomPid, dNEvents,
			dGradient, dHessian, N, 1, WORK_BLOCK_SIZE);
	gpu->MemcpyDeviceToHost(hGradient, dGradient, sizeof(real) * 2 * alignedGHCacheSize);

	real* gradientCache = hGradient;
	const real* end = gradientCache + cacheSizeGH;
	real* hessianCache = hHessian;

	// TODO Remove code duplication with CPU version from here below
	for (; gradientCache != end; ++gradientCache, ++hessianCache) {
		gradient += *gradientCache;
		hessian += *hessianCache;
	}
#endif

//  // Example of dynamic block sizes
//	unsigned int threads = (N < 512) ? nextPow2((N + 1)/ 2) : 256;
//	unsigned int blocks = (N + (threads * 2 - 1)) / (threads * 2);
//	blocks = (64 < blocks) ? 64 : blocks;
	
	gradient -= hXjY[index];
	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
	
#ifdef DP_DEBUG
	fprintf(stderr,"%5.3f %5.3f\n", *ogradient, *ohessian);
#endif
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::computeGradientAndHessian\n");
#endif 	
}

