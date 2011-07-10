/*
 * GPUCyclicCoordinateDescent.cpp
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */

#include <iostream>
#include <cmath>

#include "GPUCyclicCoordinateDescent.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncherCCD.h"

using namespace std;

GPUCyclicCoordinateDescent::GPUCyclicCoordinateDescent(int deviceNumber, InputReader* reader)
	: CyclicCoordinateDescent(reader) {
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUCylicCoordinateDescent::constructor\n");
#endif 	
    
	cout << "Running GPU version" << endl;

	gpu = new GPUInterface;
	int gpuDeviceCount = 0;
	if (gpu->Initialize()) {
		gpuDeviceCount = gpu->GetDeviceCount();
		cout << "Number of GPU devices found: " << gpuDeviceCount << endl;
	} else {
		cerr << "Unable to initialize CUDA driver library!" << endl;
		exit(-1);
	}

	if (deviceNumber < 0 || deviceNumber >= gpuDeviceCount) {
		cerr << "Unknown device number: " << deviceNumber << endl;
		exit(-1);
	}

	int numberChars = 80;
	std::vector<char> charVector(numberChars);
	gpu->GetDeviceName(deviceNumber, &charVector[0], numberChars);
	string deviceNameString(charVector.begin(), charVector.end());
	gpu->GetDeviceDescription(deviceNumber, &charVector[0]);
	string deviceDescrString(charVector.begin(), charVector.end());
	cout << "Using " << deviceNameString << ": " << deviceDescrString << endl;

	kernels = new KernelLauncherCCD(gpu);
	gpu->SetDevice(deviceNumber, kernels->getKernelsString());
	kernels->LoadKernels();
	// GPU device is already to go!
	
	hReader = reader;

//	cerr << "Memory allocate 0" << endl;
	// Allocate GPU memory for X
	cerr << "Available = " << gpu->GetAvailableMemory() << endl;
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
		}
	}
	cerr << "Available = " << gpu->GetAvailableMemory() << endl;
	cerr << "Nonzero = " << nonZero << endl;
//	dXColumnLength = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dXColumnLength, &columnLength[0], sizeof(int) * K);

//	cerr << "Memory allocate 1" << endl;
	// Allocate GPU memory for X and beta
//#ifndef NO_BETA
//	dBeta = gpu->AllocateRealMemory(J);
//	gpu->MemcpyHostToDevice(dBeta, hBeta, sizeof(REAL) * J); // Beta is never actually used on GPU
//#endif
	cerr << "Available = " << gpu->GetAvailableMemory() << endl;

	dXBeta = gpu->AllocateRealMemory(K);
	gpu->MemcpyHostToDevice(dXBeta, hXBeta, sizeof(REAL) * K);
	cerr << "Available = " << gpu->GetAvailableMemory() << endl;
	cerr << "K = " << K << endl;
//	exit(-1);

//	cerr << "Memory allocate 2" << endl;
	// Allocate GPU memory for integer vectors
	dOffs = gpu->AllocateIntMemory(K);
	gpu->MemcpyHostToDevice(dOffs, hOffs, sizeof(int) * K);
//	dEta = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dEta, hEta, sizeof(int) * K);
	dNEvents = gpu->AllocateIntMemory(N);
//	gpu->MemcpyHostToDevice(dNEvents, hNEvents, sizeof(int) * N); // Moved to computeNEvents
//	dPid = gpu->AllocateIntMemory(K);
//	gpu->MemcpyHostToDevice(dPid, hPid, sizeof(int) * K); // TODO Remove

//	cerr << "Memory allocate 3" << endl;
	// Allocate GPU memory for intermediate calculations
	dOffsExpXBeta = gpu->AllocateRealMemory(K);
//	dXOffsExpXBeta = gpu->AllocateRealMemory(K); // TODO Do we use this?  Remove


//#ifdef MERGE_TRANSFORMATION
	alignedN = getAlignedLength(N);
	cacheSizeGH = kernels->getGradientAndHessianBlocks(N);
	alignedGHCacheSize = getAlignedLength(cacheSizeGH);

	dNumerPid = gpu->AllocateRealMemory(2 * alignedN);
	dDenomPid = dNumerPid  + sizeof(real) * alignedN; // GPUPtr is void* not real*

	dGradient = gpu->AllocateRealMemory(2 * alignedGHCacheSize);
	dHessian = dGradient + sizeof(real) * alignedGHCacheSize; // GPUPtr is void* not real*
	hGradient = (real*) malloc(2 * sizeof(real) * alignedGHCacheSize);
	hHessian = hGradient + alignedGHCacheSize;
//#else
//	dDenomPid = gpu->AllocateRealMemory(N);
//	dNumerPid = gpu->AllocateRealMemory(N);
//	dT1 = gpu->AllocateRealMemory(N);
//	dGradient = gpu->AllocateRealMemory(2 * N);
//	dHessian = dGradient + sizeof(real) * N;
//	dReducedGradientHessian = gpu->AllocateRealMemory(2);
//	hGradient = (real*) malloc(sizeof(real) * N);
//	hHessian = (real*) malloc(sizeof(real) * N);
//#endif

//	cerr << "Memory allocate 5" << endl;
	// Allocate computed indices for sparse matrix operations
	dXFullRowOffsets = gpu->AllocateIntMemory(N+1);
	vector<int> rowOffsets(N + 1);
	int offset = 0;
	int currentPid = -1;
	for (int i = 0; i < K; i++) {
		int thisPid = hPid[i];
		if (thisPid != currentPid) {
			rowOffsets[thisPid] = offset;
			currentPid = thisPid;
		}
		offset++;
	}
	rowOffsets[N] = offset;
	gpu->MemcpyHostToDevice(dXFullRowOffsets, &rowOffsets[0], sizeof(int) * (N + 1));

//	cerr << "Memory allocate 6" << endl;
	dXColumnRowIndicators = (GPUPtr*) malloc(J * sizeof(GPUPtr));
//	hColumnRowLength = (int*) malloc(J * sizeof(int));
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
			dXColumnRowIndicators[j] = gpu->AllocateIntMemory(n);
			gpu->MemcpyHostToDevice(dXColumnRowIndicators[j],
					&columnRowIndicators[0], sizeof(int) * n);
		} else {
			dXColumnRowIndicators[j] = 0;
		}
	}
	
//	cerr << "Memory allocate 7" << endl;
	maxActiveWarps++;
//	dTmpCooRows = gpu->AllocateIntMemory(maxActiveWarps);
//	dTmpCooVals = gpu->AllocateRealMemory(maxActiveWarps);
	
	cout << "MaxActiveWarps = " << maxActiveWarps << endl;

//	cerr << "Done with GPU memory allocation." << endl;
	
	//delete hReader;
	
	//hReader = reader; // Keep a local copy
	
	computeRemainingStatistics(true);
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::constructor\n");
#endif 	
}

GPUCyclicCoordinateDescent::~GPUCyclicCoordinateDescent() {

//	cerr << "1" << endl;
	for (int j = 0; j < J; j++) {
		if (hXI->getNumberOfEntries(j) > 0) {
			gpu->FreeMemory(dXI[j]);
		}
	}
	free(dXI);

//#ifndef NO_BETA
//	gpu->FreeMemory(dBeta);
//#endif
	gpu->FreeMemory(dXBeta);

	gpu->FreeMemory(dOffs);
//	gpu->FreeMemory(dEta);
	gpu->FreeMemory(dNEvents);
//	gpu->FreeMemory(dPid);
	gpu->FreeMemory(dXFullRowOffsets);
	gpu->FreeMemory(dOffsExpXBeta);
//	gpu->FreeMemory(dXOffsExpXBeta);

#ifdef MERGE_TRANSFORMATION
	gpu->FreeMemory(dNumerPid);
	gpu->FreeMemory(dGradient);
	free(hGradient);
#else
	gpu->FreeMemory(dDenomPid);
	gpu->FreeMemory(dNumerPid);
	gpu->FreeMemory(dT1);
	gpu->FreeMemory(dGradient);
	gpu->FreeMemory(dReducedGradientHessian);
#endif

//	cerr << "4" << endl;
	for (int j = 0; j < J; j++) {
		if (dXColumnRowIndicators[j]) {
	//		gpu->FreeMemory(dXColumnRowIndicators[j]); // TODO Causing error under Linux
		}
	}
	free(dXColumnRowIndicators);

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
	double criterion = 0;
	for (int i = 0; i < K; i++) {
		criterion += hXBeta[i] * hEta[i];
	}

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::getObjectiveFunction\n");
#endif
    
    return criterion;
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

void GPUCyclicCoordinateDescent::computeRemainingStatistics(bool allStats) {

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering  GPUCylicCoordinateDescent::computeRemainingStatistics\n");
#endif  

    if (allStats) {
    	// NEW
//    	kernels->computeIntermediates(dOffsExpXBeta, dDenomPid, dOffs, dXBeta, dXFullRowOffsets, K, N, allStats);
    	CyclicCoordinateDescent::computeRemainingStatistics(true);
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

#ifdef MERGE_TRANSFORMATION
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

#else
	
	unsigned int threads = (N < 512) ? nextPow2((N + 1)/ 2) : 256;
	unsigned int blocks = (N + (threads * 2 - 1)) / (threads * 2);
	blocks = (64 < blocks) ? 64 : blocks;
	
	kernels->reduceSum(dGradient, dGradient, N, blocks, threads);
	kernels->reduceSum(dHessian, dGradient+(blocks*sizeof(real)), N, blocks, threads);
	gpu-> MemcpyDeviceToHost(hGradient, dGradient, 2*blocks*sizeof(real));

	for (int i = 0; i < blocks; i++)
	{	
		gradient += hGradient[i];
		hessian += hGradient[i+blocks];
	}
#endif
	

	gradient -= hXjEta[index];
	*ogradient = static_cast<double>(gradient);
	*ohessian = static_cast<double>(hessian);
	
#ifdef DP_DEBUG
	fprintf(stderr,"%5.3f %5.3f\n", *ogradient, *ohessian);
#endif
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving GPUCylicCoordinateDescent::computeGradientAndHessian\n");
#endif 	
}

