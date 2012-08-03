/*
 * BetterGPU.cpp
 *
 *  Created on: July, 2011
 *      Author: msuchard
 */

#include <iostream>
#include <cmath>
#include <numeric>

#include "BetterGPU.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncherCCD.h"

using namespace std;

BetterGPU::BetterGPU(int deviceNumber, InputReader* reader)
	: CyclicCoordinateDescent(reader) {
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering BetterGPU::constructor\n");
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

	dNEvents = gpu->AllocateIntMemory(N); // Load in computeNEvents, since this changes with cv/bootstrap

	alignedN = getAlignedLength(N);
	cacheSizeGH = kernels->getGradientAndHessianBlocks(N);
	alignedGHCacheSize = getAlignedLength(cacheSizeGH);

	dNumerPid = gpu->AllocateRealMemory(2 * alignedN);
	dDenomPid = dNumerPid  + sizeof(realTRS) * alignedN; // GPUPtr is void* not realTRS*

	dGradient = gpu->AllocateRealMemory(2 * alignedGHCacheSize);
	dHessian = dGradient + sizeof(realTRS) * alignedGHCacheSize; // GPUPtr is void* not real*
	hGradient = (realTRS*) malloc(2 * sizeof(realTRS) * alignedGHCacheSize);
	hHessian = hGradient + alignedGHCacheSize;

	computeRemainingStatistics(true, 0);
	
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving BetterGPU::constructor\n");
#endif 	
}

BetterGPU::~BetterGPU() {

	gpu->FreeMemory(dNEvents);
	gpu->FreeMemory(dNumerPid);
	gpu->FreeMemory(dGradient);

	free(hGradient);

	delete kernels;
	delete gpu;
}

void BetterGPU::computeNEvents(void) {
	CyclicCoordinateDescent::computeNEvents();
	gpu->MemcpyHostToDevice(dNEvents, hNEvents, sizeof(int) * N);

	int check = std::accumulate(hNEvents, hNEvents + N, 0);
	fprintf(stderr,"check = %d\n",check);

	vector<int> i(N);
	gpu->MemcpyDeviceToHost(&i[0], dNEvents, sizeof(int) * N);
	check = std::accumulate(i.begin(), i.end(), 0);
	fprintf(stderr,"check = %d\n",check);
}

void BetterGPU::computeRatiosForGradientAndHessian(int index) {
	computeNumeratorForGradient(index);
}

void BetterGPU::computeGradientAndHession(int index, double *ogradient,
		double *ohessian) {

	gpu->MemcpyHostToDevice(dNumerPid, numerPid, sizeof(realTRS) * 2 * alignedN); // Copy both numer and demon
	int blockUsed = kernels->computeGradientAndHessianWithReduction(dNumerPid, dDenomPid, dNEvents,
			dGradient, dHessian, N, 1, WORK_BLOCK_SIZE);
	gpu->MemcpyDeviceToHost(hGradient, dGradient, sizeof(realTRS) * 2 * alignedGHCacheSize);

	realTRS g = 0;
	realTRS h = 0;
	realTRS* gradient = hGradient;
	const realTRS* end = gradient + cacheSizeGH;
	realTRS* hessian = hHessian;

	// TODO Remove code duplication with CPU version from here below
	for (; gradient != end; ++gradient, ++hessian) {
		g += *gradient;
		h += *hessian;
	}

	double dblGradient = g;
	double dblHessian = h;

	dblGradient -= hXjEta[index];
	*ogradient = dblGradient;
	*ohessian = dblHessian;
}

