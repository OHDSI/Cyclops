/*
 * GPUContainer.h
 *
 *  Created on: May 2, 2011
 *      Author: Coki
 */

#ifndef GPUCONTAINER_H_
#define GPUCONTAINER_H_

struct GPUContainer {
	GPUInterface* gpu;
	KernelLauncherCCD* kernels;

	InputReader* hReader;
	int patients; // Number of patients per container
	int exposures; // Number of exposure levels per container
	int drugs; // Number of drugs per container

	int prev_patients;
	int prev_exposures;

	GPUPtr* dXI;
	int* columnLength;
	GPUPtr dOffs;
	GPUPtr dEta;
	GPUPtr dNEvents;
	GPUPtr dPid;
	GPUPtr dXFullRowOffsets;
	GPUPtr dBeta;
	GPUPtr dXBeta;

	GPUPtr dOffsExpXBeta;
	GPUPtr dDenomPid;
	GPUPtr dNumerPid;
	GPUPtr dT1;
	GPUPtr dXOffsExpXBeta;

#ifdef GRADIENT_HESSIAN_GPU
	GPUPtr dGradient;
	GPUPtr dHessian;
	GPUPtr dReducedGradientHessian;

	real* hGradient;
	real* hHessian;
#endif

	GPUPtr* dXColumnRowIndicators;
//	int* hColumnRowLength;

	GPUPtr dTmpCooRows;
	GPUPtr dTmpCooVals;
};

#endif /* GPUCONTAINER_H_ */
