/*
 * GPUCyclicCoordinateDescent.h
 *
 *  Created on: July, 2010
 *      Author: msuchard
 */

#ifndef GPUCYCLICCOORDINATEDESCENT_H_
#define GPUCYCLICCOORDINATEDESCENT_H_

#include "CyclicCoordinateDescent.h"
#include "InputReader.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncherCCD.h"

#define NO_BETA
#define GPU_SPARSE_PRODUCT

namespace bsccs {

#ifdef DOUBLE_PRECISION
	typedef double gpu_real;
#else
	typedef float gpu_real;
#endif


class GPUCyclicCoordinateDescent: public CyclicCoordinateDescent {
public:
	GPUCyclicCoordinateDescent(int deviceNumber, InputReader *reader);
	virtual ~GPUCyclicCoordinateDescent();

	virtual double getObjectiveFunction(void);

protected:
	
	using CyclicCoordinateDescent::hXI;
	
	virtual void resetBeta(void);

	virtual void computeNEvents(void);

	virtual void updateXBeta(double delta, int index);

	virtual void computeRemainingStatistics(bool, int index);

	virtual void computeRatiosForGradientAndHessian(int index);

	virtual void computeGradientAndHession(
			int index,
			double *gradient,
			double *hessian);

	virtual void getDenominators(void);

	virtual double computeZhangOlesConvergenceCriterion(void);

private:
	int deviceNumber;
	GPUInterface* gpu;
	KernelLauncherCCD* kernels;
	
	InputReader* hReader;

	GPUPtr* dXI;
	GPUPtr dXColumnLength;
	GPUPtr dOffs;
//	GPUPtr dEta;
	GPUPtr dNEvents;
//	GPUPtr dPid;
	GPUPtr dXFullRowOffsets;
#ifndef NO_BETA
	GPUPtr dBeta;
#endif
	GPUPtr dXBeta;

	GPUPtr dOffsExpXBeta;
	GPUPtr dDenomPid;
	GPUPtr dNumerPid;
	GPUPtr dT1;
//	GPUPtr dXOffsExpXBeta;

	GPUPtr dGradient;
	GPUPtr dHessian;
	GPUPtr dReducedGradientHessian;

	bsccs::real* hGradient;
	bsccs::real* hHessian;

	GPUPtr* dXColumnRowIndicators;

#ifdef GPU_SPARSE_PRODUCT
	GPUPtr* dNI;
	int maxNISize;
	int avgNISize;
#endif
//	int* hColumnRowLength;

	GPUPtr dTmpCooRows;
	GPUPtr dTmpCooVals;

	int alignedN;
	int cacheSizeGH;
	int alignedGHCacheSize;
};
}
#endif /* GPUCYCLICCOORDINATEDESCENT_H_ */
