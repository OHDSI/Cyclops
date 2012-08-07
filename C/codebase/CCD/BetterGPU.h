/*
 * BetterGPU.h
 *
 *  Created on: July, 2011
 *      Author: msuchard
 */

#ifndef BETTERGPU_H_
#define BETTERGPU_H_

#include "CyclicCoordinateDescent.h"
#include "InputReader.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncherCCD.h"

namespace bsccs {

#ifdef DOUBLE_PRECISION
	typedef double gpu_real;
#else
	typedef float gpu_real;
#endif


class BetterGPU : public CyclicCoordinateDescent {
public:
	BetterGPU(int deviceNumber, InputReader *reader);
	virtual ~BetterGPU();

//	virtual double getObjectiveFunction(void);

protected:
	
	using CyclicCoordinateDescent::hXI;
	
//	virtual void resetBeta(void);

	virtual void computeNEvents(void);

//	virtual void updateXBeta(double delta, int index);

//	virtual void computeRemainingStatistics(bool);

	virtual void computeRatiosForGradientAndHessian(int index);

	virtual void computeGradientAndHession(
			int index,
			double *gradient,
			double *hessian);

//	virtual void getDenominators(void);

//	virtual double computeZhangOlesConvergenceCriterion(void);

private:
	int deviceNumber;
	GPUInterface* gpu;
	KernelLauncherCCD* kernels;
	
	InputReader* hReader;

	GPUPtr dNEvents;
	GPUPtr dNumerPid;
	GPUPtr dDenomPid;

	GPUPtr dGradient;
	GPUPtr dHessian;

	int alignedN;
	int cacheSizeGH;
	int alignedGHCacheSize;

	bsccs::real* hGradient;
	bsccs::real* hHessian;
};
}
#endif /* BETTERGPU_H_ */
