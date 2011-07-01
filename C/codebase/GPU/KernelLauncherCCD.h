/*
 * @author Marc Suchard
 */

#ifndef __KernelLauncherCCD__
#define __KernelLauncherCCD__

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

#include "GPU/GPUImplDefs.h"
#include "GPU/GPUInterface.h"
#include "GPU/KernelLauncher.h"

class KernelLauncherCCD : public KernelLauncher {
	
private:
    GPUFunction fDotProduct;
    GPUFunction fUpdateXBeta;
    GPUFunction fUpdateXBetaAndFriends;
    GPUFunction fComputeIntermediates;
    GPUFunction fComputeIntermediatesMoreWork;
    GPUFunction fReduceAll;
    GPUFunction fReduceFast;
    GPUFunction fComputeAndReduceFast;
    GPUFunction fReduceRow;

    GPUFunction fComputeRatio;
    GPUFunction fComputeGradientHessian;

    GPUFunction fSpmvCooSerial;
    GPUFunction fSpmvCooFlat;
    GPUFunction fSpmvCooReduce;

    GPUFunction fSpmvCsr;
    GPUFunction fReduceTwo;
    GPUFunction fReduceSum;

    GPUFunction fClearMemory;

 //   Dim3Int bgName;

public:
    KernelLauncherCCD(GPUInterface* inGpu);

    virtual ~KernelLauncherCCD();
    
    unsigned char* getKernelsString();

// Kernel links
    void dotProduct(GPUPtr oC,
				    GPUPtr iA,
				    GPUPtr iB,
				    int length);

    void updateXBeta(GPUPtr xBeta,
					 GPUPtr xIColumn,
					 int length,
					 double delta);
    
    void updateXBetaAndFriends(GPUPtr xBeta, 
    		GPUPtr offsExpXBeta, 
    		GPUPtr denomPid, 
    		GPUPtr rowOffs,
    		GPUPtr otherOffs,
    		GPUPtr offs, 
    		GPUPtr xIColumn, 
    		int length, 
    		double delta);    

    void reduceTwo(GPUPtr oC,
				   GPUPtr iX,
				   unsigned int length);
    
    void reduceSum(
    		GPUPtr d_idata,
    		GPUPtr d_odata,
    		unsigned int size,
    		unsigned int blocks,
		unsigned int threads);    

    void computeIntermediates(
    		GPUPtr offsExpXBeta,
    		GPUPtr denomPid,
            GPUPtr offs,
            GPUPtr xBeta,
            GPUPtr pid,
            int nRows,
            int nPatients,
            bool allStats);

    void clearMemory(
    		GPUPtr dest,
    		int length);

    void computeDerivatives(
       		GPUPtr offsExpXBeta,
        	GPUPtr xI,
        	GPUPtr pid,
        	GPUPtr numerPid,
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
    		GPUPtr tmpVals);

    void computeSpmvCooIndicatorMatrix(
    		GPUPtr y,
    		GPUPtr row,
    		GPUPtr column,
    		GPUPtr x,
    		unsigned int nNonZeros,
    		GPUPtr tmpRows,
    		GPUPtr tmpVals);

    void computeSpmvCsrIndicatorMatrixNoColumns(
    		GPUPtr y,
    		GPUPtr rowOffsets,
    		GPUPtr x,
    		unsigned int nRows);

    void updateMU(void);

	virtual void SetupKernelBlocksAndGrids();

	virtual void LoadKernels();

};
#endif // __KernelLauncherCCD__
