/*
 * @author Marc Suchard
 */

#ifndef __KernelLauncher__
#define __KernelLauncher__

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

#include "GPU/GPUImplDefs.h"
#include "GPU/GPUInterface.h"

class KernelLauncher {
	
protected:
    GPUInterface* gpu;
    
public:
    KernelLauncher(GPUInterface* inGpu) {
        gpu = inGpu;
    }
    
    virtual ~KernelLauncher() {}

    virtual unsigned char* getKernelsString() = 0;

	virtual void SetupKernelBlocksAndGrids() = 0;

	virtual void LoadKernels() = 0;

};
#endif // __KernelLauncher__
