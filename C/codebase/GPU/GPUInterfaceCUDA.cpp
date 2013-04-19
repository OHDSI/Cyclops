/*
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdarg>
#include <map>
#include <iostream>

#include <cuda.h>

#include "GPU/GPUImplDefs.h"
#include "GPU/GPUImplHelper.h"
#include "GPU/GPUInterface.h"

//#define GPU_DEBUG_FLOW

#define SAFE_CUDA(call) { \
                            CUresult error = call; \
                            if(error != CUDA_SUCCESS) { \
                                fprintf(stderr, "CUDA error: \"%s\" from file <%s>, line %i.\n", \
                                        GetCUDAErrorDescription(error), __FILE__, __LINE__); \
                                exit(-1); \
                            } \
                        }

#define SAFE_CUPP(call) { \
                            SAFE_CUDA(cuCtxPushCurrent(cudaContext)); \
                            SAFE_CUDA(call); \
                            SAFE_CUDA(cuCtxPopCurrent(&cudaContext)); \
                        }

static int nGpuArchCoresPerSM[] = { -1, 8, 32 };

GPUInterface::GPUInterface() {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
#endif    
    
    cudaDevice = 0;
    cudaContext = NULL;
    cudaModule = NULL;
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
#endif    
}

GPUInterface::~GPUInterface() {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
#endif    
    
    if (cudaContext != NULL) {
        SAFE_CUDA(cuCtxPushCurrent(cudaContext));
        SAFE_CUDA(cuCtxDetach(cudaContext));
    }

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
#endif    
    
}

int GPUInterface::Initialize() {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
#endif    
    
    // Driver init; CUDA manual: "Currently, the Flags parameter must be 0."
    CUresult error = cuInit(0);
    
    int returnValue = 1;
    
    if (error == CUDA_ERROR_NO_DEVICE) {
        returnValue = 0;
    } else if (error != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA error: \"%s\" from file <%s>, line %i.\n",
                GetCUDAErrorDescription(error), __FILE__, __LINE__);
        exit(-1);
    }
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
#endif    
    
    return returnValue;
}

int GPUInterface::GetDeviceCount() {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
#endif        
    
    int numDevices = 0;
    SAFE_CUDA(cuDeviceGetCount(&numDevices));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
#endif            
    
    return numDevices;
}

void GPUInterface::SetDevice(int deviceNumber,
							unsigned char* kernelsString) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
#endif            
    
    SAFE_CUDA(cuDeviceGet(&cudaDevice, deviceNumber));
    
    SAFE_CUDA(cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice));
    
    SAFE_CUDA(cuModuleLoadData(&cudaModule, kernelsString));

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
#endif            
    
}

void GPUInterface::Synchronize() {
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::Synchronize\n");
#endif                
    
    SAFE_CUPP(cuCtxSynchronize());
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Synchronize\n");
#endif                
    
}

GPUFunction GPUInterface::GetFunction(const char* functionName) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
#endif                    
    
    GPUFunction cudaFunction; 

    SAFE_CUPP(cuModuleGetFunction(&cudaFunction, cudaModule, functionName));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
#endif                
    
    return cudaFunction;
}

//void GPUInterface::LaunchKernelIntParams(GPUFunction deviceFunction,
//                                         Dim3Int block,
//                                         Dim3Int grid,
//                                         int totalParameterCount,
//                                         ...) { // unsigned int parameters
//#ifdef GPU_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernelIntParams\n");
//#endif
//
//
//    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
//
//    SAFE_CUDA(cuFuncSetBlockShape(deviceFunction, block.x, block.y, block.z));
//
//    int offset = 0;
//    va_list parameters;
//    va_start(parameters, totalParameterCount);
//    for(int i = 0; i < totalParameterCount; i++) {
//        unsigned int param = va_arg(parameters, unsigned int);
//
//         // adjust offset alignment requirements
//        offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);
//
//        SAFE_CUDA(cuParamSeti(deviceFunction, offset, param));
//
//        offset += sizeof(param);
//    }
//    va_end(parameters);
//
//    SAFE_CUDA(cuParamSetSize(deviceFunction, offset));
//
//    SAFE_CUDA(cuLaunchGrid(deviceFunction, grid.x, grid.y));
//
//    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
//
//#ifdef GPU_DEBUG_FLOW
//    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernelIntParams\n");
//#endif
//
//}

void GPUInterface::LaunchKernelParams(GPUFunction deviceFunction,
                                         Dim3Int block,
                                         Dim3Int grid,
                                         int totalPtrParameterCount,
                                         int totalIntParameterCount,
                                         int totalFloatParameterCount,
                                         ...) { // unsigned int and float parameters
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernelParams\n");
#endif


    SAFE_CUDA(cuCtxPushCurrent(cudaContext));

    SAFE_CUDA(cuFuncSetBlockShape(deviceFunction, block.x, block.y, block.z));

    int offset = 0;
    va_list parameters;
    va_start(parameters, totalFloatParameterCount);

    for(int i = 0; i < totalPtrParameterCount; i++) {
         void* param = (void*)(size_t) va_arg(parameters, GPUPtr);

         offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);

         SAFE_CUDA(cuParamSetv(deviceFunction, offset, &param, sizeof(param)));

         offset += sizeof(param);
    }
    for(int i = 0; i < totalIntParameterCount; i++) {
        unsigned int param = va_arg(parameters, unsigned int);

         // adjust offset alignment requirements
        offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);

        SAFE_CUDA(cuParamSeti(deviceFunction, offset, param));

        offset += sizeof(param);
    }
    for(int i = 0; i < totalFloatParameterCount; i++) {
    	REAL param = (REAL) va_arg(parameters, double); // Implicit case
    
    	offset = (offset + __alignof(param) - 1) & ~(__alignof(param) - 1);

#ifdef DOUBLE_PRECISION
    	SAFE_CUDA(cuParamSetv(deviceFunction, offset, &param, sizeof(REAL)));
#else
    	SAFE_CUDA(cuParamSetf(deviceFunction, offset, param));
#endif

    	offset += sizeof(param);
    }
    va_end(parameters);

    SAFE_CUDA(cuParamSetSize(deviceFunction, offset));

    SAFE_CUDA(cuLaunchGrid(deviceFunction, grid.x, grid.y));

    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernelParams\n");
#endif

}

GPUPtr GPUInterface::AllocateMemory(int memSize) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
#endif
    
    if (memSize < 0) {
    	fprintf(stderr,"BAD ALLOC!\n");
    	exit(-1);
    }
    
    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, memSize));

#ifdef MMGPU_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + memSize));
#endif
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
#endif
    
    if (!data) {
    	fprintf(stderr, "Error allocating GPU memory!\n");
    	exit(-1);
    }
    return data;
}

GPUPtr GPUInterface::AllocateRealMemory(int length) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
#endif

    if (length < 0) {
    	fprintf(stderr,"BAD ALLOC!\n");
    	exit(-1);
    }
    
    GPUPtr data;
 
    SAFE_CUPP(cuMemAlloc(&data, SIZE_REAL * length));

#ifdef MMGPU_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + length));
#endif
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
#endif
    
    if (!data) {
    	fprintf(stderr, "Error allocating GPU memory!\n");
    	exit(-1);
    }    
    return data;
}

GPUPtr GPUInterface::AllocateIntMemory(int length) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
#endif

    if(length < 0) {
    	fprintf(stderr,"BAD ALLOC!\n");
    	exit(-1);
    }
    
    GPUPtr data;
    
    SAFE_CUPP(cuMemAlloc(&data, SIZE_INT * length));

#ifdef MMGPU_DEBUG_VALUES
    fprintf(stderr, "Allocated GPU memory %d to %d.\n", data, (data + length));
#endif
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
#endif

    if (!data) {
    	fprintf(stderr, "Error allocating GPU memory!\n");
    	exit(-1);
    }    
    return data;
}

void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
                                      const void* src,
                                      int memSize) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
#endif    
    
    if (memSize < 0) {
    	fprintf(stderr,"BAD IO\n");
    	exit(-1);
    }
    
    SAFE_CUPP(cuMemcpyHtoD(dest, src, memSize));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
#endif    
    
}

void GPUInterface::MemcpyDeviceToHost(void* dest,
                                      const GPUPtr src,
                                      int memSize) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
#endif        
    
    if (memSize < 0) {
    	fprintf(stderr,"BAD IO\n");
    	exit(-1);
    }
    
    SAFE_CUPP(cuMemcpyDtoH(dest, src, memSize));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
#endif    
    
}

//void GPUInterface::MemClear(void* dest, int memSize) {
//	SAFE_CUPP(cuMemset(dest, 0, memSize));
//}

void GPUInterface::FreeMemory(GPUPtr dPtr) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
#endif
    
    SAFE_CUPP(cuMemFree(dPtr));

#ifdef GPU_DEBUG_FLOW
    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
#endif
}

unsigned int GPUInterface::GetAvailableMemory() {
#if CUDA_VERSION >= 3020
    size_t availableMem = 0;
    size_t totalMem = 0;
    SAFE_CUPP(cuMemGetInfo(&availableMem, &totalMem));
#else
    unsigned int availableMem = 0;
    unsigned int totalMem = 0;
    SAFE_CUPP(cuMemGetInfo(&availableMem, &totalMem));
#endif
    return availableMem;
}

void GPUInterface::GetDeviceName(int deviceNumber,
                                  char* deviceName,
                                  int nameLength) {
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
#endif    
    
    CUdevice tmpCudaDevice;

    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, deviceNumber));
    
    SAFE_CUDA(cuDeviceGetName(deviceName, nameLength, tmpCudaDevice));
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
#endif        
}

void GPUInterface::GetDeviceDescription(int deviceNumber,
                                        char* deviceDescription) {    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
#endif
    
    CUdevice tmpCudaDevice;
    
    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, deviceNumber));
    
#if CUDA_VERSION >= 3020
    size_t totalGlobalMemory = 0;
#else
    unsigned int totalGlobalMemory = 0;
#endif
    int clockSpeed = 0;
    int mpCount = 0;
    int major = 0;
    int minor = 0;

    SAFE_CUDA(cuDeviceComputeCapability(&major, &minor, tmpCudaDevice));
    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, tmpCudaDevice));
    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, tmpCudaDevice));

    sprintf(deviceDescription,
            "Global memory (MB): %d | Clock speed (Ghz): %1.2f | Number of cores: %d",
            int(totalGlobalMemory / 1024.0 / 1024.0 + 0.5),
            clockSpeed / 1000000.0,
            nGpuArchCoresPerSM[major] * mpCount
//            8 * mpCount
            );
    
#ifdef GPU_DEBUG_FLOW
    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
#endif    
}

void GPUInterface::PrintfDeviceVector(GPUPtr dPtr,
                                int length, double checkValue, int *signal) {
    REAL* hPtr = (REAL*) malloc(SIZE_REAL * length);
    
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_REAL * length);
    
#ifdef DOUBLE_PRECISION
    printfVectorD(hPtr, length);
#else
    printfVectorF(hPtr,length);
#endif
    
    if (checkValue != -1) {
    	double sum = 0;
    	for(int i=0; i<length; i++) {
    		sum += hPtr[i];
    		if( (hPtr[i] > checkValue) && (hPtr[i]-checkValue > 1.0E-4)) {
    			fprintf(stderr,"Check value exception!  (%d) %2.5e > %2.5e (diff = %2.5e)\n",
    					i,hPtr[i],checkValue, (hPtr[i]-checkValue));
    			if( signal != 0 )
    				*signal = 1;  			
    		}
    		if (hPtr[i] != hPtr[i]) {
    			fprintf(stderr,"NaN found!\n");
    			//exit(0);
    			if( signal != 0 ) 
    				*signal = 1;
    		}
    	}
    	if (sum == 0) {
    		fprintf(stderr,"Zero-sum vector!\n");
    		if( signal != 0 )
    			*signal = 1;
    	}
    	
    }
    
    free(hPtr);
}

void GPUInterface::PrintfDeviceVector(GPUPtr dPtr, int length) {
	PrintfDeviceVector(dPtr,length,-1, 0);
}

void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
                             int length) {    
    int* hPtr = (int*) malloc(SIZE_INT * length);
    
    MemcpyDeviceToHost(hPtr, dPtr, SIZE_INT * length);
    
    printfInt(hPtr, length);
    
    free(hPtr);
}

const char* GPUInterface::GetCUDAErrorDescription(int errorCode) {
    
    const char* errorDesc;
    
    // from cuda.h
    switch(errorCode) {
        case CUDA_SUCCESS: errorDesc = "No errors"; break;
        case CUDA_ERROR_INVALID_VALUE: errorDesc = "Invalid value"; break;
        case CUDA_ERROR_OUT_OF_MEMORY: errorDesc = "Out of memory"; break;
        case CUDA_ERROR_NOT_INITIALIZED: errorDesc = "Driver not initialized"; break;
        case CUDA_ERROR_DEINITIALIZED: errorDesc = "Driver deinitialized"; break;
            
        case CUDA_ERROR_NO_DEVICE: errorDesc = "No CUDA-capable device available"; break;
        case CUDA_ERROR_INVALID_DEVICE: errorDesc = "Invalid device"; break;
            
        case CUDA_ERROR_INVALID_IMAGE: errorDesc = "Invalid kernel image"; break;
        case CUDA_ERROR_INVALID_CONTEXT: errorDesc = "Invalid context"; break;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: errorDesc = "Context already current"; break;
        case CUDA_ERROR_MAP_FAILED: errorDesc = "Map failed"; break;
        case CUDA_ERROR_UNMAP_FAILED: errorDesc = "Unmap failed"; break;
        case CUDA_ERROR_ARRAY_IS_MAPPED: errorDesc = "Array is mapped"; break;
        case CUDA_ERROR_ALREADY_MAPPED: errorDesc = "Already mapped"; break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU: errorDesc = "No binary for GPU"; break;
        case CUDA_ERROR_ALREADY_ACQUIRED: errorDesc = "Already acquired"; break;
        case CUDA_ERROR_NOT_MAPPED: errorDesc = "Not mapped"; break;
            
        case CUDA_ERROR_INVALID_SOURCE: errorDesc = "Invalid source"; break;
        case CUDA_ERROR_FILE_NOT_FOUND: errorDesc = "File not found"; break;
            
        case CUDA_ERROR_INVALID_HANDLE: errorDesc = "Invalid handle"; break;
            
        case CUDA_ERROR_NOT_FOUND: errorDesc = "Not found"; break;
            
        case CUDA_ERROR_NOT_READY: errorDesc = "CUDA not ready"; break;
            
        case CUDA_ERROR_LAUNCH_FAILED: errorDesc = "Launch failed"; break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: errorDesc = "Launch exceeded resources"; break;
        case CUDA_ERROR_LAUNCH_TIMEOUT: errorDesc = "Launch exceeded timeout"; break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: errorDesc =
            "Launch with incompatible texturing"; break;
            
        case CUDA_ERROR_UNKNOWN: errorDesc = "Unknown error"; break;
            
        default: errorDesc = "Unknown error";
    }
    
    return errorDesc;
}

