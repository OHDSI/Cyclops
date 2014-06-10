/*
 * @author Marc Suchard
 * @author Daniel Ayres
 */

#ifndef __GPUInterface__
#define __GPUInterface__

#ifdef HAVE_CONFIG_H
	#include "config.h"
#endif

#include "GPU/GPUImplDefs.h"

#ifdef CUDA
    #include <cuda.h>
    typedef CUdeviceptr GPUPtr;
    typedef CUfunction GPUFunction;
#else
#ifdef OPENCL
    #include <OpenCL/opencl.h>
    typedef cl_mem GPUPtr;
    typedef cl_kernel GPUFunction;
#else
	#include <cuda.h> // TODO Remove
    typedef void* GPUPtr;
    typedef void* GPUFunction;
#endif
#endif

class GPUInterface {
private:
#ifdef CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    CUmodule cudaModule;
    const char* GetCUDAErrorDescription(int errorCode);
#else
#ifdef OPENCL
    cl_device_id openClDeviceId;             // compute device id 
    cl_context openClContext;                // compute context
    cl_command_queue openClCommandQueue;     // compute command queue
    cl_program openClProgram;                // compute program
    cl_uint openClNumDevices;
    const char* GetCLErrorDescription(int errorCode);
#endif
#endif
public:
    GPUInterface();
    
    ~GPUInterface();
    
    int Initialize();

    int GetDeviceCount();

    void SetDevice(int deviceNumber,
				   unsigned char* kernelsString);
    
    void Synchronize();
    
    GPUFunction GetFunction(const char* functionName);
    
    // Does not work on Fermi
//    void LaunchKernelIntParams(GPUFunction deviceFunction,
//                               Dim3Int block,
//                               Dim3Int grid,
//                               int totalParameterCount,
//                               ...); // unsigned int parameters
    
    void LaunchKernelParams(GPUFunction deviceFunction,
                            Dim3Int block,
                            Dim3Int grid,
                            int totalPtrParameterCount,
                            int totalIntParameterCount,
                            int totalFloatParameterCount,
                            ...); // unsigned int and float parameters

    GPUPtr AllocateMemory(int memSize);
    
    GPUPtr AllocateRealMemory(int length);

    GPUPtr AllocateIntMemory(int length);

    void MemcpyHostToDevice(GPUPtr dest,
                            const void* src,
                            int memSize);

    void MemcpyDeviceToHost(void* dest,
                            const GPUPtr src,
                            int memSize);

    void MemClear(void* dest, int memSize);

    void FreeMemory(GPUPtr dPtr);
    
    unsigned int GetAvailableMemory();
    
    void GetDeviceName(int deviceNumber,
                       char* deviceName,
                       int nameLength);
    
    void GetDeviceDescription(int deviceNumber,
                              char* deviceDescription);    

    void PrintfDeviceVector(GPUPtr dPtr,
                      int length);
    
    void PrintfDeviceVector(GPUPtr dPtr,
                           int length, double checkValue);
    
    void PrintfDeviceVector(GPUPtr dPtr,
                            int length,
                            double checkValue,
                            int *signal);

    void PrintfDeviceInt(GPUPtr dPtr,
                   int length);
    
};

#endif // __GPUInterface__
