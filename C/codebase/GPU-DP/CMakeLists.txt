include_directories (${CMAKE_SOURCE_DIR}/codebase ${CUDA_INCLUDE_DIRS})

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/codebase/GPU ${CMAKE_CURRENT_SOURCE_DIR}/codebase/GPU/kernels)

set(CU_FILE_DP "../GPU/kernels/kernelAll.cu")

set(KERNEL_HEADER_FILE_DP "${CMAKE_SOURCE_DIR}/codebase/GPU-DP/kernels/CCDKernels.h")

set(CUDA_SRCS
	../GPU/GPUImplDefs.h
	../GPU/GPUImplHelper.cpp ../GPU/GPUImplHelper.h	
	../GPU/GPUInterfaceCUDA.cpp ../GPU/GPUInterface.h
	../GPU/KernelLauncher.h
	../GPU/KernelLauncherCCD.cpp ../GPU/KernelLauncherCCD.h
	)

add_definitions("-DCUDA")
add_definitions("-DDOUBLE_PRECISION")

######## Less editting below this point

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA OFF)

find_program(BIN2C bin2c)
if(NOT BIN2C)
       message(FATAL_ERROR "Could not find bin2c command")
endif(NOT BIN2C)
message(STATUS "Found bin2c: " ${BIN2C})

#if(APPLE)
#	set(CUDA_64_BIT_DEVICE_CODE OFF)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
#endif(APPLE)

set(CUDA_NVCC_FLAGS -arch compute_13)
CUDA_COMPILE_PTX(PTX_FILE_DP ${CU_FILE_DP})

add_custom_command(
	OUTPUT ${KERNEL_HEADER_FILE_DP}
	DEPENDS ${PTX_FILE_DP}
	COMMAND ${BIN2C} -st -n KERNELS_STRING ${PTX_FILE_DP} > ${KERNEL_HEADER_FILE_DP}
	COMMENT "Converting PTX to .h: ${KERNEL_HEADER_FILE_DP}")

add_custom_command(
	OUTPUT cleanPTX_DP
	COMMAND ${CMAKE_COMMAND} -E remove ${PTX_FILE_DP}
	DEPENDS ${KERNEL_HEADER_FILE_DP}
	COMMENT "Cleaning up PTX")

set(CUDA_SRCS ${CUDA_SRCS} ${KERNEL_HEADER_FILE_DP})

add_custom_target(
	kernels-dp ALL 
	DEPENDS cleanPTX_DP)

add_library(CUDADriver-dp ${CUDA_SRCS})
