include_directories (${CMAKE_SOURCE_DIR}/codebase ${CUDA_INCLUDE_DIRS})

message(STATUS "found source at: " ${CUDA_INCLUDE_DIRS})

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

set(CU_FILE "kernels/kernelAll.cu")

set(KERNEL_HEADER_FILE "${CMAKE_SOURCE_DIR}/codebase/GPU/kernels/CCDKernels.h")

set(CUDA_SRCS
	GPUImplDefs.h
	GPUImplHelper.cpp GPUImplHelper.h	
	GPUInterfaceCUDA.cpp GPUInterface.h
	KernelLauncher.h
	KernelLauncherCCD.cpp KernelLauncherCCD.h
	)

add_definitions("-DCUDA")
if(APPLE)
	set(CUDA_NVCC_FLAGS "-arch=sm_11")
else(APPLE)
	set(CUDA_NVCC_FLAGS "-arch=sm_20")
endif(APPLE)

######## Less editting below this point

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA OFF)

find_program(BIN2C bin2c)
if(NOT BIN2C)
       message(FATAL_ERROR "Could not find bin2c command")
endif(NOT BIN2C)
message(STATUS "Found bin2c: " ${BIN2C})

#if(APPLE)
#	set(CUDA_64_BIT_DEVICE_CODE OFF)
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
#endif(APPLE)

CUDA_COMPILE_PTX(PTX_FILE ${CU_FILE})

add_custom_command(
	OUTPUT ${KERNEL_HEADER_FILE}
	DEPENDS ${PTX_FILE}
	COMMAND ${BIN2C} -st -n KERNELS_STRING ${PTX_FILE} > ${KERNEL_HEADER_FILE}
	COMMENT "Converting PTX to .h: ${KERNEL_HEADER_FILE}")

add_custom_command(
	OUTPUT cleanPTX
	COMMAND ${CMAKE_COMMAND} -E remove ${PTX_FILE}
	DEPENDS ${KERNEL_HEADER_FILE}
	COMMENT "Cleaning up PTX")

set(CUDA_SRCS ${CUDA_SRCS} ${KERNEL_HEADER_FILE})

add_custom_target(
	kernels ALL 
	DEPENDS cleanPTX)

add_library(CUDADriver ${CUDA_SRCS})
