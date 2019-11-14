CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(GPU_COX)

set(RCCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(CCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/standalone/codebase)
#set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/standalone/CMakeModules)

MESSAGE(STATUS "Configuring OpenCL")
FIND_PACKAGE(OpenCL REQUIRED)
ADD_DEFINITIONS(-DHAVE_OPENCL)
#ADD_DEFINITIONS(-DOPENCL)

#INCLUDE("${CMAKE_MODULE_PATH}/build_boost_headers.cmake")
#INCLUDE_DIRECTORIES(${BoostHeaders_INCLUDE_DIRS})
#MESSAGE(STATUS "boost headers directory: ${BoostHeaders_INCLUDE_DIRS}")

#INCLUDE("${CMAKE_MODULE_PATH}/build_boost_compute.cmake")
#MESSAGE(STATUS "boost::commpute directory: ${BoostCompute_INCLUDE_DIRS}")
#ADD_DEFINITIONS(-DUSE_BOOST_COMPUTE)

include_directories(. 
    ${RCCD_SOURCE_DIR}
    ${RCCD_SOURCE_DIR}/cyclops
    ${CCD_SOURCE_DIR}
    ${CCD_SOURCE_DIR}/standalone/codebase/GPU_COX
    ${Boost_INCLUDE_DIR}          
    )

set(BASE_SOURCE_FILES
    ${RCCD_SOURCE_DIR}/cyclops/CcdInterface.cpp	
	${RCCD_SOURCE_DIR}/cyclops/CyclicCoordinateDescent.cpp	
	${RCCD_SOURCE_DIR}/cyclops/CompressedDataMatrix.cpp
	${RCCD_SOURCE_DIR}/cyclops/ModelData.cpp
	${RCCD_SOURCE_DIR}/cyclops/io/InputReader.cpp
	 ${CCD_SOURCE_DIR}/GPU_COX/io/HierarchyReader.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/io/SCCSInputReader.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/io/CLRInputReader.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/io/RTestInputReader.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/io/CoxInputReader.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/io/CCTestInputReader.cpp
	 ${CCD_SOURCE_DIR}/GPU_COX/CmdLineCcdInterface.cpp
#	 ${CCD_SOURCE_DIR}/GPU_COX/GPUCyclicCoordinateDescent.cpp
	${RCCD_SOURCE_DIR}/cyclops/engine/AbstractModelSpecifics.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/AbstractDriver.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/AbstractSelector.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/AbstractCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/ProportionSelector.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/CrossValidationSelector.cpp
    ${RCCD_SOURCE_DIR}/cyclops/drivers/GridSearchCrossValidationDriver.cpp
    ${RCCD_SOURCE_DIR}/cyclops/drivers/HierarchyGridSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/AutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/HierarchyAutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/BootstrapSelector.cpp
	${RCCD_SOURCE_DIR}/cyclops/drivers/BootstrapDriver.cpp
	${RCCD_SOURCE_DIR}/utils/HParSearch.cpp
	${RCCD_SOURCE_DIR}/tinythread/tinythread.cpp
	${RCCD_SOURCE_DIR}/cyclops/engine/ModelSpecifics.hpp
	${RCCD_SOURCE_DIR}/cyclops/engine/ModelSpecifics.h
	${RCCD_SOURCE_DIR}/cyclops/engine/GpuModelSpecifics.hpp
	${RCCD_SOURCE_DIR}/cyclops/engine/Kernels.hpp
	)
	
set(CCD_SOURCE_FILES
		${CCD_SOURCE_DIR}/GPU_COX/gpu_cox.cpp)
	
set(SINGLE_PRECISION true)
add_definitions(-DSINGLE_PRECISION)
	
#if(CUDA_FOUND)
#	set(SOURCE_FILES ${SOURCE_FILES}
#	    ../CCD/GPUCyclicCoordinateDescent.cpp
##	    ../CCD/BetterGPU.cpp
#        )
#	add_definitions(-DCUDA)
## 	if(APPLE)
## 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
## 		set(CUDA_64_BIT_DEVICE_CODE OFF)	
## 	endif(APPLE)		
#	add_definitions(-DDOUBLE_PRECISION)	
# 	if(APPLE)
# 		set(CUDA_NVCC_FLAGS -arch=sm_11)
# 	else(APPLE)	 		
# 		set(CUDA_NVCC_FLAGS -arch=sm_20) 		
# 	endif(APPLE) 	
# 	cuda_add_executable(ccd-dp ${SOURCE_FILES})
# 	target_link_libraries(ccd-dp CUDADriver-dp)
#else(CUDA_FOUND)

add_library(base_bsccs-sp ${BASE_SOURCE_FILES})
add_executable(gpu_cox-sp ${CCD_SOURCE_FILES})
target_link_libraries(gpu_cox-sp base_bsccs-sp)
target_link_libraries(gpu_cox-sp "-framework OpenCL" )

add_library(model_specifics ${RCCD_SOURCE_DIR}/cyclops/engine/AbstractModelSpecifics.cpp)

#ADD_LIBRARY(mds_opencl ${OPENCL_SOURCE_FILES})
#set_target_properties(mds_opencl PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
#TARGET_LINK_LIBRARIES(mds_opencl ${OpenCL_LIBRARIES})

	
set_target_properties(model_specifics PROPERTIES COMPILE_FLAGS "-save-temps")
 	
#endif(CUDA_FOUND)
