cmake_minimum_required(VERSION 2.6)
#project(SCCS)
project(GPU_COX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -pedantic -fPIC -std=c++0x")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -pedantic -fPIC -std=c++0x")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -g")

#set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

# OpenCL
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/standalone/CMakeModules")
find_package( OpenCL REQUIRED )
include_directories( ${OPENCL_INCLUDE_DIRS} )
#include_directories( "${CMAKE_MODULE_PATH}/khronos" )

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif(APPLE)

#find_package(CUDA)

set(R_LIBRARIES "/Library/Frameworks/R.framework/Versions/3.5/Resources/library/")
	
if(DEFINED R_LIBRARIES)
	set(BOOST_ROOT
		${R_LIBRARIES}/BH/include)
	set(RCPP_PARALLEL_ROOT
		${R_LIBRARIES}/RcppParallel)
	set(RCPP_EIGEN_ROOT
			${R_LIBRARIES}/RcppEigen)
else()
	message(FATAL_ERROR "RCPP_PARALLEL_ROOT is not defined")
endif()	
			 
include_directories(${RCPP_PARALLEL_ROOT}/include)			 
			 
find_package(Boost)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD)
#add_subdirectory(${CMAKE_SOURCE_DIR}/standalone/codebase/CCD-DP)
add_subdirectory(${CMAKE_SOURCE_DIR}/standalone/codebase/GPU_COX-SP)

#if(CUDA_FOUND)
#	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU)	
##	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU-DP)
#endif(CUDA_FOUND)

#add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/jni)
