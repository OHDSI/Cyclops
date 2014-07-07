cmake_minimum_required(VERSION 2.6)
project(SCCS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pg")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -pg")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -pg")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -pg")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

find_package(CUDA)

add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD)
add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD-DP)

if(CUDA_FOUND)
	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU)	
	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CUDARuntime)
	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU-DP)
endif(CUDA_FOUND)
