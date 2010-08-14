cmake_minimum_required(VERSION 2.6)
project(SCCS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pg")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -pg")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -pg")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

find_package(CUDA)

add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD)

if(CUDA_FOUND)
	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU)	
endif(CUDA_FOUND)
