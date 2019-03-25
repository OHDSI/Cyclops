cmake_minimum_required(VERSION 2.6)
project(SCCS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pedantic -fPIC -std=c++0x")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -pedantic -fPIC -std=c++0x")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -pg")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -pg")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif(APPLE)

#find_package(CUDA)


	
if(DEFINED R_LIBRARIES)
	set(BOOST_ROOT
		${R_LIBRARIES}/BH/include)
	set(RCPP_PARALLEL_ROOT
		${R_LIBRARIES}/RcppParallel)
else()
	message(FATAL_ERROR "RCPP_PARALLEL_ROOT is not defined")
endif()	
			 
include_directories(${RCPP_PARALLEL_ROOT}/include)			 
			 
find_package(Boost)

# add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD)
add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/CCD-DP)

#if(CUDA_FOUND)
#	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU)	
##	add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/GPU-DP)
#endif(CUDA_FOUND)

#add_subdirectory(${CMAKE_SOURCE_DIR}/codebase/jni)
