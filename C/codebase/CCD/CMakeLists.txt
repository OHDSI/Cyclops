include_directories(. ${CMAKE_SOURCE_DIR}/codebase)

#find_path(GSL_INCLUDE_DIR
#	NAMES gsl
#    DOCS "Location of GSL header file")
    
#find_path(MTL_INCLUDE_DIR
#	NAMES mtl4
#	DOCS "Location of MTL header file"
#	PATHS "/Users/msuchard/Documents/Projects")
#	
#find_path(BOOST_INCLUDE_DIR
#	NAMES boost
#	DOCS "Location of boost header file"
#	PATHS "/Users/msuchard/Documents/Projects/boost_1_42_0")
	
#find_library(GSL_LIBRARY
#	NAMES gsl
#	DOCS "Location of GSL library file")

#find_library(GSL_CBLAS_LIBRARY
#	NAMES gslcblas
#	DOCS "Location of GSL BLAS library file")

#find_library(CUBLAS_LIBRARY
#	NAMES cublas
#	DOCS "Location of CUBLAS library file")
	
#if (NOT MTL_INCLUDE_DIR)
#	message(FATAL_ERROR "MTL header file not found (and required)")
#endif (NOT MTL_INCLUDE_DIR)
#
#if (NOT BOOST_INCLUDE_DIR)
#	message(FATAL_ERROR "Boost header file not found (and required)")
#endif (NOT BOOST_INCLUDE_DIR)

#if (NOT GSL_INCLUDE_DIR)
#   message(FATAL_ERROR "GSL header file not found (and required)")
#endif (NOT GSL_INCLUDE_DIR)

#if (NOT GSL_LIBRARY)
#   message(FATAL_ERROR "GSL library not found (and required)")
#endif (NOT GSL_LIBRARY)

#if (NOT CUBLAS_LIBRARY)
#   message(FATAL_ERROR "CUBLAS library not found (and required)")
#endif (NOT CUBLAS_LIBRARY)

#set(MTL_INCLUDE_DIR "${MTL_INCLUDE_DIR}/mtl4")

#message(STATUS "Found GSL library: " ${GSL_LIBRARY})

#message(STATUS "Found CBLAS library: " ${GSL_CBLAS_LIBRARY})

#message(STATUS "Found CUBLAS library: " ${CUBLAS_LIBRARY})

#message(STATUS "Found GSL header file in:  " ${GSL_INCLUDE_DIR})

#message(STATUS "Found MTL header file in: " ${MTL_INCLUDE_DIR})

#message(STATUS "Found boost header files in: " ${BOOST_INCLUDE_DIR})



#include_directories(${GSL_INCLUDE_DIR} 
#					${CUDA_INCLUDE_DIR}
#					${MTL_INCLUDE_DIR}
#					${BOOST_INCLUDE_DIR}
#					)

set(SOURCE_FILES
	ccd.cpp
	CyclicCoordinateDescent.cpp	
	CompressedDataMatrix.cpp
	InputReader.cpp
	SCCSInputReader.cpp
	CLRInputReader.cpp
	RTestInputReader.cpp
	CCTestInputReader.cpp
	AbstractModelSpecifics.cpp
	AbstractDriver.cpp
	AbstractSelector.cpp
	CrossValidationSelector.cpp
	CrossValidationDriver.cpp
	BootstrapSelector.cpp
	BootstrapDriver.cpp)
	
if(CUDA_FOUND)
	set(SOURCE_FILES ${SOURCE_FILES}
	    GPUCyclicCoordinateDescent.cpp
#	    BetterGPU.cpp
	    )
	add_definitions(-DCUDA)	
 #	if(APPLE)
 #		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
 #		set(CUDA_64_BIT_DEVICE_CODE OFF)	
 # 	endif(APPLE)		
	if(APPLE)
		set(CUDA_NVCC_FLAGS "-arch=sm_11")
	else(APPLE)
		set(CUDA_NVCC_FLAGS "-arch=sm_20")
	endif(APPLE)
#	cuda_add_library(ccd-lib ${SOURCE_FILES})
 	cuda_add_executable(ccd ${SOURCE_FILES})
 	target_link_libraries(ccd CUDADriver)
else(CUDA_FOUND)
	add_executable(ccd ${SOURCE_FILES})
endif(CUDA_FOUND)


