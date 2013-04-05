include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

find_path(BOOST_INCLUDE_DIR
	NAMES boost
	DOCS "Location of boost header file"
	PATHS "/Users/trevorshaddox/Desktop/workspace/boost_1_51_0")
		
find_path(EIGEN_INCLUDE_DIR
	NAMES Eigen
	DOCS "Location of boost header files"
	PATHS "/Users/tshaddox/Desktop/workspace/eigen")
	
	
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})



set(SOURCE_FILES
	../CCD/ccd.cpp
	../CCD/CyclicCoordinateDescent.cpp	
	../CCD/CompressedDataMatrix.cpp
	../CCD/InputReader.cpp
	../CCD/SCCSInputReader.cpp
	../CCD/CLRInputReader.cpp
	../CCD/RTestInputReader.cpp
	../CCD/CCTestInputReader.cpp
	../CCD/AbstractDriver.cpp
	../CCD/AbstractSelector.cpp
	../CCD/CrossValidationSelector.cpp
	../CCD/CrossValidationDriver.cpp
	../CCD/BootstrapSelector.cpp
	../CCD/BootstrapDriver.cpp
	../CCD/SparseRowVector.cpp
	../CCD/MCMCDriver.cpp
	../CCD/IndependenceSampler.cpp
	../CCD/MHRatio.cpp
	../CCD/SigmaSampler.cpp
	../CCD/CredibleIntervals.cpp
	../CCD/Parameter.cpp
	../CCD/ModelLikelihood.cpp)
	
set(DOUBLE_PRECISION true)	
	
if(CUDA_FOUND)
	set(SOURCE_FILES ${SOURCE_FILES}
	    ../CCD/GPUCyclicCoordinateDescent.cpp
	    ../CCD/BetterGPU.cpp)
	add_definitions(-DCUDA)
# 	if(APPLE)
# 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
# 		set(CUDA_64_BIT_DEVICE_CODE OFF)	
# 	endif(APPLE)		
	add_definitions(-DDOUBLE_PRECISION)	
 	if(APPLE)
 		set(CUDA_NVCC_FLAGS -arch=sm_11)
 	else(APPLE)	 		
 		set(CUDA_NVCC_FLAGS -arch=sm_20) 		
 	endif(APPLE) 	
 	cuda_add_executable(ccd-dp ${SOURCE_FILES})
 #	target_link_libraries(ccd-dp CUDADriver-dp)
 	target_link_libraries(ccd-dp CUDADriver)
	target_link_libraries(ccd-dp ${BOOST_INCLUDE_LIBRARY})
	target_link_libraries(ccd-dp ${EIGEN_INCLUDE_LIBRARY})
	target_link_libraries(ccd-dp CUDARuntime)
 	
else(CUDA_FOUND)
	add_definitions(-DDOUBLE_PRECISION)	
	add_executable(ccd-dp ${SOURCE_FILES})
	target_link_libraries(ccd-dp ${BOOST_INCLUDE_LIBRARY})
endif(CUDA_FOUND)


