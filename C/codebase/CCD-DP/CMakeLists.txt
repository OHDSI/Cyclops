include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(SOURCE_FILES	
	../CCD/ccd.cpp
	../CCD/CyclicCoordinateDescent.cpp	
	../CCD/CompressedDataMatrix.cpp
	../CCD/io/InputReader.cpp
	../CCD/io/SCCSInputReader.cpp
	../CCD/io/CLRInputReader.cpp
	../CCD/io/RTestInputReader.cpp
	../CCD/io/CCTestInputReader.cpp
	../CCD/AbstractModelSpecifics.cpp
	../CCD/AbstractDriver.cpp
	../CCD/AbstractSelector.cpp
	../CCD/CrossValidationSelector.cpp
	../CCD/CrossValidationDriver.cpp
	../CCD/BootstrapSelector.cpp
	../CCD/BootstrapDriver.cpp)
	
set(DOUBLE_PRECISION true)	
	
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
	add_executable(ccd-dp ${SOURCE_FILES})
#endif(CUDA_FOUND)


