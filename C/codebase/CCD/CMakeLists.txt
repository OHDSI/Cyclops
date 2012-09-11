include_directories(. ${CMAKE_SOURCE_DIR}/codebase)

set(BASE_SOURCE_FILES
	CyclicCoordinateDescent.cpp	
	CompressedDataMatrix.cpp
	ModelData.cpp
	io/InputReader.cpp
	io/SCCSInputReader.cpp
	io/CLRInputReader.cpp
	io/RTestInputReader.cpp
	io/CoxInputReader.cpp
	io/CCTestInputReader.cpp
	AbstractModelSpecifics.cpp
	AbstractDriver.cpp
	AbstractSelector.cpp
	ProportionSelector.cpp
	CrossValidationSelector.cpp
	CrossValidationDriver.cpp
	BootstrapSelector.cpp
	BootstrapDriver.cpp)
	
set(CCD_SOURCE_FILES
    ccd.cpp)
    
set(IMPUTE_SOURCE_FILES
	imputation/ccdimpute.cpp
	imputation/ImputationPolicy.cpp
	imputation/ImputeVariables.cpp)   
	
add_library(base_bsccs ${BASE_SOURCE_FILES})	
	
if(CUDA_FOUND)
	set(CCD_SOURCE_FILES ${CCD_SOURCE_FILES}
	    GPUCyclicCoordinateDescent.cpp
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
 	cuda_add_executable(ccd ${CCD_SOURCE_FILES})
 	target_link_libraries(ccd CUDADriver)
else(CUDA_FOUND)
	add_executable(ccd ${CCD_SOURCE_FILES})
endif(CUDA_FOUND)

add_executable(ccdimpute ${IMPUTE_SOURCE_FILES})

target_link_libraries(ccd base_bsccs)
target_link_libraries(ccdimpute base_bsccs)


