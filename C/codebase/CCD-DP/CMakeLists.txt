include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(BASE_SOURCE_FILES	
	../CCD/CyclicCoordinateDescent.cpp	
	../CCD/CompressedDataMatrix.cpp
	../CCD/ModelData.cpp
	../CCD/io/InputReader.cpp
	../CCD/io/SCCSInputReader.cpp
	../CCD/io/CLRInputReader.cpp
	../CCD/io/RTestInputReader.cpp
	../CCD/io/CoxInputReader.cpp
	../CCD/io/CCTestInputReader.cpp
	../CCD/io/InputOutputSystem.cpp
	../CCD/AbstractModelSpecifics.cpp
	../CCD/AbstractDriver.cpp
	../CCD/AbstractSelector.cpp
	../CCD/ProportionSelector.cpp
	../CCD/CrossValidationSelector.cpp
	../CCD/CrossValidationDriver.cpp
	../CCD/BootstrapSelector.cpp
	../CCD/BootstrapDriver.cpp)

set(CCD_SOURCE_FILES
	../CCD/ccd.cpp)
	
set(DOUBLE_PRECISION true)	
add_definitions(-DDOUBLE_PRECISION)
	
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
	add_definitions(-DDOUBLE_PRECISION)
	add_library(base_bsccs-dp ${BASE_SOURCE_FILES})	
	add_executable(ccd-dp ${CCD_SOURCE_FILES})
	target_link_libraries(ccd-dp base_bsccs-dp ${ODBC_LIBRARIES})
#endif(CUDA_FOUND)


