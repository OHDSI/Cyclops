set(RCCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../src)
set(CCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/codebase)

include_directories(. 
    ${RCCD_SOURCE_DIR}
    ${RCCD_SOURCE_DIR}/ccd
    ${CMAKE_SOURCE_DIR}/codebase
    ${CMAKE_SOURCE_DIR}/codebase/CCD    
    )

set(BASE_SOURCE_FILES	
	${RCCD_SOURCE_DIR}/ccd/CyclicCoordinateDescent.cpp	
	${RCCD_SOURCE_DIR}/ccd/CompressedDataMatrix.cpp
	${RCCD_SOURCE_DIR}/ccd/ModelData.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/HierarchyReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/InputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/SCCSInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CLRInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/RTestInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CoxInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CCTestInputReader.cpp
	${RCCD_SOURCE_DIR}/ccd/AbstractModelSpecifics.cpp
	${RCCD_SOURCE_DIR}/ccd/AbstractDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/AbstractSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/AbstractCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/ProportionSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/CrossValidationSelector.cpp
    ${RCCD_SOURCE_DIR}/ccd/GridSearchCrossValidationDriver.cpp
    ${RCCD_SOURCE_DIR}/ccd/HierarchyGridSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/AutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/HierarchyAutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/BootstrapSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/BootstrapDriver.cpp
	${RCCD_SOURCE_DIR}/utils/HParSearch.cpp
	)
	
set(CCD_SOURCE_FILES

	${CCD_SOURCE_DIR}/CCD/ccd.cpp)
	
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
	target_link_libraries(ccd-dp base_bsccs-dp)
#endif(CUDA_FOUND)


