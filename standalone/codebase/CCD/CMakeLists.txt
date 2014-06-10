set(RCCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../src)
set(CCD_SOURCE_DIR ${CMAKE_SOURCE_DIR}/codebase)

include_directories(. 
    ${RCCD_SOURCE_DIR}
    ${RCCD_SOURCE_DIR}/ccd
    ${CMAKE_SOURCE_DIR}/codebase
    ${CMAKE_SOURCE_DIR}/codebase/CCD    
    )

set(BASE_SOURCE_FILES	
    ${RCCD_SOURCE_DIR}/ccd/CcdInterface.cpp	
	${RCCD_SOURCE_DIR}/ccd/CyclicCoordinateDescent.cpp	
	${RCCD_SOURCE_DIR}/ccd/CompressedDataMatrix.cpp
	${RCCD_SOURCE_DIR}/ccd/ModelData.cpp
	${RCCD_SOURCE_DIR}/ccd/io/InputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/HierarchyReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/SCCSInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CLRInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/RTestInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CoxInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/io/CCTestInputReader.cpp
	 ${CCD_SOURCE_DIR}/CCD/CmdLineCcdInterface.cpp
	${RCCD_SOURCE_DIR}/ccd/engine/AbstractModelSpecifics.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/AbstractDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/AbstractSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/AbstractCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/ProportionSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/CrossValidationSelector.cpp
    ${RCCD_SOURCE_DIR}/ccd/drivers/GridSearchCrossValidationDriver.cpp
    ${RCCD_SOURCE_DIR}/ccd/drivers/HierarchyGridSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/AutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/HierarchyAutoSearchCrossValidationDriver.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/BootstrapSelector.cpp
	${RCCD_SOURCE_DIR}/ccd/drivers/BootstrapDriver.cpp
	${RCCD_SOURCE_DIR}/utils/HParSearch.cpp
	)
	
set(CCD_SOURCE_FILES

	${CCD_SOURCE_DIR}/CCD/ccd.cpp)		
    
set(IMPUTE_SOURCE_FILES
	${CCD_SOURCE_DIR}/CCD/imputation/ccdimpute.cpp
	${CCD_SOURCE_DIR}/CCD/imputation/ImputationPolicy.cpp
	${CCD_SOURCE_DIR}/CCD/imputation/ImputeVariables.cpp)   
	
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

#add_executable(ccdimpute ${IMPUTE_SOURCE_FILES})

target_link_libraries(ccd base_bsccs)
#target_link_libraries(ccdimpute base_bsccs)


