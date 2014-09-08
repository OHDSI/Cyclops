include_directories(. ${CMAKE_SOURCE_DIR}/codebase)

	
set(BASE_SOURCE_FILES
	CyclicCoordinateDescent.cpp	
	CompressedDataMatrix.cpp
	ModelData.cpp
	io/HierarchyReader.cpp
	io/InputReader.cpp
	io/SCCSInputReader.cpp
	io/CLRInputReader.cpp
	io/RTestInputReader.cpp
	io/CoxInputReader.cpp
	io/CCTestInputReader.cpp
	#MCMC/BetaParameter.cpp
	#MCMC/CredibleIntervals.cpp
	#MCMC/HyperpriorParameter.cpp
	#MCMC/IndependenceSampler.cpp
	#MCMC/MCMCDriver.cpp
	#MCMC/MHRatio.cpp
	#MCMC/ModelSampler.cpp
	#MCMC/ModelSelectionDriver.cpp
	#MCMC/ModelPrior.cpp
	#MCMC/MCMCModel.cpp
	#MCMC/OneDimensionRandomWalk.cpp
	#MCMC/Parameter.cpp
	#MCMC/RandomWalk.cpp
	#MCMC/SigmaSampler.cpp
	#MCMC/TransitionKernel.cpp
	AbstractModelSpecifics.cpp
	AbstractDriver.cpp
	AbstractSelector.cpp
	AbstractCrossValidationDriver.cpp
	ProportionSelector.cpp
	CrossValidationSelector.cpp
	GridSearchCrossValidationDriver.cpp
	HierarchyGridSearchCrossValidationDriver.cpp
	AutoSearchCrossValidationDriver.cpp
	HierarchyAutoSearchCrossValidationDriver.cpp
	BootstrapSelector.cpp
	BootstrapDriver.cpp
	../utils/HParSearch.cpp)
	
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


