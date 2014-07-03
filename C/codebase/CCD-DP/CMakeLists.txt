include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(BASE_SOURCE_FILES	
	../CCD/CyclicCoordinateDescent.cpp	
	../CCD/CompressedDataMatrix.cpp
	../CCD/ModelData.cpp
	../CCD/io/HierarchyReader.cpp
	../CCD/io/InputReader.cpp
	../CCD/io/SCCSInputReader.cpp
	../CCD/io/CLRInputReader.cpp
	../CCD/io/RTestInputReader.cpp
	../CCD/io/CoxInputReader.cpp
	../CCD/io/CCTestInputReader.cpp
	../CCD/MCMC/BetaParameter.cpp
	../CCD/MCMC/CredibleIntervals.cpp
	../CCD/MCMC/HyperpriorParameter.cpp
	../CCD/MCMC/IndependenceSampler.cpp
	../CCD/MCMC/MCMCDriver.cpp
	../CCD/MCMC/ModelSampler.cpp
	../CCD/MCMC/ModelSelectionDriver.cpp
	../CCD/MCMC/ModelPrior.cpp
	../CCD/MCMC/MHRatio.cpp
	../CCD/MCMC/MCMCModel.cpp
	../CCD/MCMC/OneDimensionRandomWalk.cpp
	../CCD/MCMC/Parameter.cpp
	../CCD/MCMC/RandomWalk.cpp
	../CCD/MCMC/SigmaSampler.cpp
	../CCD/MCMC/TransitionKernel.cpp
	../CCD/AbstractModelSpecifics.cpp
	../CCD/AbstractDriver.cpp
	../CCD/AbstractSelector.cpp
	../CCD/AbstractCrossValidationDriver.cpp
	../CCD/ProportionSelector.cpp
	../CCD/CrossValidationSelector.cpp
    ../CCD/GridSearchCrossValidationDriver.cpp
    ../CCD/HierarchyGridSearchCrossValidationDriver.cpp
	../CCD/AutoSearchCrossValidationDriver.cpp
	../CCD/HierarchyAutoSearchCrossValidationDriver.cpp
	../CCD/BootstrapSelector.cpp
	../CCD/BootstrapDriver.cpp
	../utils/HParSearch.cpp
	)
	
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
	target_link_libraries(ccd-dp base_bsccs-dp)
#endif(CUDA_FOUND)


