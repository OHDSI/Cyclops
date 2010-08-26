include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(SOURCE_FILES
	../CCD/ccd.cpp
	../CCD/CyclicCoordinateDescent.cpp	
	../CCD/CompressedIndicatorMatrix.cpp
	../CCD/InputReader.cpp)
	
set(DOUBLE_PRECISION true)	
	
if(CUDA_FOUND)
	set(SOURCE_FILES ${SOURCE_FILES}
	    ../CCD/GPUCyclicCoordinateDescent.cpp)
	add_definitions(-DCUDA)
 	if(APPLE)
 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
 		set(CUDA_64_BIT_DEVICE_CODE OFF)	
 	endif(APPLE)		
 	if(DOUBLE_PRECISION)
 		add_definitions(-DDOUBLE_PRECISION)
 		set(CUDA_NVCC_FLAGS -arch compute_13)
 	else(DOUBLE_PRECISION)	
 		set(CUDA_NVCC_FLAGS -arch compute_11)
 	endif(DOUBLE_PRECISION) 	
 	cuda_add_executable(ccd-dp ${SOURCE_FILES})
 	target_link_libraries(ccd-dp CUDADriver-dp)
else(CUDA_FOUND)
	add_executable(ccd-dp ${SOURCE_FILES})
endif(CUDA_FOUND)


