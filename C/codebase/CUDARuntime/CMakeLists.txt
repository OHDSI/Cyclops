CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
find_path(CUSP_INCLUDE_DIR
	NAMES cusp
	DOCS "Location of boost header files"
	PATHS "/Users/msuchard/Dropbox/Projects/cusplibrary"
	PATHS "/Users/tshaddox/Desktop")
  
  find_library(CUBLAS_LIBRARY
	NAMES libclublas.dylib
	DOCS "Location of CUBLAS library file"
	PATHS "/usr/local/cuda/lib")
	
	message(STATUS "Found CUBLAS library: " ${CUBLAS_LIBRARY})



include_directories (${CMAKE_SOURCE_DIR}/codebase ${CUDA_INCLUDE_DIRS} ${CUSP_INCLUDE_DIR})

message("Cmake current = " ${CMAKE_CURRENT_SOURCE_DIR})
message("cmake source / codebase = "  ${CMAKE_SOURCE_DIR}/codebase)
message("CUDA include = " ${CUDA_INCLUDE_DIRS})

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)



#cuda_add_library(CUDARuntime ${CUBLAS_LIBRARY})

CUDA_ADD_LIBRARY(CUDARuntime
	CUSPEngine.h CUSPEngine.cu
	)

target_link_libraries(CUDARuntime ${CUBLAS_LIBRARY})
