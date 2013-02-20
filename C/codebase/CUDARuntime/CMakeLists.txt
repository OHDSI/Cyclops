CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
#include_directories (${CMAKE_SOURCE_DIR}/codebase ${CUDA_INCLUDE_DIRS})

message("Cmake current = " ${CMAKE_CURRENT_SOURCE_DIR})
message("cmake source / codebase = "  ${CMAKE_SOURCE_DIR}/codebase)
message("CUDA include = " ${CUDA_INCLUDE_DIRS})

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

CUDA_ADD_LIBRARY(CUDARuntime
	CUSPEngine.h CUSPEngine.cu
	)
	
	
#ADD_DEFINITION(-DMAKE_TEST)
#CUDA_ADD_EXECUTABLE(test
#  Crap.cu
#  )

