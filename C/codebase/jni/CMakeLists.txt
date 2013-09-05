include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")


set(JNI_SOURCE_FILES
	dr_inference_regression_RegressionJNIWrapper.cpp
	../CCD/ccd.cpp)

if(APPLE)
	include_directories(/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers)
endif(APPLE)
	
add_library(bsccs_jni SHARED ${JNI_SOURCE_FILES})

#IF(APPLE)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "lib" SUFFIX ".jnilib")
#ELSEIF(WIN32)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "" SUFFIX ".dll")
#ELSEIF(UNIX)
#  SET_TARGET_PROPERTIES(${SWIG_MODULE_<module_name>_REAL_NAME} PROPERTIES
#    PREFIX "lib" SUFFIX ".so")
#ENDIF(APPLE)

target_link_libraries(bsccs_jni base_bsccs)