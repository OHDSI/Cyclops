
find_package(JNI)

if (NOT JNI_FOUND)
	find_path (JNI_HEADER
		jni.h
		PATHS /usr/lib/gcc/x86_64-redhat-linux/4.4.7/ /u/local/apps/java/current/
		PATH_SUFFIXES include
	)
	message (STATUS "${JNI_HEADER}")
	set (JNI_INCLUDE_DIRS "${JNI_HEADER}")
	message (STATUS "${JNI_INCLUDE_DIRS}")

	if (JNI_HEADER)
		set (JNI_FOUND 1)
	endif(JNI_HEADER)
endif (NOT JNI_FOUND)


if (JNI_FOUND)
   message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
   message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")

include_directories(. ${CMAKE_SOURCE_DIR}/codebase ${JNI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/codebase/CCD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")

set(JNI_SOURCE_FILES
	dr_inference_regression_RegressionJNIWrapper.cpp
	../CCD/ccd.cpp)
	
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

add_definitions(-DDOUBLE_PRECISION)
target_link_libraries(bsccs_jni base_bsccs-dp ${ODBC_LIBRARIES})
else(JNI_FOUND)
	message (WARNING " No JNI found!")
endif(JNI_FOUND)
