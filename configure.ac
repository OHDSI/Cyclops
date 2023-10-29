# Process this file with autoconf to produce a configure script.
AC_INIT([Cyclops],[3.3],[msuchard@ucla.edu])

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

# platform check
RHOST=`"${R_HOME}/bin/Rscript" -e 'cat(Sys.info()[["sysname"]])'`

#################### CUDA Toolkit ####################

CUDA_LIBS=""
CUDA_CPPFLAGS=""

AC_MSG_CHECKING("checking environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
	CUDA_HOME=`find /usr/local/ -maxdepth 1 -name "cuda"`
fi

if test -z "${CUDA_HOME}"; then
	AC_MSG_NOTICE("no CUDA installation found")
else
	AC_MSG_NOTICE("using CUDA_HOME=${CUDA_HOME}")
	AC_ARG_WITH([arch],
		[AS_HELP_STRING([--with-arch=SM],[compute capability of the target device @<:@sm_70@:>@])],
		[ARCH="$withval"], [ARCH=sm_70])
	AC_MSG_RESULT([setting compute capability of the target device to ${ARCH}])
	NVCC="${CUDA_HOME}/bin/nvcc -arch=${ARCH}"
	AC_SUBST(NVCC)

	AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
	AC_CHECK_FILE([${CUDA_HOME}/lib64/libcudart.so], [BIT64="yes"])
	if test -z "${BIT64}"; then
		AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
		cu_libdir="${CUDA_HOME}/lib"
	else
		AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
		cu_libdir="${CUDA_HOME}/lib64"
	fi
	CUDA_LIBS="${CUDA_LIBS} -L${cu_libdir} -lcudart"
	CUDA_CPPFLAGS="${CUDA_CPPFLAGS} -DHAVE_CUDA -I${CUDA_HOME}/include -I${cu_libdir} -pthread -rdynamic"
fi

AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_CPPFLAGS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
