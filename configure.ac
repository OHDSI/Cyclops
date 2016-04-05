# Process this file with autoconf to produce a configure script.
AC_INIT(Cyclops, 1.1, msuchard@ucla.edu)

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

# platform check
RHOST=`"${R_HOME}/bin/Rscript" -e 'cat(Sys.info()[["sysname"]])'`

# Check for which host we are on and setup a few things
# specifically based on the host
case $RHOST in
  Darwin* )
        # Do something specific for mac
        OPENCL_LIB="-framework OpenCL"
        ;;
  Linux*)
        # Do something specific for linux
        OPENCL_LIB="-lOpenCL"
        ;;
   *BSD*)
        # Do something specific for BSD
        AC_MSG_ERROR([BSD is not currently supported])
        ;;
    *)
        #Default Case
        AC_MSG_ERROR([Your platform is not currently supported])
        ;;
esac

AC_SUBST(OPENCL_LIB)


AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
