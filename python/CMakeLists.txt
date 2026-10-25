# Build for the `cyclops` Python extension module.
#
# This is the only CMake file in the repository. The R package compiles the
# same C++ core from src/Makevars(.win), which CMake never sees, so nothing
# here affects `R CMD INSTALL`.
#
# Linux and macOS only: the core times itself with POSIX <sys/time.h>, so MSVC
# is not a supported target.

cmake_minimum_required(VERSION 3.19...3.31)

project(cyclops_python LANGUAGES CXX)

if(NOT CYCLOPS_ROOT)
    get_filename_component(CYCLOPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()
message(STATUS "Cyclops root: ${CYCLOPS_ROOT}")

if(NOT EXISTS "${CYCLOPS_ROOT}/src/cyclops/api/CyclopsApi.h")
    message(FATAL_ERROR
        "Cannot find the Cyclops C++ facade under ${CYCLOPS_ROOT}/src. "
        "Build from a full checkout of the repository, not from python/ alone.")
endif()

# --- version ----------------------------------------------------------------

file(READ "${CYCLOPS_ROOT}/DESCRIPTION" _cyclops_description)
string(REGEX MATCH "Version:[ \t]*([0-9][0-9.]*)" _ "${_cyclops_description}")
set(CYCLOPS_VERSION "${CMAKE_MATCH_1}")
if(NOT CYCLOPS_VERSION)
    message(FATAL_ERROR "Could not read Version from DESCRIPTION")
endif()

# --- dependencies -----------------------------------------------------------

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Eigen is header-only and used only for the Fisher-information and asymptotic
# variance matrices in CyclicCoordinateDescent. The R build gets it from the
# RcppEigen package; this build finds a system copy or downloads a pinned one.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found locally; fetching a pinned copy")
    include(FetchContent)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# --- sources ----------------------------------------------------------------

# Mirrors the object list in src/Makevars.in, minus the Rcpp translation units.
# cyclops/io/InputReader.cpp is deliberately omitted: it implements the
# text-file readers used only by the CLI, and every other translation unit
# references it solely from commented-out includes.
set(CYCLOPS_CORE_SOURCES
    cyclops/CcdInterface.cpp
    cyclops/CompressedDataMatrix.cpp
    cyclops/CyclicCoordinateDescent.cpp
    cyclops/ModelData.cpp
    cyclops/Timer.cpp

    cyclops/drivers/AbstractCrossValidationDriver.cpp
    cyclops/drivers/AbstractDriver.cpp
    cyclops/drivers/AbstractSelector.cpp
    cyclops/drivers/AutoSearchCrossValidationDriver.cpp
    cyclops/drivers/BootstrapDriver.cpp
    cyclops/drivers/BootstrapSelector.cpp
    cyclops/drivers/CrossValidationSelector.cpp
    cyclops/drivers/WeightBasedSelector.cpp
    cyclops/drivers/GridSearchCrossValidationDriver.cpp
    cyclops/drivers/HierarchyAutoSearchCrossValidationDriver.cpp
    cyclops/drivers/HierarchyGridSearchCrossValidationDriver.cpp
    cyclops/drivers/ProportionSelector.cpp

    cyclops/priors/CovariatePrior.cpp
    cyclops/engine/AbstractModelSpecifics.cpp

    # The language-neutral facade the bindings call.
    cyclops/api/CyclopsApi.cpp

    utils/HParSearch.cpp
)
list(TRANSFORM CYCLOPS_CORE_SOURCES PREPEND "${CYCLOPS_ROOT}/src/")

# --- target -----------------------------------------------------------------

pybind11_add_module(_cyclops MODULE src/bindings.cpp ${CYCLOPS_CORE_SOURCES})

target_include_directories(_cyclops PRIVATE
    "${CYCLOPS_ROOT}/src"
    # Several core headers include their siblings unqualified
    # (e.g. #include "ModelData.h"), so cyclops/ is on the path too.
    "${CYCLOPS_ROOT}/src/cyclops"
)

target_compile_definitions(_cyclops PRIVATE
    DOUBLE_PRECISION
    CYCLOPS_VERSION="${CYCLOPS_VERSION}"
    VERSION_INFO="${SKBUILD_PROJECT_VERSION}"
    # Selects the std::runtime_error error path in core headers that would
    # otherwise call Rcpp::stop. The R build never defines this and keeps Rcpp.
    CYCLOPS_NO_RCPP
)

target_compile_features(_cyclops PRIVATE cxx_std_17)

target_link_libraries(_cyclops PRIVATE Eigen3::Eigen Threads::Threads)

set_target_properties(_cyclops PROPERTIES CXX_EXTENSIONS OFF)

# -g1 mirrors src/Makevars.in: the templated kernels in engine/ generate
# enormous debug records otherwise.
target_compile_options(_cyclops PRIVATE
    $<$<CONFIG:Release>:-g1>
    -Wno-unknown-pragmas
)

install(TARGETS _cyclops DESTINATION cyclops)
