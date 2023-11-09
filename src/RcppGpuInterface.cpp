/*
 * RcppGpuInterface.cpp
 *
 * @author Marc Suchard
 */

 #include "Rcpp.h"

#ifdef HAVE_OPENCL
 #include <boost/compute/algorithm/reduce.hpp> // TODO Change
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
 #include <cuda_runtime_api.h>
 #include <cuda.h>
#endif // HAVE_CUDA

//' @title List available GPU devices
//'
//' @description
//' \code{listGPUDevices} list available GPU devices
//'
//' @export
// [[Rcpp::export("listGPUDevices")]]
Rcpp::CharacterVector listGPUDevices() {
	using namespace Rcpp;
	CharacterVector devices;

#ifdef HAVE_OPENCL
    for (const auto &device : boost::compute::system::devices()) {
		devices.push_back(device.name());
	}
#endif // HAVE_OPENCL;

#ifdef HAVE_CUDA
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        devices.push_back(deviceProperties.name);
    }
#endif // HAVE_CUDA;

    return devices;
}

// [[Rcpp::export(".getDefaultGPUDevice")]]
std::string getDefaultGPUDevice() {
#ifdef HAVE_OPENCL
	return boost::compute::system::default_device().name();
#else
    #ifdef HAVE_CUDA
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, 0);
        return deviceProperties.name;
    #else
        return "";
    #endif // HAVE_CUDA
#endif // HAVE_OPENCL
}
