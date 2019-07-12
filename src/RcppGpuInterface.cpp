/*
 * RcppGpuInterface.cpp
 *
 * @author Marc Suchard
 */

 #include "Rcpp.h"

#ifdef HAVE_OPENCL
 #include <boost/compute/algorithm/reduce.hpp> // TODO Change
#endif // HAVE_OPENCL

//' @export
// [[Rcpp::export("listOpenCLDevices")]]
Rcpp::CharacterVector listOpenCLDevices() {
	using namespace Rcpp;
	CharacterVector devices;

#ifdef HAVE_OPENCL
    for (const auto &device : boost::compute::system::devices()) {
		devices.push_back(device.name());
	}
#endif // HAVE_OPENCL;

    return devices;
}

// [[Rcpp::export(".getDefaultOpenCLDevice")]]
std::string getDefaultOpenCLDevice() {
#ifdef HAVE_OPENCL
	return boost::compute::system::default_device().name();
#else
    return "";
#endif // HAVE_OPENCL
}
