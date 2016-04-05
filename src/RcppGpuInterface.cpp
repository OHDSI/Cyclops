/*
 * RcppGpuInterface.cpp
 *
 * @author Marc Suchard
 */

 #include "Rcpp.h"

 #include <boost/compute/algorithm/reduce.hpp> // TODO Change

//' @export
// [[Rcpp::export("listOpenCLDevices")]]
Rcpp::CharacterVector listOpenCLDevices() {
	using namespace Rcpp;

	CharacterVector devices;
    for (const auto &device : boost::compute::system::devices()) {
		devices.push_back(device.name());
	}

    return devices;
}

// [[Rcpp::export(".getDefaultOpenCLDevice")]]
std::string getDefaultOpenCLDevice() {
	return boost::compute::system::default_device().name();
}
