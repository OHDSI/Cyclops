#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "CudaDetail.h"

template <typename DeviceVec, typename HostVec>
void resizeAndZeroCudaVec(const HostVec& hostVec, DeviceVec& deviceVec)
{
        deviceVec.resize(hostVec.size());
        thrust::fill(deviceVec.begin(), deviceVec.end(), 0.0);
}

template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDeviceCuda(const HostVec& hostVec, DeviceVec& deviceVec) 
{
	deviceVec.resize(hostVec.size());
	thrust::copy(hostVec.begin(), hostVec.end(), deviceVec.begin());
}

template <typename DeviceVec, typename HostVec>
void resizeCudaVec(const HostVec& hostVec, DeviceVec& deviceVec)
{
        deviceVec.resize(hostVec.size());
}

template <typename DeviceVec>
void resizeCudaVecSize(DeviceVec& deviceVec, int num_items)
{
        deviceVec.resize(num_items);
}

template <typename DeviceVec, typename RealType>
void fillCudaVec(DeviceVec& deviceVec, RealType val)
{
        thrust::fill(deviceVec.begin(), deviceVec.end(), val);
}

template <typename DeviceVec>
void copyCudaVec(DeviceVec& source, DeviceVec& destination)
{
        destination = source;
}

template <typename DeviceVec>
void resizeAndZeroToDeviceCuda(DeviceVec& deviceVec, int num_items)
{
        deviceVec.resize(num_items);
        thrust::fill(deviceVec.begin(), deviceVec.end(), 0.0);
}

template <typename DeviceVec>
void printCudaVec(DeviceVec& deviceVec, DeviceVec& deviceVec1, DeviceVec& deviceVec2, int num_items)
{
	for (int i = 0; i < num_items; i++) {
//		if (deviceVec[i] != 1) {
			std::cout << " i: " << i << " v: " << deviceVec[i] << " v1: " << deviceVec1[i] << " v2: " << deviceVec2[i] << '\n';
//		}
	}
}

template void resizeAndZeroCudaVec<thrust::device_vector<double>, std::vector<double>>(const std::vector<double>& hostVec, thrust::device_vector<double>& deviceVec);
template void resizeAndZeroCudaVec<thrust::device_vector<float>, std::vector<float>>(const std::vector<float>& hostVec, thrust::device_vector<float>& deviceVec);

template void resizeAndCopyToDeviceCuda<thrust::device_vector<double>, std::vector<double>>(const std::vector<double>& hostVec, thrust::device_vector<double>& deviceVec);
template void resizeAndCopyToDeviceCuda<thrust::device_vector<float>, std::vector<float>>(const std::vector<float>& hostVec, thrust::device_vector<float>& deviceVec);

template void resizeAndCopyToDeviceCuda<thrust::device_vector<int>, std::vector<int>>(const std::vector<int>& hostVec, thrust::device_vector<int>& deviceVec);
template void resizeAndCopyToDeviceCuda<thrust::device_vector<unsigned int>, std::vector<unsigned int>>(const std::vector<unsigned int>& hostVec, thrust::device_vector<unsigned int>& deviceVec);

template void resizeCudaVec<thrust::device_vector<double>, std::vector<double>>(const std::vector<double>& hostVec, thrust::device_vector<double>& deviceVec);
template void resizeCudaVec<thrust::device_vector<float>, std::vector<float>>(const std::vector<float>& hostVec, thrust::device_vector<float>& deviceVec);

template void resizeCudaVecSize<thrust::device_vector<double>>(thrust::device_vector<double>& deviceVec, int num_items);
template void resizeCudaVecSize<thrust::device_vector<float>>(thrust::device_vector<float>& deviceVec, int num_items);
template void resizeCudaVecSize<thrust::device_vector<int>>(thrust::device_vector<int>& deviceVec, int num_items);

template void fillCudaVec<thrust::device_vector<double>, double>(thrust::device_vector<double>& deviceVec, double);
template void fillCudaVec<thrust::device_vector<float>, float>(thrust::device_vector<float>& deviceVec, float);

template void copyCudaVec<thrust::device_vector<double>>(thrust::device_vector<double>& source, thrust::device_vector<double>& destination);
template void copyCudaVec<thrust::device_vector<float>>(thrust::device_vector<float>& source, thrust::device_vector<float>& destination);

template void resizeAndZeroToDeviceCuda<thrust::device_vector<double>>(thrust::device_vector<double>& deviceVec, int num_items);
template void resizeAndZeroToDeviceCuda<thrust::device_vector<float>>(thrust::device_vector<float>& deviceVec, int num_items);

template void printCudaVec<thrust::device_vector<double>>(thrust::device_vector<double>& deviceVec, thrust::device_vector<double>& deviceVec1, thrust::device_vector<double>& deviceVec2, int num_items);
template void printCudaVec<thrust::device_vector<float>>(thrust::device_vector<float>& deviceVec, thrust::device_vector<float>& deviceVec1, thrust::device_vector<float>& deviceVec2, int num_items);


/*
template <class T>
CudaDetail<T>::CudaDetail(){

}

template <class T>
CudaDetail<T>::~CudaDetail(){

}

template <class T>
void CudaDetail<T>::resizeAndCopyToDeviceCuda(std::vector<T>& hostVec, thrust::device_vector<T>& deviceVec)
{
	deviceVec.resize(hostVec.size());
	thrust::copy(hostVec.begin(), hostVec.end(), deviceVec.begin());
}

template class CudaDetail<float>;
template class CudaDetail<double>;
template class CudaDetail<int>;
template class CudaDetail<unsigned int>;
*/

