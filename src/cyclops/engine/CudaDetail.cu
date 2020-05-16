#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "CudaDetail.h"
/*
template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDeviceCuda(const HostVec& hostVec, DeviceVec& deviceVec) 
{
	deviceVec.resize(hostVec.size());
	thrust::copy(hostVec.begin(), hostVec.end(), deviceVec.begin());
}

template void resizeAndCopyToDeviceCuda<thrust::device_vector<double>, std::vector<double>>(const std::vector<double>& hostVec, thrust::device_vector<double>& deviceVec);

template void resizeAndCopyToDeviceCuda<thrust::device_vector<float>, std::vector<float>>(const std::vector<float>& hostVec, thrust::device_vector<float>& deviceVec);

template void resizeAndCopyToDeviceCuda<thrust::device_vector<int>, std::vector<int>>(const std::vector<int>& hostVec, thrust::device_vector<int>& deviceVec);

template void resizeAndCopyToDeviceCuda<thrust::device_vector<unsigned int>, std::vector<unsigned int>>(const std::vector<unsigned int>& hostVec, thrust::device_vector<unsigned int>& deviceVec);
*/

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


