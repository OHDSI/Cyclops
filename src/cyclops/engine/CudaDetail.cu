#include <stdio.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "CudaDetail.h"

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


