#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "CudaDetail.h"

template <typename RealType>
void resizeAndCopyToDeviceCuda(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec, cudaStream_t* stream)
{
	deviceVec.resize(hostVec.size());
	cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVec.data()),
			thrust::raw_pointer_cast(hostVec.data()),
			deviceVec.size()*sizeof(RealType),
			cudaMemcpyHostToDevice, stream[0]);
	cudaStreamSynchronize(stream[0]);
}

template void resizeAndCopyToDeviceCuda<double>(const std::vector<double>& hostVec, thrust::device_vector<double>& deviceVec, cudaStream_t* stream);
template void resizeAndCopyToDeviceCuda<float>(const std::vector<float>& hostVec, thrust::device_vector<float>& deviceVec, cudaStream_t* stream);
template void resizeAndCopyToDeviceCuda<int>(const std::vector<int>& hostVec, thrust::device_vector<int>& deviceVec, cudaStream_t* stream);
template void resizeAndCopyToDeviceCuda<unsigned int>(const std::vector<unsigned int>& hostVec, thrust::device_vector<unsigned int>& deviceVec, cudaStream_t* stream);

