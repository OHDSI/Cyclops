#include <vector>

template <typename RealType>
void resizeAndCopyToDeviceCuda(const  std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec, cudaStream_t* stream);

