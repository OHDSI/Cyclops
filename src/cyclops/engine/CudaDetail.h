#include <vector>
/*
template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDeviceCuda(const HostVec& hostVec, DeviceVec& deviceVec);
*/

template <class T>
class CudaDetail {

public:
	CudaDetail();
	~CudaDetail();

	void resizeAndCopyToDeviceCuda(std::vector<T>& hostVec, thrust::device_vector<T>& deviceVec);

};


