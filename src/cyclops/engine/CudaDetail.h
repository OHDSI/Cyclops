#include <vector>

template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDeviceCuda(const HostVec& hostVec, DeviceVec& deviceVec);

template <typename DeviceVec, typename HostVec>
void resizeCudaVec(const HostVec& hostVec, DeviceVec& deviceVec);

template <typename DeviceVec>
void resizeAndZeroToDeviceCuda(DeviceVec& deviceVec, int num_items);

/*
template <class T>
class CudaDetail {

public:
	CudaDetail();
	~CudaDetail();

	void resizeAndCopyToDeviceCuda(std::vector<T>& hostVec, thrust::device_vector<T>& deviceVec);

};
*/

