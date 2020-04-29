#include <vector>

template <class T>
class CudaDetail {

public:
	CudaDetail();
	~CudaDetail();

	void resizeAndCopyToDeviceCuda(std::vector<T>& hostVec, thrust::device_vector<T>& deviceVec);

};

