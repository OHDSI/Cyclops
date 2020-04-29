#include <vector>

struct CustomExp
{
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a) const {
        return exp(a);
    }
};

template <class T>
class CudaKernel {

public:

    // Allocate device arrays
    T* d_X;
    int* d_K;
    T* d_XBeta;
    T* d_ExpXBeta;
    T* d_AccDenom;
    T* d_itr;

    // Operator
    CustomExp    exp_op;

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CudaKernel(thrust::device_vector<T>& X, thrust::device_vector<int>& K, T* h_XBeta, T* h_ExpXBeta, int num_items);
    ~CudaKernel();

    void CubScanMalloc(int num_items);
    void CubScan(int num_items);
    void CubExpScanMalloc(int num_items);
    void CubExpScan(int num_items);
    void updateXBeta(unsigned int offX, unsigned int offK, unsigned int N, T delta, int gridSize, int blockSize);

};
