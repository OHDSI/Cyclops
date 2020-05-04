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
    T* d_Numer;
    T* d_Numer2;
    T* d_AccNumer;
    T* d_AccNumer2;
    T* d_itr;
    
    T* d_NWeight;
    T* d_Gradient;
    T* d_Hessian;
    T* d_G;
    T* d_H;

    // Operator
    CustomExp    exp_op;

    // Allocate temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CudaKernel(T* h_Numer, T* h_Numer2, T* h_AccDenom, T* h_NWeight, int num_items);
    CudaKernel(thrust::device_vector<T>& X, thrust::device_vector<int>& K, T* h_XBeta, T* h_ExpXBeta, int num_items);
    ~CudaKernel();

    void CubScan(T* d_in, T* d_out, int num_items);
    void CubReduce(T* d_in, T* d_out, int num_items);
    void computeAccDenomMalloc(int num_items);
    void computeAccDenom(int num_items);
    void computeAccNumerMalloc(int num_items);
    void computeAccNumer(int num_items);
    void CubExpScanMalloc(int num_items);
    void CubExpScan(int num_items);
    void updateXBeta(unsigned int offX, unsigned int offK, const unsigned int taskCount, T delta, int gridSize, int blockSize);
    void computeGradientAndHessian(size_t& N, int& gridSize, int& blockSize);

};
