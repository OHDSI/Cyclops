#include <vector>

struct CustomExp
{
    template <typename RealType>
    __host__ __device__ __forceinline__
    RealType operator()(const RealType &a) const {
        return exp(a);
    }
};

template <class RealType>
class CudaKernel {

public:

    // Device arrays
    RealType* d_XBeta;
    RealType* d_ExpXBeta;
    RealType* d_AccDenom;
    RealType* d_Numer;
    RealType* d_Numer2;
    RealType* d_AccNumer;
    RealType* d_AccNumer2;
    RealType* d_itr;
    
    RealType* d_NWeight;
    RealType* d_BufferG;
    RealType* d_BufferH;
    RealType* d_Gradient;
    RealType* d_Hessian;

    // Operator
    CustomExp    exp_op;


    CudaKernel();
    ~CudaKernel();

    void initialize(int K, int N);   
    void updateXBeta(const thrust::device_vector<RealType>& X, const thrust::device_vector<int>& K, unsigned int offX, unsigned int offK, const unsigned int taskCount, RealType delta, int gridSize, int blockSize);
    void computeGradientAndHessian(size_t& N, int& gridSize, int& blockSize);

    void CubScan(RealType* d_in, RealType* d_out, int num_items);
    void CubReduce(RealType* d_in, RealType* d_out, int num_items);

    /*
    void computeAccDenomMalloc(int num_items);
    void computeAccDenom(int num_items);
    void computeAccNumerMalloc(int num_items);
    void computeAccNumer(int num_items);

    void CubExpScanMalloc(int num_items);
    void CubExpScan(int num_items);
    */
};

