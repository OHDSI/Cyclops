#include <vector>

struct CustomExp
{
    template <typename RealType>
    __host__ __device__ __forceinline__
    RealType operator()(const RealType &a) const {
        return exp(a);
    }
};

template <typename RealType>
class CudaKernel {

public:

    // Device arrays
    RealType* d_itr; 

    // Operator
    CustomExp    exp_op;


    CudaKernel();
    ~CudaKernel();

    void initialize(int K, int N);   
    void updateXBeta(const thrust::device_vector<RealType>& X, 
		     const thrust::device_vector<int>& K, 
		     unsigned int offX, 
		     unsigned int offK, 
		     const unsigned int taskCount, 
		     RealType delta, 
		     thrust::device_vector<RealType>& dXBeta, 
		     thrust::device_vector<RealType>& dExpXBeta, 
		     int gridSize, int blockSize);
    void computeNumeratorForGradient(const thrust::device_vector<RealType>& X,
                                     const thrust::device_vector<int>& K,
                                     unsigned int offX,
                                     unsigned int offK,
                                     const unsigned int taskCount,
                                     thrust::device_vector<RealType>& dExpXBeta,
                                     thrust::device_vector<RealType>& dNumerator,
                                     thrust::device_vector<RealType>& dNumerator2,
                                     int gridSize, int blockSize);
    void computeGradientAndHessian(thrust::device_vector<RealType>& d_AccNumer, 
		    		   thrust::device_vector<RealType>& d_AccNumer2, 
				   thrust::device_vector<RealType>& d_AccDenom, 
				   thrust::device_vector<RealType>& d_NWeight, 
				   thrust::device_vector<RealType>& d_Gradient, 
				   thrust::device_vector<RealType>& d_Hessian, 
				   size_t& N);
    void computeAccumulatedNumerator(thrust::device_vector<RealType>& d_Numerator,
                                     thrust::device_vector<RealType>& d_Numerator2,
                                     thrust::device_vector<RealType>& d_AccNumer,
                                     thrust::device_vector<RealType>& d_AccNumer2,
                                     size_t& N);
    void computeAccumulatedNumerAndDenom(thrust::device_vector<RealType>& d_Denominator,
                                         thrust::device_vector<RealType>& d_Numerator,
					 thrust::device_vector<RealType>& d_Numerator2,
					 thrust::device_vector<RealType>& d_AccDenom,
					 thrust::device_vector<RealType>& d_AccNumer,
					 thrust::device_vector<RealType>& d_AccNumer2,
					 size_t& N);

    //void CubScan(thrust::device_vector<RealType>& d_in, thrust::device_vector<RealType>& d_out, int num_items);
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

