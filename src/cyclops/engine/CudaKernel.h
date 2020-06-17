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
struct functorCGH :
        public thrust::unary_function<thrust::tuple<RealType, RealType, RealType, RealType>,
                                      thrust::tuple<RealType, RealType>>
{
        typedef typename thrust::tuple<RealType, RealType, RealType, RealType> InputTuple;
        typedef typename thrust::tuple<RealType, RealType>       OutputTuple;

        __host__ __device__
                OutputTuple operator()(const InputTuple& t) const
                {
                        auto temp = thrust::get<0>(t) * thrust::get<1>(t) / thrust::get<2>(t);
                        return OutputTuple(temp, temp * (1 - thrust::get<1>(t) / thrust::get<2>(t)));
                }
};

template <typename RealType>
class CudaKernel {
	
    typedef thrust::tuple<RealType,RealType> Tup2;
    typedef typename thrust::device_vector<RealType>::iterator VecItr;
    typedef thrust::tuple<VecItr,VecItr,VecItr,VecItr> TupVec4;
    typedef thrust::zip_iterator<TupVec4> ZipVec4;

public:

    // Device arrays
    RealType* d_itr; 
    Tup2 init = thrust::make_tuple<RealType, RealType>(0, 0);

    // Operator
    CustomExp    exp_op;
    functorCGH<RealType> cGAH;

    // Declare temporary storage
    void *d_temp_storage0 = NULL;
    size_t temp_storage_bytes0 = 0;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    void *d_temp_storage_gh = NULL;
    size_t temp_storage_bytes_gh = 0;        

    CudaKernel();
    ~CudaKernel();

    void allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
                    thrust::device_vector<RealType>& d_Numerator,
		    thrust::device_vector<RealType>& d_Numerator2,
		    thrust::device_vector<RealType>& d_AccDenom,
		    thrust::device_vector<RealType>& d_AccNumer,
		    thrust::device_vector<RealType>& d_AccNumer2,
		    thrust::device_vector<RealType>& d_NWeight,
		    thrust::device_vector<RealType>& d_Gradient,
		    thrust::device_vector<RealType>& d_Hessian,
		    size_t& N,
		    thrust::device_vector<int>& indicesN);
    void updateXBeta(const thrust::device_vector<RealType>& X, 
		     const thrust::device_vector<int>& K, 
		     unsigned int offX, 
		     unsigned int offK, 
		     const unsigned int taskCount, 
		     RealType delta, 
		     thrust::device_vector<RealType>& dXBeta, 
		     thrust::device_vector<RealType>& dExpXBeta, 
		     thrust::device_vector<RealType>& dNumerator,
		     thrust::device_vector<RealType>& dNumerator2,
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
				   size_t& N
//				   ,const std::vector<int>& K,
//				   unsigned int offK,
//				   thrust::device_vector<int>& indicesN
				   );
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

    void empty4(thrust::device_vector<RealType>& d_AccNumer,
                                  thrust::device_vector<RealType>& d_AccNumer2,
                                  thrust::device_vector<RealType>& d_Buffer1,
                                  thrust::device_vector<RealType>& d_Buffer2);
    void empty2(thrust::device_vector<RealType>& d_AccDenom,
		    thrust::device_vector<RealType>& d_Buffer3);
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

