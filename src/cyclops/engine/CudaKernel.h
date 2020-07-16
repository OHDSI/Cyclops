#include <vector>
#include "../CompressedDataMatrix.h"

typedef typename bsccs::FormatType FormatType;

enum FormatTypeCuda {
        DENSE, SPARSE, INDICATOR, INTERCEPT
};

enum PriorTypeCuda {
	NOPRIOR, LAPLACE, NORMAL
};

struct CustomExp
{
	template <typename RealType>
	__host__ __device__ __forceinline__
	RealType operator()(const RealType &a) const {
	    return exp(a);
	}
};

template <typename RealType, bool isIndicator>
struct functorCGH :
		public thrust::unary_function<thrust::tuple<RealType, RealType, RealType, RealType>,
		        double2>
{
	typedef typename thrust::tuple<RealType, RealType, RealType, RealType> InputTuple;

	__host__ __device__
	double2 operator()(const InputTuple& t) const
	{
	    auto temp = thrust::get<1>(t) / thrust::get<2>(t);
	    double2 out;
	    out.x = thrust::get<0>(t) * temp;
	    if (isIndicator) {
	        out.y = out.x * (1 - temp);
	    } else {
	        out.y = thrust::get<0>(t) * (thrust::get<3>(t) / thrust::get<2>(t) - temp * temp);
	    }
	    return out;
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
	Tup2 init = thrust::make_tuple<RealType, RealType>(0, 0);
	double2 d_init;

	// Operator
	CustomExp    exp_op;
//	functorCGH<RealType> cGAH;
	functorCGH<RealType, true> compGradHessInd;
	functorCGH<RealType, false> compGradHessNInd;

	// Temporary storage required by cub::scan and cub::reduce
	void *d_temp_storage0 = NULL;
	size_t temp_storage_bytes0 = 0;
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	void *d_temp_storage_gh = NULL;
	size_t temp_storage_bytes_gh = 0;
	void *d_temp_storage_acc = NULL;
	size_t temp_storage_bytes_acc = 0;

	CudaKernel();
	~CudaKernel();

	void allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
	        thrust::device_vector<RealType>& d_Numerator,
	        thrust::device_vector<RealType>& d_Numerator2,
	        thrust::device_vector<RealType>& d_AccDenom,
	        thrust::device_vector<RealType>& d_AccNumer,
	        thrust::device_vector<RealType>& d_AccNumer2,
	        thrust::device_vector<RealType>& d_NWeight,
	        double2* dGH,
	        size_t& N,
	        thrust::device_vector<int>& indicesN);
	void updateXBeta(const thrust::device_vector<RealType>& d_X,
	        const thrust::device_vector<int>& d_K,
	        unsigned int offX,
	        unsigned int offK,
	        const unsigned int taskCount,
	        RealType delta,
	        thrust::device_vector<RealType>& d_XBeta,
	        thrust::device_vector<RealType>& d_ExpXBeta,
	        thrust::device_vector<RealType>& d_Numerator,
	        thrust::device_vector<RealType>& d_Numerator2,
	        int gridSize, int blockSize);
	void updateXBeta1(const thrust::device_vector<RealType>& d_X,
                 const thrust::device_vector<int>& d_K,
                 unsigned int offX,
                 unsigned int offK,
                 const unsigned int taskCount,
		 double2* d_GH,
		 thrust::device_vector<RealType>& d_XjY,
		 thrust::device_vector<RealType>& d_Bound,
                 thrust::device_vector<RealType>& d_Beta,
                 thrust::device_vector<RealType>& d_XBeta,
                 thrust::device_vector<RealType>& d_ExpXBeta,
                 thrust::device_vector<RealType>& d_Numerator,
                 thrust::device_vector<RealType>& d_Numerator2,
		 thrust::device_vector<RealType>& dPriorParams,
		 const int priorTypes,
                 int index, 
		 FormatType& formatType,
                 int gridSize, int blockSize);
	void computeNumeratorForGradient(const thrust::device_vector<RealType>& d_X,
	        const thrust::device_vector<int>& d_K,
	        unsigned int offX,
	        unsigned int offK,
	        const unsigned int taskCount,
	        thrust::device_vector<RealType>& d_ExpXBeta,
	        thrust::device_vector<RealType>& d_Numerator,
	        thrust::device_vector<RealType>& d_Numerator2,
		FormatType& formatType,
	        int gridSize, int blockSize);
	void processDelta(thrust::device_vector<RealType>& d_DeltaVector,
                      thrust::device_vector<RealType>& d_Bound,
                      thrust::device_vector<RealType>& d_Beta,
                      thrust::device_vector<RealType>& d_XjY,
                      double2* d_GH,
                      thrust::device_vector<RealType>& d_PriorParams,
                      std::vector<RealType>& priorTypes,
                      int index,
                      int gridSize, int blockSize);
	void computeGradientAndHessian(thrust::device_vector<RealType>& d_AccNumer,
            thrust::device_vector<RealType>& d_AccNumer2,
	        thrust::device_vector<RealType>& d_AccDenom,
	        thrust::device_vector<RealType>& d_NWeight,
	        double2* dGH,
		FormatType& formatType,
	        size_t& N
//	        ,const std::vector<int>& K,
//	        unsigned int offK,
//	        thrust::device_vector<int>& indicesN
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
	const std::vector<int>& getPriorType(std::vector<int>& priorTypes) const;


	void empty4(thrust::device_vector<RealType>& d_AccNumer,
	        thrust::device_vector<RealType>& d_AccNumer2,
	        thrust::device_vector<RealType>& d_Buffer1,
	        thrust::device_vector<RealType>& d_Buffer2);
	void empty2(thrust::device_vector<RealType>& d_AccDenom,
	        thrust::device_vector<RealType>& d_Buffer3);

	void CubScan(RealType* d_in, RealType* d_out, int num_items);
	void CubReduce(RealType* d_in, RealType* d_out, int num_items);

};

