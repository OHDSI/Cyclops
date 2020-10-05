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

template<typename RealType, bool isIndicator>
struct functorCGH
{
    typedef typename thrust::tuple<RealType, RealType> InputTuple;

    __host__ __device__
    double2 operator()(const InputTuple& accNAndN2, const InputTuple& accDAndW) const
    {
        auto temp = thrust::get<0>(accNAndN2) / thrust::get<0>(accDAndW);
        double2 out;
        out.x = thrust::get<1>(accDAndW) * temp;
        if (isIndicator) {
            out.y = out.x * (1 - temp);
        } else {
            out.y = thrust::get<1>(accDAndW) * (thrust::get<1>(accNAndN2) / thrust::get<0>(accDAndW) - temp * temp);
        }
        return out;
    }
};

template<typename RealType, bool isIndicator>
struct functorCGH1
{
    typedef typename thrust::tuple<RealType, RealType, RealType> InputTuple;

    __host__ __device__
    double2 operator()(const InputTuple& accNAndD, const RealType nEvents) const
    {
        auto temp = thrust::get<0>(accNAndD) / thrust::get<2>(accNAndD);
        double2 out;
        out.x = nEvents * temp;
        if (isIndicator) {
            out.y = out.x * (1 - temp);
        } else {
            out.y = nEvents * (thrust::get<1>(accNAndD) / thrust::get<2>(accNAndD) - temp * temp);
        }
        return out;
    }
};

template<typename RealType>
struct functorThird
{
    typedef typename thrust::tuple<RealType, RealType, RealType> InputTuple;

    __host__ __device__
    RealType operator()(InputTuple& accNAndD) const
    {
        return thrust::get<2>(accNAndD);
    }
};

/*
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
*/
template <typename RealType>
class CudaKernel {

	typedef thrust::tuple<RealType,RealType> Tup2;
	typedef typename thrust::device_vector<RealType>::iterator VecItr;
	typedef thrust::tuple<VecItr,VecItr,VecItr,VecItr> TupVec4;
	typedef thrust::zip_iterator<TupVec4> ZipVec4;

public:
	
	// Device arrays
//	Tup2 init = thrust::make_tuple<RealType, RealType>(0, 0);
	double2 d_init;

	// Operator
//	CustomExp    exp_op;
	functorCGH<RealType, true> compGradHessInd;
	functorCGH<RealType, false> compGradHessNInd;
	functorCGH1<RealType, true> compGradHessInd1;
	functorCGH1<RealType, false> compGradHessNInd1;
	functorThird<RealType> scanOutput;

	// Temporary storage required by cub::scan and cub::reduce
	void *d_temp_storage0 = NULL;
	size_t temp_storage_bytes0 = 0;
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
			double2* d_GH,
			double2* d_BlockGH,
			size_t& N,
			thrust::device_vector<int>& indicesN);
	
	void computeNumeratorForGradient(const thrust::device_vector<RealType>& d_X,
			const thrust::device_vector<int>& d_K,
			unsigned int offX,
			unsigned int offK,
			const unsigned int taskCount,
			thrust::device_vector<RealType>& d_KWeight,
			thrust::device_vector<RealType>& d_ExpXBeta,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			FormatType& formatType,
			int gridSize, int blockSize);
        
	void computeGradientAndHessian(thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_AccNumer,
			thrust::device_vector<RealType>& d_AccNumer2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			double2* d_GH,
			double2* d_BlockGH,
			FormatType& formatType,
			size_t& offCV,
			size_t& N
//			,const std::vector<int>& K,
//			unsigned int offK,
//			thrust::device_vector<int>& indicesN
			);
        
	void computeGradientAndHessian1(thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccNumer,
			thrust::device_vector<RealType>& d_AccNumer2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			double2* d_GH,
			double2* d_BlockGH,
			FormatType& formatType,
			size_t& N
			);

	void updateXBetaAndDelta(const thrust::device_vector<RealType>& d_X,
			const thrust::device_vector<int>& d_K,
			unsigned int offX,
			unsigned int offK,
			const unsigned int taskCount,
			double2* d_GH,
			thrust::device_vector<RealType>& d_XjY,
			thrust::device_vector<RealType>& d_Bound,
			thrust::device_vector<RealType>& d_KWeight,
			thrust::device_vector<RealType>& d_Beta,
			thrust::device_vector<RealType>& d_XBeta,
			thrust::device_vector<RealType>& d_ExpXBeta,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& dPriorParams,
			const int priorTypes,
			int index,
			FormatType& formatType,
			int gridSize, int blockSize);
	
	void processDelta(double2* d_GH,
			thrust::device_vector<RealType>& d_XjY,
			thrust::device_vector<RealType>& d_Delta,
			thrust::device_vector<RealType>& d_Beta,
			thrust::device_vector<RealType>& d_Bound,
			thrust::device_vector<RealType>& d_PriorParams,
			const int priorType,
			int index,
			int gridSize, int blockSize);

	void updateXBeta(const thrust::device_vector<RealType>& d_X,
			const thrust::device_vector<int>& d_K,
			unsigned int offX,
			unsigned int offK,
			const unsigned int taskCount,
			thrust::device_vector<RealType>& d_Delta,
			thrust::device_vector<RealType>& d_KWeight,
			thrust::device_vector<RealType>& d_XBeta,
			thrust::device_vector<RealType>& d_ExpXBeta,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			int index,
			FormatType& formatType,
			int gridSize, int blockSize);	
	
	void CubScan(RealType* d_in, RealType* d_out, int num_items);

	// not using
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

};

