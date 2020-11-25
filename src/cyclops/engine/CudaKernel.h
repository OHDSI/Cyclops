#include <vector>
#include "../CompressedDataMatrix.h"

typedef typename bsccs::FormatType FormatType;

enum FormatTypeCuda {
	DENSE, SPARSE, INDICATOR, INTERCEPT
};

enum PriorTypeCuda {
	NOPRIOR, LAPLACE, NORMAL
};


template<typename RealType, typename RealType2, bool isIndicator>
struct CompGradHess
{
    typedef typename thrust::tuple<RealType, RealType> InputTuple;

    __host__ __device__
    RealType2 operator()(const InputTuple& accNAndN2, const InputTuple& accDAndW) const
    {
        auto temp = thrust::get<0>(accNAndN2) / thrust::get<0>(accDAndW);
        RealType2 out;
        out.x = thrust::get<1>(accDAndW) * temp;
        if (isIndicator) {
            out.y = out.x * (1 - temp);
        } else {
            out.y = thrust::get<1>(accDAndW) * (thrust::get<1>(accNAndN2) / thrust::get<0>(accDAndW) - temp * temp);
        }
        return out;
    }
};

template<typename RealType, typename RealType2, bool isIndicator>
struct CompGradHess1
{
    typedef typename thrust::tuple<RealType, RealType, RealType> InputTuple;

    __host__ __device__
    RealType2 operator()(const InputTuple& accNAndD, const RealType nEvents) const
    {
        auto temp = thrust::get<0>(accNAndD) / thrust::get<2>(accNAndD);
        RealType2 out;
        out.x = nEvents * temp;
        if (isIndicator) {
            out.y = out.x * (1 - temp);
        } else {
            out.y = nEvents * (thrust::get<1>(accNAndD) / thrust::get<2>(accNAndD) - temp * temp);
        }
        return out;
    }
};


template <typename RealType, typename RealType2>
class CudaKernel {

public:

	cudaStream_t* stream;
	int CVFolds;
	int fold;

	// Operator
	CompGradHess<RealType, RealType2, true> compGradHessInd;
	CompGradHess<RealType, RealType2, false> compGradHessNInd;
	CompGradHess1<RealType, RealType2, true> compGradHessInd1;
	CompGradHess1<RealType, RealType2, false> compGradHessNInd1;

	// Temporary storage required by cub::scan and cub::reduce
	void *d_temp_storage_accd = NULL;
	size_t temp_storage_bytes_accd = 0;
	void *d_temp_storage_gh = NULL;
	size_t temp_storage_bytes_gh = 0;

	CudaKernel();
	~CudaKernel();

	void allocStreams(int streamCVFolds);

	void setFold(int currentFold);

	void resizeAndCopyToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec);

	void resizeAndFillToDevice(thrust::device_vector<RealType>& deviceVec, RealType val, int num_items);

	void copyFromHostToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec);

	void copyFromDeviceToHost(const thrust::device_vector<RealType>& deviceVec, std::vector<RealType>& hostVec);

	void copyFromDeviceToDevice(const thrust::device_vector<RealType>& source, thrust::device_vector<RealType>& destination);

	void allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			size_t& N);
	
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
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			FormatType& formatType,
			size_t& offCV,
			size_t& N
			);
        
	void computeGradientAndHessian1(thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			FormatType& formatType,
			size_t& offCV,
			size_t& N
			);

	void computeGradientAndHessianBackwards(thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_NWeight,
			thrust::device_vector<RealType>& d_YWeight,
			thrust::device_vector<RealType>& d_Y,
			RealType2* d_GH,
			FormatType& formatType,
			size_t& offCV,
			size_t& N);

	void updateXBetaAndDelta(const thrust::device_vector<RealType>& d_X,
			const thrust::device_vector<int>& d_K,
			unsigned int offX,
			unsigned int offK,
			const unsigned int taskCount,
			RealType2* d_GH,
			thrust::device_vector<RealType>& d_XjY,
			thrust::device_vector<RealType>& d_Bound,
			thrust::device_vector<RealType>& d_BoundBuffer,
			thrust::device_vector<RealType>& d_KWeight,
			thrust::device_vector<RealType>& d_Beta,
			thrust::device_vector<RealType>& d_BetaBuffer,
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
	
	void computeAccumlatedDenominator(thrust::device_vector<RealType>& d_Denominator, 
			thrust::device_vector<RealType>& d_AccDenom, 
			int N);


	// not using
	void processDelta(double2* d_GH,
                        thrust::device_vector<RealType>& d_XjY,
                        thrust::device_vector<RealType>& d_Delta,
                        thrust::device_vector<RealType>& d_Beta,
                        thrust::device_vector<RealType>& d_BetaBuffer,
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
                        RealType d_Delta,
                        thrust::device_vector<RealType>& d_KWeight,
			thrust::device_vector<RealType>& d_Beta,
			thrust::device_vector<RealType>& d_BetaBuffer,
                        thrust::device_vector<RealType>& d_XBeta,
                        thrust::device_vector<RealType>& d_ExpXBeta,
                        thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccDenom,
                        thrust::device_vector<RealType>& d_Numerator,
                        thrust::device_vector<RealType>& d_Numerator2,
                        int index, size_t& N,
                        FormatType& formatType,
                        int gridSize, int blockSize);

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

