#include <string>
#include <vector>
#include "../CompressedDataMatrix.h"

// #define DEBUG_GPU_COX

typedef typename bsccs::FormatType FormatType;

enum FormatTypeCuda {
	DENSE, SPARSE, INDICATOR, INTERCEPT
};

enum PriorTypeCuda {
	NOPRIOR, LAPLACE, NORMAL
};

template<typename RealType>
struct Tuple2Plus
{
	typedef typename thrust::tuple<RealType, RealType> Tuple2;
	__host__ __device__
	Tuple2 operator()(const Tuple2& lhs, const Tuple2& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
				thrust::get<1>(lhs) + thrust::get<1>(rhs));
	}
};

template<typename RealType>
struct Tuple3Plus
{
	typedef typename thrust::tuple<RealType, RealType, RealType> Tuple3;
	__host__ __device__
	Tuple3 operator()(const Tuple3& lhs, const Tuple3& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
				thrust::get<1>(lhs) + thrust::get<1>(rhs),
				thrust::get<2>(lhs) + thrust::get<2>(rhs));
	}
};

template<typename RealType>
struct Tuple6Plus
{
	typedef typename thrust::tuple<RealType, RealType, RealType, RealType, RealType, RealType> Tuple6;
	__host__ __device__
	Tuple6 operator()(const Tuple6& lhs, const Tuple6& rhs)
	{
		return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
				thrust::get<1>(lhs) + thrust::get<1>(rhs),
				thrust::get<2>(lhs) + thrust::get<2>(rhs),
				thrust::get<3>(lhs) + thrust::get<3>(rhs),
				thrust::get<4>(lhs) + thrust::get<4>(rhs),
				thrust::get<5>(lhs) + thrust::get<5>(rhs));
	}
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
        RealType2 out;
        if (thrust::get<2>(accNAndD) == static_cast<RealType>(0)) {
            out.x = static_cast<RealType>(0);
            out.y = static_cast<RealType>(0);
        } else {
            auto temp = thrust::get<0>(accNAndD) / thrust::get<2>(accNAndD);
            out.x = nEvents * temp;
            if (isIndicator) {
                out.y = out.x * (1 - temp);
            } else {
                out.y = nEvents * (thrust::get<1>(accNAndD) / thrust::get<2>(accNAndD) - temp * temp);
            }
        }
        return out;
    }
};

template<typename RealType, typename RealType2, bool isIndicator>
struct CompGradHess2
{
    typedef typename thrust::tuple<RealType, RealType, RealType, RealType> InputTuple;

    __host__ __device__
    RealType2 operator()(const InputTuple& Inputs) const
    {
        RealType2 out;
        if (thrust::get<0>(Inputs) == static_cast<RealType>(0)) {
            out.x = static_cast<RealType>(0);
            out.y = static_cast<RealType>(0);
        } else {
            auto temp = thrust::get<1>(Inputs) / thrust::get<0>(Inputs);
            out.x = thrust::get<3>(Inputs) * temp;
            if (isIndicator) {
                out.y = out.x * (1 - temp);
            } else {
                out.y = thrust::get<3>(Inputs) * (thrust::get<2>(Inputs) / thrust::get<0>(Inputs) - temp * temp);
            }
        }
        return out;
    }
};

template<typename RealType>
struct TwoWayScan
{
    typedef typename thrust::tuple<RealType, RealType, RealType, RealType> InputTuple; // denom.begin(), denom.rbegin(), Y, YWeight
    typedef typename thrust::tuple<RealType, RealType> OutputTuple; // accDenom, decDenom

    __host__ __device__
    OutputTuple operator()(const InputTuple& Inputs) const {
        RealType d = thrust::get<0>(Inputs);
        RealType dr = 0;
        if (thrust::get<2>(Inputs) > static_cast<RealType>(1)) {
            dr = thrust::get<1>(Inputs) / thrust::get<3>(Inputs);
        }
        return thrust::make_tuple(d, dr);
    }
};

template<typename RealType>
struct ScansAddition
{
    typedef typename thrust::tuple<RealType, RealType> InputTuple; // accDenom, decDenom
    typedef typename thrust::tuple<RealType, RealType> ConditionTuple; // Y, YWeight

    __host__ __device__
    RealType operator()(const InputTuple& Inputs, const ConditionTuple& Conditions) const {
        RealType accDenom = thrust::get<0>(Inputs);
        if (thrust::get<0>(Conditions) == static_cast<RealType>(1)) {
            accDenom += thrust::get<1>(Inputs) * thrust::get<1>(Conditions);
        }
        return accDenom;
    }
};

template<typename RealType>
struct BackwardScans
{
    typedef typename thrust::tuple<RealType, RealType, RealType, RealType, RealType> InputTuple; // denom, numer, numer2, Y, YWeight
    typedef typename thrust::tuple<RealType, RealType, RealType> OutputTuple;

    __host__ __device__
    OutputTuple operator()(const InputTuple& Inputs) const {
        RealType d = 0;
        RealType n = 0;
        RealType n2 = 0;
        if (thrust::get<3>(Inputs) > static_cast<RealType>(1)) {
            d = thrust::get<0>(Inputs) / thrust::get<4>(Inputs);
            n = thrust::get<1>(Inputs) / thrust::get<4>(Inputs);
            n2 = thrust::get<2>(Inputs) / thrust::get<4>(Inputs);
        }
        return thrust::make_tuple(d, n, n2);
    }
};

template<typename RealType>
struct TwoWayScans
{
    typedef typename thrust::tuple<RealType, RealType, RealType, RealType,
    RealType, RealType, RealType, RealType> InputTuple; // denom, denom.r, numer, numer.r, numer2, numer2.r, Y, YWeight
    typedef typename thrust::tuple<RealType, RealType, RealType,
    RealType, RealType, RealType> OutputTuple; // denom, denom.r, numer, numer.r, numer2, numer2.r

    __host__ __device__
    OutputTuple operator()(const InputTuple& Inputs) const {
        RealType d = thrust::get<0>(Inputs);
        RealType n = thrust::get<2>(Inputs);
        RealType n2 = thrust::get<4>(Inputs);
        RealType dr = 0;
        RealType nr = 0;
        RealType n2r = 0;
        if (thrust::get<6>(Inputs) > static_cast<RealType>(1)) {
            dr = thrust::get<1>(Inputs) / thrust::get<7>(Inputs);
            nr = thrust::get<3>(Inputs) / thrust::get<7>(Inputs);
            n2r = thrust::get<5>(Inputs) / thrust::get<7>(Inputs);
        }
        return thrust::make_tuple(d, dr, n, nr, n2, n2r);
    }
};

template<typename RealType, typename RealType2, bool isIndicator>
struct TwoWayReduce
{
    typedef typename thrust::tuple<RealType, RealType, RealType, RealType, RealType, RealType, RealType, RealType, RealType> InputTuple; // AccDenom, AccNumer, AccNumer2, DecDenom, DecNumer, DecNumer2, NWeight, Y, YWeight

    __host__ __device__
    RealType2 operator()(const InputTuple& Inputs) const {

        RealType d = thrust::get<0>(Inputs);
        RealType n = thrust::get<1>(Inputs);
        RealType n2 = thrust::get<2>(Inputs);
        if (thrust::get<7>(Inputs) == static_cast<RealType>(1)) {
            d += thrust::get<3>(Inputs) * thrust::get<8>(Inputs);
            n += thrust::get<4>(Inputs) * thrust::get<8>(Inputs);
            n2 += thrust::get<5>(Inputs) * thrust::get<8>(Inputs);
        }

        RealType2 out;
        if (d == static_cast<RealType>(0)) {
            out.x = static_cast<RealType>(0);
            out.y = static_cast<RealType>(0);
        } else {
            auto temp = n / d;
            out.x = thrust::get<6>(Inputs) * temp;
            if (isIndicator) {
                out.y = out.x * (1 - temp);
            } else {
                out.y = thrust::get<6>(Inputs) * (n2 / d - temp * temp);
            }
        }

        return out;
    }
};


template <typename RealType, typename RealType2>
class CudaKernel {

	typedef typename thrust::device_vector<RealType>::iterator VecItr;
	typedef typename thrust::reverse_iterator<VecItr> RItr;

	typedef thrust::tuple<RealType, RealType> Tup2;
	typedef thrust::tuple<RealType, RealType, RealType> Tup3;
	typedef thrust::tuple<RealType, RealType, RealType, RealType> Tup4;
	typedef thrust::tuple<RealType, RealType, RealType, RealType, RealType, RealType> Tup6;

	typedef thrust::tuple<VecItr, VecItr, VecItr, VecItr> TupVec4;
	typedef thrust::zip_iterator<TupVec4> ZipVec4;
	typedef thrust::tuple<VecItr, RItr, RItr, RItr> NRTupVec4;
	typedef thrust::zip_iterator<NRTupVec4> NRZipVec4;
	typedef thrust::tuple<RItr, RItr, RItr, RItr, RItr> RTupVec5;
	typedef thrust::zip_iterator<RTupVec5> RZipVec5;
	typedef thrust::tuple<VecItr, RItr, VecItr, RItr, VecItr, RItr, RItr, RItr> NRTupVec8;
	typedef thrust::zip_iterator<NRTupVec8> NRZipVec8;
	typedef thrust::tuple<VecItr, VecItr, VecItr, VecItr, VecItr, VecItr, VecItr, VecItr, VecItr> TupVec9;
	typedef thrust::zip_iterator<TupVec9> ZipVec9;

public:

	std::string desiredDeviceName;
	cudaStream_t* stream;
	int CVFolds;
	int fold;
	int devIndex;

	RealType2 d_init;

	// Operator
	Tuple2Plus<RealType> tuple2Plus;
	Tuple3Plus<RealType> tuple3Plus;
	Tuple6Plus<RealType> tuple6Plus;

	CompGradHess<RealType, RealType2, true> compGradHessInd;
	CompGradHess<RealType, RealType2, false> compGradHessNInd;
	CompGradHess1<RealType, RealType2, true> compGradHessInd1;
	CompGradHess1<RealType, RealType2, false> compGradHessNInd1;
	CompGradHess2<RealType, RealType2, true> compGradHessInd2;
	CompGradHess2<RealType, RealType2, false> compGradHessNInd2;

	TwoWayScan<RealType> twoWayScan;
	ScansAddition<RealType> scansAddition;
	BackwardScans<RealType> backwardScans;
	TwoWayScans<RealType> twoWayScans;
	TwoWayReduce<RealType, RealType2, true> fineGrayInd;
	TwoWayReduce<RealType, RealType2, false> fineGrayNInd;

	// Temporary storage required by cub::scan and cub::reduce
	void *d_temp_storage_accd = NULL;
	size_t temp_storage_bytes_accd = 0;
	void *d_temp_storage_faccd = NULL;
	size_t temp_storage_bytes_faccd = 0;
	void *d_temp_storage_accn = NULL;
	size_t temp_storage_bytes_accn = 0;
	void *d_temp_storage_gh = NULL;
	size_t temp_storage_bytes_gh = 0;
	void *d_temp_storage_fs = NULL;
	size_t temp_storage_bytes_fs = 0;
	void *d_temp_storage_bs = NULL;
	size_t temp_storage_bytes_bs = 0;
	void *d_temp_storage_fgh = NULL;
	size_t temp_storage_bytes_fgh = 0;

	thrust::device_vector<RealType> d_AccNumer;
	thrust::device_vector<RealType> d_AccNumer2;
/*
	RealType* betaIn = NULL;
	RealType* betaOut = NULL;
	RealType* boundIn = NULL;
	RealType* boundOut = NULL;
	RealType* temp = NULL;
*/
	CudaKernel(const std::string& deviceName);
	~CudaKernel();

	cudaStream_t* getStream();

	const std::string getDeviceName();

	void allocStreams(int streamCVFolds);

	void setFold(int currentFold);

	void resizeAndCopyToDeviceInt(const std::vector<int>& hostVec, thrust::device_vector<int>& deviceVec);

	void resizeAndCopyToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec);

	void resizeAndFillToDevice(thrust::device_vector<RealType>& deviceVec, RealType val, int num_items);

	void copyFromHostToDevice(const std::vector<RealType>& hostVec, thrust::device_vector<RealType>& deviceVec);

	void copyFromDeviceToHost(const thrust::device_vector<RealType>& deviceVec, std::vector<RealType>& hostVec);

	void copyFromDeviceToDevice(const thrust::device_vector<RealType>& source, thrust::device_vector<RealType>& destination);
/*
	void setBounds(thrust::device_vector<RealType>& d_Bound, thrust::device_vector<RealType>& d_BoundBuffer, RealType val, size_t& J);

	void resetBeta(thrust::device_vector<RealType>& dBeta, thrust::device_vector<RealType>& dBetaBuffer, size_t& J);

	void getBeta(std::vector<RealType>& hBeta);

	void getBound();
*/
	void allocTempStorage(thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			size_t& N);

	void allocTempStorageByKey(thrust::device_vector<int>& d_Key,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			size_t& N);

	void allocTempStorageFG(thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_AccNumer,
			thrust::device_vector<RealType>& d_AccNumer2,
			thrust::device_vector<RealType>& d_DecDenom,
			thrust::device_vector<RealType>& d_DecNumer,
			thrust::device_vector<RealType>& d_DecNumer2,
			thrust::device_vector<RealType>& d_NWeight,
			thrust::device_vector<RealType>& d_YWeight,
			thrust::device_vector<RealType>& d_Y,
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

	void computeGradientAndHessianByKey(thrust::device_vector<int>& d_Key,
			thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_NWeight,
			RealType2* d_GH,
			FormatType& formatType,
			size_t& offCV,
			size_t& N
			);

	void computeTwoWayGradientAndHessian(thrust::device_vector<RealType>& d_Numerator,
			thrust::device_vector<RealType>& d_Numerator2,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccNumer,
			thrust::device_vector<RealType>& d_AccNumer2,
			thrust::device_vector<RealType>& d_AccDenom,
			thrust::device_vector<RealType>& d_DecNumer,
			thrust::device_vector<RealType>& d_DecNumer2,
			thrust::device_vector<RealType>& d_DecDenom,
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

	void computeAccumlatedDenominatorByKey(thrust::device_vector<int>& d_Key,
			thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccDenom,
			int N);

	void computeTwoWayAccumlatedDenominator(thrust::device_vector<RealType>& d_Denominator,
			thrust::device_vector<RealType>& d_AccDenom, 
			thrust::device_vector<RealType>& d_DecDenom, 
			thrust::device_vector<RealType>& d_YWeight, 
			thrust::device_vector<RealType>& d_Y,
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

