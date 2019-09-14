/*
 * GpuModelSpecifics.hpp
 *
 *  Created on: Apr 4, 2016
 *      Author: msuchard
 */

#ifndef GPUMODELSPECIFICS_HPP_
#define GPUMODELSPECIFICS_HPP_


// #define USE_VECTOR
#undef USE_VECTOR

// #define GPU_DEBUG
#undef GPU_DEBUG
//#define USE_LOG_SUM
#define TIME_DEBUG

#include <Rcpp.h>

#include "ModelSpecifics.h"
#include "Iterators.h"

#include <boost/compute/algorithm/reduce.hpp>

namespace bsccs {

namespace compute = boost::compute;

namespace detail {

namespace constant {
    static const int updateXBetaBlockSize = 256; // 512; // Appears best on K40
    static const int updateAllXBetaBlockSize = 32;
    int exactCLRBlockSize = 32;
    int exactCLRSyncBlockSize = 32;
    static const int maxBlockSize = 256;
}; // namespace constant

template <typename DeviceVec, typename HostVec>
DeviceVec allocateAndCopyToDevice(const HostVec& hostVec, const compute::context& context, compute::command_queue& queue) {
    DeviceVec deviceVec(hostVec.size(), context);
    compute::copy(std::begin(hostVec), std::end(hostVec), std::begin(deviceVec), queue);
    return std::move(deviceVec);
}

template <typename DeviceVec, typename HostVec>
void resizeAndCopyToDevice(const HostVec& hostVec, DeviceVec& deviceVec, compute::command_queue& queue) {
    deviceVec.resize(hostVec.size());
    compute::copy(std::begin(hostVec), std::end(hostVec), std::begin(deviceVec), queue);
}

template <typename HostVec, typename DeviceVec>
void compare(const HostVec& host, const DeviceVec& device, const std::string& error, double tolerance = 1E-10) {
    bool valid = true;

    for (size_t i = 0; i < host.size(); ++i) {
        auto h = host[i];
        auto d = device[i];
        if (std::abs(h - d) > tolerance) {
            std::cerr << "@ " << i << " : " << h << " - " << d << " = " << (h - d) << std::endl;
            valid = false;
        }
    }
    if (!valid) {
        //forward_exception_to_r(error);
        Rcpp::stop(error);
        // throw new std::logic_error(error);
    }
}

template <int D, class T>
int getAlignedLength(T n) {
    return (n / D) * D + (n % D == 0 ? 0 : D);
}

}; // namespace detail

struct SourceCode {
    std::string body;
    std::string name;

    SourceCode(std::string body, std::string name) : body(body), name(name) { }
};

template <typename RealType>
class AllGpuColumns {
public:
    typedef compute::vector<RealType> DataVector;
    typedef compute::vector<int> IndicesVector;
    typedef compute::uint_ UInt;
    typedef compute::vector<UInt> dStartsVector;
    typedef std::vector<UInt> hStartsVector;

    AllGpuColumns(const compute::context& context) : indices(context), data(context), // {
    		ddataStarts(context), dindicesStarts(context), dtaskCounts(context) {
        // Do nothing
    }

    virtual ~AllGpuColumns() { }

    void initialize(const CompressedDataMatrix<RealType>& mat,
                    compute::command_queue& queue,
                    size_t K, bool pad) {
        std::vector<RealType> flatData;
        std::vector<int> flatIndices;

        std::cerr << "AGC start" << std::endl;

        UInt dataStart = 0;
        UInt indicesStart = 0;

        for (int j = 0; j < mat.getNumberOfColumns(); ++j) {
            const auto& column = mat.getColumn(j);
            const auto format = column.getFormatType();

            dataStarts.push_back(dataStart);
            indicesStarts.push_back(indicesStart);
            formats.push_back(format);

            // Data vector
            if (format == FormatType::SPARSE ||
                format == FormatType::DENSE) {
                appendAndPad(column.getDataVector(), flatData, dataStart, pad);
            }

            // Indices vector
            if (format == FormatType::INDICATOR ||
                format == FormatType::SPARSE) {
                appendAndPad(column.getColumnsVector(), flatIndices, indicesStart, pad);
            }

            // Task count
            if (format == FormatType::DENSE ||
                format == FormatType::INTERCEPT) {
                taskCounts.push_back(K);
            } else { // INDICATOR, SPARSE
                taskCounts.push_back(column.getNumberOfEntries());
            }
        }

        detail::resizeAndCopyToDevice(flatData, data, queue);
        detail::resizeAndCopyToDevice(flatIndices, indices, queue);
        detail::resizeAndCopyToDevice(dataStarts, ddataStarts, queue);
        detail::resizeAndCopyToDevice(indicesStarts, dindicesStarts, queue);
        detail::resizeAndCopyToDevice(taskCounts, dtaskCounts, queue);


    	std::cerr << "AGC end " << flatData.size() << " " << flatIndices.size() << std::endl;
    }

    UInt getDataOffset(int column) const {
        return dataStarts[column];
    }

    UInt getIndicesOffset(int column) const {
        return indicesStarts[column];
    }

    UInt getTaskCount(int column) const {
    	return taskCounts[column];
    }

    const DataVector& getData() const {
        return data;
    }

    const IndicesVector& getIndices() const {
        return indices;
    }

    const dStartsVector& getDataStarts() const {
    	return ddataStarts;
    }

    const dStartsVector& getIndicesStarts() const {
    	return dindicesStarts;
    }

    const dStartsVector& getTaskCounts() const {
    	return dtaskCounts;
    }

    const std::vector<FormatType> getFormat() const{
    	return formats;
    }


private:

    template <class T>
    void appendAndPad(const T& source, T& destination, UInt& length, bool pad) {
        for (auto x : source) {
            destination.push_back(x);
        }
        if (pad) {
            auto i = source.size();
            const auto end = detail::getAlignedLength<16>(i);
            for (; i < end; ++i) {
                destination.push_back(typename T::value_type());
            }
            length += end;
        } else {
            length += source.size();
        }
    }


    IndicesVector indices;
    DataVector data;
	hStartsVector taskCounts;
	hStartsVector dataStarts;
	hStartsVector indicesStarts;

	dStartsVector dtaskCounts;
	dStartsVector ddataStarts;
	dStartsVector dindicesStarts;

	//std::vector<UInt> taskCounts;
    //std::vector<UInt> dataStarts;
    //std::vector<UInt> indicesStarts;
    std::vector<FormatType> formats;
};

template <typename RealType>
class GpuColumn {
public:
    typedef compute::vector<RealType> DataVector;
    typedef compute::vector<int> IndicesVector;
    typedef compute::uint_ UInt;

    //GpuColumn(const GpuColumn<RealType>& copy);

    GpuColumn(const CompressedDataColumn<RealType>& column,
              const compute::context& context,
              compute::command_queue& queue,
              size_t denseLength)
        : format(column.getFormatType()), indices(context), data(context) {

            // Data vector
            if (format == FormatType::SPARSE ||
                format == FormatType::DENSE) {
                const auto& columnData = column.getDataVector();
                detail::resizeAndCopyToDevice(columnData, data, queue);
            }

            // Indices vector
            if (format == FormatType::INDICATOR ||
                format == FormatType::SPARSE) {
                const auto& columnIndices = column.getColumnsVector();
                detail::resizeAndCopyToDevice(columnIndices, indices, queue);
            }

            // Task count
            if (format == FormatType::DENSE ||
                format == FormatType::INTERCEPT) {
                tasks = static_cast<UInt>(denseLength);
            } else { // INDICATOR, SPARSE
                tasks = static_cast<UInt>(column.getNumberOfEntries());
            }
        }

    virtual ~GpuColumn() { }

    const IndicesVector& getIndicesVector() const { return indices; }
    const DataVector& getDataVector() const { return data; }
    UInt getTaskCount() const { return tasks; }

private:
    FormatType format;
    IndicesVector indices;
    DataVector data;
    UInt tasks;
};

template <class BaseModel, typename RealType, class BaseModelG>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, RealType>, BaseModelG {
public:

    using ModelSpecifics<BaseModel, RealType>::modelData;
    using ModelSpecifics<BaseModel, RealType>::hX;
    using ModelSpecifics<BaseModel, RealType>::hNtoK;
    using ModelSpecifics<BaseModel, RealType>::hPid;
    using ModelSpecifics<BaseModel, RealType>::hPidInternal;
    using ModelSpecifics<BaseModel, RealType>::accReset;
    using ModelSpecifics<BaseModel, RealType>::hXjY;
    using ModelSpecifics<BaseModel, RealType>::hXjX;
    using ModelSpecifics<BaseModel, RealType>::sparseIndices;
    using ModelSpecifics<BaseModel, RealType>::K;
    using ModelSpecifics<BaseModel, RealType>::J;
    using ModelSpecifics<BaseModel, RealType>::N;

#ifdef CYCLOPS_DEBUG_TIMING
    using ModelSpecifics<BaseModel, RealType>::duration;
#endif

    using ModelSpecifics<BaseModel, RealType>::norm;
    using ModelSpecifics<BaseModel, RealType>::boundType;
    using ModelSpecifics<BaseModel, RealType>::hXt;
    using ModelSpecifics<BaseModel, RealType>::logLikelihoodFixedTerm;

	using ModelSpecifics<BaseModel, RealType>::offsExpXBeta;
	using ModelSpecifics<BaseModel, RealType>::hXBeta;
	using ModelSpecifics<BaseModel, RealType>::hY;
	using ModelSpecifics<BaseModel, RealType>::hOffs;
	using ModelSpecifics<BaseModel, RealType>::denomPid;
	using ModelSpecifics<BaseModel, RealType>::denomPid2;
	using ModelSpecifics<BaseModel, RealType>::numerPid;
	using ModelSpecifics<BaseModel, RealType>::numerPid2;
	using ModelSpecifics<BaseModel, RealType>::hNWeight;
	using ModelSpecifics<BaseModel, RealType>::hKWeight;

    using ModelSpecifics<BaseModel, RealType>::syncCV;
    using ModelSpecifics<BaseModel, RealType>::syncCVFolds;

    using ModelSpecifics<BaseModel, RealType>::accDenomPid;
    using ModelSpecifics<BaseModel, RealType>::accNumerPid;
    using ModelSpecifics<BaseModel, RealType>::accNumerPid2;

    //using ModelSpecifics<BaseModel, WeightType>::hBeta;
    //using ModelSpecifics<BaseModel, WeightType>::algorithmType;

    /*
    using ModelSpecifics<BaseModel, WeightType>::hNWeightPool;
    using ModelSpecifics<BaseModel, WeightType>::hKWeightPool;
    using ModelSpecifics<BaseModel, WeightType>::accDenomPidPool;
    using ModelSpecifics<BaseModel, WeightType>::accNumerPid2Pool;
    using ModelSpecifics<BaseModel, WeightType>::accResetPool;
    using ModelSpecifics<BaseModel, WeightType>::hPidPool;
    using ModelSpecifics<BaseModel, WeightType>::hPidInternalPool;
    using ModelSpecifics<BaseModel, WeightType>::hXBetaPool;
    using ModelSpecifics<BaseModel, WeightType>::offsExpXBetaPool;
    using ModelSpecifics<BaseModel, WeightType>::denomPidPool;
    using ModelSpecifics<BaseModel, WeightType>::numerPidPool;
    using ModelSpecifics<BaseModel, WeightType>::numerPid2Pool;
    using ModelSpecifics<BaseModel, WeightType>::hXjYPool;
    using ModelSpecifics<BaseModel, WeightType>::hXjXPool;
    using ModelSpecifics<BaseModel, WeightType>::logLikelihoodFixedTermPool;
    using ModelSpecifics<BaseModel, WeightType>::normPool;
    using ModelSpecifics<BaseModel, WeightType>::useLogSum;
    */

    int tpb = 256; // threads-per-block  // Appears best on K40
    int maxWgs = 16;
    int tpb0 = 16;
    int tpb1 = 16;

	bool double_precision = false;

    GpuModelSpecifics(const ModelData<RealType>& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,RealType>(input),
      device(compute::system::find_device(deviceName)),
      ctx(device),
      queue(ctx, device
          , compute::command_queue::enable_profiling
      ),
      dColumns(ctx), dColumnsXt(ctx),
      dY(ctx), dBeta(ctx), dXBeta(ctx), dExpXBeta(ctx), dDenominator(ctx),
	  dDenominator2(ctx), dAccDenominator(ctx), dNorm(ctx), dOffs(ctx),
	  dBound(ctx), dXjY(ctx), dXjX(ctx), dNtoK(ctx),
	  //  dFixBeta(ctx), dAllDelta(ctx), dRealVector1(ctx),
	  dBuffer(ctx), dBuffer1(ctx), dKWeight(ctx), dNWeight(ctx),
      dId(ctx), dIntVector1(ctx), dIntVector2(ctx),
	  dXBetaVector(ctx), dOffsExpXBetaVector(ctx), dDenomPidVector(ctx),
	  dDenomPid2Vector(ctx), dNWeightVector(ctx), dKWeightVector(ctx), dPidVector(ctx),
	  dAccDenomPidVector(ctx), dAccNumerPidVector(ctx), dAccNumerPid2Vector(ctx),
	  dAccResetVector(ctx), dPidInternalVector(ctx), dNumerPidVector(ctx),
	  dNumerPid2Vector(ctx), dNormVector(ctx), dXjXVector(ctx), dXjYVector(ctx),
	  dDeltaVector(ctx), dBoundVector(ctx), dPriorParams(ctx), dBetaVector(ctx),
	  dAllZero(ctx), dDoneVector(ctx), dIndexListWithPrior(ctx), dCVIndices(ctx),
	  dSMStarts(ctx), dSMScales(ctx), dSMIndices(ctx), dLogX(ctx), dKStrata(ctx){

        std::cerr << "ctor GpuModelSpecifics" << std::endl;

        // Get device ready to compute
        std::cerr << "Using: " << device.name() << std::endl;
    }

    virtual ~GpuModelSpecifics() {
        std::cerr << "dtor GpuModelSpecifics" << std::endl;
    }

    virtual void deviceInitialization() {
    	RealType blah = 0;
    	if (sizeof(blah)==8) {
    		double_precision = true;
    	}

#ifdef TIME_DEBUG
        std::cerr << "start dI" << std::endl;
#endif
        //isNvidia = compute::detail::is_nvidia_device(queue.get_device());

        std::cout << "maxWgs: " << maxWgs << "\n";

        int need = 0;

        // Copy data
        dColumns.initialize(hX, queue, K, true);
        //this->initializeMmXt();
        //dColumnsXt.initialize(*hXt, queue, K, true);
        formatList.resize(J);

        for (size_t j = 0; j < J /*modelData.getNumberOfColumns()*/; ++j) {

#ifdef TIME_DEBUG
          //  std::cerr << "dI " << j << std::endl;
#endif
        	FormatType format = hX.getFormatType(j);
            //const auto& column = modelData.getColumn(j);
            // columns.emplace_back(GpuColumn<real>(column, ctx, queue, K));
            need |= (1 << format);

            indicesFormats[format].push_back(j);
            formatList[j] = format;
        }
            // Rcpp::stop("done");

        std::vector<FormatType> neededFormatTypes;
        for (int t = 0; t < 4; ++t) {
            if (need & (1 << t)) {
                neededFormatTypes.push_back(static_cast<FormatType>(t));
            }
        }

        auto& inputY = modelData.getYVectorRef();
        detail::resizeAndCopyToDevice(inputY, dY, queue);

        // Internal buffers
        //detail::resizeAndCopyToDevice(hBeta, dBeta, queue);
        detail::resizeAndCopyToDevice(hXBeta, dXBeta, queue);  hXBetaKnown = true; dXBetaKnown = true;
        detail::resizeAndCopyToDevice(offsExpXBeta, dExpXBeta, queue);
        detail::resizeAndCopyToDevice(denomPid, dDenominator, queue);
        detail::resizeAndCopyToDevice(denomPid2, dDenominator2, queue);
        detail::resizeAndCopyToDevice(accDenomPid, dAccDenominator, queue);
        std::vector<int> myHPid;
        for (int i=0; i<K; i++) {
        	myHPid.push_back(hPid[i]);
        }
        detail::resizeAndCopyToDevice(myHPid, dId, queue);
        detail::resizeAndCopyToDevice(hOffs, dOffs, queue);
        detail::resizeAndCopyToDevice(hKWeight, dKWeight, queue);
        detail::resizeAndCopyToDevice(hNWeight, dNWeight, queue);

        std::cerr << "Format types required: " << need << std::endl;

		// shadily sets hNWeight to determine right block size TODO do something about this
    	if (BaseModel::exactCLR) {
    		if (hNWeight.size() < N + 1) { // Add +1 for extra (zero-weight stratum)
    			hNWeight.resize(N + 1);
    		}

    		std::fill(hNWeight.begin(), hNWeight.end(), static_cast<RealType>(0));
    		for (size_t k = 0; k < K; ++k) {
    			hNWeight[hPid[k]] += hY[k];
    		}

    		int clrSize = 32;

    		RealType maxCases = 0;
    		for (int i=0; i<N; i++) {
    			if (hNWeight[i] > maxCases) {
    				maxCases = hNWeight[i];
    			}
    		}

    		while (maxCases >= clrSize) {
    			clrSize = clrSize * 2;
    		}
    		if (clrSize > detail::constant::maxBlockSize) {
    			clrSize = detail::constant::maxBlockSize;
    		}

    		detail::constant::exactCLRBlockSize = clrSize;

    		std::cout << "exactCLRBlockSize: " << detail::constant::exactCLRBlockSize << "\n";
    	}

        buildAllKernels(neededFormatTypes);
        std::cout << "built all kernels \n";

        //printAllKernels(std::cerr);
    }

    virtual void resetBeta() {
    	/*
    	if (syncCV) {
    		std::vector<real> temp;
    		//temp.resize(J*syncCVFolds, 0.0);
    		int size = layoutByPerson ? cvIndexStride : syncCVFolds;
    		temp.resize(J*size, 0.0);
    		detail::resizeAndCopyToDevice(temp, dBetaVector, queue);
    	} else {
    	*/
    		std::vector<RealType> temp;
    		temp.resize(J, 0.0);
    		detail::resizeAndCopyToDevice(temp, dBeta, queue);
    	//}
    }

    void computeFixedTermsInGradientAndHessian(bool useCrossValidation) {
#ifdef CYCLOPS_DEBUG_TIMING
			auto start = bsccs::chrono::steady_clock::now();
#endif

    	if (ModelSpecifics<BaseModel,RealType>::sortPid()) {
    		ModelSpecifics<BaseModel,RealType>::doSortPid(useCrossValidation);
    	}
    	if (ModelSpecifics<BaseModel,RealType>::allocateXjY()) {
    		computeXjY(useCrossValidation);
    	}
    	if (ModelSpecifics<BaseModel,RealType>::allocateXjX()) {
    	    ModelSpecifics<BaseModel,RealType>::computeXjX(useCrossValidation);
    		detail::resizeAndCopyToDevice(hXjX, dXjX, queue);
    	}
    	if (ModelSpecifics<BaseModel,RealType>::allocateNtoKIndices()) {
    		ModelSpecifics<BaseModel,RealType>::computeNtoKIndices(useCrossValidation);
    		detail::resizeAndCopyToDevice(hNtoK, dNtoK, queue);
    	}

#ifdef CYCLOPS_DEBUG_TIMING
			auto end = bsccs::chrono::steady_clock::now();
			///////////////////////////"
			auto name = "computeFixedTermsInGradientAndHessian";
			duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

    }

    virtual void computeRemainingStatistics(bool useWeights) {
        //std::cerr << "GPU::cRS called" << std::endl;
    	if (syncCV) {
    		//computeRemainingStatistics();
    	} else {
    		hBuffer.resize(K);

#ifdef CYCLOPS_DEBUG_TIMING
    		auto start = bsccs::chrono::steady_clock::now();
#endif

    		// get kernel
    		auto& kernel = kernelComputeRemainingStatistics;

    		// set kernel args
    		const auto taskCount = K;
    		int dK = K;
    		kernel.set_arg(0, dK);
    		kernel.set_arg(1, dXBeta);
    		kernel.set_arg(2, dExpXBeta);
    		kernel.set_arg(3, dDenominator);
    		kernel.set_arg(4, dY);
    		kernel.set_arg(5, dOffs);
    		kernel.set_arg(6, dId);

    		// set work size, no looping
    		size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
    		if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
    			++workGroups;
    		}
    		auto localWorkSize = detail::constant::updateXBetaBlockSize;
    		auto globalWorkSize = workGroups * localWorkSize;

    		if (BaseModelG::useNWeights) {
    			kernel.set_arg(7, dNtoK);
    			kernel.set_arg(8, dNWeight);
    			localWorkSize = tpb;
    			globalWorkSize = N*localWorkSize;
    			kernel.set_arg(9, dDenominator2);
    		}
    		/*
    				std::vector<int> myNtoK;
    				myNtoK.resize(dNtoK.size());
    				compute::copy(std::begin(dNtoK), std::end(dNtoK), std::begin(myNtoK), queue);
    				std::cout << "NtoK: ";
    				for (auto x:myNtoK) {
    					std::cout << x << " ";
    				}
    				std::cout << "\n";

    				std::vector<real> myNWeight;
    				myNWeight.resize(dNWeight.size());
    				compute::copy(std::begin(dNWeight), std::end(dNWeight), std::begin(myNWeight), queue);
    				std::cout << "NWeight: ";
    				for (auto x:myNWeight) {
    					std::cout << x << " ";
    				}
    				std::cout << "\n";
    		 */

    		// run kernel
    		queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    		queue.finish();

//    		std::vector<RealType> hDenominator;
//    		hDenominator.resize(dDenominator.size());
//    		compute::copy(std::begin(dDenominator), std::end(dDenominator), std::begin(hDenominator), queue);
//    		std::cout << "denom: ";
//    		for (auto x:hDenominator) {
//    		    std::cout << x << " ";
//    		}
//    		std::cout << "\n";


#ifdef CYCLOPS_DEBUG_TIMING
    		auto end = bsccs::chrono::steady_clock::now();
    		///////////////////////////"
    		duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif
    	}
    }

    void computeXjY(bool useCrossValidation) {
    	if (!syncCV) {
    		ModelSpecifics<BaseModel,RealType>::computeXjY(useCrossValidation);
    		detail::resizeAndCopyToDevice(hXjY, dXjY, queue);

    		std::cout << "XjY: ";
    		for (auto x:hXjY) {
    			std::cout << x << " ";
    		}
    		std::cout << "\n";
    	} else {
    		int size = layoutByPerson ? cvIndexStride : syncCVFolds;

    		dXjYVector.resize(J*size);

    		for (int i = FormatType::INTERCEPT; i >= FormatType::DENSE; --i) {
    			FormatType formatType = (FormatType)i;

    			std::vector<int> indices;
    			for (int index=0; index<J; index++) {
    				int formatType1 = formatList[index];
    				if (formatType1 == formatType) {
    					indices.push_back(index);
    				}
    			}

    			if (indices.size() > 0) {
    				auto& kernel = kernelComputeXjY[formatType];
    				kernel.set_arg(0, dColumns.getDataStarts());
    				kernel.set_arg(1, dColumns.getIndicesStarts());
    				kernel.set_arg(2, dColumns.getTaskCounts());
    				kernel.set_arg(3, dColumns.getData());
    				kernel.set_arg(4, dColumns.getIndices());
    				kernel.set_arg(5, dY);
    				kernel.set_arg(6, dKWeightVector);
    				kernel.set_arg(7, dXjYVector);
    				kernel.set_arg(8, cvIndexStride);
    				int dJ = J;
    				kernel.set_arg(9, dJ);
    				int length = indices.size();
    				kernel.set_arg(10, length);

    				detail::resizeAndCopyToDevice(indices, dIntVector1, queue);
    				kernel.set_arg(11, dIntVector1);

    				size_t globalWorkSize = syncCVFolds * tpb;
    				size_t localWorkSize = tpb;

    				queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    				queue.finish();
    			}
    		}
    	}
    }

    virtual void runCCDexactCLR(bool useWeights) {

    }

    virtual void runCCDStratified(bool useWeights) {

    }

    virtual void runCCDNonStratified(bool useWeights) {
    	int wgs = maxWgs; // for reduction across strata

    	if (useWeights) {
    		if (dBuffer.size() < 2*wgs*cvIndexStride) {
    			dBuffer.resize(2*wgs*cvIndexStride);
    		}
    	} else {
    		if (dBuffer.size() < 2*wgs) {
    			dBuffer.resize(2*wgs);
    		}

    		for (int index = 0; index < J; index++) {
#ifdef CYCLOPS_DEBUG_TIMING
    			auto start = bsccs::chrono::steady_clock::now();
#endif
    			FormatType formatType = hX.getFormatType(index);

    			auto& kernel = kernelGradientHessianNoWeight[formatType];

    			const auto taskCount = dColumns.getTaskCount(index);

    			//std::cout << "kernel 0 called\n";
    			kernel.set_arg(0, dColumns.getDataOffset(index));
    			kernel.set_arg(1, dColumns.getIndicesOffset(index));
    			kernel.set_arg(2, taskCount);
    			kernel.set_arg(3, dColumns.getData());
    			kernel.set_arg(4, dColumns.getIndices());
    			kernel.set_arg(5, dY);
    			kernel.set_arg(6, dXBeta);
    			kernel.set_arg(7, dExpXBeta);
    			kernel.set_arg(8, dDenominator);
    			kernel.set_arg(9, dBuffer);
    			kernel.set_arg(10, dId);
    			kernel.set_arg(11, dKWeight);

    			size_t globalWorkSize = wgs*tpb;
    			size_t localWorkSize = tpb;

#ifdef CYCLOPS_DEBUG_TIMING
    			auto end = bsccs::chrono::steady_clock::now();
    			///////////////////////////"
    			auto name = "compGradHessArgsG" + getFormatTypeExtension(formatType) + " ";
    			duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

    			// run kernel
#ifdef CYCLOPS_DEBUG_TIMING
    			start = bsccs::chrono::steady_clock::now();
#endif
    			queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, localWorkSize);
    			queue.finish();
#ifdef CYCLOPS_DEBUG_TIMING
    			end = bsccs::chrono::steady_clock::now();
    			///////////////////////////"
    			name = "compGradHessKernelG" + getFormatTypeExtension(formatType) + " ";
    			duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif


    			////////////////////////// Start Process Delta
#ifdef CYCLOPS_DEBUG_TIMING
    			start = bsccs::chrono::steady_clock::now();
#endif

//    			hBuffer.resize(J);
//    			compute::copy(std::begin(dPriorParams), std::begin(dPriorParams)+J, std::begin(hBuffer), queue);
//    			std::cout << "priors: ";
//    			for (auto x:hBuffer) {
//    				std::cout << x << " ";
//    			}
//    			std::cout << "\n";


    			int priorType = priorTypes[index];
    			auto& kernel1 = kernelProcessDeltaBuffer[priorType];

    			kernel1.set_arg(0, dBuffer);
    			if (dDeltaVector.size() < J) {
    				dDeltaVector.resize(J);
    			}
    			kernel1.set_arg(1, dDeltaVector);
    			kernel1.set_arg(2, wgs);
    			kernel1.set_arg(3, dBound);
    			kernel1.set_arg(4, dPriorParams);
    			kernel1.set_arg(5, dXjY);
    			kernel1.set_arg(6, index);
    			kernel1.set_arg(7, dBeta);
    			//int dN = N;
    			kernel1.set_arg(8, wgs);
#ifdef CYCLOPS_DEBUG_TIMING
    			end = bsccs::chrono::steady_clock::now();
    			///////////////////////////"
    			name = "compProcessDeltaArgsG" + getFormatTypeExtension(formatType) + " ";
    			duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
    			///////// run kernel
#ifdef CYCLOPS_DEBUG_TIMING
    			start = bsccs::chrono::steady_clock::now();
#endif

    			queue.enqueue_1d_range_kernel(kernel1, 0, tpb, tpb);
    			queue.finish();

#ifdef CYCLOPS_DEBUG_TIMING
    			end = bsccs::chrono::steady_clock::now();
    			///////////////////////////"
    			name = "compProcessDeltaKernelG" + getFormatTypeExtension(formatType) + " ";
    			duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

    			hBuffer.resize(J);
//    			compute::copy(std::begin(dBuffer), std::begin(dBuffer)+2*wgs, std::begin(hBuffer), queue);
    			compute::copy(std::begin(dDeltaVector), std::begin(dDeltaVector)+J, std::begin(hBuffer), queue);
    			std::cout << "delta " << index << ": " << hBuffer[index] << "\n";
//    			std::cout << "hBuffer: ";
//    			for (auto x:hBuffer) {
//    				std::cout << x << " ";
//    			}
//    			std::cout << "\n";
    		}
    	}
    }

    virtual void runCCD(bool useWeights) {

    	 int wgs = maxWgs; // for reduction across strata

    	 if (BaseModel::exactCLR) {
    		 runCCDexactCLR(useWeights);
    	 } else if (BaseModelG::useNWeights) {
    		 runCCDStratified(useWeights);
    	 } else {
    		 runCCDNonStratified(useWeights);
    	 }

    	 for (int index = 0; index < J; index++) {
#ifdef CYCLOPS_DEBUG_TIMING
    		 auto start = bsccs::chrono::steady_clock::now();
#endif
    		 FormatType formatType = hX.getFormatType(index);
    	 }

    }

    void turnOnSyncCV(int foldToCompute) {
    	ModelSpecifics<BaseModel, RealType>::turnOnSyncCV(foldToCompute);

    	std::cout << "start turn on syncCV\n";

    	syncCV = true;
    	pad = true;
    	syncCVFolds = foldToCompute;

    	layoutByPerson = false;
    	if (!layoutByPerson) multiprocessors = syncCVFolds;

    	tpb0 = 16;

    	if (BaseModel::exactCLR) {
    		detail::constant::exactCLRSyncBlockSize = detail::constant::maxBlockSize/tpb0;

    		if (detail::constant::exactCLRSyncBlockSize > detail::constant::exactCLRBlockSize) {
    			detail::constant::exactCLRSyncBlockSize = detail::constant::exactCLRBlockSize;
    		}

    		std::cout << "exactCLRSyncBlockSize: " << detail::constant::exactCLRSyncBlockSize << "\n";
    	}

        if (pad) {
        	// layout by person
        	cvBlockSize = tpb0;
        	cvIndexStride = detail::getAlignedLength<16>(syncCVFolds);
        } else {
        	// do not use
        	cvIndexStride = K;
        	//cvIndexStride = syncCVFolds;
        }

        tpb1 = tpb / tpb0;
        if (!layoutByPerson) cvIndexStride = detail::getAlignedLength<16>(K);

        std::cout << "cvStride: " << cvIndexStride << "\n";

    	//int dataStart = 0;
    	int garbage = 0;

    	std::vector<RealType> blah(cvIndexStride, 0);
    	std::vector<int> blah1(cvIndexStride, 0);
        //std::vector<real> hNWeightTemp;
        std::vector<RealType> hKWeightTemp;
        //std::vector<real> accDenomPidTemp;
        //std::vector<real> accNumerPidTemp;
        //std::vector<real> accNumerPid2Temp;
        //std::vector<int> accResetTemp;
        std::vector<int> hPidTemp;
        //std::vector<int> hPidInternalTemp;
        std::vector<RealType> hXBetaTemp;
        std::vector<RealType> offsExpXBetaTemp;
        std::vector<RealType> denomPidTemp;
        std::vector<RealType> denomPid2Temp;
        //std::vector<real> numerPidTemp;
        //std::vector<real> numerPid2Temp;
        //std::vector<real> hXjYTemp;
        //std::vector<real> hXjXTemp;
        //std::vector<real> logLikelihoodFixedTermTemp;
        //std::vector<IndexVectorPtr> sparseIndicesTemp;
        //std::vector<real> normTemp;
        //std::vector<int> cvIndexOffsets;


        if (layoutByPerson) {
        	for (int i=0; i<K; i++) {
        		//std::fill(std::begin(blah), std::end(blah), static_cast<real>(hKWeight[i]));
        		//appendAndPad(blah, hKWeightTemp, garbage, pad);

        		std::fill(std::begin(blah), std::end(blah), hXBeta[i]);
        		appendAndPad(blah, hXBetaTemp, garbage, pad);

        		std::fill(std::begin(blah), std::end(blah), offsExpXBeta[i]);
        		appendAndPad(blah, offsExpXBetaTemp, garbage, pad);

        		std::fill(std::begin(blah), std::end(blah), denomPid[i]);
        		appendAndPad(blah, denomPidTemp, garbage, pad);

        		std::fill(std::begin(blah), std::end(blah), denomPid2[i]);
        		appendAndPad(blah, denomPid2Temp, garbage, pad);

        		std::fill(std::begin(blah1), std::end(blah1), hPid[i]);
        		appendAndPad(blah1, hPidTemp, garbage, pad);
        	}
        } else {
        	for (int i=0; i<K; i++) {
        		blah1.push_back(hPid[i]);
        	}
        	for (int i=0; i<syncCVFolds; i++) {
        		appendAndPad(hXBeta, hXBetaTemp, garbage, pad);
        		appendAndPad(offsExpXBeta, offsExpXBetaTemp, garbage, pad);
        		appendAndPad(denomPid, denomPidTemp, garbage, pad);
        		appendAndPad(denomPid2, denomPid2Temp, garbage, pad);
        		appendAndPad(blah1, hPidTemp, garbage, pad);
        	}
        }
        //detail::resizeAndCopyToDevice(hNWeightTemp, dNWeightVector, queue);
        //detail::resizeAndCopyToDevice(hKWeightTemp, dKWeightVector, queue);
        //detail::resizeAndCopyToDevice(accDenomPidTemp, dAccDenomPidVector, queue);
        //detail::resizeAndCopyToDevice(accNumerPidTemp, dAccNumerPidVector, queue);
        //detail::resizeAndCopyToDevice(accNumerPid2Temp, dAccNumerPid2Vector, queue);
        //detail::resizeAndCopyToDevice(accResetTemp, dAccResetVector, queue);
        detail::resizeAndCopyToDevice(hPidTemp, dPidVector, queue);
        //detail::resizeAndCopyToDevice(hPidInternalTemp, dPidInternalVector, queue);
        detail::resizeAndCopyToDevice(hXBetaTemp, dXBetaVector, queue);
        detail::resizeAndCopyToDevice(offsExpXBetaTemp, dOffsExpXBetaVector, queue);
        detail::resizeAndCopyToDevice(denomPidTemp, dDenomPidVector, queue);
        detail::resizeAndCopyToDevice(denomPid2Temp, dDenomPid2Vector, queue);
        //detail::resizeAndCopyToDevice(numerPidTemp, dNumerPidVector, queue);
        //detail::resizeAndCopyToDevice(numerPid2Temp, dNumerPid2Vector, queue);
        //detail::resizeAndCopyToDevice(hXjYTemp, dXjYVector, queue);
        //detail::resizeAndCopyToDevice(hXjXTemp, dXjXVector, queue);
        //detail::resizeAndCopyToDevice(logLikelihoodFixedTermTemp, dLogLikelihoodFixedTermVector, queue);
        //detail::resizeAndCopyToDevice(sparseIndicesTemp, dSpareIndicesVector, queue);
        //detail::resizeAndCopyToDevice(normTemp, dNormVector, queue);
        //detail::resizeAndCopyToDevice(cvIndexOffsets, dcvIndexOffsets, queue);

        // AllZero
        std::vector<int> hAllZero;
        hAllZero.push_back(0);
        detail::resizeAndCopyToDevice(hAllZero, dAllZero, queue);

        // Done vector
        std::vector<int> hDone;
        std::vector<int> hCVIndices;
        int a = layoutByPerson ? cvIndexStride : syncCVFolds;
        hDone.resize(a, 0);
        for (int i=0; i<syncCVFolds; i++) {
        	hDone[i] = 1;
        	hCVIndices.push_back(i);
        }
    	activeFolds = syncCVFolds;

        detail::resizeAndCopyToDevice(hDone, dDoneVector, queue);
        detail::resizeAndCopyToDevice(hCVIndices, dCVIndices, queue);

        // build needed syncCV kernels
        int need = 0;
        for (size_t j = 0; j < J /*modelData.getNumberOfColumns()*/; ++j) {
            FormatType format = hX.getFormatType(j);

            //const auto& column = modelData.getColumn(j);
            // columns.emplace_back(GpuColumn<real>(column, ctx, queue, K));
            need |= (1 << format);
        }
        std::vector<FormatType> neededFormatTypes;
        for (int t = 0; t < 4; ++t) {
            if (need & (1 << t)) {
                neededFormatTypes.push_back(static_cast<FormatType>(t));
            }
        }
        std::cerr << "Format types required: " << need << std::endl;
        buildAllSyncCVKernels(neededFormatTypes);
        std::cout << "built all syncCV kernels \n";

        //printAllSyncCVKernels(std::cerr);
    }

    void turnOffSyncCV() {
    	syncCV = false;
    	initialized = false;
    }

    bool isGPU() {return true;};

    void setBounds(double initialBound) {
    	if (syncCV) {
    		std::vector<RealType> temp;
    		//temp.resize(J*syncCVFolds, initialBound);
    		// layout by person
    		int size = layoutByPerson ? cvIndexStride : syncCVFolds;
    		temp.resize(J*size, initialBound);
    		detail::resizeAndCopyToDevice(temp, dBoundVector, queue);
    	} else {
    		std::vector<RealType> temp;
    		temp.resize(J, initialBound);
    		detail::resizeAndCopyToDevice(temp, dBound, queue);
    	}
    }

    void setPriorTypes(std::vector<int>& inTypes) {
    	priorTypes.resize(J);
    	for (int i=0; i<J; i++) {
    		priorTypes[i] = inTypes[i];
    	}
    }

    void setPriorParams(std::vector<double>& inParams) {
    	std::vector<RealType> temp;
    	temp.resize(J, 0.0);
    	for (int i=0; i<J; i++) {
    		temp[i] = inParams[i];
    	}
    	detail::resizeAndCopyToDevice(temp, dPriorParams, queue);
    }

private:

    void buildAllKernels(const std::vector<FormatType>& neededFormatTypes) {
        buildAllGradientHessianKernels(neededFormatTypes);
        std::cout << "built gradhessian kernels \n";
//        buildAllUpdateXBetaKernels(neededFormatTypes);
//        std::cout << "built updateXBeta kernels \n";
//        buildAllGetGradientObjectiveKernels();
//        std::cout << "built getGradObjective kernels \n";
//        buildAllGetLogLikelihoodKernels();
//        std::cout << "built getLogLikelihood kernels \n";
        buildAllComputeRemainingStatisticsKernels();
        std::cout << "built computeRemainingStatistics kernels \n";
//        buildEmptyKernel();
//        std::cout << "built empty kernel\n";
//        //buildReduceCVBufferKernel();
//        //std::cout << "built reduceCVBuffer kernel\n";
        buildAllProcessDeltaKernels();
        std::cout << "built ProcessDelta kernels \n";
//        //buildAllDoItAllKernels(neededFormatTypes);
//        //std::cout << "built doItAll kernels\n";
//        buildAllDoItAllNoSyncCVKernels(neededFormatTypes);
//        std::cout << "built doItAllNoSyncCV kernels\n";
    }

    void buildAllSyncCVKernels(const std::vector<FormatType>& neededFormatTypes) {
//        buildAllSyncCVGradientHessianKernels(neededFormatTypes);
//        std::cout << "built syncCV gradhessian kernels \n";
//        buildAllSyncCVUpdateXBetaKernels(neededFormatTypes);
//        std::cout << "built syncCV updateXBeta kernels \n";
//        buildAllSyncCVGetGradientObjectiveKernels();
//        std::cout << "built syncCV getGradObjective kernels \n";
//        //buildAllGetLogLikelihoodKernels();
//        //std::cout << "built getLogLikelihood kernels \n";
//        buildAllSyncCVComputeRemainingStatisticsKernels();
//        std::cout << "built computeRemainingStatistics kernels \n";
//        buildReduceCVBufferKernel();
//        std::cout << "built reduceCVBuffer kernel\n";
//        buildAllProcessDeltaSyncCVKernels();
//        std::cout << "built ProcessDelta kernels \n";
//        buildAllDoItAllKernels(neededFormatTypes);
//        std::cout << "built doItAll kernels\n";
//        buildPredLogLikelihoodKernel();
//        std::cout << "built pred likelihood kernel\n";
        buildAllComputeXjYKernels(neededFormatTypes);
        std::cout << "built xjy kernels\n";
    }

    void buildAllGradientHessianKernels(const std::vector<FormatType>& neededFormatTypes) {
        int b = 0;
        for (FormatType formatType : neededFormatTypes) {
            buildGradientHessianKernel(formatType, true); ++b;
            buildGradientHessianKernel(formatType, false); ++b;
        }
    }

    void buildAllProcessDeltaKernels() {
    	buildProcessDeltaKernel(0);
    	buildProcessDeltaKernel(1);
    	buildProcessDeltaKernel(2);
    }

    void buildAllComputeRemainingStatisticsKernels() {
    	//for (FormatType formatType : neededFormatTypes) {
    		buildComputeRemainingStatisticsKernel();
    	//}
    }

    void buildAllComputeXjYKernels(const std::vector<FormatType>& neededFormatTypes) {
    	int b = 0;
    	for (FormatType formatType : neededFormatTypes) {
    		buildXjYKernel(formatType); ++b;
    	}
    }

    void buildGradientHessianKernel(FormatType formatType, bool useWeights) {

        auto isNvidia = compute::detail::is_nvidia_device(queue.get_device());
        isNvidia = false;
        std::stringstream options;

        if (BaseModel::exactCLR) {
        } else if (BaseModelG::useNWeights) {
        } else {

            if (double_precision) {
            	options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;
            } else {
                options << "-DREAL=float -DTMP_REAL=float -DTPB=" << tpb;
            }
            options << " -cl-mad-enable";

            std::cout << "double precision: " << double_precision << " tpb: " << tpb << "\n";

        	auto source = writeCodeForGradientHessianKernel(formatType, useWeights, isNvidia);
        	std::cout << source.body;

        	// CCD Kernel
        	auto program = compute::program::build_with_source(source.body, ctx, options.str());
        	std::cout << "program built\n";
        	auto kernel = compute::kernel(program, source.name);

        	if (useWeights) {
        		kernelGradientHessianWeighted[formatType] = std::move(kernel);
        	} else {
        		kernelGradientHessianNoWeight[formatType] = std::move(kernel);
        	}
        }
    }

    void buildProcessDeltaKernel(int priorType) {
        std::stringstream options;

        if (double_precision) {
#ifdef USE_VECTOR
        options << "-DREAL=double -DTMP_REAL=double2 -DTPB=" << tpb;
#else
        options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;
#endif // USE_VECTOR
        } else {
#ifdef USE_VECTOR
            options << "-DREAL=float -DTMP_REAL=float2 -DTPB=" << tpb;
#else
            options << "-DREAL=float -DTMP_REAL=float -DTPB=" << tpb;
#endif // USE_VECTOR
        }
        options << " -cl-mad-enable";

    	auto source = writeCodeForProcessDeltaKernel(priorType);
    	std::cout << source.body;
    	auto program = compute::program::build_with_source(source.body, ctx, options.str());
    	auto kernel = compute::kernel(program, source.name);

    	kernelProcessDeltaBuffer[priorType] = std::move(kernel);
    }

    void buildComputeRemainingStatisticsKernel() {
        	std::stringstream options;
        	if (BaseModelG::useNWeights) {
        		if (double_precision) {
    #ifdef USE_VECTOR
        			options << "-DREAL=double -DTMP_REAL=double2 -DTPB=" << tpb;
    #else
        			options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;
    #endif // USE_VECTOR
        		} else {
    #ifdef USE_VECTOR
        			options << "-DREAL=float -DTMP_REAL=float2 -DTPB=" << tpb;
    #else
        			options << "-DREAL=float -DTMP_REAL=float -DTPB=" << tpb;
    #endif // USE_VECTOR
        		}
        	} else {
        		options << "-DREAL=" << (double_precision ? "double" : "float");
        	}

        	options << " -cl-mad-enable";

            auto source = writeCodeForComputeRemainingStatisticsKernel();

            if (BaseModelG::useNWeights) {
            	source = writeCodeForStratifiedComputeRemainingStatisticsKernel(BaseModel::efron);
            }
            auto program = compute::program::build_with_source(source.body, ctx, options.str());
            std::cout << "program built\n";
            auto kernel = compute::kernel(program, source.name);

            kernelComputeRemainingStatistics = std::move(kernel);
        }


    void buildXjYKernel(FormatType formatType) {
    	std::stringstream options;
    	if (double_precision) {
    		options << "-DREAL=double -DTMP_REAL=double -DTPB=" << tpb;
        } else {

            options << "-DREAL=float -DTMP_REAL=float -DTPB=" << tpb;
        }
        options << " -cl-mad-enable";

         auto source = writeCodeForComputeXjYKernel(formatType, layoutByPerson);
         //std::cout << source.body;
         auto program = compute::program::build_with_source(source.body, ctx, options.str());
         auto kernel = compute::kernel(program, source.name);

         kernelComputeXjY[formatType] = std::move(kernel);
    }

    std::string getFormatTypeExtension(FormatType formatType) {
        switch (formatType) {
        case DENSE:
            return "Den";
        case SPARSE:
            return "Spa";
        case INDICATOR:
            return "Ind";
        case INTERCEPT:
            return "Icp";
        default: return "";
        }
    }

    SourceCode writeCodeForGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia);

    SourceCode writeCodeForProcessDeltaKernel(int priorType);

    SourceCode writeCodeForComputeRemainingStatisticsKernel();

    SourceCode writeCodeForStratifiedComputeRemainingStatisticsKernel(bool efron);

    SourceCode writeCodeForComputeXjYKernel(FormatType formatType, bool layoutByPerson);


    template <class T>
       void appendAndPad(const T& source, T& destination, int& length, bool pad) {
           for (auto x : source) {
               destination.push_back(x);
           }
           if (pad) {
               auto i = source.size();
               const auto end = detail::getAlignedLength<16>(i);
               for (; i < end; ++i) {
                   destination.push_back(typename T::value_type());
               }
               length += end;
           } else {
               length += source.size();
           }
       }

    // boost::compute objects
    const compute::device device;
    const compute::context ctx;
    compute::command_queue queue;
    compute::program program;

    std::map<FormatType, compute::kernel> kernelGradientHessianWeighted;
    std::map<FormatType, compute::kernel> kernelGradientHessianNoWeight;
    std::map<int, compute::kernel> kernelProcessDeltaBuffer;
    std::map<FormatType, compute::kernel> kernelComputeXjY;

    compute::kernel kernelComputeRemainingStatistics;

    std::map<FormatType, std::vector<int>> indicesFormats;
    std::vector<FormatType> formatList;

    // vectors of columns
    // std::vector<GpuColumn<real> > columns;
    AllGpuColumns<RealType> dColumns;
    AllGpuColumns<RealType> dColumnsXt;


    // CPU storage
    std::vector<RealType> hBuffer0;
    std::vector<RealType> hBuffer;
    std::vector<RealType> hBuffer1;
    std::vector<RealType> xMatrix;
    std::vector<RealType> expXMatrix;
	std::vector<RealType> hFirstRow;
	std::vector<RealType> hOverflow;

    // device storage
    compute::vector<RealType> dY;
    compute::vector<RealType> dBeta;
    compute::vector<RealType> dXBeta;
    compute::vector<RealType> dExpXBeta;
    compute::vector<RealType> dDenominator;
    compute::vector<RealType> dDenominator2;
    compute::vector<RealType> dAccDenominator;
    compute::vector<RealType> dNorm;
    compute::vector<RealType> dOffs;
    //compute::vector<int>  dFixBeta;
    //compute::vector<RealType> dAllDelta;

    compute::vector<RealType> dBound;
    compute::vector<RealType> dXjY;
    compute::vector<RealType> dXjX;

    // for exactCLR
    std::vector<int> subjects;
    int totalCases;
    int maxN;
    int maxCases;
    //compute::vector<RealType>  dRealVector1;
    compute::vector<int>  dIntVector1;
    compute::vector<int>  dIntVector2;
    bool initialized = false;
    compute::vector<int> dNtoK;
    compute::vector<int> dAllZero;
    compute::vector<RealType> dLogX;
    compute::vector<int> dKStrata;

#ifdef USE_VECTOR
    compute::vector<compute::double2_> dBuffer;
#else
    compute::vector<RealType> dBuffer;
    compute::vector<RealType> dBuffer1;
#endif // USE_VECTOR
    compute::vector<RealType> dKWeight;	//TODO make these weighttype
    compute::vector<RealType> dNWeight; //TODO make these weighttype
    compute::vector<int> dId;

    bool dXBetaKnown;
    bool hXBetaKnown;

    // syhcCV
//    bool syncCV;
//    int syncCVFolds;
    bool layoutByPerson;
    int cvBlockSize;
    int cvIndexStride;
    bool pad;
    int activeFolds;
    int multiprocessors = device.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS)*4/5;

    compute::vector<RealType> dNWeightVector;
    compute::vector<RealType> dKWeightVector;
    compute::vector<RealType> dAccDenomPidVector;
    compute::vector<RealType> dAccNumerPidVector;
    compute::vector<RealType> dAccNumerPid2Vector;
    compute::vector<int> dAccResetVector;
    compute::vector<int> dPidVector;
    compute::vector<int> dPidInternalVector;
    compute::vector<RealType> dXBetaVector;
    compute::vector<RealType> dOffsExpXBetaVector;
    compute::vector<RealType> dDenomPidVector;
    compute::vector<RealType> dDenomPid2Vector;
    compute::vector<RealType> dNumerPidVector;
    compute::vector<RealType> dNumerPid2Vector;
    compute::vector<RealType> dXjYVector;
    compute::vector<RealType> dXjXVector;
    //compute::vector<real> dLogLikelihoodFixedTermVector;
    //compute::vector<IndexVectorPtr> dSparseIndicesVector;
    compute::vector<RealType> dNormVector;
    compute::vector<RealType> dDeltaVector;
    compute::vector<RealType> dBoundVector;
    compute::vector<RealType> dPriorParams;
    compute::vector<RealType> dBetaVector;
    compute::vector<int> dDoneVector;
    compute::vector<int> dCVIndices;
    compute::vector<int> dSMStarts;
    compute::vector<int> dSMScales;
    compute::vector<int> dSMIndices;

    std::vector<int> hSMStarts;
    std::vector<int> hSMScales;
    std::vector<int> hSMIndices;

    std::vector<int> hSMScales0;
    std::vector<int> hSMIndices0;

    std::vector<RealType> priorTypes;
    compute::vector<int> dIndexListWithPrior;
    std::vector<int> indexListWithPriorStarts;
    std::vector<int> indexListWithPriorLengths;

};

static std::string timesX(const std::string& arg, const FormatType formatType) {
    return (formatType == INDICATOR || formatType == INTERCEPT) ?
        arg : arg + " * x";
}

static std::string weight(const std::string& arg, bool useWeights) {
    return useWeights ? "w * " + arg : arg;
}

static std::string weightK(const std::string& arg, bool useWeights) {
    return useWeights ? "wK * " + arg : arg;
}

static std::string weightN(const std::string& arg, bool useWeights) {
    return useWeights ? "wN * " + arg : arg;
}

struct GroupedDataG {
public:
	std::string getGroupG(const std::string& groups, const std::string& person) {
		return groups + "[" + person + "]";
	}
};

struct GroupedWithTiesDataG : GroupedDataG {
public:
};

struct OrderedDataG {
public:
	std::string getGroupG(const std::string& groups, const std::string& person) {
		return person;
	}
};

struct OrderedWithTiesDataG {
public:
	std::string getGroupG(const std::string& groups, const std::string& person) {
		return groups + "[" + person + "]";
	}
};

struct IndependentDataG {
public:
	std::string getGroupG(const std::string& groups, const std::string& person) {
		return person;
	}
};

struct FixedPidG {
};

struct SortedPidG {
};

struct NoFixedLikelihoodTermsG {
	// TODO throw error like in ModelSpecifics.h?
};

#define Fraction std::complex

struct GLMProjectionG {
public:
	const static bool denomRequiresStratumReduction = false; // TODO not using now
	const static bool logisticDenominator = false; // TODO not using now
	const static bool useNWeights = false;

	std::string gradientNumeratorContribG(
			const std::string& x, const std::string& predictor,
			const std::string& xBeta, const std::string& y) {
		return predictor + "*" + x;
	}

	std::string logLikeNumeratorContribG(
			const std::string& yi, const std::string& xBetai) {
		return yi + "*" + xBetai;
	}

	std::string gradientNumerator2ContribG(
			const std::string& x, const std::string& predictor) {
		return predictor + "*" + x + "*" + x;
	}

};

struct SurvivalG {
public:
	// TODO incrementFisherInformation
	// TODO incrementMMGradientAndHessian
};

struct LogisticG {
public:

	// TODO incrementFisherInformation

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		// assume exists: numer, denom, w if weighted
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

	// TODO incrementMMGradientAndHessian
	// TODO is the other increment G + H deprecated?

};

struct SelfControlledCaseSeriesG : public GroupedDataG, GLMProjectionG, FixedPidG, SurvivalG {
public:

	std::string logLikeFixedTermsContribG(
			const std::string& yi, const std::string& offseti,
			const std::string& logoffseti) {
		return yi + "*" + "log(" + offseti + ")";
	}

	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

	// TODO incrementMMGradientAndHessian

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
    	return offs + "*" + "exp(" + xBeta + ")";
    }

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta,
			const std::string& y, const std::string& k) {
    	return offs + "[" +  k + "]" + "*" + "exp(" + xBeta + ")";
    }

	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log("  + denom + ")";
	}

	std::string logPredLikeContribG(
			const std::string& y, const std::string& weight,
			const std::string& xBeta, const std::string& denominator) {
	    return y + "*" + weight + "*"  + "(" + xBeta + "- log(" + denominator + "))";
	}

	std::string logPredLikeContribG(
			const std::string& ji, const std::string& weighti,
			const std::string& xBetai, const std::string& denoms,
			const std::string& groups, const std::string& i) {
		return ji + "*" + weighti + "*" + "(" + xBetai + "- log(" + denoms + "[" + getGroupG(groups, i) + "]))";
	}

	// TODO predictEstimate

};

struct ConditionalPoissonRegressionG : public GroupedDataG, GLMProjectionG, FixedPidG, SurvivalG {
public:

	// outputs logLikeFixedTerm
	std::string logLikeFixedTermsContribG(
			const std::string& yi, const std::string& offseti,
			const std::string& logoffseti) {
		std::stringstream code;
		code << "logLikeFixedTerm = (REAL)0.0;";
		code << "for (int i=2; i<=(int)" + yi + "; i++)";
			code << "logLikeFixedTerm -= log((REAL)i);";
		return(code.str());
	} // TODO not sure if this works

	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string observationCountG(const std::string& yi) {
		return "(REAL)" + yi;
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}


    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta,
			const std::string& y, const std::string& k) {
		return "exp(" + xBeta + ")";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}

	std::string logPredLikeContribG(
			const std::string& y, const std::string& weight,
			const std::string& xBeta, const std::string& denominator) {
		return y + "*" + weight + "*" +  "(" + xBeta + "- log(" + denominator + "))";
	}

	std::string logPredLikeContribG(
			const std::string& ji, const std::string& weighti,
			const std::string& xBetai, const std::string& denoms,
			const std::string& groups, const std::string& i) {
		return ji + "*" + weighti + "*" + "(" + xBetai + " - log(" + denoms + "[" + getGroupG(groups, i) + "]))";
	}

	// TODO predictEstimate

};

struct ConditionalLogisticRegressionG : public GroupedDataG, GLMProjectionG, FixedPidG, SurvivalG {
public:
	const static bool denomRequiresStratumReduction = true;
	const static bool useNWeights = true;

	// TODO logLikeFixedTermsContrib throw error?

	std::string getDenomNullValueG () {
	    return "(REAL)0.0";
	}

	std::string observationCountG(const std::string& yi) {
		return "(REAL)" + yi;
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta,
			const std::string& y, const std::string& k) {
		return "exp(" + xBeta + ")";
    }

	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}

	std::string logPredLikeContribG(
			const std::string& y, const std::string& weight,
			const std::string& xBeta, const std::string& denominator) {
		return y + "*" + weight + "*" +  "(" + xBeta + "- log(" + denominator + "))";
	}

	std::string logPredLikeContribG(
			const std::string& ji, const std::string& weighti,
			const std::string& xBetai, const std::string& denoms,
			const std::string& groups, const std::string& i) {
		return ji + "*" + weighti + "*" + "(" + xBetai + " - log(" + denoms + "[" + getGroupG(groups, i) + "]))";
	}

	//TODO predictEstimate

};

// TODO add efron properly

struct EfronConditionalLogisticRegressionG : public GroupedDataG, GLMProjectionG, FixedPidG, SurvivalG {
public:
	const static bool denomRequiresStratumReduction = true;
	const static bool useNWeights = true;
	const static bool efron = true;

	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}



};

struct TiedConditionalLogisticRegressionG : public GroupedWithTiesDataG, GLMProjectionG, FixedPidG, SurvivalG {
public:
	const static bool denomRequiresStratumReduction = false;

	// TODO logLikeFixedTermsContrib throw error?

	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string observationCountG(const std::string& yi) {
		return "(REAL)" + yi;
	}

    // same as lr, do not use
	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		// assume exists: numer, denom
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta,
			const std::string& y, const std::string& k) {
		return "exp(" + xBeta + ")";
    }

	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}

	std::string logPredLikeContribG(
			const std::string& y, const std::string& weight,
			const std::string& xBeta, const std::string& denominator) {
		return y + "*" + weight + "*" +  "(" + xBeta + "- log(" + denominator + "))";
	}

	std::string logPredLikeContribG(
			const std::string& ji, const std::string& weighti,
			const std::string& xBetai, const std::string& denoms,
			const std::string& groups, const std::string& i) {
		return ji + "*" + weighti + "*" + "(" + xBetai + " - log(" + denoms + "[" + getGroupG(groups, i) + "]))";
	}

	//TODO predictEstimate

};

struct LogisticRegressionG : public IndependentDataG, GLMProjectionG, LogisticG, FixedPidG,
	NoFixedLikelihoodTermsG {
public:
	const static bool logisticDenominator = true;

	std::string getDenomNullValueG () {
		return "(REAL)1.0";
	}

	std::string observationCountG(const std::string& yi) {
		return "(REAL)1.0";
	}

	std::string setIndependentDenominatorG(const std::string& expXBeta) {
	    return "(REAL)1.0 + " + expXBeta;
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta,
			const std::string& y, const std::string& k) {
		return "exp(" + xBeta + ")";
    }

    // outputs logLikeDenominatorContrib
    std::string logLikeDenominatorContribG(
    		const std::string& ni, const std::string& denom) {
    	std::stringstream code;
    	code << "REAL logLikeDenominatorContrib;";
    	code << "if (" + ni + " == (REAL)0.0)  {";
    	code << "	logLikeDenominatorContrib = 0.0;";
    	code << "} else {";
    	code << "	logLikeDenominatorContrib = " + ni + "* log((" + denom + "- (REAL)1.0)/" + ni + "+ (REAL)1.0);";
    	code << "}";
    	return(code.str());
	}

	std::string logPredLikeContribG(
			const std::string& y, const std::string& weight,
			const std::string& xBeta, const std::string& denominator) {
		return y + "*" + weight + "*" +  "(" + xBeta + "- log(" + denominator + "))";
	}

	std::string logPredLikeContribG(
			const std::string& ji, const std::string& weighti,
			const std::string& xBetai, const std::string& denoms,
			const std::string& groups, const std::string& i) {
		return ji + "*" + weighti + "*" + "(" + xBetai + " - log(" + denoms + "[" + getGroupG(groups, i) + "]))";
	}

	//TODO predictEstimate

	//TODO incrementGradientAndHessian2

};

// TODO transcribe rest of BaseModelG's from BaseModel once figure out how to handle cox

struct CoxProportionalHazardsG : public OrderedDataG, GLMProjectionG, SortedPidG, NoFixedLikelihoodTermsG, SurvivalG {
public:
	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}

};

struct StratifiedCoxProportionalHazardsG : public CoxProportionalHazardsG {
public:
};

struct BreslowTiedCoxProportionalHazardsG : public OrderedWithTiesDataG, GLMProjectionG, SortedPidG, NoFixedLikelihoodTermsG, SurvivalG {
public:
	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}


	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL g = numer / denom;      \n";
        code << "       REAL gradient = " << weight("g", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = " << weight("g * ((REAL)1.0 - g)", useWeights) << ";\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("(nume2 / denom - g * g)", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return ni + "*" + "log(" + denom + ")";
	}
};

struct LeastSquaresG : public IndependentDataG, FixedPidG, NoFixedLikelihoodTermsG  {
public:
	const static bool denomRequiresStratumReduction = false;
	const static bool logisticDenominator = false;
	const static bool useNWeights = false;

	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		return("");
	}

	std::string logLikeNumeratorContribG() {
		std::stringstream code;
		code << "(y-xb)*(y-xb)";
		return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "(REAL)0.0";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return "log(" + denom + ")";
	}

};

struct PoissonRegressionG : public IndependentDataG, GLMProjectionG, FixedPidG {
public:
	std::string getDenomNullValueG () {
		return "(REAL)0.0";
	}

	std::string incrementGradientAndHessianG(FormatType formatType, bool useWeights) {
		std::stringstream code;
        code << "       REAL gradient = " << weight("numer", useWeights) << ";\n";
        if (formatType == INDICATOR || formatType == INTERCEPT) {
            code << "       REAL hessian  = gradient;\n";
        } else {
            code << "       REAL nume2 = " << timesX("numer", formatType) << ";\n" <<
                    "       REAL hessian  = " << weight("nume2", useWeights) << ";\n";
        }
        return(code.str());
	}

    std::string getOffsExpXBetaG(
    		const std::string& offs, const std::string& xBeta) {
		return "exp(" + xBeta + ")";
    }


	std::string logLikeDenominatorContribG(
			const std::string& ni, const std::string& denom) {
		return denom;
	}

};


} // namespace bsccs

#include "Kernels.hpp"

#endif /* GPUMODELSPECIFICS_HPP_ */













