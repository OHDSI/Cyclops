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
    typedef std::vector<UInt> StartsVector;

    AllGpuColumns(const compute::context& context) : indices(context), data(context) {
    		//dataStarts(context), indicesStarts(context), taskCounts(context) {
        // Do nothing
    }

    virtual ~AllGpuColumns() { }

    void initialize(const CompressedDataMatrix& mat,
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

    const StartsVector& getDataStarts() const {
    	return dataStarts;
    }

    const StartsVector& getIndicesStarts() const {
    	return indicesStarts;
    }

    const StartsVector& getTaskCounts() const {
    	return taskCounts;
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
	StartsVector taskCounts;
	StartsVector dataStarts;
	StartsVector indicesStarts;

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

    GpuColumn(const CompressedDataColumn& column,
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


template <class BaseModel, typename WeightType>
class GpuModelSpecifics : public ModelSpecifics<BaseModel, WeightType> {
public:

    using ModelSpecifics<BaseModel, WeightType>::modelData;
    using ModelSpecifics<BaseModel, WeightType>::offsExpXBeta;
    using ModelSpecifics<BaseModel, WeightType>::hXBeta;
    using ModelSpecifics<BaseModel, WeightType>::hY;
    using ModelSpecifics<BaseModel, WeightType>::hNtoK;
    using ModelSpecifics<BaseModel, WeightType>::hNWeight;
    using ModelSpecifics<BaseModel, WeightType>::hKWeight;
    using ModelSpecifics<BaseModel, WeightType>::hPid;
    using ModelSpecifics<BaseModel, WeightType>::hPidInternal;
    using ModelSpecifics<BaseModel, WeightType>::hOffs;
    using ModelSpecifics<BaseModel, WeightType>::denomPid;
    using ModelSpecifics<BaseModel, WeightType>::hXjY;
    using ModelSpecifics<BaseModel, WeightType>::hXjX;
    using ModelSpecifics<BaseModel, WeightType>::K;
    using ModelSpecifics<BaseModel, WeightType>::J;
    using ModelSpecifics<BaseModel, WeightType>::N;
    using ModelSpecifics<BaseModel, WeightType>::duration;
    //using ModelSpecifics<BaseModel, WeightType>::hBeta;
    using ModelSpecifics<BaseModel, WeightType>::algorithmType;
    using ModelSpecifics<BaseModel, WeightType>::norm;
    using ModelSpecifics<BaseModel, WeightType>::boundType;

    const static int tpb = 128; // threads-per-block  // Appears best on K40
    const static int maxWgs = 2;  // work-group-size

    // const static int globalWorkSize = tpb * wgs;

    GpuModelSpecifics(const ModelData& input,
                      const std::string& deviceName)
    : ModelSpecifics<BaseModel,WeightType>(input),
      device(compute::system::find_device(deviceName)),
      ctx(device),
      queue(ctx, device
          , compute::command_queue::enable_profiling
      ),
      dColumns(ctx),
      dY(ctx), dBeta(ctx), dXBeta(ctx), dExpXBeta(ctx), dDenominator(ctx), dBuffer(ctx), dKWeight(ctx),
      dId(ctx), dNorm(ctx), dOffs(ctx), dFixBeta(ctx), dIndices(ctx), dVector1(ctx), dVector2(ctx),
      dBuffer1(ctx), dXMatrix(ctx), dExpXMatrix(ctx), dOverflow0(ctx), dOverflow1(ctx),
	  dXBetaKnown(false), hXBetaKnown(false){

        std::cerr << "ctor GpuModelSpecifics" << std::endl;

        // Get device ready to compute
        std::cerr << "Using: " << device.name() << std::endl;
    }

    virtual ~GpuModelSpecifics() {
        std::cerr << "dtor GpuModelSpecifics" << std::endl;
    }

    virtual void deviceInitialization() {
#ifdef TIME_DEBUG
        std::cerr << "start dI" << std::endl;
#endif

        int need = 0;

        // Copy data
        dColumns.initialize(modelData, queue, K, true);

        for (size_t j = 0; j < J /*modelData.getNumberOfColumns()*/; ++j) {

#ifdef TIME_DEBUG
          //  std::cerr << "dI " << j << std::endl;
#endif

            const auto& column = modelData.getColumn(j);
            // columns.emplace_back(GpuColumn<real>(column, ctx, queue, K));
            need |= (1 << column.getFormatType());
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
        detail::resizeAndCopyToDevice(hPidInternal, dId, queue);
        detail::resizeAndCopyToDevice(hOffs, dOffs, queue);

        if (BaseModel::exactCLR) {
        }

        std::cerr << "Format types required: " << need << std::endl;

        buildAllKernels(neededFormatTypes);
        std::cout << "built all kernels \n";

        printAllKernels(std::cerr);
    }

    virtual void computeRemainingStatistics(bool useWeights) {

        //std::cerr << "GPU::cRS called" << std::endl;

        // Currently RS only computed on CPU and then copied

        hBuffer.resize(K);
        //std::cout << "before cRS offsExpXBeta: " << offsExpXBeta[0] << ' ' << offsExpXBeta[1] << '\n';


        //std::cout << "hExpXBeta: " << hXBeta[0] << ' ' << hXBeta[1] << '\n';
    	//compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue);
/*
        if (BaseModel::likelihoodHasDenominator) {
            compute::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(hBuffer), queue);
            //compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
            //compute::copy(std::begin(dDenominator), std::end(dDenominator), std::begin(denomPid), queue);
        }
        */

        //std::cout << "dXBeta: " << exp(hXBeta[0]) << ' ' << exp(hXBeta[1]) << '\n';

        /*
        compute::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(hBuffer), queue);
        std::cout << "dExpXBeta: " << hBuffer[0] << ' ' << hBuffer[1] << '\n';

        ModelSpecifics<BaseModel, WeightType>::computeRemainingStatistics(useWeights);
        std::cout << "after cRS offsExpXBeta: " << offsExpXBeta[0] << ' ' << offsExpXBeta[1] << '\n';
		*/

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif
        /*
        if (algorithmType == AlgorithmType::MM) {
        	compute::copy(std::begin(hBeta), std::end(hBeta), std::begin(dBeta), queue);
        }
        */

        /*
        if (BaseModel::likelihoodHasDenominator) {
            compute::copy(std::begin(offsExpXBeta), std::end(offsExpXBeta), std::begin(dExpXBeta), queue);
            //compute::copy(std::begin(denomPid), std::end(denomPid), std::begin(dDenominator), queue);
        }
        */

        //compute::copy(std::begin(dDenominator), std::end(dDenominator), std::begin(hBuffer), queue);
        //std::cout << "before kernel dDenominator: " << hBuffer[0] << " " << hBuffer[1] << '\n';

        auto& kernel = kernelComputeRemainingStatistics[FormatType::DENSE];
        //         auto& column = columns[index];
        //         const auto taskCount = column.getTaskCount();

        const auto taskCount = K;
        int dK = K;
        kernel.set_arg(0, dK);
        size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
        if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
        	++workGroups;
        }
        const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
        queue.finish();

        //hBuffer.resize(K);
        //compute::copy(std::begin(dDenominator), std::end(dDenominator), std::begin(hBuffer), queue);
        //std::cout << "after kernel dDenominator: " << hBuffer[0] << " " << hBuffer[1] << '\n';



#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        duration["compRSG          "] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();;
#endif

    }

    void computeGradientAndHessian(int index, double *ogradient,
                                           double *ohessian, bool useWeights) {

#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::computeGradientAndHessian(index, ogradient, ohessian, useWeights);
        std::cerr << *ogradient << " & " << *ohessian << std::endl;
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif
        FormatType formatType = modelData.getFormatType(index);
        double gradient = 0.0;
        double hessian = 0.0;

        if (BaseModel::exactCLR) {
        	auto& kernel = (useWeights) ? // Double-dispatch
        	                            kernelGradientHessianWeighted[formatType] :
        	                            kernelGradientHessianNoWeight[formatType];
        	if (!initialized) {
        		totalCases = 0;
        		for (int i=0; i < N; ++i) {
        			totalCases += hNWeight[i];
        		}
        		int temp = 0;
        		maxN = 0;
        		subjects.resize(N);
        		for (int i = 0; i < N; ++i) {
        			int newN = hNtoK[i+1] - hNtoK[i];
        			subjects[i] = newN;
        			if (newN > maxN) maxN = newN;
        		}

        		// indices vector
        		std::vector<int> hIndices;
        		hIndices.resize(3*(N+totalCases));
        		temp = 0;
        		for (int i=0; i < N; ++i) {
        			hIndices[temp] = 0;
        			hIndices[temp+1] = 0;
        			hIndices[temp+2] = 0;
        			temp += 3;
        			for (int j = 3; j < 3*(hNWeight[i]+1); ++j) {
        				hIndices[temp] = i+1;
        				++temp;
        			}
        		}
        		detail::resizeAndCopyToDevice(hIndices, dIndices, queue);

        		// constant vectors
        		std::vector<int> hVector1;
        		std::vector<int> hVector2;
        		hVector1.resize(3*(N+totalCases));
        		hVector2.resize(3*(N+totalCases));
        		for (int i=0; i < N+totalCases; ++i) {
        			hVector1[3*i] = 0;
        			hVector1[3*i+1] = 1;
        			hVector1[3*i+2] = 1;
        			hVector2[3*i] = 0;
        			hVector2[3*i+1] = 0;
        			hVector2[3*i+2] = 1;
        		}
        		detail::resizeAndCopyToDevice(hVector1, dVector1, queue);
        		detail::resizeAndCopyToDevice(hVector2, dVector2, queue);

        		// overflow vectors
        		std::vector<int> hOverflow;
        		hOverflow.resize(N+1);
        		for (int i=0; i < N+1; ++i) {
        			hOverflow[i] = 0;
        		}
        		detail::resizeAndCopyToDevice(hOverflow, dOverflow0, queue);
        		detail::resizeAndCopyToDevice(hOverflow, dOverflow1, queue);

        		initialized = true;
        	}

        	kernel.set_arg(2, dIndices);
        	kernel.set_arg(5, dVector1);
        	kernel.set_arg(6, dVector2);
        	kernel.set_arg(10, dOverflow0);
        	kernel.set_arg(11, dOverflow1);

        	// B0 and B1
    	    int temp = 0;
        	hBuffer.resize(3*(N+totalCases));
    	    for (int i=0; i < 3*(N+totalCases); ++i) {
    	    	hBuffer[i] = 0;
    	    }
    	    for (int i=0; i < N; ++i) {
    	    	hBuffer[temp] = 1;
    	        temp += 3*(hNWeight[i]+1);
    	    }
    	    detail::resizeAndCopyToDevice(hBuffer, dBuffer, queue);
    	    detail::resizeAndCopyToDevice(hBuffer, dBuffer1, queue);
    	    kernel.set_arg(0, dBuffer);
    	    kernel.set_arg(1, dBuffer1);

            compute::copy(std::begin(dExpXBeta), std::end(dExpXBeta), std::begin(offsExpXBeta), queue);
            GenericIterator x(modelData, index);
    	    auto expX = offsExpXBeta.begin();

    	    // X and ExpX matrices
    	    xMatrix.resize((N+1) * maxN);
    	    expXMatrix.resize((N+1) * maxN);
    	    for (int j = 0; j < maxN; ++j) {
    	    	xMatrix[j*(N+1)] = 0;
    	    	expXMatrix[j*(N+1)] = 0;
    	    }

    	    for (int i = 1; i < (N+1); ++i) {
    	        for (int j = 0; j < maxN; ++j) {
    	            if (j < subjects[i-1]) {
    	                xMatrix[j*(N+1) + i] = x.value();
    	                expXMatrix[j*(N+1) + i] = *expX;
    	                ++expX;
    	                ++x;
    	            } else {
    	                xMatrix[j*(N+1) + i] = 0;
    	                expXMatrix[j*(N+1) + i] = -1;
    	            }
    	        }
    	    }
    	    detail::resizeAndCopyToDevice(xMatrix, dXMatrix, queue);
    	    detail::resizeAndCopyToDevice(expXMatrix, dExpXMatrix, queue);
    	    int dN = N;
    	    kernel.set_arg(3, dXMatrix);
    	    kernel.set_arg(4, dExpXMatrix);
    	    kernel.set_arg(7, dN);

    	    compute::uint_ taskCount = 3*(N+totalCases);
    	    size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
    	    if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
    	    	++workGroups;
    	    }
    	    const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;
    	    kernel.set_arg(12, taskCount);
    	    for (int i=0; i < maxN; ++i) {
    	        /*
        	    kernel.set_arg(0, dBuffer);
        	    kernel.set_arg(1, dBuffer1);
            	kernel.set_arg(2, dIndices);
        	    kernel.set_arg(3, dXMatrix);
        	    kernel.set_arg(4, dExpXMatrix);
            	kernel.set_arg(5, dVector1);
            	kernel.set_arg(6, dVector2);
        	    kernel.set_arg(7, dN);
    	         */
                if (dKWeight.size() == 0) {
                    kernel.set_arg(9, 0);
                } else {
                    kernel.set_arg(9, dKWeight); // TODO Only when dKWeight gets reallocated
                }
                /*
            	kernel.set_arg(10, dOverflow0);
            	kernel.set_arg(11, dOverflow1);
        	    kernel.set_arg(12, taskCount);
                 */

        	    kernel.set_arg(8, i);
    	        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
    	        queue.finish();
    	    }

    	    if (maxN%2 == 0) {
    	    	compute::copy(std::begin(dBuffer), std::end(dBuffer), std::begin(hBuffer), queue);
    	    } else {
    	    	compute::copy(std::begin(dBuffer1), std::end(dBuffer1), std::begin(hBuffer), queue);
    	    }

    	    temp = 0;
    	    for (int i=0; i<N; ++i) {
    	    	temp += hNWeight[i]+1;
    	        std::cout<<"new values" << i << ": " << hBuffer[3*temp-3] <<" | "<< hBuffer[3*temp-2] << " | " << hBuffer[3*temp-1] << '\n';
    	    	gradient -= (real)(-hBuffer[3*temp-2]/hBuffer[3*temp-3]);
    	    	hessian -= (real)((hBuffer[3*temp-2]/hBuffer[3*temp-3]) * (hBuffer[3*temp-2]/hBuffer[3*temp-3]) - hBuffer[3*temp-1]/hBuffer[3*temp-3]);
    	    }
        } else {
/*
        if (!dXBetaKnown) {
            compute::copy(std::begin(hXBeta), std::end(hXBeta), std::begin(dXBeta), queue);
            dXBetaKnown = true;
        }
        */

        auto& kernel = (useWeights) ? // Double-dispatch
                            kernelGradientHessianWeighted[formatType] :
                            kernelGradientHessianNoWeight[formatType];

        // auto& column = columns[index];
        // const auto taskCount = column.getTaskCount();

        const auto taskCount = dColumns.getTaskCount(index);

        const auto wgs = maxWgs;
        const auto globalWorkSize = tpb * wgs;

        size_t loops = taskCount / globalWorkSize;
        if (taskCount % globalWorkSize != 0) {
            ++loops;
        }

        // std::cerr << dBuffer.get_buffer() << std::endl;

#ifdef USE_VECTOR
        if (dBuffer.size() < maxWgs) {
            dBuffer.resize(maxWgs, queue);
            //compute::fill(std::begin(dBuffer), std::end(dBuffer), 0.0, queue); // TODO Not needed
            kernel.set_arg(9, dBuffer); // Can get reallocated.
            hBuffer.resize(2 * maxWgs);
        }
#else

        if (dBuffer.size() < 2 * maxWgs) {
            dBuffer.resize(2 * maxWgs, queue);
            //compute::fill(std::begin(dBuffer), std::end(dBuffer), 0.0, queue); // TODO Not needed
            kernel.set_arg(9, dBuffer); // Can get reallocated.
            hBuffer.resize(2 * maxWgs);
        }
#endif

        // std::cerr << dBuffer.get_buffer() << std::endl << std::endl;

        if (dKWeight.size() == 0) {
            kernel.set_arg(11, 0);
        } else {
            kernel.set_arg(11, dKWeight); // TODO Only when dKWeight gets reallocated
        }

//         kernel.set_arg(0, 0);
//         kernel.set_arg(1, 0);
//         kernel.set_arg(2, taskCount);
//
//         kernel.set_arg(3, column.getDataVector());
//         kernel.set_arg(4, column.getIndicesVector());


        kernel.set_arg(0, dColumns.getDataOffset(index));
        kernel.set_arg(1, dColumns.getIndicesOffset(index));
        kernel.set_arg(2, taskCount);

        kernel.set_arg(3, dColumns.getData());
        kernel.set_arg(4, dColumns.getIndices());


//         std::cerr << "loop= " << loops << std::endl;
//         std::cerr << "n   = " << taskCount << std::endl;
//         std::cerr << "gWS = " << globalWorkSize << std::endl;
//         std::cerr << "tpb = " << tpb << std::endl;
//
        // std::cerr << kernel.get_program().source() << std::endl;


//         compute::vector<real> tmpR(taskCount, ctx);
//         compute::vector<int> tmpI(taskCount, ctx);

        // kernel.set_arg(0, tmpR);
        // kernel.set_arg(1, tmpI);

//         kernel.set_arg(4, tmpR);
//         kernel.set_arg(5, tmpR);
//         kernel.set_arg(6, tmpR);
//         kernel.set_arg(7, tmpR);
        // kernel.set_arg(8, tmpR);
        // kernel.set_arg(9, dBuffer); // TODO Why is this necessary?

//         kernel.set_arg(9, tmpI);
//         kernel.set_arg(10, tmpR);

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, tpb);
        queue.finish();

//         for (int i = 0; i < wgs; ++i) {
//             std::cerr << ", " << dBuffer[i];
//         }
//         std::cerr << std::endl;

        // Get result
#ifdef USE_VECTOR
        compute::copy(std::begin(dBuffer), std::end(dBuffer), reinterpret_cast<compute::double2_ *>(hBuffer.data()), queue);

        double gradient = 0.0;
        double hessian = 0.0;

        for (int i = 0; i < 2 * wgs; i += 2) { // TODO Use SSE
            gradient += hBuffer[i + 0];
            hessian  += hBuffer[i + 1];
        }

        if (BaseModel::precomputeGradient) { // Compile-time switch
            gradient -= hXjY[index];
        }

        if (BaseModel::precomputeHessian) { // Compile-time switch
            hessian += static_cast<real>(2.0) * hXjX[index];
        }

        *ogradient = gradient;
        *ohessian = hessian;
#else
        compute::copy(std::begin(dBuffer), std::end(dBuffer), std::begin(hBuffer), queue);

        for (int i = 0; i < wgs; ++i) { // TODO Use SSE
            gradient += hBuffer[i];
            hessian  += hBuffer[i + wgs];
        }
        }

        if (BaseModel::precomputeGradient) { // Compile-time switch
            gradient -= hXjY[index];
        }

        if (BaseModel::precomputeHessian) { // Compile-time switch
            hessian += static_cast<real>(2.0) * hXjX[index];
        }

        *ogradient = gradient;
        *ohessian = hessian;
#endif


#ifdef GPU_DEBUG
        std::cerr << gradient << " & " << hessian << std::endl << std::endl;
#endif // GPU_DEBUG

//         for (auto x : dBuffer) {
//             std::cerr << x << std::endl;
//         }
// //         for(int i = 0; i < wgs; ++i) {
// //             std::cerr << dBuffer[i] << std::endl;
// //         }
//         std::cerr << (-hXjY[index]) << "  " << "0.0" << std::endl;
//
//
//         Rcpp::stop("out");

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "compGradHessG" + getFormatTypeExtension(formatType) + " ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

    }

	virtual void computeMMGradientAndHessian(
			std::vector<GradientHessian>& gh,
			const std::vector<bool>& fixBeta,
			bool useWeights) {
#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif
        if (!dXBetaKnown) {
        	//compute::copy(std::begin(hBeta), std::end(hBeta), std::begin(dBeta), queue);
            compute::copy(std::begin(hXBeta), std::end(hXBeta), std::begin(dXBeta), queue);
            dXBetaKnown = true;
        }

        FormatType formatType = modelData.getFormatType(0);

        auto& kernel = (useWeights) ? // Double-dispatch
        		kernelGradientHessianMMWeighted[formatType] :
				kernelGradientHessianMMNoWeight[formatType];

        //auto& kernel = (useWeights) ? // Double-dispatch
        //		kernelGradientHessianWeighted[formatType] :
		//		kernelGradientHessianNoWeight[formatType];

    	const auto wgs = maxWgs;
    	const auto globalWorkSize = tpb * wgs;

    	if (dBuffer.size() < 2 * maxWgs * J) {
    		dBuffer.resize(2 * maxWgs * J, queue);
    		//compute::fill(std::begin(dBuffer), std::end(dBuffer), 0.0, queue); // TODO Not needed
    		kernel.set_arg(9, dBuffer); // Can get reallocated.
    		hBuffer.resize(2 * maxWgs * J);
    	}

    	if (dKWeight.size() == 0) {
    		kernel.set_arg(11, 0);
    	} else {
    		kernel.set_arg(11, dKWeight); // TODO Only when dKWeight gets reallocated
    	}

    	/*
    	std::vector<int> hDataStarts;
    	hDataStarts.resize(J);
    	compute::copy(std::begin(dColumns.getIndicesStarts()), std::end(dColumns.getIndicesStarts()), std::begin(hDataStarts), queue);
        for (auto x : hDataStarts) {
            std::cerr << " " << x;
        }
        std::cerr << "\n";
        */
/*
    	kernel.set_arg(0, dColumns.getDataStarts());
    	kernel.set_arg(1, dColumns.getIndicesStarts());
    	kernel.set_arg(2, dColumns.getTaskCounts());
	    //detail::resizeAndCopyToDevice(fixBeta, dFixBeta, queue);
    	dFixBeta.resize(J);
    	for (int i=0; i<J; ++i) {
    		dFixBeta[i] = fixBeta[i];
    	}
        //compute::copy(std::begin(fixBeta), std::end(fixBeta), std::begin(dFixBeta), queue);
    	kernel.set_arg(13, dFixBeta);
    	int dJ = J;
    	kernel.set_arg(14, dJ);
 */


    	kernel.set_arg(3, dColumns.getData());
    	kernel.set_arg(4, dColumns.getIndices());
    	kernel.set_arg(6, dXBeta);
    	kernel.set_arg(7, dExpXBeta);
	    if (norm.size() == 0) {
	        this->initializeMM(boundType, fixBeta);
		    //dNorm.resize(norm.size(), queue);
        	//compute::copy(std::begin(norm), std::end(norm), std::begin(dNorm), queue);
		    //std::cout << "size: " << dNorm.size() << " dNorm0: "<< dNorm[0] << '\n';
		    detail::resizeAndCopyToDevice(norm, dNorm, queue);
	    	kernel.set_arg(12, dNorm);
	    }
    	//kernel.set_arg(12, dBeta);

		//queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, tpb);
		//queue.finish();


    	for (int index = 0; index < J; ++index) {

    		if (fixBeta[index]) continue;

    		// auto& column = columns[index];
    		// const auto taskCount = column.getTaskCount();

    		const auto taskCount = dColumns.getTaskCount(index);

    		size_t loops = taskCount / globalWorkSize;
    		if (taskCount % globalWorkSize != 0) {
    			++loops;
    		}
    		kernel.set_arg(0, dColumns.getDataOffset(index));
    		kernel.set_arg(1, dColumns.getIndicesOffset(index));
    		kernel.set_arg(2, taskCount);
    		kernel.set_arg(13, index);
    		queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, tpb);
    		queue.finish();
    	}



    	// Get result

    	compute::copy(std::begin(dBuffer), std::end(dBuffer), std::begin(hBuffer), queue);

    	for (int j = 0; j < J; ++j) {
    		for (int i = 0; i < wgs; ++i) { // TODO Use SSE
    			gh[j].first += hBuffer[i + 2*j*wgs];
    			gh[j].second += hBuffer[i + wgs + 2*j*wgs];
    		}
    	}

    	if (BaseModel::precomputeGradient) { // Compile-time switch
    		for (int j=0; j < J; ++j) {
    			gh[j].first -= hXjY[j];
    		}
    	}

    	if (BaseModel::precomputeHessian) { // Compile-time switch
    		for (int j = 0; j < J; ++j) {
    			gh[j].second += static_cast<real>(2.0) * hXjX[j];
    		}
    	}

#ifdef CYCLOPS_DEBUG_TIMING
    	auto end = bsccs::chrono::steady_clock::now();
    	///////////////////////////"
    	auto name = "compGradHessG" + getFormatTypeExtension(formatType) + " ";
    	duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

	}

    virtual void updateXBeta(real realDelta, int index, bool useWeights) {
#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::updateXBeta(realDelta, index, useWeights);
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        auto& kernel = kernelUpdateXBeta[modelData.getFormatType(index)];
//         auto& column = columns[index];
//         const auto taskCount = column.getTaskCount();
        const auto taskCount = dColumns.getTaskCount(index);

        kernel.set_arg(0, dColumns.getDataOffset(index));
        kernel.set_arg(1, dColumns.getIndicesOffset(index));
        kernel.set_arg(2, taskCount);
        kernel.set_arg(3, realDelta);
        kernel.set_arg(4, dColumns.getData());
        kernel.set_arg(5, dColumns.getIndices());

        size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
        if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
            ++workGroups;
        }
        const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
        queue.finish();

        hXBetaKnown = false; // dXBeta was just updated

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "updateXBetaG" + getFormatTypeExtension(modelData.getFormatType(index)) + "  ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

#ifdef GPU_DEBUG
        // Compare results:
        detail::compare(hXBeta, dXBeta, "xBeta not equal");
        detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
        detail::compare(denomPid, dDenominator, "denominator not equal");
#endif // GPU_DEBUG
    }
/*
    virtual void computeXBeta(real* allDelta, bool useWeights) {
#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::updateXBeta(realDelta, index, useWeights);
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        auto& kernel = kernelUpdateXBeta[FormatType::DENSE];
//         auto& column = columns[index];
//         const auto taskCount = column.getTaskCount();
        for (int index = 0; index < J; ++index) {
        	real delta = allDelta[index];
        	const auto taskCount = dColumns.getTaskCount(index);

        	kernel.set_arg(0, dColumns.getDataOffset(index));
        	kernel.set_arg(1, dColumns.getIndicesOffset(index));
        	kernel.set_arg(2, taskCount);
        	kernel.set_arg(3, delta);
        	kernel.set_arg(4, dColumns.getData());
        	kernel.set_arg(5, dColumns.getIndices());

        	size_t workGroups = taskCount / detail::constant::updateXBetaBlockSize;
        	if (taskCount % detail::constant::updateXBetaBlockSize != 0) {
        		++workGroups;
        	}
        	const size_t globalWorkSize = workGroups * detail::constant::updateXBetaBlockSize;

        	queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, detail::constant::updateXBetaBlockSize);
        	queue.finish();
        }

        hXBetaKnown = false; // dXBeta was just updated

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "ComputeXBetaG" + getFormatTypeExtension(FormatType::DENSE) + "  ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif

#ifdef GPU_DEBUG
        // Compare results:
        detail::compare(hXBeta, dXBeta, "xBeta not equal");
        detail::compare(offsExpXBeta, dExpXBeta, "expXBeta not equal");
        detail::compare(denomPid, dDenominator, "denominator not equal");
#endif // GPU_DEBUG
    }
    */

    virtual double getGradientObjective(bool useCrossValidation) {
#ifdef GPU_DEBUG
        ModelSpecifics<BaseModel, WeightType>::getGradientObjective(useCrossValidation);
        std::cerr << *ogradient << " & " << *ohessian << std::endl;
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto start = bsccs::chrono::steady_clock::now();
#endif

        FormatType formatType = FormatType::DENSE;
        auto& kernel = (useCrossValidation) ? // Double-dispatch
                            kernelGetGradientObjectiveWeighted[formatType] :
							kernelGetGradientObjectiveNoWeight[formatType];

        const auto wgs = maxWgs;
        const auto globalWorkSize = tpb * wgs;

        dBuffer.resize(2 * maxWgs, queue);
        kernel.set_arg(3, dBuffer); // Can get reallocated.
        hBuffer.resize(2 * maxWgs);

        if (dKWeight.size() == 0) {
            kernel.set_arg(4, 0);
        } else {
            kernel.set_arg(4, dKWeight); // TODO Only when dKWeight gets reallocated
        }

        queue.enqueue_1d_range_kernel(kernel, 0, globalWorkSize, tpb);
        queue.finish();

        compute::copy(std::begin(dBuffer), std::end(dBuffer), std::begin(hBuffer), queue);

        double objective = 0.0;

        for (int i = 0; i < wgs; ++i) { // TODO Use SSE
        	objective += hBuffer[i];
        }

#ifdef GPU_DEBUG
        std::cerr << gradient << " & " << hessian << std::endl << std::endl;
#endif // GPU_DEBUG

#ifdef CYCLOPS_DEBUG_TIMING
        auto end = bsccs::chrono::steady_clock::now();
        ///////////////////////////"
        auto name = "compGradObj" + getFormatTypeExtension(formatType) + " ";
        duration[name] += bsccs::chrono::duration_cast<chrono::TimingUnits>(end - start).count();
#endif
        return(objective);
    }

    virtual void setWeights(double* inWeights, bool useCrossValidation) {
        // Currently only computed on CPU and then copied to GPU
        ModelSpecifics<BaseModel, WeightType>::setWeights(inWeights, useCrossValidation);

        detail::resizeAndCopyToDevice(hKWeight, dKWeight, queue);
    }

    virtual const RealVector& getXBeta() {
        if (!hXBetaKnown) {
            compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
            hXBetaKnown = true;
        }
        return ModelSpecifics<BaseModel,WeightType>::getXBeta();
    }

    virtual const RealVector& getXBetaSave() {
        return ModelSpecifics<BaseModel,WeightType>::getXBetaSave();
    }

    virtual void saveXBeta() {
        if (!hXBetaKnown) {
            compute::copy(std::begin(dXBeta), std::end(dXBeta), std::begin(hXBeta), queue);
            hXBetaKnown = true;
        }
        ModelSpecifics<BaseModel,WeightType>::saveXBeta();
    }

    virtual void zeroXBeta() {

        //std::cerr << "GPU::zXB called" << std::endl;

        ModelSpecifics<BaseModel,WeightType>::zeroXBeta(); // touches hXBeta

        dXBetaKnown = false;
    }

    virtual void axpyXBeta(const double beta, const int j) {

        //std::cerr << "GPU::aXB called" << std::endl;

        ModelSpecifics<BaseModel,WeightType>::axpyXBeta(beta, j); // touches hXBeta

        dXBetaKnown = false;
    }

    virtual void computeNumeratorForGradient(int index) {
    }


private:

    void buildAllUpdateXBetaKernels(const std::vector<FormatType>& neededFormatTypes) {
        for (FormatType formatType : neededFormatTypes) {
            buildUpdateXBetaKernel(formatType);
        }
    }

    void buildAllComputeRemainingStatisticsKernels(const std::vector<FormatType>& neededFormatTypes) {
    	for (FormatType formatType : neededFormatTypes) {
    		buildComputeRemainingStatisticsKernel(formatType);
    	}
    }

    void buildAllComputeXBetaKernels(const std::vector<FormatType>& neededFormatTypes) {
    	for (FormatType formatType : neededFormatTypes) {
    		buildComputeXBetaKernel(formatType);
    	}
    }

    void buildAllGradientHessianKernels(const std::vector<FormatType>& neededFormatTypes) {
        int b = 0;
        for (FormatType formatType : neededFormatTypes) {
            buildGradientHessianKernel(formatType, true); ++b;
            buildGradientHessianKernel(formatType, false); ++b;
        }
    }

    void buildAllGetGradientObjectiveKernels(const std::vector<FormatType>& neededFormatTypes) {
        int b = 0;
        for (FormatType formatType : neededFormatTypes) {
        	buildGetGradientObjectiveKernel(formatType, true); ++b;
        	buildGetGradientObjectiveKernel(formatType, false); ++b;
        }
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

    SourceCode writeCodeForUpdateXBetaKernel(FormatType formatType);

    SourceCode writeCodeForMMGradientHessianKernel(FormatType formatType, bool useWeights, bool isNvidia);

    SourceCode writeCodeForGetGradientObjective(FormatType formatType, bool useWeights, bool isNvidia);

    SourceCode writeCodeForComputeRemainingStatisticsKernel(FormatType formatType);

    SourceCode writeCodeForComputeXBetaKernel(FormatType formatType);

    SourceCode writeCodeForGradientHessianKernelExactCLR(FormatType formatType, bool useWeights, bool isNvidia);

    void buildGradientHessianKernel(FormatType formatType, bool useWeights) {

        std::stringstream options;

        if (sizeof(real) == 8) {
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
        options << " -cl-mad-enable -cl-fast-relaxed-math";

//         compute::vector<compute::double2_> buf(10, ctx);
//
//         compute::double2_ sum = compute::double2_{0.0, 0.0};
//         compute::reduce(buf.begin(), buf.end(), &sum, queue);
//
//         std::cerr << sum << std::endl;
//
//         auto cache = compute::program_cache::get_global_cache(ctx);
//         auto list = cache->get_keys();
//         std::cerr << "list size = " << list.size() << std::endl;
//         for (auto a : list) {
//             std::cerr << a.first << ":" << a.second << std::endl;
//             auto p = cache->get(a.first, a.second);
//             if (p) {
//                 std::cerr << p->source() << std::endl;
//             }
//         }
//
//         Rcpp::stop("out");

        const auto isNvidia = compute::detail::is_nvidia_device(queue.get_device());
        std::cout << "formatType: " << formatType << " isNvidia: " << isNvidia << '\n';

//         std::cerr << queue.get_device().name() << " " << queue.get_device().vendor() << std::endl;
//         std::cerr << "isNvidia = " << isNvidia << std::endl;
//         Rcpp::stop("out");

        if (BaseModel::exactCLR) {
        	// CCD Kernel
        	auto source = writeCodeForGradientHessianKernelExactCLR(formatType, useWeights, isNvidia);
        	std::cout << source.body;
        	auto program = compute::program::build_with_source(source.body, ctx, options.str());
        	auto kernel = compute::kernel(program, source.name);

        	kernel.set_arg(2, dIndices);
        	kernel.set_arg(5, dVector1);
        	kernel.set_arg(6, dVector2);
        	kernel.set_arg(9, dKWeight);

        	// MM Kernel
        	source = writeCodeForMMGradientHessianKernel(formatType, useWeights, isNvidia);
        	program = compute::program::build_with_source(source.body, ctx, options.str());
        	auto kernelMM = compute::kernel(program, source.name);
        	kernelMM.set_arg(5, dY);
        	kernelMM.set_arg(6, dXBeta);
        	kernelMM.set_arg(7, dExpXBeta);
        	kernelMM.set_arg(8, dDenominator);
        	kernelMM.set_arg(9, dBuffer);  // TODO Does not seem to stick
        	kernelMM.set_arg(10, dId);
        	kernelMM.set_arg(11, dKWeight); // TODO Does not seem to stick

        	if (useWeights) {
        		kernelGradientHessianWeighted[formatType] = std::move(kernel);
        		kernelGradientHessianMMWeighted[formatType] = std::move(kernelMM);
        	} else {
        		kernelGradientHessianNoWeight[formatType] = std::move(kernel);
        		kernelGradientHessianMMNoWeight[formatType] = std::move(kernelMM);
        	}
        } else {
        	// CCD Kernel
        	auto source = writeCodeForGradientHessianKernel(formatType, useWeights, isNvidia);
        	auto program = compute::program::build_with_source(source.body, ctx, options.str());
        	auto kernel = compute::kernel(program, source.name);
        	kernel.set_arg(5, dY);
        	kernel.set_arg(6, dXBeta);
        	kernel.set_arg(7, dExpXBeta);
        	kernel.set_arg(8, dDenominator);
        	kernel.set_arg(9, dBuffer);  // TODO Does not seem to stick
        	kernel.set_arg(10, dId);
        	kernel.set_arg(11, dKWeight); // TODO Does not seem to stick
        	// Rcpp::stop("cGH");

        	// MM Kernel
        	source = writeCodeForMMGradientHessianKernel(formatType, useWeights, isNvidia);
        	program = compute::program::build_with_source(source.body, ctx, options.str());
        	auto kernelMM = compute::kernel(program, source.name);
        	kernelMM.set_arg(5, dY);
        	kernelMM.set_arg(6, dXBeta);
        	kernelMM.set_arg(7, dExpXBeta);
        	kernelMM.set_arg(8, dDenominator);
        	kernelMM.set_arg(9, dBuffer);  // TODO Does not seem to stick
        	kernelMM.set_arg(10, dId);
        	kernelMM.set_arg(11, dKWeight); // TODO Does not seem to stick

        	if (useWeights) {
        		kernelGradientHessianWeighted[formatType] = std::move(kernel);
        		kernelGradientHessianMMWeighted[formatType] = std::move(kernelMM);
        	} else {
        		kernelGradientHessianNoWeight[formatType] = std::move(kernel);
        		kernelGradientHessianMMNoWeight[formatType] = std::move(kernelMM);
        	}
        }
    }

    void buildUpdateXBetaKernel(FormatType formatType) {

        std::stringstream options;
        options << "-DREAL=" << (sizeof(real) == 8 ? "double" : "float");

        auto source = writeCodeForUpdateXBetaKernel(formatType);

        // std::cerr << source.body << std::endl;

        auto program = compute::program::build_with_source(source.body, ctx, options.str());
        std::cout << "built updateXBeta program \n";
        auto kernel = compute::kernel(program, source.name);

        // Rcpp::stop("uXB");

        // Run-time constant arguments.
        kernel.set_arg(6, dY);
        kernel.set_arg(7, dXBeta);
        kernel.set_arg(8, dExpXBeta);
        kernel.set_arg(9, dDenominator);
        kernel.set_arg(10, dId);

        kernelUpdateXBeta[formatType] = std::move(kernel);
    }

    void buildComputeRemainingStatisticsKernel(FormatType formatType) {
    	std::stringstream options;
    	options << "-DREAL=" << (sizeof(real) == 8 ? "double" : "float");

        auto source = writeCodeForComputeRemainingStatisticsKernel(formatType);
        std::cout << source.body;
        auto program = compute::program::build_with_source(source.body, ctx, options.str());
        std::cout << "built computeRemainingStatistics program \n";
        auto kernel = compute::kernel(program, source.name);

        int dK = K;
        kernel.set_arg(0, dK);
        kernel.set_arg(1, dXBeta);
        kernel.set_arg(2, dExpXBeta);
        kernel.set_arg(3, dDenominator);
        kernel.set_arg(4, dId);

        kernelComputeRemainingStatistics[formatType] = std::move(kernel);
    }

    void buildComputeXBetaKernel(FormatType formatType) {
      	std::stringstream options;
      	options << "-DREAL=" << (sizeof(real) == 8 ? "double" : "float");

      	auto source = writeCodeForComputeXBetaKernel(formatType);
      	std::cout << source.body;
      	auto program = compute::program::build_with_source(source.body, ctx, options.str());
      	std::cout << "built computeXBeta program \n";
      	auto kernel = compute::kernel(program, source.name);
        kernel.set_arg(6, dY);
        kernel.set_arg(7, dXBeta);
        kernel.set_arg(8, dExpXBeta);
        kernel.set_arg(9, dDenominator);
        kernel.set_arg(10, dId);
      	kernelComputeXBeta[formatType] = std::move(kernel);
    }

    void buildGetGradientObjectiveKernel(FormatType formatType, bool useWeights) {
    	std::stringstream options;
        if (sizeof(real) == 8) {
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
        options << " -cl-mad-enable -cl-fast-relaxed-math";

         const auto isNvidia = compute::detail::is_nvidia_device(queue.get_device());

         auto source = writeCodeForGetGradientObjective(formatType, useWeights, isNvidia);
         std::cout << source.body;
         auto program = compute::program::build_with_source(source.body, ctx, options.str());
         auto kernel = compute::kernel(program, source.name);

         int dK = K;

         // Run-time constant arguments.
         kernel.set_arg(0, dK);
         kernel.set_arg(1, dY);
         kernel.set_arg(2, dXBeta);
         kernel.set_arg(3, dBuffer);  // TODO Does not seem to stick
         kernel.set_arg(4, dKWeight); // TODO Does not seem to stick

         if (useWeights) {
             kernelGetGradientObjectiveWeighted[formatType] = std::move(kernel);
         } else {
        	 kernelGetGradientObjectiveNoWeight[formatType] = std::move(kernel);
         }
    }

    void printKernel(compute::kernel& kernel, std::ostream& stream) {

        auto program = kernel.get_program();
        auto buildOptions = program.get_build_info<std::string>(CL_PROGRAM_BUILD_OPTIONS, device);

        stream // TODO Change to R
            << "Options: " << buildOptions << std::endl
            << program.source()
            << std::endl;
    }

    void buildAllKernels(const std::vector<FormatType>& neededFormatTypes) {
        buildAllGradientHessianKernels(neededFormatTypes);
        std::cout << "built gradhessian kernels \n";
        buildAllUpdateXBetaKernels(neededFormatTypes);
        std::cout << "built updateXBeta kernels \n";
        buildAllGetGradientObjectiveKernels(neededFormatTypes);
        std::cout << "built getGradObjective kernels \n";
        buildAllComputeRemainingStatisticsKernels(neededFormatTypes);
        std::cout << "built computeRemainingStatistics kernels \n";
        buildAllComputeXBetaKernels(neededFormatTypes);
        std::cout << "built computeXBeta kernels \n";
    }

    void printAllKernels(std::ostream& stream) {
    	for (auto& entry : kernelGradientHessianWeighted) {
    		printKernel(entry.second, stream);
    	}

    	for (auto& entry : kernelGradientHessianNoWeight) {
    		printKernel(entry.second, stream);
    	}
    	for (auto& entry : kernelGradientHessianMMWeighted) {
    		printKernel(entry.second, stream);
    	}

    	for (auto& entry : kernelGradientHessianMMNoWeight) {
    		printKernel(entry.second, stream);
    	}

        for (auto& entry : kernelUpdateXBeta) {
            printKernel(entry.second, stream);
        }

        for (auto& entry: kernelGetGradientObjectiveWeighted) {
        	printKernel(entry.second, stream);
        }

        for (auto& entry: kernelGetGradientObjectiveNoWeight) {
        	printKernel(entry.second, stream);
        }

        for (auto& entry: kernelComputeRemainingStatistics) {
        	printKernel(entry.second, stream);
        }

        for (auto& entry: kernelComputeXBeta) {
        	printKernel(entry.second, stream);
        }
    }

    // boost::compute objects
    const compute::device device;
    const compute::context ctx;
    compute::command_queue queue;
    compute::program program;

    std::map<FormatType, compute::kernel> kernelGradientHessianWeighted;
    std::map<FormatType, compute::kernel> kernelGradientHessianNoWeight;
    std::map<FormatType, compute::kernel> kernelUpdateXBeta;
    std::map<FormatType, compute::kernel> kernelGradientHessianMMWeighted;
    std::map<FormatType, compute::kernel> kernelGradientHessianMMNoWeight;
    std::map<FormatType, compute::kernel> kernelGetGradientObjectiveWeighted;
    std::map<FormatType, compute::kernel> kernelGetGradientObjectiveNoWeight;
    std::map<FormatType, compute::kernel> kernelComputeRemainingStatistics;
    std::map<FormatType, compute::kernel> kernelComputeXBeta;

    // vectors of columns
    // std::vector<GpuColumn<real> > columns;
    AllGpuColumns<real> dColumns;

    std::vector<real> hBuffer;
    std::vector<real> xMatrix;
    std::vector<real> expXMatrix;

    // Internal storage
    compute::vector<real> dY;
    compute::vector<real> dBeta;
    compute::vector<real> dXBeta;
    compute::vector<real> dExpXBeta;
    compute::vector<real> dDenominator;
    compute::vector<real> dNorm;
    compute::vector<real> dOffs;
    compute::vector<bool> dFixBeta;

    // for exactCLR
    std::vector<int> subjects;
    int totalCases;
    int maxN;
    compute::vector<int>  dVector1;
    compute::vector<int>  dVector2;
    compute::vector<int>  dIndices;
    compute::vector<real> dXMatrix;
    compute::vector<real> dExpXMatrix;
    bool initialized = false;
    compute::vector<int> dOverflow0;
    compute::vector<int> dOverflow1;

#ifdef USE_VECTOR
    compute::vector<compute::double2_> dBuffer;
#else
    compute::vector<real> dBuffer;
    compute::vector<real> dBuffer1;
#endif // USE_VECTOR
    compute::vector<real> dKWeight;
    compute::vector<int> dId;

    bool dXBetaKnown;
    bool hXBetaKnown;
};

} // namespace bsccs


#include "Kernels.hpp"

#endif /* GPUMODELSPECIFICS_HPP_ */
