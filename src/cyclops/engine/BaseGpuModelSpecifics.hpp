//
// Created by Jianxiao Yang on 2019-12-18.
//

#ifndef BASEGPUMODELSPECIFICS_HPP
#define BASEGPUMODELSPECIFICS_HPP

#ifdef HAVE_OPENCL
#include <boost/compute/algorithm/reduce.hpp>
#endif // HAVE_OPENCL

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif //HAVE_CUDA

#include "ModelSpecifics.hpp"

namespace bsccs {

#ifdef HAVE_OPENCL
    namespace compute = boost::compute;
#endif // HAVE_OPENCL

    namespace detail {

        namespace constant {
            static const int updateXBetaBlockSize = 256; // 512; // Appears best on K40
            static const int updateAllXBetaBlockSize = 32;
            int exactCLRBlockSize = 32;
            int exactCLRSyncBlockSize = 32;
            static const int maxBlockSize = 256;
        }; // namespace constant

#ifdef HAVE_OPENCL
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
#endif // HAVE_OPENCL

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
//        Rcpp::stop(error);
                // throw new std::logic_error(error);
            }
        }

        template <int D, class T>
        int getAlignedLength(T n) {
            return (n / D) * D + (n % D == 0 ? 0 : D);
        }
    }

    struct SourceCode {
        std::string body;
        std::string name;

        SourceCode(std::string body, std::string name) : body(body), name(name) { }
    };

#ifdef HAVE_OPENCL
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

/*
        typedef thrust::device_vector<RealType> DataVector;
        typedef thrust::device_vector<int> IndicesVector;
        typedef unsigned int UInt;
        typedef thrust::device_vector<UInt> dStartsVector;
        typedef std::vector<UInt> hStartsVector;

        AllGpuColumns() {
            // Do nothing
//		std::cout << "ctor AGC \n";
        }
*/
        virtual ~AllGpuColumns() {
//		std::cout << "dtor AGC \n";       
	}

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

/*
	        resizeAndCopyToDeviceCuda(flatData, data);
            resizeAndCopyToDeviceCuda(flatIndices, indices);
            resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
            resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
            resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);

            CudaDetail<RealType> rdetail;
            CudaDetail<int> idetail;
            CudaDetail<UInt> udetail;
            rdetail.resizeAndCopyToDeviceCuda(flatData, data);
            idetail.resizeAndCopyToDeviceCuda(flatIndices, indices);
            udetail.resizeAndCopyToDeviceCuda(dataStarts, ddataStarts);
            udetail.resizeAndCopyToDeviceCuda(indicesStarts, dindicesStarts);
            udetail.resizeAndCopyToDeviceCuda(taskCounts, dtaskCounts);
*/
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

    template<class BaseModel, typename RealType>
    class BaseGpuModelSpecifics : public ModelSpecifics<BaseModel, RealType> {
    public:

        using ModelSpecifics<BaseModel, RealType>::modelData;
        using ModelSpecifics<BaseModel, RealType>::hX;
//        using ModelSpecifics<BaseModel, RealType>::hNtoK;
        using ModelSpecifics<BaseModel, RealType>::hPid;
        using ModelSpecifics<BaseModel, RealType>::hPidInternal;
        using ModelSpecifics<BaseModel, RealType>::accReset;
        using ModelSpecifics<BaseModel, RealType>::hXjY;
//        using ModelSpecifics<BaseModel, RealType>::hXjX;
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
//        using ModelSpecifics<BaseModel, RealType>::logLikelihoodFixedTerm;

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

        using ModelSpecifics<BaseModel, RealType>::accDenomPid;
        using ModelSpecifics<BaseModel, RealType>::accNumerPid;
        using ModelSpecifics<BaseModel, RealType>::accNumerPid2;

        BaseGpuModelSpecifics(const ModelData<RealType>& input,
                              const std::string& deviceName) :
                ModelSpecifics<BaseModel,RealType>(input),
                device(compute::system::find_device(deviceName)),
                ctx(device),
                queue(ctx, device,
                        compute::command_queue::enable_profiling
                ),
                dColumns(),
                dBeta(ctx), dXBeta(ctx), dExpXBeta(ctx),
                dDenominator(ctx), dDenominator2(ctx), dAccDenominator(ctx),
                dOffs(ctx), dKWeight(ctx), dNWeight(ctx), dId(ctx),
                dY(ctx) {
            std::cerr << "ctor BaseGpuModelSpecifics" << std::endl;
        }

        virtual ~BaseGpuModelSpecifics() {
            std::cerr << "dtor BaseGpuModelSpecifics" << std::endl;
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

//            std::vector<FormatType> neededFormatTypes;
            for (int t = 0; t < 4; ++t) {
                if (need & (1 << t)) {
                    neededFormatTypes.push_back(static_cast<FormatType>(t));
                }
            }

            auto& inputY = modelData.getYVectorRef();
            detail::resizeAndCopyToDevice(inputY, dY, queue);

            // Internal buffers
//            detail::resizeAndCopyToDevice(hBeta, dBeta, queue);
            detail::resizeAndCopyToDevice(hXBeta, dXBeta, queue);
//            hXBetaKnown = true; dXBetaKnown = true;
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

        }

        virtual void setWeights(double* inWeights, bool useCrossValidation) {
            // Currently only computed on CPU and then copied to GPU
            ModelSpecifics<BaseModel, RealType>::setWeights(inWeights, useCrossValidation);

            detail::resizeAndCopyToDevice(hKWeight, dKWeight, queue);
            detail::resizeAndCopyToDevice(hNWeight, dNWeight, queue);
        }

    protected:

        // boost::compute objects
        const compute::device device;
        const compute::context ctx;
        compute::command_queue queue;

        // vectors of columns
        // std::vector<GpuColumn<real> > columns;
        AllGpuColumns<RealType> dColumns;
//        AllGpuColumns<RealType> dColumnsXt;

        std::map<FormatType, std::vector<int>> indicesFormats;
        std::vector<FormatType> formatList;
        std::vector<FormatType> neededFormatTypes;

        // device storage
        compute::vector<RealType> dY;
        compute::vector<RealType> dBeta;
        compute::vector<RealType> dXBeta;
        compute::vector<RealType> dExpXBeta;
        compute::vector<RealType> dDenominator;
        compute::vector<RealType> dDenominator2;
        compute::vector<RealType> dAccDenominator;
        compute::vector<RealType> dOffs;
        compute::vector<RealType> dKWeight;	//TODO make these weighttype
        compute::vector<RealType> dNWeight; //TODO make these weighttype
        compute::vector<int> dId;

        const int tpb = 256; // threads-per-block  // Appears best on K40
        const int maxWgs = 16;
        const int tpb0 = 16;
        const int tpb1 = 16;

        bool double_precision = false;

        int multiprocessors = device.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS)*4/5;

    };
#endif // HAVE_OPENCL
}
#endif //BASEGPUMODELSPECIFICS_HPP
