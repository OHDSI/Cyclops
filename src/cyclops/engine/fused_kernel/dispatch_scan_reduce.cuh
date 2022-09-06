
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceScanReduce provides device-wide, parallel operations for computing a prefix scan and transform-reduction across a sequence of data items residing within device-accessible memory.
 */

#pragma once

#include <iterator>

#include "agent_scan_reduce.cuh"
#include <cub/thread/thread_operators.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/config.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename            ScanTileStateT>     ///< Tile status interface type
__global__ void DeviceScanReduceInitKernel(
    ScanTileStateT      tile_state,         ///< [in] Tile status interface
    int                 num_tiles)          ///< [in] Number of tiles
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename                ScanTileStateT,         ///< Tile status interface type
    typename                NumSelectedIteratorT>   ///< Output iterator type for recording the number of items selected
__global__ void DeviceScanReduceCompactInitKernel(
    ScanTileStateT          tile_state,             ///< [in] Tile status interface
    int                     num_tiles,              ///< [in] Number of tiles
    NumSelectedIteratorT    d_num_selected_out)     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
{
    // Initialize tile status
    tile_state.InitializeStatus(num_tiles);

    // Initialize d_num_selected_out
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
        *d_num_selected_out = 0;
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename            ChainedPolicyT,          ///< Chained tuning policy
    typename            ScanInputIteratorT,      ///< Random-access input iterator type for reading scan inputs \iterator
    typename            TransformInputIteratorT, ///< Random-access input iterator type for reading transform inputs \iterator
    typename            OutputIteratorT,         ///< Output iterator type for recording the reduced aggregate \iterator
    typename            ScanTileStateT,          ///< Tile status interface type
    typename            ScanOpT,                 ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            ReductionOpT,            ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename            TransformOpT,            ///< Transformation functor on scan output
    typename            InitValueT,              ///< Initial value to seed the exclusive scan (cub::NullType for inclusive scans)
    typename            OffsetT>                 ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanReducePolicyT::BLOCK_THREADS))
__global__ void DeviceScanReduceKernel(
    ScanInputIteratorT      d_in,               ///< Input data
    TransformInputIteratorT d_trans_in,         ///< Input data
    OutputIteratorT         d_block_sum,        ///< Output data (block reduction)
    ScanTileStateT          tile_state,         ///< Tile status interface
    int                     start_tile,         ///< The starting tile for the current grid
    ScanOpT                 scan_op,            ///< Binary scan functor
    ReductionOpT            reduction_op,       ///< Binary reduction functor
    TransformOpT            transform_op,       ///< Transformation functor on scan output
    InitValueT              init_value,         ///< Initial value to seed the exclusive scan
    OffsetT                 num_items)          ///< Total number of scan items for the entire problem
{
    using RealInitValueT = typename InitValueT::value_type;
    typedef typename ChainedPolicyT::ActivePolicy::ScanReducePolicyT ScanReducePolicyT;

    // Thread block type for scanning input tiles
    typedef AgentScanReduce<
        ScanReducePolicyT,
        ScanInputIteratorT,
        TransformInputIteratorT,
        OutputIteratorT,
        ScanOpT,
        ReductionOpT,
        TransformOpT,
        RealInitValueT,
        OffsetT> AgentScanReduceT;

    // Shared memory for AgentScan
    __shared__ typename AgentScanReduceT::TempStorage temp_storage;

    RealInitValueT real_init_value = init_value;

    // Process tiles
    AgentScanReduceT(temp_storage, d_in, d_trans_in, d_block_sum, scan_op, reduction_op, transform_op, real_init_value).ConsumeRange(
        num_items,
        tile_state,
        start_tile);
}

/**
 * Reduce a single tile kernel entry point (single-block).  Can be used to aggregate privatized thread block reductions from a previous multi-block reduction pass.
 */
template <
    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                ScanInputIteratorT,         ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                ReductionOpT,               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename                OutputT>                    ///< Data element type that is convertible to the \p value type of \p OutputIteratorT
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanReducePolicyT::BLOCK_THREADS), 1)
__global__ void DeviceScanReduceSingleTileKernel(
    ScanInputIteratorT      d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    OffsetT                 num_items,                  ///< [in] Total number of input data items
    ReductionOpT            reduction_op,               ///< [in] Binary reduction functor
    OutputT                 init)                       ///< [in] The initial value of the reduction
{
    // Thread block type for reducing input tiles
    typedef AgentReduce<
            typename ChainedPolicyT::ActivePolicy::ScanReducePolicyT,
            ScanInputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT>
        AgentReduceT;

    // Shared memory storage
    __shared__ typename AgentReduceT::TempStorage temp_storage;

    // Check if empty problem
    if (num_items == 0)
    {
        if (threadIdx.x == 0)
            *d_out = init;
        return;
    }

    // Consume input tiles
    OutputT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(
        OffsetT(0),
        num_items);

    // Output result
    if (threadIdx.x == 0)
        *d_out = reduction_op(init, block_aggregate);
}

/******************************************************************************
 * Policy
 ******************************************************************************/

template <
    typename ScanInputT> ///< Data type
struct DeviceScanReducePolicy
{
    // For large values, use timesliced loads/stores to fit shared memory.
    static constexpr bool LargeValues = sizeof(ScanInputT) > 128;
    static constexpr BlockLoadAlgorithm ScanTransposedLoad =
      LargeValues ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED
                  : BLOCK_LOAD_WARP_TRANSPOSE;
    static constexpr BlockStoreAlgorithm1 ScanTransposedStore =
      LargeValues ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED1
                  : BLOCK_STORE_WARP_TRANSPOSE1;

    /// SM350
    struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
    {
        // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
        typedef AgentScanReducePolicy<
                128, 12,                                        ///< Threads per block, items per thread
                ScanInputT,
                4,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED1,
                BLOCK_SCAN_RAKING,
                BLOCK_REDUCE_WARP_REDUCTIONS>
            ScanReducePolicyT;

        // SingleTilePolicy
        typedef ScanReducePolicyT SingleTilePolicy;
    };

    /// SM520
    struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
    {
        // Titan X: 32.47B items/s @ 48M 32-bit T
        typedef AgentScanReducePolicy<
                128, 12,                                        ///< Threads per block, items per thread
                ScanInputT,
                4,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                ScanTransposedStore,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_REDUCE_WARP_REDUCTIONS>
            ScanReducePolicyT;

        // SingleTilePolicy
        typedef ScanReducePolicyT SingleTilePolicy;
    };

    /// SM600
    struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
    {
        typedef AgentScanReducePolicy<
                128, 15,                                        ///< Threads per block, items per thread
                ScanInputT,
                4,
                ScanTransposedLoad,
                LOAD_DEFAULT,
                ScanTransposedStore,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_REDUCE_WARP_REDUCTIONS>
            ScanReducePolicyT;

        // SingleTilePolicy
        typedef ScanReducePolicyT SingleTilePolicy;
    };

    /// MaxPolicy
    typedef Policy600 MaxPolicy;
};


/******************************************************************************
 * Dispatch
 ******************************************************************************/


/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename ScanInputIteratorT,      ///< Random-access input iterator type for reading scan inputs \iterator
    typename TransformInputIteratorT, ///< Random-access input iterator type for reading transformation inputs \iterator
    typename OutputIteratorT,         ///< Output iterator type for recording the reduced aggregate \iterator
    typename ScanOpT,                 ///< Binary scan functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename ReductionOpT,            ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename TransformOpT,            ///< Transformation functor on scan output
    typename InitValueT,              ///< The init_value element type for ScanOpT (cub::NullType for inclusive scans)
    typename OffsetT,                 ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceScanReducePolicy<
      // Accumulator type.
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 cub::detail::value_t<ScanInputIteratorT>,
                                 typename InitValueT::value_type>>>
struct DispatchScanReduce:
    SelectedPolicy
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The input value type
    using InputT = cub::detail::value_t<ScanInputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT::value_type if provided, otherwise the
    // input iterator's value type.
    using ScanOutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 InputT,
                                 typename InitValueT::value_type>;
    using ReduceOutputT =
      cub::detail::non_void_value_t<OutputIteratorT,
                                        cub::detail::value_t<ScanInputIteratorT>>;


    void*                    d_temp_storage;         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&                  temp_storage_bytes;     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    ScanInputIteratorT       d_in;                   ///< [in] Iterator to the input sequence of data items
    TransformInputIteratorT  d_trans_in;             ///< [in] Iterator to the additional input sequence of data items
    OutputIteratorT          d_sum;                  ///< [out] Pointer to the output aggregate
    ScanOpT                  scan_op;                ///< [in] Binary scan functor
    ReductionOpT             reduction_op;           ///< [in] Binary reduction functor
    TransformOpT             transform_op;           ///< [in] Transformation functor on scan output
    InitValueT               init_value;             ///< [in] Initial value to seed the exclusive scan
    OffsetT                  num_items;              ///< [in] Total number of input items (i.e., the length of \p d_in)
    cudaStream_t             stream;                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                     debug_synchronous;
    int                      ptx_version;

    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchScanReduce(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ScanInputIteratorT      d_in,                   ///< [in] Iterator to the input sequence of data items
        TransformInputIteratorT d_trans_in,             ///< [in] Pointer to the additional input sequence of data
        OutputIteratorT         d_sum,                  ///< [out] Pointer to the output aggregate
        OffsetT                 num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        ScanOpT                 scan_op,                ///< [in] Binary scan functor
        ReductionOpT            reduction_op,           ///< [in] Binary reduction functor
        TransformOpT            transform_op,           ///< [in] Transformation functor on scan output
        InitValueT              init_value,             ///< [in] Initial value to seed the exclusive scan
        cudaStream_t            stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous,
        int                     ptx_version
    ):
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_in(d_in),
        d_trans_in(d_trans_in),
        d_sum(d_sum),
        scan_op(scan_op),
        reduction_op(reduction_op),
        transform_op(transform_op),
        init_value(init_value),
        num_items(num_items),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}

    template <typename ActivePolicyT, typename InitKernel, typename ScanReduceKernel, typename DeviceReduceSingleTileKernel>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke(InitKernel init_kernel, ScanReduceKernel scan_reduce_kernel, DeviceReduceSingleTileKernel single_tile_kernel)
    {
#ifndef CUB_RUNTIME_ENABLED

        (void)init_kernel;
        (void)scan_reduce_kernel;
        (void)single_tile_kernel;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else

        typedef typename ActivePolicyT::ScanReducePolicyT Policy;
        typedef typename cub::ScanTileState<InputT> ScanTileStateT;

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
            int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Run grids in epochs (in case number of tiles exceeds max x-dimension
            int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[2];
            if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors
            allocation_sizes[1] =
            {
                    scan_grid_size * sizeof(ReduceOutputT)    // bytes needed for privatized block reductions (reduction)
            };

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void* allocations[2] = {};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }
            ReduceOutputT* d_block_sum = (ReduceOutputT*) allocations[1];

            // Return if empty problem
            if (num_items == 0)
                break;

            // Construct the tile status interface
            ScanTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(init_kernel,
                tile_state,
                num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;


            // Get SM occupancy for scan_kernel
            int scan_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                scan_sm_occupancy,            // out
                scan_reduce_kernel,
                Policy::BLOCK_THREADS))) break;

            for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
            {
                // Log scan_reduce_kernel configuration
                if (debug_synchronous) _CubLog("Invoking %d scan_reduce_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    start_tile, scan_grid_size, Policy::BLOCK_THREADS, (long long) stream, Policy::ITEMS_PER_THREAD, scan_sm_occupancy);

                // Invoke scan_reduce_kernel
                THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                    scan_grid_size, Policy::BLOCK_THREADS, 0, stream
                ).doit(scan_reduce_kernel,
                    d_in,
                    d_trans_in,
                    d_block_sum,
                    tile_state,
                    start_tile,
                    scan_op,
                    reduction_op,
                    transform_op,
                    init_value,
                    num_items);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }

            // Log single_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), %d items per thread\n",
                Policy::BLOCK_THREADS,
                (long long) stream,
                Policy::ITEMS_PER_THREAD);

            // Invoke DeviceReduceSingleTileKernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                1, Policy::BLOCK_THREADS, 0, stream
            ).doit(single_tile_kernel,
                d_block_sum,
                d_sum,
                scan_grid_size,
                reduction_op,
                ReduceOutputT());

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke()
    {
        typedef typename DispatchScanReduce::MaxPolicy MaxPolicyT;
        typedef typename cub::ScanTileState<InputT> ScanTileStateT;
        typedef typename ActivePolicyT::SingleTilePolicy SingleTilePolicyT;

        // Ensure kernels are instantiated.
        return Invoke<ActivePolicyT>(
            DeviceScanReduceInitKernel<ScanTileStateT>,
            DeviceScanReduceKernel<MaxPolicyT, ScanInputIteratorT, TransformInputIteratorT, ReduceOutputT*, ScanTileStateT, ScanOpT, ReductionOpT, TransformOpT, InitValueT, OffsetT>,
            DeviceScanReduceSingleTileKernel<MaxPolicyT, ReduceOutputT*, OutputIteratorT, OffsetT, ReductionOpT, ReduceOutputT>
        );
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        ScanInputIteratorT      d_in,                   ///< [in] Iterator to the input sequence of data items
        TransformInputIteratorT d_trans_in,             ///< [in] Iterator to the additional input sequence of data items
        OutputIteratorT         d_sum,                  ///< [out] Pointer to the output aggregate
        ScanOpT                 scan_op,                ///< [in] Binary scan functor
        ReductionOpT            reduction_op,           ///< [in] Binary reduction functor
        TransformOpT            transform_op,           ///< [in] Transformation functor on scan output
        InitValueT              init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT                 num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t            stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                    debug_synchronous)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename DispatchScanReduce::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchScanReduce dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_trans_in,
                d_sum,
                num_items,
                scan_op,
                reduction_op,
                transform_op,
                init_value,
                stream,
                debug_synchronous,
                ptx_version
            );
            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};



CUB_NAMESPACE_END
